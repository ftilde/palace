use std::fmt::{Display, Write};

use ash::vk;
use futures::StreamExt;
use id::{Id, Identify};

use crate::{
    array::TensorMetaData,
    dim::DynDimension,
    dtypes::{DType, ElementType, ScalarType},
    operator::OperatorDescriptor,
    operators::tensor::TensorOperator,
    storage::gpu::{InplaceHandle, InplaceResult, WriteHandle},
    task::{Request, RequestStream},
    vulkan::{
        pipeline::{AsDescriptors, ComputePipeline, DescriptorConfig},
        shader::{Config, ShaderDefines},
        state::RessourceId,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

#[derive(id::Identify, Clone, Copy)]
pub enum UnaryOp {
    Abs,
    Neg,
    Cast(DType),
}

impl UnaryOp {
    fn dtype(&self, input: DType) -> Result<DType, crate::Error> {
        Ok(match self {
            UnaryOp::Abs => input,
            UnaryOp::Cast(output) => {
                if input.vec_size() == output.vec_size() {
                    *output
                } else {
                    return Err(format!("{:?} cannot be converted to {:?}", input, output).into());
                }
            }
            UnaryOp::Neg => match input.scalar {
                ScalarType::U8 | ScalarType::U16 | ScalarType::U32 => {
                    return Err(format!("Value of type {:?} cannot be negated", input).into())
                }
                ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::F32 => input,
            },
        })
    }
}
struct WriteUnary(UnaryOp, NodeId, u32);
impl Display for WriteUnary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = WriteValue(self.1, self.2);
        match self.0 {
            UnaryOp::Abs => write!(f, "abs({})", v),
            UnaryOp::Cast(output) => write!(f, "{}({})", output.scalar.glsl_type(), v),
            UnaryOp::Neg => write!(f, "-{}", v),
        }
    }
}

#[derive(id::Identify, Clone, Copy)]
pub enum BinOp {
    Add,
    Mul,
    Max,
}

impl BinOp {
    fn dtype(&self, input1: DType, input2: DType) -> Result<DType, crate::Error> {
        if input1 == input2 {
            Ok(input1)
        } else {
            Err(format!(
                "Mismatched dtypes {:?} and {:?} for binary op",
                input1, input2
            )
            .into())
        }
    }
}

const VEC_LOOP_VARIABLE_NAME: &'static str = "i";
struct VecLoop(u32);
impl Display for VecLoop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == 1 {
            Ok(())
        } else {
            write!(f, "for(int i=0; i<{}; ++i)", self.0)
        }
    }
}

struct WriteValue(NodeId, u32);
impl Display for WriteValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.1 == 1 {
            write!(f, "{}", self.0)
        } else {
            write!(f, "{}[{}]", self.0, VEC_LOOP_VARIABLE_NAME)
        }
    }
}

struct WriteBin(BinOp, NodeId, NodeId, u32);
impl Display for WriteBin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let l = WriteValue(self.1, self.3);
        let r = WriteValue(self.2, self.3);
        match self.0 {
            BinOp::Add => write!(f, "{} + {}", l, r),
            BinOp::Mul => write!(f, "{} * {}", l, r),
            BinOp::Max => write!(f, "max({}, {})", l, r),
        }
    }
}

#[derive(id::Identify, Clone, Copy)]
pub enum ConstValue {
    F32(f32),
}

impl Display for ConstValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstValue::F32(v) => write!(f, "{}", v),
        }
    }
}

impl ConstValue {
    fn dtype(&self) -> DType {
        match self {
            ConstValue::F32(_) => DType::scalar(ScalarType::F32),
        }
    }
}

#[derive(id::Identify, Clone)]
enum Node {
    Const(ConstValue),
    UnaryOp(DType, UnaryOp, NodeId),
    BinOp(DType, BinOp, NodeId, NodeId),
    Read(DType, InputId),
}

#[derive(Clone)]
struct OrderedSet<T>(Vec<T>);

impl<T: Identify> OrderedSet<T> {
    fn add(&mut self, item: T) -> usize {
        let pos = if let Some(pos) = self
            .0
            .iter()
            .enumerate()
            .find_map(|(i, n)| (n.id() == item.id()).then_some(i))
        {
            pos
        } else {
            let pos = self.0.len();
            self.0.push(item);
            pos
        };
        pos
    }

    fn merge(mut self, other: OrderedSet<T>) -> Self {
        for l in other.0.into_iter() {
            self.add(l);
        }
        self
    }
}

//TODO: Rc for cheap clone?
#[derive(Clone)]
pub struct JitTensorOperator<D: DynDimension> {
    root: NodeId,
    metadata: Option<TensorMetaData<D>>,
    dtype: DType,
    operators: OrderedSet<TensorOperator<D, DType>>,
    nodes: OrderedSet<Node>,
}

impl<D: DynDimension> id::Identify for JitTensorOperator<D> {
    fn id(&self) -> Id {
        // Note: All information (including operators via there ids) is present in the operation
        // tree `node`
        self.root.0
    }
}

impl<D: DynDimension> JitTensorOperator<D> {
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn metadata(&self) -> Option<TensorMetaData<D>> {
        self.metadata.clone()
    }
    pub fn with_md(mut self, new_md: TensorMetaData<D>) -> Result<Self, crate::Error> {
        self.metadata = if let Some(md) = self.metadata {
            if new_md == md {
                Some(md)
            } else {
                return Err(
                    format!("Mismatch with existing metadata {:?}, {:?}", md, new_md,).into(),
                );
            }
        } else {
            Some(new_md)
        };
        Ok(self)
    }
    pub fn unary_op(op: UnaryOp, inner: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        if let UnaryOp::Cast(target) = &op {
            if *target == inner.dtype {
                return Ok(inner);
            }
        }

        Ok({
            let dtype = op.dtype(inner.dtype)?;
            let node = Node::UnaryOp(dtype, op, inner.root);
            let root = NodeId(node.id());
            let mut nodes = inner.nodes;
            nodes.add(node);
            Self {
                root,
                metadata: inner.metadata.clone(),
                dtype,
                operators: inner.operators,
                nodes,
            }
        })
    }
    pub fn bin_op(
        op: BinOp,
        l: JitTensorOperator<D>,
        r: JitTensorOperator<D>,
    ) -> Result<Self, crate::Error> {
        Ok({
            let dtype = op.dtype(l.dtype, r.dtype)?;

            let node = Node::BinOp(dtype, op, l.root, r.root);
            let root = NodeId(node.id());
            let mut nodes = l.nodes.merge(r.nodes);
            nodes.add(node);
            Self {
                metadata: match (l.metadata, r.metadata) {
                    (None, None) => None,
                    (None, r) => r,
                    (l, None) => l,
                    (l, r) => {
                        if l == r {
                            l
                        } else {
                            return Err(format!("Metadata mismatch {:?}, {:?}", l, r).into());
                        }
                    }
                },
                dtype,
                operators: l.operators.merge(r.operators),
                nodes,
                root,
            }
        })
    }
}

impl<D: DynDimension> From<ConstValue> for JitTensorOperator<D> {
    fn from(c: ConstValue) -> Self {
        let node = Node::Const(c);
        let root = NodeId(node.id());
        let nodes = OrderedSet(vec![node]);
        Self {
            root,
            nodes,
            metadata: None,
            dtype: c.dtype(),
            operators: OrderedSet(Vec::new()),
        }
    }
}

impl<D: DynDimension> From<f32> for JitTensorOperator<D> {
    fn from(value: f32) -> Self {
        ConstValue::F32(value).into()
    }
}

impl<D: DynDimension> From<TensorOperator<D, DType>> for JitTensorOperator<D> {
    fn from(c: TensorOperator<D, DType>) -> Self {
        let dtype = c.chunks.dtype();
        let id = c.chunks.id();
        let metadata = Some(c.metadata.clone());

        let node = Node::Read(dtype, InputId(id));
        let root = NodeId(node.id());
        let nodes = OrderedSet(vec![node]);
        Self {
            root,
            nodes,
            metadata,
            dtype,
            operators: OrderedSet(vec![c]),
        }
    }
}

impl<D: DynDimension> JitTensorOperator<D> {
    pub fn add(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::Add, self, other)
    }
    pub fn mul(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::Mul, self, other)
    }
    pub fn max(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::Max, self, other)
    }

    pub fn abs(self) -> Result<Self, crate::Error> {
        Self::unary_op(UnaryOp::Abs, self)
    }
    pub fn neg(self) -> Result<Self, crate::Error> {
        Self::unary_op(UnaryOp::Neg, self)
    }
    pub fn cast(self, to: DType) -> Result<Self, crate::Error> {
        Self::unary_op(UnaryOp::Cast(to), self)
    }
}

#[derive(id::Identify, Clone)]
struct InputId(Id);
impl Display for InputId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "input{}", self.0.raw())
    }
}

#[derive(id::Identify, Copy, Clone)]
struct NodeId(Id);

impl Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0.raw())
    }
}

pub fn jit<D: DynDimension>(op: TensorOperator<D, DType>) -> JitTensorOperator<D> {
    op.into()
}

fn compile<D: DynDimension>(
    nodes: &Vec<Node>,
    root: NodeId,
    root_dtype: DType,
    inputs: &Vec<TensorOperator<D, DType>>,
) -> Result<(String, Config), crate::Error> {
    let mut shader = String::new();
    let mut config = Config::new();

    writeln!(&mut shader, "layout (local_size_x = 256) in;")?;

    for (i, input) in inputs.iter().enumerate() {
        writeln!(
            &mut shader,
            r#"
            layout(std430, binding = {}) readonly buffer InputBuffer{}{{
                {} values[BRICK_MEM_SIZE];
            }} input{};
            "#,
            i,
            i,
            input.chunks.dtype().glsl_type(),
            i,
        )?;
    }

    writeln!(
        &mut shader,
        r#"
            layout(std430, binding = {}) buffer OutputBuffer{{
                {} values[BRICK_MEM_SIZE];
            }} output_buf;
            "#,
        inputs.len(),
        root_dtype.glsl_type(),
    )?;

    writeln!(
        &mut shader,
        r#"
            void main() {{
                uint gID = gl_GlobalInvocationID.x;

                if(gID < BRICK_MEM_SIZE) {{
            "#
    )?;

    for (i, input) in inputs.iter().enumerate() {
        writeln!(
            &mut shader,
            "{} {} = input{}.values[gID];",
            input.chunks.dtype().glsl_type(),
            InputId(input.chunks.id()),
            i
        )?;
    }

    for node in nodes {
        let res_id = NodeId(node.id());
        let dtype = match node {
            Node::Const(c) => {
                writeln!(&mut shader, "{} {} = {};", c.dtype().glsl_type(), res_id, c)?;
                c.dtype()
            }
            Node::UnaryOp(t, o, a) => {
                //let param = translate(a, w, value_ids)?;
                writeln!(
                    &mut shader,
                    "{} {}; {} {} = {};",
                    t.glsl_type(),
                    res_id,
                    VecLoop(t.size),
                    WriteValue(res_id, t.size),
                    WriteUnary(*o, *a, t.size)
                )?;
                *t
            }
            Node::BinOp(t, o, l, r) => {
                //let param_l = translate(l, w, value_ids)?;
                //let param_r = translate(r, w, value_ids)?;
                writeln!(
                    &mut shader,
                    "{} {}; {} {} = {};",
                    t.glsl_type(),
                    res_id,
                    VecLoop(t.size),
                    WriteValue(res_id, t.size),
                    WriteBin(*o, *l, *r, t.size)
                )?;
                *t
            }
            Node::Read(t, v) => {
                writeln!(&mut shader, "{} {} = {};", t.glsl_type(), res_id, v)?;
                *t
            }
        };
        config = config.ext(dtype.glsl_ext());
    }

    writeln!(
        &mut shader,
        r#"
                    output_buf.values[gID] = {};
                }}
            }}
            "#,
        root,
    )?;

    Ok((shader, config))
}
impl<D: DynDimension> JitTensorOperator<D> {
    pub fn compile(mut self) -> Result<TensorOperator<D, DType>, crate::Error> {
        enum MaybeInplaceResult<'a, 'inv> {
            Inplace(InplaceResult<'a, 'inv>),
            NotInplace,
        }
        enum MaybeInplaceHandle<'a> {
            Inplace(InplaceHandle<'a>),
            NotInplace(WriteHandle<'a>),
        }

        let dtype = self.dtype;
        let Some(metadata) = self.metadata.clone() else {
            return Err("No metadata information in JitOperator".into());
        };

        if let &[Node::Read(_, _)] = self.nodes.0.as_slice() {
            assert_eq!(self.operators.0.len(), 1);
            return Ok(self.operators.0.pop().unwrap());
        }

        let inplace_operator_index = self
            .operators
            .0
            .iter()
            .position(|o| o.dtype().element_layout() == dtype.element_layout());

        Ok(TensorOperator::with_state(
            OperatorDescriptor::new("jit").dependent_on_data(&self),
            dtype,
            metadata.clone(),
            (self, metadata),
            move |ctx, positions, (jit_operator, metadata)| {
                async move {
                    let device = ctx.preferred_device();

                    let access_info = DstBarrierInfo {
                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                    };
                    let m = metadata.clone().into_dyn();

                    let num_chunk_elements = m.num_chunk_elements();

                    let pipeline = device.request_state(
                        RessourceId::new("pipeline")
                            .of(ctx.current_op())
                            .dependent_on(jit_operator),
                        || {
                            let (shader, config) = compile(
                                &jit_operator.nodes.0,
                                jit_operator.root,
                                dtype,
                                &jit_operator.operators.0,
                            )?;
                            //println!("{}", shader.as_str());
                            ComputePipeline::new(
                                device,
                                (
                                    shader.as_str(),
                                    ShaderDefines::new().add("BRICK_MEM_SIZE", num_chunk_elements),
                                    config,
                                ),
                                true,
                            )
                        },
                    )?;

                    let mut brick_stream = ctx
                        .submit_unordered_with_data(positions.iter().map(|(pos, _)| {
                            (
                                ctx.group(
                                    jit_operator
                                        .operators
                                        .0
                                        .iter()
                                        .enumerate()
                                        .filter(|(i, _)| Some(*i) != inplace_operator_index)
                                        .map(|(_, input)| {
                                            input.chunks.request_gpu(device.id, *pos, access_info)
                                        }),
                                ),
                                *pos,
                            )
                        }))
                        .then_req_with_data(*ctx, |(input, pos)| {
                            if let Some(inplace_operator_index) = inplace_operator_index {
                                let output = jit_operator.operators.0[inplace_operator_index]
                                    .chunks
                                    .request_inplace_gpu(
                                        device.id,
                                        pos,
                                        ctx.current_op_desc().unwrap(),
                                        dtype,
                                        num_chunk_elements,
                                        access_info,
                                    )
                                    .map(|v| MaybeInplaceResult::Inplace(v));
                                (output, (input, pos))
                            } else {
                                //let output = ctx
                                //    .alloc_slot_gpu(device, pos, num_chunk_elements)
                                //    .map(|v| MaybeInplaceResult::NotInplace(v));
                                let output = Request::ready(MaybeInplaceResult::NotInplace);
                                (output, (input, pos))
                            }
                        })
                        .then_req_with_data(*ctx, |(output, (inputs, pos))| {
                            let output = match output {
                                MaybeInplaceResult::Inplace(v) => {
                                    v.alloc().map(|v| MaybeInplaceHandle::Inplace(v))
                                }
                                MaybeInplaceResult::NotInplace => ctx
                                    .alloc_slot_gpu(device, pos, num_chunk_elements)
                                    .map(|v| MaybeInplaceHandle::NotInplace(v)),
                            };
                            (output, inputs)
                        });

                    while let Some((output, inputs)) = brick_stream.next().await {
                        device.with_cmd_buffer(|cmd| {
                            let mut descriptors: Vec<&dyn AsDescriptors> = Vec::new();
                            match &output {
                                MaybeInplaceHandle::Inplace(h) => {
                                    let i = inplace_operator_index.unwrap();
                                    descriptors.extend(
                                        inputs[..i].iter().map(|v| v as &dyn AsDescriptors),
                                    );
                                    match h {
                                        InplaceHandle::Inplace(rw, _) => descriptors.push(rw),
                                        InplaceHandle::New(r, _) => descriptors.push(r),
                                    }
                                    descriptors.extend(
                                        inputs[i..].iter().map(|v| v as &dyn AsDescriptors),
                                    );
                                    match h {
                                        InplaceHandle::Inplace(rw, _) => descriptors.push(rw),
                                        InplaceHandle::New(_, w) => descriptors.push(w),
                                    }
                                }
                                MaybeInplaceHandle::NotInplace(w) => {
                                    descriptors
                                        .extend(inputs.iter().map(|v| v as &dyn AsDescriptors));
                                    descriptors.push(w);
                                }
                            };

                            let descriptor_config = DescriptorConfig::from_vec(descriptors);

                            let global_size = num_chunk_elements;

                            unsafe {
                                let mut pipeline = pipeline.bind(cmd);

                                pipeline.push_descriptor_set(0, descriptor_config);
                                pipeline.dispatch(global_size);
                            }
                        });

                        unsafe {
                            match output {
                                MaybeInplaceHandle::Inplace(h) => h.initialized(
                                    *ctx,
                                    SrcBarrierInfo {
                                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                        access: vk::AccessFlags2::SHADER_WRITE,
                                    },
                                ),
                                MaybeInplaceHandle::NotInplace(w) => w.initialized(
                                    *ctx,
                                    SrcBarrierInfo {
                                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                        access: vk::AccessFlags2::SHADER_WRITE,
                                    },
                                ),
                            }
                        };
                    }

                    Ok(())
                }
                .into()
            },
        ))
    }
}

impl<D: DynDimension> TryInto<TensorOperator<D, DType>> for JitTensorOperator<D> {
    type Error = crate::Error;
    fn try_into(self) -> Result<TensorOperator<D, DType>, Self::Error> {
        self.compile()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        test_util::compare_tensor,
        vec::{LocalVoxelPosition, VoxelPosition},
    };

    #[test]
    fn two_inputs() {
        let s = [4, 4, 4];
        let size = VoxelPosition::from(s);
        let brick_size = LocalVoxelPosition::from(s);

        let input_fn_1 = |v: VoxelPosition| (v.x().raw * v.y().raw * v.z().raw) as f32;
        let input_fn_2 = |v: VoxelPosition| (v.x().raw + v.y().raw + v.z().raw) as f32;

        let input1 =
            jit(crate::operators::rasterize_function::voxel(size, brick_size, input_fn_1).into());
        let input2 =
            jit(crate::operators::rasterize_function::voxel(size, brick_size, input_fn_2).into());

        let output = input1
            .cast(ScalarType::I32.into())
            .unwrap()
            .neg()
            .unwrap()
            .add(
                input2
                    .clone()
                    .cast(ScalarType::I32.into())
                    .unwrap()
                    .mul(input2.cast(ScalarType::I32.into()).unwrap())
                    .unwrap(),
            )
            .unwrap()
            .cast(ScalarType::F32.into())
            .unwrap()
            .compile()
            .unwrap();

        let expected = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            let input1 = input_fn_1(v);
            let input2 = input_fn_2(v);

            (-(input1 as i32) + (input2 as i32) * (input2 as i32)) as f32
        });

        compare_tensor(output.try_into().unwrap(), expected);
    }
}
