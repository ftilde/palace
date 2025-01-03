use std::fmt::{Display, Write};

use ash::vk;
use futures::StreamExt;
use id::{Id, Identify};

use crate::{
    array::TensorMetaData,
    dim::DynDimension,
    dtypes::{DType, ElementType, ScalarType},
    op_descriptor,
    operator::{DataParam, OperatorDescriptor, OperatorParameter},
    operators::tensor::TensorOperator,
    storage::gpu::{InplaceHandle, InplaceResult, WriteHandle},
    task::{Request, RequestStream},
    vulkan::{
        pipeline::{AsDescriptors, ComputePipelineBuilder, DescriptorConfig},
        shader::{Config, Shader},
        DstBarrierInfo, SrcBarrierInfo,
    },
};

#[derive(id::Identify, Clone, Copy, Debug)]
pub enum UnaryOp {
    Abs,
    Neg,
    Cast(DType),
    Index(u32),
    Splat(u32),
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
            UnaryOp::Index(i) => {
                if *i < input.size {
                    input.scalar.into()
                } else {
                    return Err(format!(
                        "Index {} out of range for vector value of size {}",
                        i, input.size
                    )
                    .into());
                }
            }
            UnaryOp::Splat(size) => {
                if input.size == 1 {
                    DType {
                        scalar: input.scalar,
                        size: *size,
                    }
                } else {
                    return Err(
                        format!("Cannot splat non-scalar value (dim {})", input.size).into(),
                    );
                }
            }
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
            UnaryOp::Index(i) => write!(f, "{}[{}]", v, i),
            UnaryOp::Splat(_i) => write!(f, "{}", WriteValue(self.1, 1)),
        }
    }
}

#[derive(id::Identify, Clone, Copy, Debug)]
pub enum BinOp {
    Add,
    Mul,
    Sub,
    Div,
    Max,
    Min,
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
            BinOp::Sub => write!(f, "{} - {}", l, r),
            BinOp::Mul => write!(f, "{} * {}", l, r),
            BinOp::Div => write!(f, "{} / {}", l, r),
            BinOp::Max => write!(f, "max({}, {})", l, r),
            BinOp::Min => write!(f, "min({}, {})", l, r),
        }
    }
}

#[derive(id::Identify, Clone, Copy, Debug)]
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

#[derive(id::Identify, Clone, Debug)]
enum Instruction {
    Const(ConstValue),
    Unary(DType, UnaryOp, InstructionOffset),
    Binary(DType, BinOp, InstructionOffset, InstructionOffset),
    Read(DType, InputId),
}

impl Instruction {
    fn dtype(&self) -> DType {
        match self {
            Instruction::Const(const_value) => const_value.dtype(),
            Instruction::Unary(dtype, ..) => *dtype,
            Instruction::Binary(dtype, ..) => *dtype,
            Instruction::Read(dtype, ..) => *dtype,
        }
    }
}

#[derive(Clone, Identify)]
struct OrderedSet<T: Identify>(Vec<T>);

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
}

//TODO: Rc for cheap clone?
#[derive(Identify, Clone)]
pub struct JitTensorOperator<D: DynDimension> {
    metadata: Option<TensorMetaData<D>>,
    dtype: DType,
    operators: OrderedSet<TensorOperator<D, DType>>,
    instructions: Vec<Instruction>,
}

impl<D: DynDimension> OperatorParameter for JitTensorOperator<D> {
    fn data_longevity(&self) -> crate::storage::DataLongevity {
        self.operators
            .0
            .iter()
            .map(|o| o.data_longevity())
            .min()
            .unwrap_or(crate::storage::DataLongevity::Stable)
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
            let op = Instruction::Unary(dtype, op, InstructionOffset(1));
            let mut ops = inner.instructions;
            ops.push(op);
            Self {
                metadata: inner.metadata.clone(),
                dtype,
                operators: inner.operators,
                instructions: ops,
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

            let mut inputs = l.operators;
            let old_inputs = r.operators;

            let mut ops = l.instructions;
            let mut new_ops = r.instructions;
            for op in &mut new_ops {
                if let Instruction::Read(_, input_id) = op {
                    *input_id = InputId(inputs.add(old_inputs.0[input_id.0].clone()));
                }
            }

            let op = Instruction::Binary(
                dtype,
                op,
                InstructionOffset(new_ops.len() + 1),
                InstructionOffset(1),
            );
            ops.extend(new_ops);
            ops.push(op);

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
                operators: inputs,
                instructions: ops,
            }
        })
    }
}

impl<D: DynDimension> From<ConstValue> for JitTensorOperator<D> {
    fn from(c: ConstValue) -> Self {
        let op = Instruction::Const(c);
        let ops = vec![op];
        Self {
            instructions: ops,
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
        let metadata = Some(c.metadata.clone());

        let op = Instruction::Read(dtype, InputId(0));
        let ops = vec![op];
        Self {
            instructions: ops,
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
    pub fn sub(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::Sub, self, other)
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

#[derive(id::Identify, Clone, Debug)]
struct InputId(usize);

impl Display for InputId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "input{}", self.0)
    }
}

#[derive(Copy, Clone)]
struct NodeId(usize);

impl Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

#[derive(id::Identify, Copy, Clone, Debug)]
struct InstructionOffset(usize);

pub fn jit<D: DynDimension>(op: TensorOperator<D, DType>) -> JitTensorOperator<D> {
    op.into()
}

fn compile(
    instructions: &Vec<Instruction>,
    input_types: &Vec<DType>,
) -> Result<(String, Config), crate::Error> {
    let mut shader = String::new();
    let mut config = Config::new();

    writeln!(&mut shader, "#include<size_util.glsl>")?;

    for (i, dtype) in input_types.iter().enumerate() {
        writeln!(
            &mut shader,
            r#"
            layout(std430, binding = {}) readonly buffer InputBuffer{}{{
                {} values[BRICK_MEM_SIZE];
            }} input{};
            "#,
            i,
            i,
            dtype.glsl_type(),
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
        input_types.len(),
        instructions.last().unwrap().dtype().glsl_type(),
    )?;

    writeln!(
        &mut shader,
        r#"
            void main() {{
                uint gID = global_position_linear;

                if(gID < BRICK_MEM_SIZE) {{
            "#
    )?;

    for (i, dtype) in input_types.iter().enumerate() {
        writeln!(
            &mut shader,
            "{} {} = input{}.values[gID];",
            dtype.glsl_type(),
            InputId(i),
            i
        )?;
    }

    struct Node<'a> {
        instr: &'a Instruction,
        id: Id,
        instruction_number: usize,
    }

    impl Identify for Node<'_> {
        fn id(&self) -> Id {
            self.id
        }
    }

    // Perform a collapse-of-identical-instructions optimization step:
    // Calculate an id for all instructions and, based on that collect them in a new list
    // ("nodes"), which automatically deduplicates.

    let mut nodes: OrderedSet<Node> = OrderedSet(Vec::new());
    let mut instruction_to_node: Vec<NodeId> = Vec::new();

    for (i, instr) in instructions.iter().enumerate() {
        let id = match instr {
            Instruction::Const(const_value) => const_value.id(),
            Instruction::Unary(dtype, unary_op, offset) => {
                let node_num = instruction_to_node[i - offset.0];
                Id::combine(&[dtype.id(), unary_op.id(), nodes.0[node_num.0].id])
            }
            Instruction::Binary(dtype, bin_op, offset_l, offset_r) => {
                let node_num_l = instruction_to_node[i - offset_l.0];
                let node_num_r = instruction_to_node[i - offset_r.0];
                Id::combine(&[
                    dtype.id(),
                    bin_op.id(),
                    nodes.0[node_num_l.0].id,
                    nodes.0[node_num_r.0].id,
                ])
            }
            Instruction::Read(dtype, input_id) => Id::combine(&[dtype.id(), input_id.id()]),
        };

        let node_id = nodes.add(Node {
            instr: &instr,
            id,
            instruction_number: i,
        });

        instruction_to_node.push(NodeId(node_id));
    }

    let input_node_id = |own_instruction_number: usize, input_offset: InstructionOffset| {
        let input_instruction_number = own_instruction_number - input_offset.0;
        instruction_to_node[input_instruction_number]
    };

    for (i, node) in nodes.0.iter().enumerate() {
        let res_id = NodeId(i);
        let dtype = match node.instr {
            Instruction::Const(c) => {
                writeln!(&mut shader, "{} {} = {};", c.dtype().glsl_type(), res_id, c)?;
                c.dtype()
            }
            Instruction::Unary(t, o, a) => {
                writeln!(
                    &mut shader,
                    "{} {}; {} {} = {};",
                    t.glsl_type(),
                    res_id,
                    VecLoop(t.size),
                    WriteValue(res_id, t.size),
                    WriteUnary(*o, input_node_id(node.instruction_number, *a), t.size)
                )?;
                *t
            }
            Instruction::Binary(t, o, l, r) => {
                writeln!(
                    &mut shader,
                    "{} {}; {} {} = {};",
                    t.glsl_type(),
                    res_id,
                    VecLoop(t.size),
                    WriteValue(res_id, t.size),
                    WriteBin(
                        *o,
                        input_node_id(node.instruction_number, *l),
                        input_node_id(node.instruction_number, *r),
                        t.size
                    )
                )?;
                *t
            }
            Instruction::Read(t, v) => {
                writeln!(&mut shader, "{} {} = {};", t.glsl_type(), res_id, v)?;
                *t
            }
        };
        config = config.ext(dtype.glsl_ext());
    }

    let root_node_id = NodeId(nodes.0.len() - 1);

    writeln!(
        &mut shader,
        r#"
                    output_buf.values[gID] = {};
                }}
            }}
            "#,
        root_node_id,
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

        let Some(metadata) = self.metadata.clone() else {
            return Err("No metadata information in JitOperator".into());
        };

        if let &[Instruction::Read(_, _)] = self.instructions.as_slice() {
            assert_eq!(self.operators.0.len(), 1);
            return Ok(self.operators.0.pop().unwrap());
        }

        Ok(TensorOperator::with_state(
            op_descriptor!(),
            self.dtype,
            metadata.clone(),
            (self, DataParam(metadata)),
            |ctx, positions, (jit_operator, metadata)| {
                async move {
                    let inplace_operator_index = jit_operator.operators.0.iter().position(|o| {
                        o.dtype().element_layout() == jit_operator.dtype.element_layout()
                    });

                    let device = ctx.preferred_device();

                    let access_info = DstBarrierInfo {
                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                    };
                    let m = (*metadata).clone().into_dyn();

                    let num_chunk_elements = m.num_chunk_elements();

                    let input_dtypes = jit_operator
                        .operators
                        .0
                        .iter()
                        .map(|op| op.dtype())
                        .collect::<Vec<_>>();
                    let pipeline = device.request_state(
                        (
                            &input_dtypes,
                            &jit_operator.instructions,
                            num_chunk_elements,
                        ),
                        |device, (input_dtypes, instructions, num_chunk_elements)| {
                            let (shader, config) = compile(instructions, input_dtypes)?;
                            //println!("{}", shader.as_str());
                            ComputePipelineBuilder::new(
                                Shader::new(shader.as_str())
                                    .define("BRICK_MEM_SIZE", num_chunk_elements)
                                    .with_config(config),
                            )
                            .use_push_descriptor(true)
                            .build(device)
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
                                        jit_operator.dtype,
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
                                    .alloc_slot_gpu(device, pos, &m.chunk_size)
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
                                pipeline.dispatch(device, global_size);
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
