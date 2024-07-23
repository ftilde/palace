use std::fmt::{Display, Write};

use ash::vk;
use futures::StreamExt;
use id::{Id, Identify};

use crate::{
    array::TensorMetaData,
    dim::DynDimension,
    dtypes::DType,
    operator::OperatorDescriptor,
    operators::tensor::TensorOperator,
    task::RequestStream,
    vulkan::{
        pipeline::{AsDescriptors, ComputePipeline, DescriptorConfig},
        shader::ShaderDefines,
        state::RessourceId,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

#[derive(id::Identify, Clone, Copy)]
pub enum UnaryOp {
    Abs,
    Neg,
}

impl UnaryOp {
    fn dtype(&self, input: DType) -> Result<DType, crate::Error> {
        Ok(match self {
            UnaryOp::Abs => input,
            UnaryOp::Neg => match input {
                DType::U8 | DType::U16 | DType::U32 | DType::U8Vec4 => {
                    return Err(format!("Value of type {:?} cannot be negated", input).into())
                }
                DType::I8 | DType::I16 | DType::I32 | DType::F32 | DType::F32Vec4A2 => input,
            },
        })
    }
}
struct WriteUnary(UnaryOp, ValueId);
impl Display for WriteUnary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            UnaryOp::Abs => write!(f, "abs({})", self.1),
            UnaryOp::Neg => write!(f, "-{}", self.1),
        }
    }
}

#[derive(id::Identify, Clone, Copy)]
pub enum BinOp {
    Add,
    Mul,
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
struct WriteBin(BinOp, ValueId, ValueId);
impl Display for WriteBin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            BinOp::Add => write!(f, "{} + {}", self.1, self.2),
            BinOp::Mul => write!(f, "{} * {}", self.1, self.2),
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
            ConstValue::F32(_) => DType::F32,
        }
    }
}

#[derive(id::Identify, Clone)]
enum Node {
    Const(ConstValue),
    UnaryOp(DType, UnaryOp, Box<Node>),
    BinOp(DType, BinOp, Box<Node>, Box<Node>),
    Read(DType, InputId),
}

#[derive(Clone)]
struct OperatorSet<D: DynDimension>(Vec<TensorOperator<D, DType>>);

impl<D: DynDimension> OperatorSet<D> {
    fn add(&mut self, op: TensorOperator<D, DType>) -> usize {
        let pos = if let Some(pos) = self
            .0
            .iter()
            .enumerate()
            .find_map(|(i, n)| (n.id() == op.id()).then_some(i))
        {
            pos
        } else {
            let pos = self.0.len();
            self.0.push(op.clone());
            pos
        };
        pos
    }

    fn merge(mut self, other: OperatorSet<D>) -> Self {
        for l in other.0.into_iter() {
            self.add(l);
        }
        self
    }
}

//TODO: Rc for cheap clone?
#[derive(Clone)]
pub struct JitTensorOperator<D: DynDimension> {
    node: Node,
    metadata: Option<TensorMetaData<D>>,
    dtype: DType,
    operators: OperatorSet<D>,
}

impl<D: DynDimension> id::Identify for JitTensorOperator<D> {
    fn id(&self) -> Id {
        // Note: All information (including operators via there ids) is present in the operation
        // tree `node`
        self.node.id()
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
        Ok({
            let dtype = op.dtype(inner.dtype)?;
            Self {
                metadata: inner.metadata.clone(),
                dtype,
                node: Node::UnaryOp(dtype, op, Box::new(inner.node)),
                operators: inner.operators,
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
                node: Node::BinOp(dtype, op, Box::new(l.node), Box::new(r.node)),
                operators: l.operators.merge(r.operators),
            }
        })
    }
}

impl<D: DynDimension> From<ConstValue> for JitTensorOperator<D> {
    fn from(c: ConstValue) -> Self {
        Self {
            node: Node::Const(c),
            metadata: None,
            dtype: c.dtype(),
            operators: OperatorSet(Vec::new()),
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
        Self {
            metadata,
            dtype,
            node: Node::Read(dtype, InputId(id)),
            operators: OperatorSet(vec![c]),
        }
    }
}

impl<D: DynDimension> JitTensorOperator<D> {
    pub fn add(self, other: JitTensorOperator<D>) -> Result<Self, crate::Error> {
        Self::bin_op(BinOp::Add, self, other)
    }
}

impl<D: DynDimension> JitTensorOperator<D> {
    pub fn abs(self) -> Result<Self, crate::Error> {
        Self::unary_op(UnaryOp::Abs, self)
    }
}

#[derive(id::Identify, Clone)]
struct InputId(Id);
impl Display for InputId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "input{}", self.0.raw())
    }
}

struct ValueId(usize);

impl Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

#[derive(Default)]
struct ValueIdGenerator {
    next: usize,
}
impl ValueIdGenerator {
    fn next(&mut self) -> ValueId {
        self.next += 1;
        ValueId(self.next)
    }
}

fn translate(
    n: &Node,
    w: &mut dyn Write,
    value_ids: &mut ValueIdGenerator,
) -> Result<ValueId, std::fmt::Error> {
    let res_id = value_ids.next();
    match n {
        Node::Const(c) => {
            writeln!(w, "{} {} = {};", c.dtype().glsl_type(), res_id, c)?;
        }
        Node::UnaryOp(t, o, a) => {
            let param = translate(a, w, value_ids)?;
            writeln!(
                w,
                "{} {} = {};",
                t.glsl_type(),
                res_id,
                WriteUnary(*o, param)
            )?;
        }
        Node::BinOp(t, o, l, r) => {
            let param_l = translate(l, w, value_ids)?;
            let param_r = translate(r, w, value_ids)?;
            writeln!(
                w,
                "{} {} = {};",
                t.glsl_type(),
                res_id,
                WriteBin(*o, param_l, param_r)
            )?;
        }
        Node::Read(t, v) => {
            writeln!(w, "{} {} = {};", t.glsl_type(), res_id, v)?;
        }
    }
    Ok(res_id)
}

pub fn jit<D: DynDimension>(op: TensorOperator<D, DType>) -> JitTensorOperator<D> {
    op.into()
}

fn compile<D: DynDimension>(
    node: &Node,
    inputs: &Vec<TensorOperator<D, DType>>,
) -> Result<String, crate::Error> {
    let mut shader = String::new();

    writeln!(&mut shader, "#version 450")?;
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
                float values[BRICK_MEM_SIZE];
            }} output_buf;
            "#,
        inputs.len(),
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

    let mut value_ids = ValueIdGenerator::default();
    let res = translate(&node, &mut shader, &mut value_ids)?;

    writeln!(
        &mut shader,
        r#"
                    output_buf.values[gID] = {};
                }}
            }}
            "#,
        res,
    )?;

    Ok(shader)
}
impl<D: DynDimension> JitTensorOperator<D> {
    pub fn compile(self) -> Result<TensorOperator<D, DType>, crate::Error> {
        let dtype = self.dtype;
        let Some(metadata) = self.metadata.clone() else {
            return Err("No metadata information in JitOperator".into());
        };

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
                        RessourceId::new("pipeline").of(ctx.current_op()),
                        || {
                            let shader = compile(&jit_operator.node, &jit_operator.operators.0)?;
                            ComputePipeline::new(
                                device,
                                (
                                    shader.as_str(),
                                    ShaderDefines::new().add("BRICK_MEM_SIZE", num_chunk_elements),
                                ),
                                true,
                            )
                        },
                    )?;

                    let mut brick_stream = ctx
                        .submit_unordered_with_data(positions.iter().map(|(pos, _)| {
                            (
                                ctx.group(jit_operator.operators.0.iter().map(|input| {
                                    input.chunks.request_gpu(device.id, *pos, access_info)
                                })),
                                *pos,
                            )
                        }))
                        .then_req_with_data(*ctx, |(input, pos)| {
                            let output = ctx.alloc_slot_gpu(device, pos, num_chunk_elements);
                            (output, input)
                        });

                    while let Some((gpu_brick_out, inputs)) = brick_stream.next().await {
                        device.with_cmd_buffer(|cmd| {
                            let mut descriptors = inputs
                                .iter()
                                .map(|v| v as &dyn AsDescriptors)
                                .collect::<Vec<_>>();
                            descriptors.push(&gpu_brick_out);
                            let descriptor_config = DescriptorConfig::from_vec(descriptors);

                            let global_size = num_chunk_elements;

                            unsafe {
                                let mut pipeline = pipeline.bind(cmd);

                                pipeline.push_descriptor_set(0, descriptor_config);
                                pipeline.dispatch(global_size);
                            }
                        });

                        unsafe {
                            gpu_brick_out.initialized(
                                *ctx,
                                SrcBarrierInfo {
                                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                    access: vk::AccessFlags2::SHADER_WRITE,
                                },
                            )
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
