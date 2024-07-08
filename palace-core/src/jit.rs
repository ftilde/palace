use std::fmt::{Display, Write};

use ash::vk;
use futures::StreamExt;
use id::Identify;

use crate::{
    dim::Dimension,
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
}

impl UnaryOp {
    fn dtype(&self, input: DType) -> DType {
        input
    }
}
struct WriteUnary(UnaryOp, ValueId);
impl Display for WriteUnary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            UnaryOp::Abs => write!(f, "abs({})", self.1),
        }
    }
}

#[derive(id::Identify, Clone, Copy)]
pub enum BinOp {
    Add,
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
        }
    }
}

#[derive(id::Identify, Clone, Copy)]
pub enum ConstValue {
    F32(f32),
}

impl<D: Dimension> From<f32> for Node<D> {
    fn from(value: f32) -> Self {
        Self::Const(ConstValue::F32(value))
    }
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

#[derive(id::Identify)]
pub enum Node<D: Dimension> {
    Const(ConstValue),
    UnaryOp(UnaryOp, Box<Node<D>>),
    BinOp(BinOp, Box<Node<D>>, Box<Node<D>>),
    Read(TensorOperator<D, DType>),
}

impl<D: Dimension> Node<D> {
    pub fn add(self, other: Node<D>) -> Self {
        Self::BinOp(BinOp::Add, Box::new(self), Box::new(other))
    }
}

impl<D: Dimension> Node<D> {
    pub fn abs(self) -> Self {
        Self::UnaryOp(UnaryOp::Abs, Box::new(self))
    }
}

enum InternalNode {
    Const(DType, ConstValue),
    UnaryOp(DType, UnaryOp, Box<InternalNode>),
    BinOp(DType, BinOp, Box<InternalNode>, Box<InternalNode>),
    Read(DType, InputId),
}

impl InternalNode {
    fn dtype(&self) -> DType {
        match self {
            InternalNode::Const(d, _) => *d,
            InternalNode::UnaryOp(d, _, _) => *d,
            InternalNode::BinOp(d, _, _, _) => *d,
            InternalNode::Read(d, _) => *d,
        }
    }
}

fn type_check<D: Dimension>(
    n: &Node<D>,
    input_ops: &mut Vec<TensorOperator<D, DType>>,
) -> Result<InternalNode, crate::Error> {
    Ok(match n {
        Node::Const(c) => InternalNode::Const(c.dtype(), *c),
        Node::UnaryOp(op, inner) => {
            let v = type_check(inner, input_ops)?;
            InternalNode::UnaryOp(op.dtype(v.dtype()), *op, Box::new(v))
        }
        Node::BinOp(op, l, r) => {
            let l = type_check(l, input_ops)?;
            let r = type_check(r, input_ops)?;
            InternalNode::BinOp(
                op.dtype(l.dtype(), r.dtype())?,
                *op,
                Box::new(l),
                Box::new(r),
            )
        }
        Node::Read(v) => {
            let dtype = v.chunks.dtype();
            let pos = if let Some(pos) = input_ops
                .iter()
                .enumerate()
                .find_map(|(i, n)| (n.id() == v.id()).then_some(i))
            {
                pos
            } else {
                let pos = input_ops.len();
                input_ops.push(v.clone());
                pos
            };

            InternalNode::Read(dtype, InputId(pos))
        }
    })
}

struct InputId(usize);
impl Display for InputId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "input{}.values[gID]", self.0)
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
    n: &InternalNode,
    w: &mut dyn Write,
    value_ids: &mut ValueIdGenerator,
) -> Result<ValueId, std::fmt::Error> {
    let res_id = value_ids.next();
    match n {
        InternalNode::Const(t, c) => {
            writeln!(w, "{} {} = {};", t.glsl_type(), res_id, c)?;
        }
        InternalNode::UnaryOp(t, o, a) => {
            let param = translate(a, w, value_ids)?;
            writeln!(
                w,
                "{} {} = {};",
                t.glsl_type(),
                res_id,
                WriteUnary(*o, param)
            )?;
        }
        InternalNode::BinOp(t, o, l, r) => {
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
        InternalNode::Read(t, v) => {
            writeln!(w, "{} {} = {};", t.glsl_type(), res_id, v)?;
        }
    }
    Ok(res_id)
}

pub fn jit<D: Dimension>(op: TensorOperator<D, DType>) -> Node<D> {
    Node::Read(op)
}

impl<D: Dimension> Node<D> {
    fn compile(&self) -> Result<(DType, String, Vec<TensorOperator<D, DType>>), crate::Error> {
        let mut inputs = Vec::new();
        let out = type_check(self, &mut inputs)?;

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

        let mut value_ids = ValueIdGenerator::default();
        let res = translate(&out, &mut shader, &mut value_ids)?;

        writeln!(
            &mut shader,
            r#"
                    output_buf.values[gID] = {};
                }}
            }}
            "#,
            res,
        )?;

        Ok((out.dtype(), shader, inputs))
    }
}

impl<D: Dimension> TryInto<TensorOperator<D, DType>> for Node<D> {
    type Error = crate::Error;
    fn try_into(self) -> Result<TensorOperator<D, DType>, Self::Error> {
        let (dtype, shader, inputs) = self.compile()?;

        let first = inputs
            .first()
            .ok_or("Need at least one input operator".to_owned())?;
        let metadata = first.metadata;
        for input in &inputs {
            if input.metadata != metadata {
                return Err(format!(
                    "Metadata of {:?} ({:?}) does not match selected output metadata ({:?})",
                    input.id(),
                    input.metadata,
                    metadata
                )
                .into());
            }
        }

        Ok(TensorOperator::with_state(
            OperatorDescriptor::new("jit").dependent_on_data(&self),
            dtype,
            metadata,
            (shader, inputs),
            move |ctx, positions, (shader, inputs)| {
                async move {
                    let device = ctx.preferred_device();

                    let access_info = DstBarrierInfo {
                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                    };
                    let m = metadata;

                    let pipeline = device.request_state(
                        RessourceId::new("pipeline").of(ctx.current_op()),
                        || {
                            ComputePipeline::new(
                                device,
                                (
                                    shader.as_str(),
                                    ShaderDefines::new().add("BRICK_MEM_SIZE", m.chunk_size.hmul()),
                                ),
                                true,
                            )
                        },
                    );

                    let mut brick_stream = ctx
                        .submit_unordered_with_data(positions.iter().map(|(pos, _)| {
                            (
                                ctx.group(inputs.iter().map(|input| {
                                    input.chunks.request_gpu(device.id, *pos, access_info)
                                })),
                                *pos,
                            )
                        }))
                        .then_req_with_data(*ctx, |(input, pos)| {
                            let brick_info = m.chunk_info(pos);

                            let output = ctx.alloc_slot_gpu(device, pos, brick_info.mem_elements());
                            (output, (input, brick_info))
                        });

                    while let Some((gpu_brick_out, (inputs, brick_info))) =
                        brick_stream.next().await
                    {
                        device.with_cmd_buffer(|cmd| {
                            let mut descriptors = inputs
                                .iter()
                                .map(|v| v as &dyn AsDescriptors)
                                .collect::<Vec<_>>();
                            descriptors.push(&gpu_brick_out);
                            let descriptor_config = DescriptorConfig::from_vec(descriptors);

                            let global_size = brick_info.mem_elements();

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
