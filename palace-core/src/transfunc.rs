use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};
use id::Identify;

use futures::StreamExt;

use crate::{
    array::ChunkIndex,
    dim::{DynDimension, D4},
    dtypes::StaticElementType,
    op_descriptor,
    operator::{OperatorDescriptor, OperatorNetworkNode},
    operators::{
        array::{from_rc, ArrayOperator},
        tensor::TensorOperator,
    },
    task::RequestStream,
    vec::Vector,
    vulkan::{
        pipeline::{ComputePipelineBuilder, DescriptorConfig},
        shader::Shader,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

pub type TransFuncTableOperator = ArrayOperator<StaticElementType<Vector<D4, u8>>>;

#[derive(Clone, Identify)]
pub struct TransFuncOperator {
    pub table: TransFuncTableOperator,
    pub min: f32,
    pub max: f32,
}

impl OperatorNetworkNode for TransFuncOperator {
    fn descriptor(&self) -> OperatorDescriptor {
        self.table.descriptor()
    }
}

impl TransFuncOperator {
    pub fn data(&self) -> TransFuncData {
        TransFuncData {
            len: self.table.metadata.dimensions[0].raw,
            min: self.min,
            max: self.max,
        }
    }
    pub fn gen(min: f32, max: f32, len: usize, g: impl FnMut(usize) -> Vector<D4, u8>) -> Self {
        let vals = (0..len).map(g).collect::<Vec<_>>();
        let table = from_rc(std::rc::Rc::from(vals));
        Self { table, min, max }
    }
    pub fn gen_normalized(
        min: f32,
        max: f32,
        len: usize,
        mut g: impl FnMut(f32) -> Vector<D4, u8>,
    ) -> Self {
        let vals = (0..len)
            .map(|i| g(i as f32 / len as f32))
            .collect::<Vec<_>>();
        let table = from_rc(std::rc::Rc::from(vals));
        Self { table, min, max }
    }
    pub fn grey_ramp(min: f32, max: f32) -> Self {
        Self::gen(min, max, 256, |i| Vector::fill(i as u8))
    }
    pub fn red_ramp(min: f32, max: f32) -> Self {
        Self::gen(min, max, 256, |i| Vector::from([i as u8, 0, 0, i as u8]))
    }
    pub fn normalized(mut self) -> Self {
        self.min = 0.0;
        self.max = 1.0;
        self
    }
}

#[derive(Copy, Clone, AsStd140, GlslStruct)]
pub struct TransFuncData {
    pub len: u32,
    pub min: f32,
    pub max: f32,
}

pub fn apply<'op, D: DynDimension>(
    input: TensorOperator<D, StaticElementType<f32>>,
    tf: TransFuncOperator,
) -> TensorOperator<D, StaticElementType<Vector<D4, u8>>> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        tf_min: f32,
        tf_max: f32,
        tf_len: u32,
    }
    const SHADER: &'static str = r#"
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include <util.glsl>
#include <size_util.glsl>

layout(std430, binding = 0) readonly buffer InputBuffer{
    float values[BRICK_MEM_SIZE];
} sourceData;

layout(std430, binding = 1) buffer TFTableBuffer {
    u8vec4 values[];
} tf_table;

layout(std430, binding = 2) buffer OutputBuffer{
    u8vec4 values[BRICK_MEM_SIZE];
} outputData;

declare_push_consts(consts);

//TODO: deduplicate, move to module
u8vec4 classify(float val) {
    float norm = (val-consts.tf_min)/(consts.tf_max - consts.tf_min);
    uint index = min(uint(max(0.0, norm) * consts.tf_len), consts.tf_len - 1);
    return tf_table.values[index];
}

void main()
{
    uint gID = global_position_linear;

    if(gID < BRICK_MEM_SIZE) {
        float v = sourceData.values[gID];
        outputData.values[gID] = classify(v);
    }
}
"#;

    TensorOperator::with_state(
        op_descriptor!(),
        Default::default(),
        input.metadata.clone(),
        (input, tf),
        move |ctx, positions, (input, tf)| {
            async move {
                let device = ctx.preferred_device();

                let access_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                };
                let m = input.metadata.clone().into_dyn();

                assert_eq!(tf.table.metadata.dimension_in_chunks()[0].raw, 1);
                let tf_data_gpu = ctx
                    .submit(
                        tf.table
                            .chunks
                            .request_gpu(device.id, ChunkIndex(0), access_info),
                    )
                    .await;

                let pipeline =
                    device.request_state(m.num_chunk_elements(), |device, chunk_elements| {
                        ComputePipelineBuilder::new(
                            Shader::new(SHADER)
                                .define("BRICK_MEM_SIZE", chunk_elements)
                                .push_const_block::<PushConstants>(),
                        )
                        .use_push_descriptor(true)
                        .build(device)
                    })?;

                let mut brick_stream = ctx
                    .submit_unordered_with_data(positions.iter().map(|(pos, _)| {
                        (input.chunks.request_gpu(device.id, *pos, access_info), *pos)
                    }))
                    .then_req_with_data(*ctx, |(input, pos)| {
                        let output = ctx.alloc_slot_gpu(device, pos, &m.chunk_size);
                        (output, input)
                    });

                while let Some((output_chunk, input_chunk)) = brick_stream.next().await {
                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config =
                            DescriptorConfig::new([&input_chunk, &tf_data_gpu, &output_chunk]);

                        let global_size = m.num_chunk_elements();

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            let tf_data = tf.data();
                            let consts = PushConstants {
                                tf_min: tf_data.min,
                                tf_max: tf_data.max,
                                tf_len: tf_data.len,
                            };
                            pipeline.push_constant(consts);

                            pipeline.push_descriptor_set(0, descriptor_config);
                            pipeline.dispatch(device, global_size);
                        }
                    });

                    unsafe {
                        output_chunk.initialized(
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
    )
}
