use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};

use crate::{
    array::{TensorEmbeddingData, VolumeMetaData},
    operator::OperatorDescriptor,
    operators::tensor::TensorOperator,
    vec::Vector,
    vulkan::{
        pipeline::{ComputePipeline, DescriptorConfig},
        shader::ShaderDefines,
        state::RessourceId,
        SrcBarrierInfo,
    },
};

use super::volume::{LODVolumeOperator, VolumeOperator};

pub fn ball(base_metadata: VolumeMetaData) -> LODVolumeOperator<f32> {
    rasterize_lod(
        base_metadata,
        r#"{
            vec3 centered = pos_normalized-vec3(0.5);
            vec3 sq = centered*centered;
            float d_sq = sq.x + sq.y + sq.z;
            result = clamp(10*(0.5 - sqrt(d_sq)), 0.0, 1.0);
        }"#,
    )
}

pub fn rasterize_lod(base_metadata: VolumeMetaData, body: &str) -> LODVolumeOperator<f32> {
    let mut levels = Vec::new();
    let mut spacing = Vector::fill(1.0f32);
    //TODO: maybe we want to compute the spacing based on dimension reduction instead? could be
    //more accurate...
    loop {
        let md = VolumeMetaData {
            dimensions: (base_metadata.dimensions.raw().f32() / spacing)
                .map(|v| v.ceil() as u32)
                .global(),
            chunk_size: base_metadata.chunk_size,
        };
        levels.push(rasterize(md, body).embedded(TensorEmbeddingData { spacing }));
        spacing = spacing.scale(2.0);
        if md
            .chunk_size
            .raw()
            .zip(md.dimensions.raw(), |c, d| c >= d)
            .all()
        {
            break;
        }
    }

    LODVolumeOperator { levels }
}

pub fn rasterize(metadata: VolumeMetaData, body: &str) -> VolumeOperator<f32> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        offset: cgmath::Vector3<u32>,
        mem_dim: cgmath::Vector3<u32>,
        logical_dim: cgmath::Vector3<u32>,
        vol_dim: cgmath::Vector3<u32>,
    }

    let shader = format!(
        "{}{}{}",
        r#"
#version 450

#include <util.glsl>

layout (local_size_x = 256) in;

layout(std430, binding = 0) buffer OutputBuffer{
    float values[BRICK_MEM_SIZE];
} outputData;

declare_push_consts(consts);

void main()
{
    uint gID = gl_GlobalInvocationID.x;

    if(gID < BRICK_MEM_SIZE) {
        uvec3 out_local = from_linear3(gID, consts.mem_dim);
        float result = 0.0;
        uvec3 pos_voxel = out_local + consts.offset;
        vec3 pos_normalized = vec3(pos_voxel)/vec3(consts.vol_dim);

        if(all(lessThan(out_local, consts.logical_dim))) {
        "#,
        body,
        r#"
        } else {
            result = NaN;
        }

        outputData.values[gID] = result;
    }
}
"#
    );

    TensorOperator::with_state(
        OperatorDescriptor::new("rasterize_gpu")
            .dependent_on_data(body)
            .dependent_on_data(&metadata),
        metadata,
        (metadata, shader),
        move |ctx, positions, (metadata, shader)| {
            async move {
                let device = ctx.preferred_device();

                let m = metadata;

                let pipeline = device.request_state(
                    RessourceId::new("pipeline")
                        .of(ctx.current_op())
                        .dependent_on(&m.chunk_size),
                    || {
                        ComputePipeline::new(
                            device,
                            (
                                shader.as_str(),
                                ShaderDefines::new()
                                    .push_const_block::<PushConstants>()
                                    .add("BRICK_MEM_SIZE", m.chunk_size.hmul()),
                            ),
                            true,
                        )
                    },
                );

                for (pos, _) in positions {
                    let brick_info = m.chunk_info(pos);

                    let gpu_brick_out = ctx
                        .submit(ctx.alloc_slot_gpu(device, pos, brick_info.mem_elements()))
                        .await;
                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config = DescriptorConfig::new([&gpu_brick_out]);

                        let global_size = brick_info.mem_elements();

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            pipeline.push_constant(PushConstants {
                                offset: brick_info.begin.into_elem::<u32>().into(),
                                logical_dim: brick_info
                                    .logical_dimensions
                                    .into_elem::<u32>()
                                    .into(),
                                mem_dim: m.chunk_size.into_elem::<u32>().into(),
                                vol_dim: m.dimensions.into_elem::<u32>().into(),
                            });
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
    )
}
