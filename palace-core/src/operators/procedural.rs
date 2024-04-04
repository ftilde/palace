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
        r#"float run(vec3 pos) {
            vec3 centered = pos-vec3(0.5);
            vec3 sq = centered*centered;
            float d_sq = sq.x + sq.y + sq.z;
            return clamp(10*(0.5 - sqrt(d_sq)), 0.0, 1.0);
        }"#,
    )
}

pub fn full(base_metadata: VolumeMetaData) -> LODVolumeOperator<f32> {
    rasterize_lod(
        base_metadata,
        r#"float run(vec3 pos) {
            return 1.0;
        }"#,
    )
}

pub fn mandelbulb(base_metadata: VolumeMetaData) -> LODVolumeOperator<f32> {
    rasterize_lod(
        base_metadata,
        r#"
vec3 vec_pow(vec3 v, float n) {
    float r = length(v);
    float p = atan(v.y, v.x);
    float t = atan(length(v.xy), v.z);

    return pow(r, n) * vec3(sin(n*t)*cos(n*p), sin(n*t)*sin(n*p), cos(n*t));
}

float run(vec3 pos) {
    float outside_radius = 2.3;
    vec3 centered = (pos-vec3(0.5)) * outside_radius;

    vec3 v = vec3(centered);
    int i;
    int max_i = 10;
    int min_i = 2;
    for(i=0; i<max_i; i++) {
        if(length(v) > outside_radius) {
            break;
        }

        v = vec_pow(v, 8.0) + centered;
    }
    return float(max(i-min_i, 0))/float(max_i-min_i);
}
        "#,
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

pub fn rasterize(metadata: VolumeMetaData, gen_fn: &str) -> VolumeOperator<f32> {
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

//float run(vec3 pos) {
//  ...
//}
"#,
        gen_fn,
        r#"

void main()
{
    uint gID = gl_GlobalInvocationID.x;

    if(gID < BRICK_MEM_SIZE) {
        uvec3 out_local = from_linear3(gID, consts.mem_dim);
        float result = 0.0;
        uvec3 pos_voxel = out_local + consts.offset;
        vec3 pos_normalized = vec3(pos_voxel)/vec3(consts.vol_dim);

        if(all(lessThan(out_local, consts.logical_dim))) {
            result = run(pos_normalized);
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
            .dependent_on_data(gen_fn)
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
