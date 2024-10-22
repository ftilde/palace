use ash::vk;
use id::Identify;

use crate::{
    array::{ImageMetaData, TensorEmbeddingData, TensorMetaData, VolumeMetaData},
    dim::Dimension,
    dtypes::StaticElementType,
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

use super::{
    tensor::{LODImageOperator, LODTensorOperator},
    volume::LODVolumeOperator,
};

pub fn ball(base_metadata: VolumeMetaData) -> LODVolumeOperator<StaticElementType<f32>> {
    rasterize_lod(
        base_metadata,
        r#"float run(float[3] pos_normalized, uint[3] pos_voxel) {
            vec3 centered = to_glsl(pos_normalized)-vec3(0.5);
            vec3 sq = centered*centered;
            float d_sq = sq.x + sq.y + sq.z;
            return clamp(10*(0.5 - sqrt(d_sq)), 0.0, 1.0);
        }"#,
    )
}

pub fn full(base_metadata: VolumeMetaData) -> LODVolumeOperator<StaticElementType<f32>> {
    rasterize_lod(
        base_metadata,
        r#"float run(float[3] pos_normalized, uint[3] pos_voxel) {
            return 1.0;
        }"#,
    )
}

pub fn mandelbulb(base_metadata: VolumeMetaData) -> LODVolumeOperator<StaticElementType<f32>> {
    rasterize_lod(
        base_metadata,
        r#"
vec3 vec_pow(vec3 v, float n) {
    float r = length(v);
    float p = atan(v.y, v.x);
    float t = atan(length(v.xy), v.z);

    return pow(r, n) * vec3(sin(n*t)*cos(n*p), sin(n*t)*sin(n*p), cos(n*t));
}

float run(float[3] pos_normalized, uint[3] pos_voxel) {
    float outside_radius = 2.3;
    vec3 centered = (to_glsl(pos_normalized)-vec3(0.5)) * outside_radius;

    vec3 v = centered;
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

pub fn mandelbrot(base_metadata: ImageMetaData) -> LODImageOperator<StaticElementType<f32>> {
    rasterize_lod(
        base_metadata,
        r#"
vec2 complex_square(vec2 v) {
    return vec2(v.x*v.x - v.y*v.y, 2*v.x*v.y);
}
float run(float[2] pos_normalized, uint[2] pos_voxel) {
    float outside_radius = 5;
    vec2 centered = (to_glsl(pos_normalized)-vec2(0.8, 0.5)) * outside_radius;

    vec2 v = centered;
    int i;
    int max_i = 50;
    int min_i = 2;
    for(i=0; i<max_i; i++) {
        if(length(v) > outside_radius) {
            break;
        }

        v = complex_square(v) + centered;
    }
    return float(max(i-min_i, 0))/float(max_i-min_i);
}
        "#,
    )
}

pub fn rasterize_lod<D: Dimension>(
    base_metadata: TensorMetaData<D>,
    body: &str,
) -> LODTensorOperator<D, StaticElementType<f32>> {
    let mut levels = Vec::new();
    let mut spacing = Vector::fill(1.0f32);
    //TODO: maybe we want to compute the spacing based on dimension reduction instead? could be
    //more accurate...
    loop {
        let md = TensorMetaData {
            dimensions: (base_metadata.dimensions.raw().f32() / spacing)
                .map(|v| v.ceil() as u32)
                .global(),
            chunk_size: base_metadata.chunk_size,
        };
        levels.push(rasterize(md, body).embedded(TensorEmbeddingData { spacing }));
        spacing = spacing.scale(2.0);
        let chunk_size: Vector<D, _> = md.chunk_size;
        let dimensions: Vector<D, _> = md.dimensions;
        if chunk_size.raw().zip(&dimensions.raw(), |c, d| c >= d).all() {
            break;
        }
    }

    LODTensorOperator { levels }
}

pub fn rasterize<D: Dimension>(
    metadata: TensorMetaData<D>,
    gen_fn: &str,
) -> TensorOperator<D, StaticElementType<f32>> {
    #[derive(Clone, bytemuck::Zeroable)]
    #[repr(C)]
    struct PushConstants<D: Dimension> {
        offset: Vector<D, u32>,
        mem_dim: Vector<D, u32>,
        logical_dim: Vector<D, u32>,
        vol_dim: Vector<D, u32>,
    }

    impl<D: Dimension> Copy for PushConstants<D> where Vector<D, u32>: Copy {}
    //TODO: This is fine for the current layout, but we really want a better general approach
    unsafe impl<D: Dimension> bytemuck::Pod for PushConstants<D> where PushConstants<D>: Copy {}

    let shader = format!(
        "{}{}{}",
        r#"
#version 450

#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <vec.glsl>

layout(scalar, push_constant) uniform PushConsts {
    uint[N] offset;
    uint[N] mem_dim;
    uint[N] logical_dim;
    uint[N] vol_dim;
} consts;

layout (local_size_x = 256) in;

layout(std430, binding = 0) buffer OutputBuffer{
    float values[BRICK_MEM_SIZE];
} outputData;

//float run(uint[N] pos_normalized, float[n] pos_voxel) {
//  ...
//}
"#,
        gen_fn,
        r#"

void main()
{
    uint gID = gl_GlobalInvocationID.x;

    if(gID < BRICK_MEM_SIZE) {
        uint[N] out_local = from_linear(gID, consts.mem_dim);
        float result = 0.0;
        uint[N] pos_voxel = add(out_local, consts.offset);
        float[N] pos_normalized = div(to_float(pos_voxel),to_float(consts.vol_dim));

        if(all(less_than(out_local, consts.logical_dim))) {
            result = run(pos_normalized, pos_voxel);
        } else {
            result = NaN;
        }

        outputData.values[gID] = result;
    }
}
"#
    );

    let shader_id = shader.id();

    TensorOperator::with_state(
        OperatorDescriptor::new("rasterize_gpu")
            .dependent_on_data(gen_fn)
            .dependent_on_data(&metadata),
        Default::default(),
        metadata,
        (metadata, shader),
        move |ctx, positions, (metadata, shader)| {
            async move {
                let device = ctx.preferred_device();

                let m = metadata;

                let pipeline = device.request_state(
                    RessourceId::new("pipeline")
                        .of(ctx.current_op())
                        .dependent_on(&shader_id)
                        .dependent_on(&m.chunk_size)
                        .dependent_on(&D::N),
                    || {
                        ComputePipeline::new(
                            device,
                            (
                                shader.as_str(),
                                ShaderDefines::new()
                                    .add("BRICK_MEM_SIZE", m.chunk_size.hmul())
                                    .add("N", D::N),
                            ),
                            true,
                        )
                    },
                )?;

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

                            pipeline.push_constant_pod(PushConstants {
                                offset: brick_info.begin.into_elem::<u32>(),
                                logical_dim: brick_info.logical_dimensions.into_elem::<u32>(),
                                mem_dim: m.chunk_size.into_elem::<u32>(),
                                vol_dim: m.dimensions.into_elem::<u32>(),
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
