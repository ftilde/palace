use ash::vk;

use crate::{
    array::{ImageMetaData, TensorEmbeddingData, TensorMetaData, VolumeMetaData},
    dim::{Dimension, D2, D3},
    dtypes::StaticElementType,
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    operators::tensor::TensorOperator,
    vec::Vector,
    vulkan::{
        pipeline::{ComputePipelineBuilder, DescriptorConfig},
        shader::Shader,
        SrcBarrierInfo,
    },
};

use super::tensor::{LODImageOperator, LODTensorOperator, LODVolumeOperator};

pub fn ball(
    base_metadata: VolumeMetaData,
    base_embedding_data: TensorEmbeddingData<D3>,
) -> LODVolumeOperator<StaticElementType<f32>> {
    rasterize_lod(
        base_metadata,
        base_embedding_data,
        r#"float run(float[3] pos_normalized, uint[3] pos_voxel) {
            vec3 centered = to_glsl(pos_normalized)-vec3(0.5);
            vec3 sq = centered*centered;
            float d_sq = sq.x + sq.y + sq.z;
            return clamp(10*(0.5 - sqrt(d_sq)), 0.0, 1.0);
        }"#,
    )
}

pub fn full(
    base_metadata: VolumeMetaData,
    base_embedding_data: TensorEmbeddingData<D3>,
) -> LODVolumeOperator<StaticElementType<f32>> {
    rasterize_lod(
        base_metadata,
        base_embedding_data,
        r#"float run(float[3] pos_normalized, uint[3] pos_voxel) {
            return 1.0;
        }"#,
    )
}

pub fn mandelbulb(
    base_metadata: VolumeMetaData,
    base_embedding_data: TensorEmbeddingData<D3>,
) -> LODVolumeOperator<StaticElementType<f32>> {
    rasterize_lod(
        base_metadata,
        base_embedding_data,
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

pub fn mandelbrot(
    base_metadata: ImageMetaData,
    base_embedding_data: TensorEmbeddingData<D2>,
) -> LODImageOperator<StaticElementType<f32>> {
    rasterize_lod(
        base_metadata,
        base_embedding_data,
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
    base_embedding_data: TensorEmbeddingData<D>,
    body: &'static str,
) -> LODTensorOperator<D, StaticElementType<f32>> {
    let mut levels = Vec::new();
    let mut spacing_mult = Vector::fill(1.0);
    //TODO: maybe we want to compute the spacing based on dimension reduction instead? could be
    //more accurate...
    loop {
        let md = TensorMetaData {
            dimensions: (base_metadata.dimensions.raw().f32() / spacing_mult)
                .map(|v| v.ceil() as u32)
                .global(),
            chunk_size: base_metadata.chunk_size,
        };
        levels.push(rasterize(md, body).embedded(TensorEmbeddingData {
            spacing: base_embedding_data.spacing * spacing_mult,
        }));
        spacing_mult = spacing_mult.scale(2.0);
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
    gen_fn: &'static str,
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

    let shader_parts = vec![
        r#"
#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <vec.glsl>
#include <size_util.glsl>

layout(scalar, push_constant) uniform PushConsts {
    uint[N] offset;
    uint[N] mem_dim;
    uint[N] logical_dim;
    uint[N] vol_dim;
} consts;

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
    uint gID = global_position_linear;

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
"#,
    ];

    TensorOperator::with_state(
        op_descriptor!(),
        Default::default(),
        metadata,
        (DataParam(metadata), DataParam(shader_parts)),
        |ctx, positions, (metadata, shader_parts)| {
            async move {
                let device = ctx.preferred_device();

                let m = metadata;

                let pipeline = device.request_state(
                    (&shader_parts, m.chunk_size.hmul()),
                    |device, (shader_parts, chunk_size)| {
                        ComputePipelineBuilder::new(
                            Shader::from_parts(shader_parts.to_vec())
                                .define("BRICK_MEM_SIZE", chunk_size)
                                .define("N", D::N),
                        )
                        .use_push_descriptor(true)
                        .build(device)
                    },
                )?;

                for (pos, _) in positions {
                    let brick_info = m.chunk_info(pos);

                    let gpu_brick_out = ctx
                        .submit(ctx.alloc_slot_gpu(device, pos, &brick_info.mem_dimensions))
                        .await;
                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config = DescriptorConfig::new([&gpu_brick_out]);

                        let global_size = brick_info.mem_dimensions.raw();

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            pipeline.push_constant_pod(PushConstants {
                                offset: brick_info.begin.into_elem::<u32>(),
                                logical_dim: brick_info.logical_dimensions.into_elem::<u32>(),
                                mem_dim: m.chunk_size.into_elem::<u32>(),
                                vol_dim: m.dimensions.into_elem::<u32>(),
                            });
                            pipeline.push_descriptor_set(0, descriptor_config);
                            pipeline.dispatch_dyn(device, global_size);
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        data::{LocalVoxelPosition, VoxelPosition},
        operators::rechunk::rechunk,
        test_util::*,
    };

    #[test]
    fn test_rasterize_gpu() {
        let size = VoxelPosition::fill(5.into());

        let fill_expected = |comp: &mut ndarray::ArrayViewMut3<f32>| {
            for z in 0..size.z().raw {
                for y in 0..size.y().raw {
                    for x in 0..size.x().raw {
                        let pos = VoxelPosition::from([z, y, x]);
                        comp[pos.as_index()] = x as f32 + y as f32 + z as f32;
                    }
                }
            }
        };
        for chunk_size in [[5, 1, 1], [4, 4, 1], [2, 3, 4], [1, 1, 1], [5, 5, 5]] {
            let input = rasterize(
                crate::array::VolumeMetaData {
                    dimensions: size,
                    chunk_size: chunk_size.into(),
                },
                r#"float run(float[3] pos_normalized, uint[3] pos_voxel) { return float(pos_voxel[0] + pos_voxel[1] + pos_voxel[2]); }"#,
            );
            let output = rechunk(input, LocalVoxelPosition::from(chunk_size).into_elem());
            compare_tensor_fn(output, fill_expected);
        }
    }
}
