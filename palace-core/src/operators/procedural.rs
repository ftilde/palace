use std::borrow::Cow;

use ash::vk;

use crate::{
    array::{ImageMetaData, TensorEmbeddingData, TensorMetaData, VolumeMetaData},
    dim::{DynDimension, D2, D3},
    dtypes::StaticElementType,
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    operators::tensor::TensorOperator,
    vec::Vector,
    vulkan::{
        pipeline::{ComputePipelineBuilder, DescriptorConfig, DynPushConstants},
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

pub fn rasterize_lod<D: DynDimension>(
    base_metadata: TensorMetaData<D>,
    base_embedding_data: TensorEmbeddingData<D>,
    body: impl Into<Cow<'static, str>>,
) -> LODTensorOperator<D, StaticElementType<f32>> {
    let mut levels = Vec::new();
    let nd = base_metadata.dim().n();
    let mut spacing_mult = Vector::fill_with_len(1.0, nd);
    let body = body.into();
    //TODO: maybe we want to compute the spacing based on dimension reduction instead? could be
    //more accurate...
    loop {
        let md = TensorMetaData {
            dimensions: (base_metadata.dimensions.raw().f32() / spacing_mult.clone())
                .map(|v| v.ceil() as u32)
                .global(),
            chunk_size: base_metadata.chunk_size.clone(),
        };
        levels.push(
            rasterize(md.clone(), body.clone()).embedded(TensorEmbeddingData {
                spacing: base_embedding_data.spacing.clone() * spacing_mult.clone(),
            }),
        );
        spacing_mult = spacing_mult.scale(2.0);
        let chunk_size: Vector<D, _> = md.chunk_size;
        let dimensions: Vector<D, _> = md.dimensions.clone();
        if chunk_size.raw().zip(&dimensions.raw(), |c, d| c >= d).all() {
            break;
        }
    }

    LODTensorOperator { levels }
}

pub fn rasterize<D: DynDimension>(
    metadata: TensorMetaData<D>,
    gen_fn: impl Into<Cow<'static, str>>,
) -> TensorOperator<D, StaticElementType<f32>> {
    let nd = metadata.dim().n();

    let push_constants = DynPushConstants::new()
        .vec::<u32>(nd, "offset")
        .vec::<u32>(nd, "logical_dim")
        .vec::<u32>(nd, "mem_dim")
        .vec::<u32>(nd, "vol_dim");

    let shader_parts = vec![
        r#"
#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <vec.glsl>
#include <size_util.glsl>

declare_push_consts(consts);

layout(std430, binding = 0) buffer OutputBuffer{
    float values[BRICK_MEM_SIZE];
} outputData;

//float run(uint[N] pos_normalized, float[n] pos_voxel) {
//  ...
//}
"#
        .into(),
        gen_fn.into(),
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
"#
        .into(),
    ];

    TensorOperator::with_state(
        op_descriptor!(),
        Default::default(),
        metadata.clone(),
        (
            DataParam(metadata),
            DataParam(shader_parts),
            DataParam(push_constants),
        ),
        |ctx, positions, loc, (metadata, shader_parts, push_constants)| {
            async move {
                let device = ctx.preferred_device(loc);

                let m = metadata;

                let pipeline = device.request_state(
                    (&push_constants, &shader_parts, m.chunk_size.clone()),
                    |device, (push_constants, shader_parts, chunk_size)| {
                        ComputePipelineBuilder::new(
                            Shader::from_parts(shader_parts.to_vec())
                                .push_const_block_dyn(&push_constants)
                                .define("BRICK_MEM_SIZE", chunk_size.hmul())
                                .define("N", chunk_size.len()),
                        )
                        .use_push_descriptor(true)
                        .build(device)
                    },
                )?;

                for pos in positions {
                    let brick_info = m.chunk_info(pos);

                    let gpu_brick_out = ctx
                        .submit(ctx.alloc_slot_gpu(device, pos, &brick_info.mem_dimensions))
                        .await;
                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config = DescriptorConfig::new([&gpu_brick_out]);

                        let global_size = brick_info.mem_dimensions.raw();

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            pipeline.push_constant_dyn(&push_constants, |consts| {
                                consts.vec(&brick_info.begin.raw())?;
                                consts.vec(&brick_info.logical_dimensions.raw())?;
                                consts.vec(&m.chunk_size.raw())?;
                                consts.vec(&m.dimensions.raw())?;
                                Ok(())
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
