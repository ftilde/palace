use crevice::std140::AsStd140;
use futures::StreamExt;

use crate::{
    array::{ImageMetaData, VolumeMetaData},
    data::{BrickPosition, ChunkCoordinate, GlobalCoordinate, LocalCoordinate, Vector},
    operator::OperatorId,
    operators::tensor::TensorOperator,
    vulkan::{pipeline::ComputePipeline, state::RessourceId},
};

use super::{scalar::ScalarOperator, volume::VolumeOperator};

pub fn slice_projection_mat_z<'a>(
    input_data: ScalarOperator<'a, VolumeMetaData>,
    output_data: ScalarOperator<'a, ImageMetaData>,
    selected_slice: ScalarOperator<'a, GlobalCoordinate>,
) -> ScalarOperator<'a, mint::ColumnMatrix3<f32>> {
    crate::operators::scalar::scalar(
        OperatorId::new("slice_projection_mat_z")
            .dependent_on(&input_data)
            .dependent_on(&output_data)
            .dependent_on(&selected_slice),
        (input_data, output_data, selected_slice),
        move |ctx, (input_data, output_data, selected_slice), _| {
            async move {
                let (input_data, output_data, selected_slice) = futures::join! {
                    ctx.submit(input_data.request_scalar()),
                    ctx.submit(output_data.request_scalar()),
                    ctx.submit(selected_slice.request_scalar()),
                };

                let vol_dim = input_data.dimensions.map(|v| v.raw as f32);
                let img_dim = output_data.dimensions.map(|v| v.raw as f32);

                let out = mint::ColumnMatrix3 {
                    x: mint::Vector3 {
                        x: vol_dim.x() / img_dim.x(),
                        y: 0.0,
                        z: 0.0,
                    },
                    y: mint::Vector3 {
                        x: 0.0,
                        y: vol_dim.y() / img_dim.y(),
                        z: 0.0,
                    },
                    z: mint::Vector3 {
                        x: 0.0,
                        y: 0.0,
                        z: selected_slice.raw as f32 + 0.5, //For +0.5 see below
                    },
                };
                ctx.write(out)
            }
            .into()
        },
    )
}

pub fn render_slice<'a>(
    input: VolumeOperator<'a>,
    result_metadata: ScalarOperator<'a, ImageMetaData>,
    projection_mat: ScalarOperator<'a, mint::ColumnMatrix3<f32>>,
) -> VolumeOperator<'a> {
    const SHADER: &'static str = r#"
#version 450

#extension GL_KHR_shader_subgroup_arithmetic : require

layout (local_size_x = 1024) in;

layout(std430, binding = 0) readonly buffer InputBuffer{
float values[];
} sourceData;

layout(std430, binding = 1) buffer OutputBuffer{
uint value;
} sum;

layout(std140, push_constant) uniform PushConstants
{
uvec3 mem_dim;
uvec3 logical_dim;
float norm_factor;
} consts;

uvec3 from_linear(uint linear_pos, uvec3 size) {
uvec3 vec_pos;
vec_pos.x = linear_pos % size.x;
linear_pos /= size.x;
vec_pos.y = linear_pos % size.y;
linear_pos /= size.y;
vec_pos.z = linear_pos;

return vec_pos;
}

#define atomic_add(mem, value) {\
uint initial = 0;\
uint new = 0;\
do {\
    initial = mem;\
    new = floatBitsToUint(uintBitsToFloat(initial) + (value));\
    if (new == initial) {\
        break;\
    }\
} while(atomicCompSwap(mem, initial, new) != initial);\
}

shared uint shared_sum;

void main()
{
uint gID = gl_GlobalInvocationID.x;
if(gl_LocalInvocationIndex == 0) {
    shared_sum = floatBitsToUint(0.0);
}
barrier();

float val;

uvec3 local = from_linear(gID, consts.mem_dim);

if(local.x < consts.logical_dim.x && local.y < consts.logical_dim.y && local.z < consts.logical_dim.z) {
    val = sourceData.values[gID] * consts.norm_factor;
} else {
    val = 0.0;
}

float sg_sum = subgroupAdd(val);

if(gl_SubgroupInvocationID == 0) {
    atomic_add(shared_sum, sg_sum);
}

barrier();

if(gl_LocalInvocationIndex == 0) {
    atomic_add(sum.value, uintBitsToFloat(shared_sum));
}
}
"#;

    const N_CHANNELS: u32 = 4;
    fn full_info(r: ImageMetaData) -> VolumeMetaData {
        VolumeMetaData {
            dimensions: [r.dimensions.y(), r.dimensions.x(), N_CHANNELS.into()].into(),
            chunk_size: [r.chunk_size.y(), r.chunk_size.x(), N_CHANNELS.into()].into(),
        }
    }

    TensorOperator::with_state(
        OperatorId::new("sliceviewer")
            .dependent_on(&input)
            .dependent_on(&result_metadata)
            .dependent_on(&projection_mat),
        result_metadata.clone(),
        (input, result_metadata, projection_mat),
        move |ctx, result_metadata, _| {
            async move {
                let r = ctx.submit(result_metadata.request_scalar()).await;
                let m = full_info(r);
                ctx.write(m)
            }
            .into()
        },
        move |ctx, positions, (input, result_metadata, projection_mat), _| {
            async move {
                let device = ctx.vulkan_device();

                let m_in = ctx.submit(input.metadata.request_scalar()).await;

                let _pipeline = device
                    .request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(device, SHADER)
                    });

                let (m, transform) = futures::join! {
                    ctx.submit(result_metadata.request_scalar()),
                    ctx.submit(projection_mat.request_scalar()),
                };
                let m = full_info(m);

                let requests = positions.into_iter().map(|pos| {
                    let out_info = m.chunk_info(pos);
                    let out_begin_pixel = out_info.begin().raw().map(|v| v as f32).drop_dim(2);
                    let out_end_pixel = out_info.end().raw().map(|v| v as f32).drop_dim(2);

                    let out_begin_voxel = transform * out_begin_pixel.to_homogeneous_coord();
                    let out_end_voxel = transform * out_end_pixel.to_homogeneous_coord();
                    let llb = out_begin_voxel.zip(out_end_voxel, |a, b| a.min(b).floor() as u32);
                    let urb = out_begin_voxel.zip(out_end_voxel, |a, b| a.max(b).floor() as u32);

                    let llb_brick = m_in.chunk_pos(llb.global());
                    // Clamp to valid range:
                    let urb_brick = m_in
                        .chunk_pos(urb.global())
                        .zip(m_in.dimension_in_bricks() - Vector::fill(1u32), |a, b| {
                            a.min(b)
                        });

                    let brick_region_size = urb_brick + Vector::fill(1u32) - llb_brick;

                    let low = llb_brick.raw();
                    let high = urb_brick.raw();

                    let in_brick_positions = itertools::iproduct! {
                        low.z()..=high.z(),
                        low.y()..=high.y(),
                        low.x()..=high.x()
                    }
                    .map(|(z, y, x)| BrickPosition::from([z, y, x]))
                    .collect::<Vec<_>>();
                    let intersecting_bricks = ctx.group(
                        in_brick_positions
                            .iter()
                            .map(|pos| input.bricks.request(*pos)),
                    );

                    (intersecting_bricks, (pos, llb_brick, brick_region_size))
                });

                let mut stream = ctx.submit_unordered_with_data(requests);
                while let Some((intersecting_bricks, (pos, llb_brick, brick_region_size))) =
                    stream.next().await
                {
                    let out_info = m.chunk_info(pos);

                    let mut tile_out = ctx.alloc_slot(pos, out_info.mem_elements()).unwrap();
                    let mut tile = crate::data::chunk_mut(&mut tile_out, &out_info);

                    #[derive(Copy, Clone, AsStd140)]
                    struct PushConstants {
                        transform: mint::ColumnMatrix3<f32>,
                        vol_dim: mint::Vector3<u32>,
                        chunk_dim: mint::Vector3<u32>,
                        brick_region_size: mint::Vector3<u32>,
                        llb_brick: mint::Vector3<u32>,
                        //num_chunk_elems: u32,
                    }

                    let out_begin = out_info.begin();
                    let out_end = out_info.end();
                    let consts = PushConstants {
                        transform,
                        vol_dim: m_in.dimensions.raw().into(),
                        chunk_dim: m_in.chunk_size.raw().into(),
                        brick_region_size: brick_region_size.raw().into(),
                        llb_brick: llb_brick.raw().into(),
                    };
                    for y in out_begin.0[0].raw..out_end.0[0].raw {
                        for x in out_begin.0[1].raw..out_end.0[1].raw {
                            //TODO: Maybe revisit this +0.5 -0.5 business.
                            let pos = (Vector::<2, f32>::from([y as f32, x as f32])
                                + Vector::fill(0.5))
                            .to_homogeneous_coord();
                            let sample_pos = consts.transform * pos - Vector::fill(0.5);
                            let sample_pos = sample_pos.map(|v| v.round() as u32).global();

                            let c = if sample_pos.x().raw < consts.vol_dim.x
                                && sample_pos.y().raw < consts.vol_dim.y
                                && sample_pos.z().raw < consts.vol_dim.z
                            {
                                let sample_brick = sample_pos.raw() / consts.chunk_dim.into();
                                let llb_brick: Vector<3, u32> = consts.llb_brick.into();
                                let brick_region_size: Vector<3, u32> =
                                    consts.brick_region_size.into();
                                let urb_brick: Vector<3, u32> =
                                    brick_region_size + consts.llb_brick.into();
                                if llb_brick.x() <= sample_brick.x()
                                    && sample_brick.x() < urb_brick.x()
                                    && llb_brick.y() <= sample_brick.y()
                                    && sample_brick.y() < urb_brick.y()
                                    && llb_brick.z() <= sample_brick.z()
                                    && sample_brick.z() < urb_brick.z()
                                {
                                    let sample_brick_region = sample_brick - llb_brick;
                                    let sample_brick_pos_linear = crate::data::to_linear(
                                        sample_brick_region.into_elem::<ChunkCoordinate>(),
                                        brick_region_size.into_elem(),
                                    );

                                    let brick_begin = sample_brick * m_in.chunk_size.raw();
                                    let brick = &intersecting_bricks[sample_brick_pos_linear];
                                    let local = (sample_pos - brick_begin).local();
                                    let local_index =
                                        crate::data::to_linear(local, m_in.chunk_size);
                                    let v = brick[local_index];
                                    Vector::from([v, v, v, 1.0])
                                } else {
                                    Vector::from([0.0, 0.0, 0.0, 0.0])
                                }
                            } else {
                                Vector::from([0.0, 0.0, 0.0, 0.0])
                            };

                            for channel in 0..4 {
                                let p: Vector<3, LocalCoordinate> = Vector::from([
                                    y - out_begin.0[0].raw,
                                    x - out_begin.0[1].raw,
                                    channel,
                                ]);
                                tile[p.as_index()].write(c.0[channel as usize]);
                            }
                        }
                    }
                    unsafe { tile_out.initialized() };
                }

                Ok(())
            }
            .into()
        },
    )
}

#[cfg(test)]
mod test {
    use crate::{
        array::ImageMetaData,
        data::{GlobalCoordinate, Vector, VoxelPosition},
        operators::{volume::VolumeOperatorState, volume_gpu::VoxelRasterizerGLSL},
        test_util::compare_volume,
    };

    fn test_sliceviewer_configuration(
        img_size: Vector<2, GlobalCoordinate>,
        vol_size: Vector<3, GlobalCoordinate>,
    ) {
        let num_channels = 4;
        let img_size_c = VoxelPosition::from([img_size.y(), img_size.x(), num_channels.into()]);
        for z in 0..vol_size.z().raw {
            let fill_expected = |comp: &mut ndarray::ArrayViewMut3<f32>| {
                for y in 0..img_size_c.0[0].raw {
                    for x in 0..img_size_c.0[1].raw {
                        for c in 0..img_size_c.0[2].raw {
                            let pos = VoxelPosition::from([y, x, c]);
                            let voxel_y = (y as f32 + 0.5) / img_size.y().raw as f32
                                * vol_size.y().raw as f32
                                - 0.5;
                            let voxel_x = (x as f32 + 0.5) / img_size.x().raw as f32
                                * vol_size.x().raw as f32
                                - 0.5;
                            let val = if c == num_channels - 1 {
                                1.0
                            } else {
                                voxel_x.round() + voxel_y.round() + z as f32
                            };
                            comp[pos.as_index()] = val;
                        }
                    }
                }
            };

            let input = VoxelRasterizerGLSL {
                metadata: crate::array::VolumeMetaData {
                    dimensions: vol_size,
                    chunk_size: (vol_size / Vector::fill(2u32)).local(),
                },
                body: r#"result = float(pos_voxel.x + pos_voxel.y + pos_voxel.z);"#.to_owned(),
            };

            let img_meta = ImageMetaData {
                dimensions: img_size,
                chunk_size: (img_size / Vector::fill(3u32)).local(),
            };
            let input = input.operate();
            let slice_proj = super::slice_projection_mat_z(
                input.metadata.clone(),
                crate::operators::scalar::constant_hash(img_meta),
                crate::operators::scalar::constant_hash(z.into()),
            );
            let slice = super::render_slice(
                input,
                crate::operators::scalar::constant_hash(img_meta),
                slice_proj,
            );
            compare_volume(slice, fill_expected);
        }
    }
    #[test]
    fn test_sliceviewer() {
        for img_size in [[5, 3], [6, 6], [10, 20]] {
            for vol_size in [[5, 3, 2], [6, 6, 6], [2, 10, 20]] {
                test_sliceviewer_configuration(img_size.into(), vol_size.into())
            }
        }
    }
}
