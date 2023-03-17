use crevice::std140::AsStd140;
use futures::StreamExt;

use crate::{
    array::VolumeMetaData,
    data::{BrickPosition, GlobalCoordinate, LocalCoordinate, Vector, AABB},
    id::Id,
    operator::OperatorId,
    operators::tensor::TensorOperator,
    vulkan::{pipeline::ComputePipeline, state::RessourceId},
};

use super::volume::VolumeOperator;

pub struct SliceViewerState {
    pub metadata: VolumeMetaData,
}

impl SliceViewerState {
    fn new(size: Vector<2, GlobalCoordinate>, chunk_size: Vector<2, LocalCoordinate>) -> Self {
        let n_channels = 4;
        Self {
            metadata: VolumeMetaData {
                dimensions: [size.y(), size.x(), n_channels.into()].into(),
                chunk_size: [chunk_size.y(), chunk_size.x(), n_channels.into()].into(),
            },
        }
    }
    fn projection_mat(&self) -> mint::ColumnMatrix3<f32> {
        todo!()
    }
    fn operate<'a>(&'a self, input: VolumeOperator<'a>) -> VolumeOperator<'a> {
        #[derive(Copy, Clone, AsStd140)]
        struct PushConstants {
            offset: mint::Vector3<u32>,
            mem_dim: mint::Vector3<u32>,
            logical_dim: mint::Vector3<u32>,
            vol_dim: mint::Vector3<u32>,
            num_chunk_elems: u32,
        }

        let m = self.metadata;

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

        TensorOperator::with_state(
            OperatorId::new("sliceviewer").dependent_on(Id::hash(&m)),
            (),
            input,
            move |ctx, _, _| async move { ctx.write(m) }.into(),
            move |ctx, positions, input, _| {
                async move {
                    let device = ctx.vulkan_device();

                    let m_in = ctx.submit(input.metadata.request_scalar()).await;

                    let _pipeline = device
                        .request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                            ComputePipeline::new(device, SHADER)
                        });

                    let transform = self.projection_mat();

                    let requests = positions.into_iter().map(|pos| {
                        let out_info = m.chunk_info(pos);
                        let out_begin_pixel = out_info.begin().raw().map(|v| v as f32).drop_dim(2);
                        let out_end_pixel = out_info.end().raw().map(|v| v as f32).drop_dim(2);

                        let out_begin_voxel = transform * out_begin_pixel.to_homogeneous_coord();
                        let out_end_voxel = transform * out_end_pixel.to_homogeneous_coord();

                        let out_begin_voxel: Vector<3, f32> =
                            mint::Vector3::from(out_begin_voxel).into();

                        let out_end_voxel: Vector<3, f32> =
                            mint::Vector3::from(out_end_voxel).into();

                        let llb =
                            out_begin_voxel.zip(out_end_voxel, |a, b| a.min(b).floor() as u32);
                        let urb =
                            out_begin_voxel.zip(out_end_voxel, |a, b| a.max(b).floor() as u32);

                        let llb_brick = m.chunk_pos(llb.global()).raw();
                        let urb_brick = m.chunk_pos(urb.global()).raw();

                        let in_brick_positions = itertools::iproduct! {
                            llb_brick.z()..=urb_brick.z(),
                            llb_brick.y()..=urb_brick.y(),
                            llb_brick.x()..=urb_brick.x()
                        }
                        .map(|(z, y, x)| BrickPosition::from([z, y, x]))
                        .collect::<Vec<_>>();
                        let intersecting_bricks = ctx.group(
                            in_brick_positions
                                .iter()
                                .map(|pos| input.bricks.request(*pos)),
                        );

                        (intersecting_bricks, (pos, in_brick_positions))
                    });

                    let mut stream = ctx.submit_unordered_with_data(requests);
                    while let Some((intersecting_bricks, (pos, in_brick_positions))) =
                        stream.next().await
                    {
                        let out_info = m.chunk_info(pos);

                        let mut tile_out = ctx.alloc_slot(pos, out_info.mem_elements()).unwrap();
                        let mut tile = crate::data::chunk_mut(&mut tile_out, &out_info);

                        let out_begin = out_info.begin();
                        let out_end = out_info.end();
                        for y in out_begin.y().raw..out_end.y().raw {
                            for x in out_begin.x().raw..out_end.x().raw {
                                let pos = Vector::from([y as f32, x as f32]).to_homogeneous_coord();
                                let sample_pos = transform * pos;

                                let mut val = None;
                                for (brick, b_pos) in
                                    intersecting_bricks.iter().zip(&in_brick_positions)
                                {
                                    let in_info = m_in.chunk_info(*b_pos);
                                    let begin = in_info.begin().map(|v| v.raw as f32);
                                    let end = in_info.end().map(|v| v.raw as f32);
                                    let bb = AABB::new(begin, end);

                                    if bb.contains(sample_pos) {
                                        let local = (sample_pos - begin)
                                            .map(|v| (v.round() as u32))
                                            .global();

                                        let chunk = crate::data::chunk(brick, &in_info);
                                        val = Some(chunk[local.as_index()]);

                                        break;
                                    }
                                }
                                let c = if let Some(v) = val {
                                    Vector::from([v, v, v, 1.0])
                                } else {
                                    Vector::from([0.0, 0.0, 0.0, 0.0])
                                };

                                for channel in 0..4 {
                                    let p: Vector<3, LocalCoordinate> =
                                        Vector::from([y, x, channel]);
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
}
