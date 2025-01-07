use ash::vk;
use derive_more::From;
use futures::StreamExt;
use itertools::Itertools;
use std::hash::Hash;

use crate::{
    chunk_utils::ChunkCopyPipeline,
    data::{ChunkCoordinate, GlobalCoordinate, LocalCoordinate},
    dim::DynDimension,
    dtypes::{DType, ElementType},
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    storage::DataVersionType,
    task::RequestStream,
    vec::Vector,
    vulkan::{DstBarrierInfo, SrcBarrierInfo},
};

use super::tensor::TensorOperator;

#[derive(Copy, Clone, From, Hash, Debug, id::Identify, PartialEq, Eq)]
pub enum ChunkSize {
    Fixed(LocalCoordinate),
    //Relative(f32),
    Full,
}

impl ChunkSize {
    pub fn apply(self, global_size: GlobalCoordinate) -> LocalCoordinate {
        match self {
            ChunkSize::Fixed(l) => l,
            //RechunkArgument::Relative(f) => ((global_size.raw as f32 * f).round() as u32).into(),
            ChunkSize::Full => global_size.local(),
        }
    }
}

pub fn rechunk<D: DynDimension, T: ElementType>(
    input: TensorOperator<D, T>,
    chunk_size: Vector<D, ChunkSize>,
) -> TensorOperator<D, T> {
    let md = &input.metadata;

    // Early return in case of matched sizes
    if md.chunk_size == chunk_size.zip(&md.dimensions, |b, d| b.apply(d)) {
        return input;
    }

    TensorOperator::with_state(
        op_descriptor!(),
        input.chunks.dtype(),
        {
            let mut m = input.metadata.clone();
            m.chunk_size = chunk_size.zip(&m.dimensions, |v, d| v.apply(d));
            m
        },
        (input, DataParam(chunk_size)),
        |ctx, mut positions, (input, chunk_size)| {
            async move {
                let device = ctx.preferred_device();

                let dtype: DType = input.chunks.dtype().into();

                let nd = input.metadata.dimensions.len();

                let m_in = input.metadata.clone();
                let m_out = {
                    let mut m_out = m_in.clone();
                    m_out.chunk_size = chunk_size.zip(&m_in.dimensions, |v, d| v.apply(d));
                    m_out
                };

                positions.sort_by_key(|(v, _)| v.0);

                let pipeline = device.request_state(
                    (&dtype, &m_in.chunk_size),
                    |device, (dtype, chunk_size)| {
                        ChunkCopyPipeline::new(device, *dtype, chunk_size.clone())
                    },
                )?;

                let caches = positions.into_iter().map(|(pos, _)| {
                    let out_info = m_out.chunk_info(pos);
                    let out_begin = out_info.begin();
                    let out_end = out_info.end();

                    let in_begin_brick = m_in.chunk_pos(out_begin);
                    let in_end_brick = m_in.chunk_pos(&out_end.map(|v| v - 1u32));

                    let in_brick_positions = (0..nd)
                        .into_iter()
                        .map(|i| in_begin_brick[i].raw..=in_end_brick[i].raw)
                        .multi_cartesian_product()
                        .map(|coordinates| {
                            m_in.chunk_index(
                                &Vector::<D, ChunkCoordinate>::try_from(coordinates).unwrap(),
                            )
                        })
                        .collect::<Vec<_>>();

                    let tile_done =
                        ctx.access_state_cache::<u8>(pos, "tile_done", in_brick_positions.len());
                    (tile_done, (pos, in_brick_positions))
                });

                let mut stream = ctx
                    .submit_unordered_with_data(caches)
                    .then_req_with_data(*ctx, |(tile_done, (pos, in_brick_positions))| {
                        let mut tile_done = unsafe {
                            tile_done.init(|r| {
                                crate::data::fill_uninit(r, 0);
                            })
                        };
                        let reuse_res =
                            ctx.alloc_try_reuse_gpu(device, pos, m_out.num_chunk_elements());

                        if reuse_res.new {
                            //println!("lost prev version :(");
                            tile_done.fill(0);
                        }

                        let in_brick_positions_to_fill: Vec<_> = in_brick_positions
                            .into_iter()
                            .enumerate()
                            .filter_map(|(i, position)| {
                                (tile_done[i] == 0).then_some((i, position))
                            })
                            .collect();

                        let brick_requests =
                            ctx.group(in_brick_positions_to_fill.iter().map(|(_, pos)| {
                                input.chunks.request_gpu(
                                    device.id,
                                    *pos,
                                    DstBarrierInfo {
                                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                        access: vk::AccessFlags2::SHADER_READ,
                                    },
                                )
                            }));

                        (
                            brick_requests,
                            (
                                reuse_res.request,
                                pos,
                                in_brick_positions_to_fill,
                                tile_done,
                            ),
                        )
                    })
                    .then_req_with_data(
                        *ctx,
                        |(in_bricks, (out_request, pos, in_brick_positions_to_fill, tile_done))| {
                            (
                                out_request,
                                (in_bricks, pos, in_brick_positions_to_fill, tile_done),
                            )
                        },
                    );

                while let Some((
                    gpu_brick_out,
                    (intersecting_bricks, pos, in_brick_positions, mut tile_done),
                )) = stream.next().await
                {
                    let out_info = m_out.chunk_info(pos);

                    let out_begin = out_info.begin();
                    let out_end = out_info.end();

                    //let skip = tile_done.len() - intersecting_bricks.len();
                    //if skip > 0 {
                    //    println!("skipping {}", skip);
                    //}

                    for (gpu_brick_in, (tile_index, in_brick_pos)) in intersecting_bricks
                        .iter()
                        .zip(in_brick_positions.into_iter())
                    {
                        let in_info = m_in.chunk_info(in_brick_pos);

                        let in_begin = in_info.begin();
                        let in_end = in_info.end();

                        let overlap_begin = in_begin.zip(out_begin, |i, o| i.max(o));
                        let overlap_end = in_end.zip(&out_end, |i, o| i.min(o));
                        let overlap_size =
                            (&overlap_end - &overlap_begin).map(LocalCoordinate::interpret_as);

                        let in_chunk_begin = in_info.in_chunk(&overlap_begin);

                        let out_chunk_begin = out_info.in_chunk(&overlap_begin);

                        //TODO initialization of outside regions
                        unsafe {
                            pipeline.run(
                                device,
                                gpu_brick_in,
                                &gpu_brick_out,
                                &in_chunk_begin,
                                &out_chunk_begin,
                                &m_out.chunk_size,
                                &overlap_size,
                            )
                        };

                        if gpu_brick_in.version == DataVersionType::Final {
                            tile_done[tile_index] = 1;
                        }
                    }
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
        test_util::*,
    };

    #[test]
    fn test_rechunk_gpu() {
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let input = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            crate::data::to_linear(&v, &size) as f32
        });

        let fill_expected = |comp: &mut ndarray::ArrayViewMut3<f32>| {
            for z in 0..size.z().raw {
                for y in 0..size.y().raw {
                    for x in 0..size.x().raw {
                        let pos = VoxelPosition::from([z, y, x]);
                        let val = crate::data::to_linear(&pos, &size) as f32;
                        comp[pos.as_index()] = val
                    }
                }
            }
        };
        for chunk_size in [[5, 1, 1], [4, 4, 1], [2, 3, 4], [1, 1, 1], [5, 5, 5]] {
            let output = rechunk(
                input.clone(),
                LocalVoxelPosition::from(chunk_size).into_elem(),
            );
            compare_tensor_fn(output, fill_expected);
        }
    }
}
