use ash::vk;
use derive_more::From;
use futures::StreamExt;
use itertools::Itertools;
use std::hash::Hash;

use crate::{
    data::{ChunkCoordinate, GlobalCoordinate, LocalCoordinate},
    dim::DynDimension,
    dtypes::{DType, ElementType},
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    storage::DataVersionType,
    task::RequestStream,
    vec::Vector,
    vulkan::{
        pipeline::{ComputePipelineBuilder, DescriptorConfig, DynPushConstants},
        shader::Shader,
        DstBarrierInfo, SrcBarrierInfo,
    },
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

    let nd = input.dim().n();

    let push_constants = DynPushConstants::new()
        .vec::<u32>(nd, "mem_size_in")
        .vec::<u32>(nd, "mem_size_out")
        .vec::<u32>(nd, "begin_in")
        .vec::<u32>(nd, "begin_out")
        .vec::<u32>(nd, "region_size")
        .scalar::<u32>("global_size");

    const SHADER: &'static str = r#"
#include <util.glsl>
#include <vec.glsl>
#include <size_util.glsl>

layout(std430, binding = 0) readonly buffer InputBuffer{
    T values[BRICK_MEM_SIZE_IN];
} sourceData;

layout(std430, binding = 1) buffer OutputBuffer{
    T values[];
} outputData;

declare_push_consts(constants);

void main() {
    uint gID = global_position_linear;

    if(gID < constants.global_size) {
        uint[N] region_pos = from_linear(gID, constants.region_size);

        uint[N] in_pos = add(constants.begin_in, region_pos);
        uint[N] out_pos = add(constants.begin_out, region_pos);

        uint in_index = to_linear(in_pos, constants.mem_size_in);
        uint out_index = to_linear(out_pos, constants.mem_size_out);

        outputData.values[out_index] = sourceData.values[in_index];
    }
}
"#;
    TensorOperator::with_state(
        op_descriptor!(),
        input.chunks.dtype(),
        {
            let mut m = input.metadata.clone();
            m.chunk_size = chunk_size.zip(&m.dimensions, |v, d| v.apply(d));
            m
        },
        (input, DataParam(chunk_size), DataParam(push_constants)),
        |ctx, mut positions, (input, chunk_size, push_constants)| {
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
                    (&push_constants, &m_in.num_chunk_elements(), &dtype, &nd),
                    |device, (push_constants, num_elements, dtype, nd)| {
                        ComputePipelineBuilder::new(
                            Shader::new(SHADER)
                                .define("BRICK_MEM_SIZE_IN", num_elements)
                                .define("N", nd)
                                .define("T", dtype.glsl_type())
                                .push_const_block_dyn(push_constants)
                                .ext(dtype.glsl_ext())
                                .ext(Some(crate::vulkan::shader::ext::SCALAR_BLOCK_LAYOUT)),
                        )
                        .use_push_descriptor(true)
                        .build(device)
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

                    device.with_cmd_buffer(|cmd| {
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

                            let descriptor_config =
                                DescriptorConfig::new([gpu_brick_in, &gpu_brick_out]);

                            let global_size = overlap_size.hmul();

                            //TODO initialization of outside regions
                            unsafe {
                                let mut pipeline = pipeline.bind(cmd);

                                pipeline.push_constant_dyn(&push_constants, |consts| {
                                    consts.vec(&m_in.chunk_size.raw())?;
                                    consts.vec(&m_out.chunk_size.raw())?;
                                    consts.vec(&in_chunk_begin.raw())?;
                                    consts.vec(&out_chunk_begin.raw())?;
                                    consts.vec(&overlap_size.raw())?;
                                    consts.scalar(global_size as u32)?;
                                    Ok(())
                                });

                                pipeline.push_descriptor_set(0, descriptor_config);
                                pipeline.dispatch(device, global_size);
                            }

                            if gpu_brick_in.version == DataVersionType::Final {
                                tile_done[tile_index] = 1;
                            }
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
