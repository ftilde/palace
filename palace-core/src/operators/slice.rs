use ash::vk;
use itertools::Itertools;

use crate::{
    array::TensorMetaData,
    data::{ChunkCoordinate, GlobalCoordinate, LocalCoordinate},
    dim::DynDimension,
    dtypes::{DType, ElementType},
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    vec::Vector,
    vulkan::{
        pipeline::{ComputePipelineBuilder, DescriptorConfig, DynPushConstants},
        shader::Shader,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::{rechunk::ChunkSize, tensor::TensorOperator};

#[derive(Copy, Clone)]
pub enum Range {
    FromTo(u32, u32),
}

impl Range {
    fn apply(&self, size: u32) -> (u32, u32) {
        match self {
            Range::FromTo(begin, end) => (*begin, begin + (end - begin).min(size)),
        }
    }
}

impl From<u32> for Range {
    fn from(value: u32) -> Self {
        Self::FromTo(value, value + 1)
    }
}

impl From<std::ops::Range<u32>> for Range {
    fn from(value: std::ops::Range<u32>) -> Self {
        Self::FromTo(value.start, value.end)
    }
}

pub fn slice<D: DynDimension, T: ElementType>(
    input: TensorOperator<D, T>,
    range: Vector<D, Range>,
    chunk_size: Vector<D, ChunkSize>,
) -> TensorOperator<D, T> {
    let md = &input.metadata;

    let nd = input.dim().n();

    let range = range.zip(&md.dimensions, |r, d| r.apply(d.raw));
    let new_dimensions = range.map(|(from, to)| GlobalCoordinate::from(to - from));
    let new_chunk_size = chunk_size.zip(&md.dimensions, |s, d| s.apply(d));
    let offset = range.map(|(from, _)| from);

    let out_md = TensorMetaData {
        dimensions: new_dimensions,
        chunk_size: new_chunk_size,
    };

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
        out_md.clone(),
        (
            input,
            DataParam(out_md),
            DataParam(offset),
            DataParam(push_constants),
        ),
        |ctx, mut positions, (input, m_out, offset, push_constants)| {
            async move {
                let device = ctx.preferred_device();

                let dtype: DType = input.chunks.dtype().into();

                let nd = input.metadata.dimensions.len();

                let m_in = input.metadata.clone();

                let out_chunk_size = &m_out.chunk_size;

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

                let _ = ctx
                    .run_unordered(positions.into_iter().map(|(pos, _)| {
                        let out_chunk_size = &out_chunk_size;
                        let m_in = &m_in;
                        async move {
                            let out_info = m_out.chunk_info(pos);
                            let out_begin = out_info.begin().clone() + offset.0.clone();
                            let out_end = out_info.end() + offset.0.clone();

                            let input_begin_chunk = m_in.chunk_pos(&out_begin);
                            let input_end_chunk = m_in.chunk_pos(&out_end);
                            let input_dim_in_chunks = m_in.dimension_in_chunks();

                            let in_brick_positions = (0..nd)
                                .into_iter()
                                .map(|i| {
                                    input_begin_chunk[i].raw
                                        ..(input_end_chunk[i].raw + 1)
                                            .min(input_dim_in_chunks[i].raw)
                                })
                                .multi_cartesian_product()
                                .map(|coordinates| {
                                    m_in.chunk_index(
                                        &Vector::<D, ChunkCoordinate>::try_from(coordinates)
                                            .unwrap(),
                                    )
                                })
                                .collect::<Vec<_>>();

                            let in_bricks = ctx
                                .submit(ctx.group(in_brick_positions.iter().map(|pos| {
                                    input.chunks.request_gpu(
                                        device.id,
                                        *pos,
                                        DstBarrierInfo {
                                            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                            access: vk::AccessFlags2::SHADER_READ,
                                        },
                                    )
                                })))
                                .await;

                            let gpu_chunk_out = ctx
                                .submit(ctx.alloc_slot_gpu(device, pos, &out_chunk_size))
                                .await;

                            for (gpu_chunk_in, chunk_pos) in
                                in_bricks.into_iter().zip(in_brick_positions.into_iter())
                            {
                                let in_info = m_in.chunk_info(chunk_pos);

                                let in_begin = in_info.begin();
                                let in_end = in_info.end();

                                let overlap_begin = in_begin.zip(&out_begin, |i, o| i.max(o));
                                let overlap_end = in_end.zip(&out_end, |i, o| i.min(o));
                                let overlap_size = (&overlap_end - &overlap_begin)
                                    .map(LocalCoordinate::interpret_as);

                                let in_chunk_begin = in_info.in_chunk(&overlap_begin);

                                let out_chunk_begin =
                                    out_info.in_chunk(&(overlap_begin - offset.0.clone()));

                                let descriptor_config =
                                    DescriptorConfig::new([&gpu_chunk_in, &gpu_chunk_out]);

                                let global_size = overlap_size.hmul();

                                //TODO initialization of outside regions
                                device.with_cmd_buffer(|cmd| unsafe {
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
                                });
                            }

                            unsafe {
                                gpu_chunk_out.initialized(
                                    *ctx,
                                    SrcBarrierInfo {
                                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                        access: vk::AccessFlags2::SHADER_WRITE,
                                    },
                                )
                            };
                        }
                        .into()
                    }))
                    .await;

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
    fn test_slice_gpu() {
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let input = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            crate::data::to_linear(&v, &size) as f32
        });

        let slice_pos = 2;

        let fill_expected = |comp: &mut ndarray::ArrayViewMut3<f32>| {
            let z = slice_pos;
            for y in 0..size.y().raw {
                for x in 0..size.x().raw {
                    let value_pos = VoxelPosition::from([z, y, x]);
                    let index_pos = VoxelPosition::from([0, y, x]);
                    let val = crate::data::to_linear(&value_pos, &size) as f32;
                    comp[index_pos.as_index()] = val
                }
            }
        };
        for chunk_size in [[5, 1, 1], [4, 4, 1], [2, 3, 4], [1, 1, 1], [5, 5, 5]] {
            let output = slice(
                input.clone(),
                Vector::new([slice_pos.into(), (0..5).into(), (0..5).into()]),
                LocalVoxelPosition::from(chunk_size).into_elem(),
            );
            compare_tensor_fn(output, fill_expected);
        }
    }
}
