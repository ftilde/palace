use ash::vk;
use itertools::Itertools;

use crate::{
    array::{TensorEmbeddingData, TensorMetaData},
    chunk_utils::ChunkCopyPipeline,
    data::{ChunkCoordinate, GlobalCoordinate, LocalCoordinate},
    dim::{DDyn, DynDimension, SmallerDim},
    dtypes::{DType, ElementType},
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    vec::Vector,
    vulkan::{DstBarrierInfo, SrcBarrierInfo},
};

use super::{
    rechunk::ChunkSize,
    tensor::{EmbeddedTensorOperator, TensorOperator},
};

#[derive(Copy, Clone)]
pub enum Range {
    Scalar(u32),
    FromTo(u32, u32),
}

impl Range {
    fn apply(&self, size: u32) -> (u32, u32) {
        match self {
            Range::Scalar(at) => {
                let start = (*at).min(size);
                let end = (at + 1).min(size);
                (start, end)
            }
            Range::FromTo(begin, end) => ((*begin).min(size), (*end).min(size)),
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

pub fn try_squash_dim<D: DynDimension + SmallerDim, T: ElementType>(
    input: TensorOperator<D, T>,
    dim: usize,
) -> Result<TensorOperator<D::Smaller, T>, crate::Error> {
    if input.metadata.dimensions[dim].raw != 1 {
        return Err(format!(
            "Dimensions must be one to squash, but is {}",
            input.metadata.dimensions[dim].raw
        )
        .into());
    }
    if input.metadata.chunk_size[dim].raw != 1 {
        return Err(format!(
            "Chunk size must be one to squash, but is {}",
            input.metadata.chunk_size[dim].raw
        )
        .into());
    }

    let new_md = TensorMetaData {
        dimensions: input.metadata.dimensions.drop_dim(dim),
        chunk_size: input.metadata.chunk_size.drop_dim(dim),
    };

    Ok(TensorOperator {
        metadata: new_md,
        chunks: input.chunks,
    })
}

pub fn slice_and_squash<D: DynDimension + SmallerDim, T: ElementType>(
    input: TensorOperator<D, T>,
    range: Vector<D, Range>,
) -> Result<TensorOperator<DDyn, T>, crate::Error> {
    let mut input = slice(input.into_dyn(), range.clone().into_dyn())?;
    for (dim, arg) in range.into_iter().enumerate().rev() {
        if matches!(arg, Range::Scalar(_)) {
            input = try_squash_dim(input, dim).unwrap();
        }
    }
    Ok(input)
}
pub fn squash_embedding_data_for_slice<D: DynDimension + SmallerDim>(
    input: TensorEmbeddingData<D>,
    range: Vector<D, Range>,
) -> TensorEmbeddingData<DDyn> {
    let mut ed = input.into_dyn();
    for (dim, arg) in range.into_iter().enumerate().rev() {
        if matches!(arg, Range::Scalar(_)) {
            ed = ed.drop_dim(dim)
        }
    }
    ed
}

pub fn slice_and_squash_embedded<D: DynDimension + SmallerDim, T: ElementType>(
    input: EmbeddedTensorOperator<D, T>,
    range: Vector<D, Range>,
) -> Result<EmbeddedTensorOperator<DDyn, T>, crate::Error> {
    Ok(
        slice_and_squash(input.inner, range.clone())?.embedded(squash_embedding_data_for_slice(
            input.embedding_data.into_dyn(),
            range.into_dyn(),
        )),
    )
}

pub fn slice<D: DynDimension, T: ElementType>(
    input: TensorOperator<D, T>,
    range: Vector<D, Range>,
) -> Result<TensorOperator<D, T>, crate::Error> {
    let actual_range = range.zip(&input.metadata.dimensions, |r, d| r.apply(d.raw));
    let chunk_size = input
        .metadata
        .chunk_size
        .zip(&actual_range, |c, (from, to)| {
            ChunkSize::Fixed((to - from).min(c.raw).into())
        });

    slice_and_rechunk(input, range, chunk_size)
}

pub fn slice_and_rechunk<D: DynDimension, T: ElementType>(
    input: TensorOperator<D, T>,
    range: Vector<D, Range>,
    chunk_size: Vector<D, ChunkSize>,
) -> Result<TensorOperator<D, T>, crate::Error> {
    let md = &input.metadata;

    let range = range.zip(&md.dimensions, |r, d| r.apply(d.raw));
    let new_dimensions = range.map(|(from, to)| GlobalCoordinate::from(to - from));
    let new_chunk_size = chunk_size.zip(&md.dimensions, |s, d| s.apply(d));
    let offset = range.map(|(from, _)| from);

    if new_dimensions.hmul() == 0 {
        return Err("Slicing would result in a zero size tensor".into());
    }

    let out_md = TensorMetaData {
        dimensions: new_dimensions,
        chunk_size: new_chunk_size,
    };
    Ok(TensorOperator::with_state(
        op_descriptor!(),
        input.chunks.dtype(),
        out_md.clone(),
        (input, DataParam(out_md), DataParam(offset)),
        |ctx, mut positions, (input, m_out, offset)| {
            async move {
                let device = ctx.preferred_device();

                let dtype: DType = input.chunks.dtype().into();

                let nd = input.metadata.dimensions.len();

                let m_in = input.metadata.clone();

                let out_chunk_size = &m_out.chunk_size;

                positions.sort_by_key(|(v, _)| v.0);

                let pipeline = device.request_state(
                    (&dtype, &m_in.chunk_size),
                    |device, (dtype, chunk_size)| {
                        ChunkCopyPipeline::new(device, *dtype, chunk_size.clone())
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
                            let out_last = out_end.clone() - Vector::fill_with_len(1u32, nd);

                            let input_begin_chunk = m_in.chunk_pos(&out_begin);
                            let input_last_chunk = m_in.chunk_pos(&out_last);
                            let input_dim_in_chunks = m_in.dimension_in_chunks();

                            let in_brick_positions = (0..nd)
                                .into_iter()
                                .map(|i| {
                                    input_begin_chunk[i].raw
                                        ..(input_last_chunk[i].raw + 1)
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

                                //TODO initialization of outside regions
                                unsafe {
                                    pipeline.run(
                                        device,
                                        &gpu_chunk_in,
                                        &gpu_chunk_out,
                                        &in_chunk_begin,
                                        &out_chunk_begin,
                                        &m_out.chunk_size,
                                        &overlap_size,
                                    )
                                };
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
    ))
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
            let output = slice_and_rechunk(
                input.clone(),
                Vector::new([slice_pos.into(), (0..5).into(), (0..5).into()]),
                LocalVoxelPosition::from(chunk_size).into_elem(),
            )
            .unwrap();
            compare_tensor_fn(output, fill_expected);
        }
    }
}
