use std::collections::VecDeque;

use ash::vk;
use id::Identify;
use itertools::Itertools;

use crate::{
    array::{ChunkIndex, TensorEmbeddingData, TensorMetaData},
    coordinate::{ChunkCoordinate, LocalCoordinate},
    data::GlobalCoordinate,
    dim::{DynDimension, LargerDim, D1, D3},
    dtypes::{DType, StaticElementType},
    op_descriptor,
    operator::{DataParam, Operator, OperatorDescriptor, OperatorNetworkNode},
    operators::tensor::{EmbeddedTensorOperator, LODTensorOperator, TensorOperator},
    vec::Vector,
    vulkan::{
        pipeline::{ComputePipelineBuilder, DescriptorConfig, DynPushConstants},
        shader::Shader,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::{random_walker_weights, SolverConfig, WeightFunction};

type D = D3;

pub fn hierchical_random_walker(
    tensor: LODTensorOperator<D3, StaticElementType<f32>>,
    points_fg: TensorOperator<D1, DType>,
    points_bg: TensorOperator<D1, DType>,
    weight_function: WeightFunction,
    min_edge_weight: f32,
    cfg: SolverConfig,
) -> LODTensorOperator<D3, StaticElementType<f32>> {
    let mut levels = tensor.levels.into_iter().rev();
    let root_level = levels.next().unwrap();

    assert!(root_level.inner.metadata.is_single_chunk());

    let root_seeds = super::rasterize_seed_points(
        points_fg.clone(),
        points_bg.clone(),
        root_level.metadata,
        root_level.embedding_data,
    );
    let root_result = super::random_walker(
        root_level.inner,
        root_seeds.into(),
        weight_function,
        min_edge_weight,
        cfg,
    )
    .embedded(root_level.embedding_data);

    let mut output = VecDeque::new();
    output.push_front(root_result.clone());

    let mut prev_result = root_result;
    for level in levels {
        let result = level_step(
            level,
            prev_result,
            points_fg.clone(),
            points_bg.clone(),
            weight_function,
            min_edge_weight,
            cfg,
        );
        output.push_front(result.clone());
        prev_result = result;
    }

    LODTensorOperator {
        levels: output.into(),
    }
}

#[derive(Clone, Identify)]
struct ExpandedChunkOperator<D: DynDimension, E> {
    inner: Operator<E>,
    metadata: ExpandedMetaData<D>,
}

impl<D: DynDimension, E> OperatorNetworkNode for ExpandedChunkOperator<D, E> {
    fn descriptor(&self) -> OperatorDescriptor {
        self.inner.descriptor()
    }
}

struct ExpandedChunkInfo<D: DynDimension> {
    mem_size: Vector<D, LocalCoordinate>,
    size_before: Vector<D, LocalCoordinate>,
    size_after: Vector<D, LocalCoordinate>,
    size_core: Vector<D, LocalCoordinate>,
    begin: Vector<D, GlobalCoordinate>,
}

#[derive(Clone, Identify)]
struct ExpandedMetaData<D: DynDimension> {
    base: TensorMetaData<D>,
    expansion_by: Vector<D, LocalCoordinate>,
}

impl<D: DynDimension> ExpandedMetaData<D> {
    fn mem_size(&self) -> Vector<D, LocalCoordinate> {
        &self.base.chunk_size + &self.expansion_by.scale(2.into())
    }
    fn chunk(&self, pos: ChunkIndex) -> ExpandedChunkInfo<D> {
        let chunk_base = self.base.chunk_info(pos);
        let begin = chunk_base
            .begin()
            .saturating_sub(&self.expansion_by.global());
        let end = (chunk_base.end() + self.expansion_by.global()).min(&self.base.dimensions);
        let size_before = (chunk_base.begin() - &begin).local();
        let size_core = chunk_base.logical_dimensions.clone();
        let size_after = (end - chunk_base.end()).local();
        let mem_size = self.mem_size();

        ExpandedChunkInfo {
            mem_size,
            size_before,
            size_after,
            size_core,
            begin,
        }
    }
}

impl<D: DynDimension> ExpandedChunkInfo<D> {
    fn logical_size(&self) -> Vector<D, LocalCoordinate> {
        &(&self.size_before + &self.size_after) + &self.size_core
    }

    fn end(&self) -> Vector<D, GlobalCoordinate> {
        &self.begin + &self.logical_size()
    }
}

fn expand<D: DynDimension>(
    input: TensorOperator<D, StaticElementType<f32>>,
    expansion_by: Vector<D, LocalCoordinate>,
) -> ExpandedChunkOperator<D, StaticElementType<f32>> {
    let original_metadata = input.metadata.clone();

    let nd = input.dim().n();

    assert!(expansion_by
        .zip(&input.metadata.chunk_size, |l, r| l < r)
        .all());

    let push_constants = DynPushConstants::new()
        .vec::<u32>(nd, "tensor_dim_in")
        .vec::<u32>(nd, "tensor_dim_out")
        .vec::<u32>(nd, "to_in_offset")
        .vec::<u32>(nd, "to_out_offset")
        .vec::<u32>(nd, "overlap_dim")
        .scalar::<u32>("overlap_size");

    let m_out = ExpandedMetaData {
        base: original_metadata,
        expansion_by,
    };

    let op = Operator::with_state(
        op_descriptor!(),
        Default::default(),
        (input, DataParam(m_out.clone()), DataParam(push_constants)),
        |ctx, mut positions, (input, m_out, push_constants)| {
            async move {
                let device = ctx.preferred_device();

                let nd = input.dim().n();

                let m_in = &input.metadata;

                let out_chunk_size = m_out.mem_size();
                let out_mem_size = out_chunk_size.hmul();

                positions.sort_by_key(|(v, _)| v.0);

                let pipeline = device.request_state(
                    (m_in.chunk_size.hmul(), out_mem_size, nd, &push_constants),
                    |device, (mem_size_in, mem_size_out, nd, push_constants)| {
                        ComputePipelineBuilder::new(
                            Shader::new(include_str!("expand_sample.glsl"))
                                .define("N", nd)
                                .define("BRICK_MEM_SIZE_IN", mem_size_in)
                                .define("BRICK_MEM_SIZE_OUT", mem_size_out)
                                .push_const_block_dyn(&push_constants),
                        )
                        .use_push_descriptor(true)
                        .build(device)
                    },
                )?;

                ctx.run_unordered(positions.into_iter().map(|(pos, _)| {
                    let out_chunk_size = &out_chunk_size;
                    async move {
                        let chunk_pos = m_in.chunk_pos_from_index(pos);
                        let dim_in_chunks = m_in.dimension_in_chunks();

                        let in_brick_positions = (0..nd)
                            .into_iter()
                            .map(|i| {
                                chunk_pos[i].raw.saturating_sub(1)
                                    ..(chunk_pos[i].raw + 1).min(dim_in_chunks[i].raw)
                            })
                            .multi_cartesian_product()
                            .map(|coordinates| {
                                Vector::<D, ChunkCoordinate>::try_from(coordinates).unwrap()
                            })
                            .collect::<Vec<_>>();

                        let in_bricks = ctx
                            .submit(ctx.group(in_brick_positions.iter().map(|pos| {
                                input.chunks.request_gpu(
                                    device.id,
                                    m_in.chunk_index(pos),
                                    DstBarrierInfo {
                                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                        access: vk::AccessFlags2::SHADER_READ,
                                    },
                                )
                            })))
                            .await;

                        let gpu_chunk_out = ctx
                            .submit(ctx.alloc_slot_gpu(device, pos, out_mem_size))
                            .await;

                        let out_chunk = m_out.chunk(pos);
                        let region_end = out_chunk.end();
                        let region_begin = out_chunk.begin;

                        for (chunk, chunk_pos) in
                            in_bricks.into_iter().zip(in_brick_positions.into_iter())
                        {
                            let read_chunk = m_in.chunk_info_vec(&chunk_pos);
                            let overlap_begin = region_begin.clone().max(read_chunk.begin());
                            let overlap_end = region_end.min(&read_chunk.end());

                            let descriptor_config = DescriptorConfig::new([&chunk, &gpu_chunk_out]);

                            let tensor_dim_in = m_in.chunk_size.raw();
                            let tensor_dim_out = out_chunk_size.clone().raw();
                            let to_in_offset =
                                (overlap_begin.clone() - read_chunk.begin().clone()).raw();
                            let to_out_offset =
                                (overlap_begin.clone() - region_begin.clone()).raw();
                            let overlap_dim = (overlap_end - overlap_begin.clone()).raw();
                            let overlap_size = overlap_dim.hmul() as u32;

                            device.with_cmd_buffer(|cmd| unsafe {
                                let mut pipeline = pipeline.bind(cmd);

                                pipeline.push_constant_dyn(&push_constants, |w| {
                                    w.vec(&tensor_dim_in)?;
                                    w.vec(&tensor_dim_out)?;
                                    w.vec(&to_in_offset)?;
                                    w.vec(&to_out_offset)?;
                                    w.vec(&overlap_dim)?;
                                    w.scalar(overlap_size)?;

                                    Ok(())
                                });
                                pipeline.push_descriptor_set(0, descriptor_config);
                                pipeline.dispatch_dyn(device, overlap_dim);
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
    );

    ExpandedChunkOperator {
        inner: op,
        metadata: m_out,
    }
}

fn expanded_seeds(
    upper_result: EmbeddedTensorOperator<D3, StaticElementType<f32>>,
    points_fg: TensorOperator<D1, DType>,
    points_bg: TensorOperator<D1, DType>,
    out_md: ExpandedMetaData<D>,
    out_ed: TensorEmbeddingData<D>,
) -> ExpandedChunkOperator<D3, StaticElementType<f32>> {
    todo!()
}

fn run_rw(
    weights: ExpandedChunkOperator<<D3 as LargerDim>::Larger, StaticElementType<f32>>,
    seeds: ExpandedChunkOperator<D3, StaticElementType<f32>>,
    cfg: SolverConfig,
) -> ExpandedChunkOperator<D3, StaticElementType<f32>> {
    todo!()
}

fn shrink(
    input: ExpandedChunkOperator<D3, StaticElementType<f32>>,
) -> TensorOperator<D3, StaticElementType<f32>> {
    let nd = input.metadata.base.dim().n();

    let push_constants = DynPushConstants::new()
        .vec::<u32>(nd, "tensor_dim_in")
        .vec::<u32>(nd, "tensor_dim_out")
        .vec::<u32>(nd, "to_in_offset")
        .scalar::<u32>("tensor_out_size");

    TensorOperator::with_state(
        op_descriptor!(),
        Default::default(),
        input.metadata.base.clone(),
        (input, DataParam(push_constants)),
        |ctx, positions, (input, push_constants)| {
            async move {
                let device = ctx.preferred_device();

                let m_in = &input.metadata;
                let m_out = &m_in.base;
                let nd = m_out.dim().n();

                let out_chunk_size = m_out.chunk_size;
                let in_chunk_size = m_in.mem_size();

                let pipeline = device.request_state(
                    (
                        in_chunk_size.hmul(),
                        out_chunk_size.hmul(),
                        nd,
                        &push_constants,
                    ),
                    |device, (mem_size_in, mem_size_out, nd, push_constants)| {
                        ComputePipelineBuilder::new(
                            Shader::new(include_str!("shrink_sample.glsl"))
                                .define("N", nd)
                                .define("BRICK_MEM_SIZE_IN", mem_size_in)
                                .define("BRICK_MEM_SIZE_OUT", mem_size_out)
                                .push_const_block_dyn(&push_constants),
                        )
                        .use_push_descriptor(true)
                        .build(device)
                    },
                )?;

                ctx.run_unordered(positions.into_iter().map(|(pos, _)| {
                    async move {
                        let in_chunk = ctx
                            .submit(input.inner.request_gpu(
                                device.id,
                                pos,
                                DstBarrierInfo {
                                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                    access: vk::AccessFlags2::SHADER_READ,
                                },
                            ))
                            .await;

                        let gpu_chunk_out = ctx
                            .submit(ctx.alloc_slot_gpu(device, pos, out_chunk_size.hmul()))
                            .await;

                        let descriptor_config = DescriptorConfig::new([&in_chunk, &gpu_chunk_out]);

                        let chunk_info = m_in.chunk(pos);
                        let out_size = m_out.chunk_size;

                        device.with_cmd_buffer(|cmd| unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            pipeline.push_constant_dyn(&push_constants, |w| {
                                w.vec(&m_in.mem_size().raw())?;
                                w.vec(&out_size.raw())?;
                                w.vec(&chunk_info.size_before.raw())?;
                                w.scalar(out_size.hmul() as u32)?;

                                Ok(())
                            });
                            pipeline.push_descriptor_set(0, descriptor_config);
                            pipeline.dispatch_dyn(device, out_size.raw());
                        });

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

fn level_step(
    current_level: EmbeddedTensorOperator<D3, StaticElementType<f32>>,
    upper_result: EmbeddedTensorOperator<D3, StaticElementType<f32>>,
    points_fg: TensorOperator<D1, DType>,
    points_bg: TensorOperator<D1, DType>,
    weight_function: WeightFunction,
    min_edge_weight: f32,
    cfg: SolverConfig,
) -> EmbeddedTensorOperator<D3, StaticElementType<f32>> {
    let expansion_by = current_level
        .metadata
        .chunk_size
        .map(|s| ((s.raw as f32 * 0.125) as u32).into());
    let weights = random_walker_weights(
        current_level.inner.clone(),
        weight_function,
        min_edge_weight,
    );
    let expanded_weights = expand(weights, expansion_by.push_dim_small(0.into()));
    let seeds = expanded_seeds(
        upper_result,
        points_fg,
        points_bg,
        ExpandedMetaData {
            base: current_level.metadata,
            expansion_by,
        },
        current_level.embedding_data,
    );

    let expanded_result = run_rw(expanded_weights, seeds, cfg);
    shrink(expanded_result).embedded(current_level.embedding_data)
}

#[cfg(test)]
mod test {
    use crate::{
        test_util::compare_tensor,
        vec::{LocalVoxelPosition, VoxelPosition},
    };

    use super::*;

    #[test]
    fn expand_shrink() {
        let size = VoxelPosition::from([100, 100, 100]);
        let brick_size = LocalVoxelPosition::from([32, 32, 32]);
        let expansion = LocalVoxelPosition::from([7, 7, 7]);

        let vol = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            (v.x() + v.y() + v.z()).raw as f32
        });

        let expanded = expand(vol.clone(), expansion);

        let result = shrink(expanded);

        compare_tensor(vol, result);
    }
}
