use std::{alloc::Layout, collections::VecDeque};

use ash::vk;
use gpu_allocator::MemoryLocation::GpuOnly;
use id::Identify;
use itertools::Itertools;

use crate::{
    aabb::AABB,
    array::{ChunkIndex, TensorEmbeddingData, TensorMetaData},
    chunk_utils::ChunkCopyPipeline,
    coordinate::{ChunkCoordinate, LocalCoordinate},
    data::GlobalCoordinate,
    dim::{DynDimension, LargerDim, D1},
    dtypes::{DType, ScalarType, StaticElementType},
    op_descriptor,
    operator::{DataParam, OpaqueOperator, Operator, OperatorDescriptor, OperatorNetworkNode},
    operators::{
        randomwalker::random_walker_on_chunk,
        resample::resample_transform,
        tensor::{EmbeddedTensorOperator, LODTensorOperator, TensorOperator},
    },
    storage::gpu::InplaceHandle,
    vec::Vector,
    vulkan::{
        memory::TempRessource,
        pipeline::{ComputePipelineBuilder, DescriptorConfig, DynPushConstants, NullBuf},
        shader::Shader,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::SolverConfig;

pub fn hierarchical_random_walker_solver<D: DynDimension + LargerDim>(
    weights: LODTensorOperator<D::Larger, StaticElementType<f32>>,
    points_fg: TensorOperator<D1, DType>,
    points_bg: TensorOperator<D1, DType>,
    cfg: SolverConfig,
) -> LODTensorOperator<D, StaticElementType<f32>> {
    let mut levels = weights.levels.into_iter().rev();
    let root_level = levels.next().unwrap();

    assert!(root_level.inner.metadata.is_single_chunk());

    let root_seeds = super::rasterize_seed_points(
        points_fg.clone(),
        points_bg.clone(),
        root_level.metadata.clone().pop_dim_small(),
        root_level.embedding_data.clone().pop_dim_small(),
    );
    let root_result = super::random_walker_single_chunk(root_level.inner, root_seeds.into(), cfg)
        .embedded(root_level.embedding_data.pop_dim_small());

    let mut output = VecDeque::new();
    output.push_front(root_result.clone());

    let mut prev_result = root_result;
    for level in levels {
        let result = level_step(
            level,
            prev_result,
            points_fg.clone(),
            points_bg.clone(),
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
    fn chunk_info(&self, pos: ChunkIndex) -> ExpandedChunkInfo<D> {
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

    assert!(expansion_by
        .zip(&input.metadata.chunk_size, |l, r| l < r)
        .all());

    let m_out = ExpandedMetaData {
        base: original_metadata,
        expansion_by,
    };

    let op = Operator::with_state(
        op_descriptor!(),
        Default::default(),
        (input, DataParam(m_out.clone())),
        |ctx, mut positions, loc, (input, m_out)| {
            async move {
                let device = ctx.preferred_device(loc);

                let nd = input.dim().n();

                let m_in = &input.metadata;

                let out_chunk_size = m_out.mem_size();

                positions.sort();

                let pipeline = device.request_state(&m_in.chunk_size, |device, chunk_size| {
                    ChunkCopyPipeline::new(device, ScalarType::F32.into(), chunk_size.clone())
                })?;

                let _ = ctx
                    .run_unordered(positions.into_iter().map(|pos| {
                        let out_chunk_size = &out_chunk_size;
                        async move {
                            let chunk_pos = m_in.chunk_pos_from_index(pos);
                            let dim_in_chunks = m_in.dimension_in_chunks();

                            let in_brick_positions = (0..nd)
                                .into_iter()
                                .map(|i| {
                                    chunk_pos[i].raw.saturating_sub(1)
                                        ..(chunk_pos[i].raw + 2).min(dim_in_chunks[i].raw)
                                })
                                .multi_cartesian_product()
                                .map(|coordinates| {
                                    Vector::<D, ChunkCoordinate>::try_from(coordinates).unwrap()
                                })
                                .collect::<Vec<_>>();

                            assert!(in_brick_positions.len() <= 3usize.pow(nd as u32));

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
                                .submit(ctx.alloc_slot_gpu(device, pos, &out_chunk_size))
                                .await;

                            // This is useful for debugging sometimes..
                            //device.with_cmd_buffer(|cmd| unsafe {
                            //    cmd.functions().cmd_fill_buffer(
                            //        cmd.raw(),
                            //        gpu_chunk_out.buffer,
                            //        0,
                            //        vk::WHOLE_SIZE,
                            //        f32::NAN.to_bits(),
                            //    );
                            //});

                            //ctx.submit(device.barrier(
                            //    SrcBarrierInfo {
                            //        stage: vk::PipelineStageFlags2::TRANSFER,
                            //        access: vk::AccessFlags2::TRANSFER_WRITE,
                            //    },
                            //    DstBarrierInfo {
                            //        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            //        access: vk::AccessFlags2::SHADER_READ
                            //            | vk::AccessFlags2::SHADER_WRITE,
                            //    },
                            //))
                            //.await;

                            let out_chunk = m_out.chunk_info(pos);
                            let region_end = out_chunk.end();
                            let region_begin = out_chunk.begin;

                            for (chunk, chunk_pos) in
                                in_bricks.into_iter().zip(in_brick_positions.into_iter())
                            {
                                let read_chunk = m_in.chunk_info_vec(&chunk_pos);
                                let overlap_begin = region_begin.clone().max(read_chunk.begin());
                                let overlap_end = region_end.min(&read_chunk.end());

                                let chunk_dim_out = out_chunk_size.clone();
                                let to_in_offset =
                                    overlap_begin.clone() - read_chunk.begin().clone();
                                let to_out_offset = overlap_begin.clone() - region_begin.clone();
                                let overlap_size = overlap_end - overlap_begin.clone();

                                //TODO initialization of outside regions
                                unsafe {
                                    pipeline.run(
                                        device,
                                        &chunk,
                                        &gpu_chunk_out,
                                        &to_in_offset.local(),
                                        &to_out_offset.local(),
                                        &chunk_dim_out,
                                        &overlap_size.local(),
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
    );

    ExpandedChunkOperator {
        inner: op,
        metadata: m_out,
    }
}

fn run_rw<D: DynDimension + LargerDim>(
    weights: ExpandedChunkOperator<D::Larger, StaticElementType<f32>>,
    init_values: ExpandedChunkOperator<D, StaticElementType<f32>>,
    upper_result: EmbeddedTensorOperator<D, StaticElementType<f32>>,
    points_fg: TensorOperator<D1, DType>,
    points_bg: TensorOperator<D1, DType>,
    current_ed: TensorEmbeddingData<D>,
    cfg: SolverConfig,
) -> ExpandedChunkOperator<D, StaticElementType<f32>> {
    let metadata = init_values.metadata.clone();
    let inner = Operator::with_state(
        op_descriptor!(),
        Default::default(),
        (
            weights,
            init_values,
            upper_result,
            points_fg,
            points_bg,
            DataParam(current_ed),
            DataParam(cfg),
        ),
        |ctx,
         positions,
         loc,
         (weights, init_values, upper_result, points_fg, points_bg, current_ed, cfg)| {
            async move {
                let device = ctx.preferred_device(loc);
                let upper_metadata = &upper_result.metadata;
                let current_metadata = &init_values.metadata;
                let num_chunks = upper_metadata.dimension_in_chunks().hmul();

                let nd = upper_result.dim().n();
                let push_constants = DynPushConstants::new()
                    .scalar::<u64>("page_table_root")
                    .vec::<f32>(nd, "grid_to_grid_scale")
                    .vec::<u32>(nd, "out_tensor_size")
                    .vec::<u32>(nd, "out_chunk_size_memory")
                    .vec::<u32>(nd, "out_chunk_size_logical")
                    .vec::<u32>(nd, "out_begin")
                    .vec::<u32>(nd, "in_dimensions")
                    .vec::<u32>(nd, "in_chunk_size")
                    .scalar::<u32>("num_points_fg")
                    .scalar::<u32>("num_points_bg");

                let pipeline = device.request_state(
                    (
                        &push_constants,
                        num_chunks,
                        upper_metadata.chunk_size.hmul(),
                        nd,
                    ),
                    |device, (push_constants, num_chunks, mem_size, nd)| {
                        ComputePipelineBuilder::new(
                            Shader::new(include_str!("seeds_hierarchical.glsl"))
                                .define("NUM_CHUNKS", num_chunks)
                                .define("BRICK_MEM_SIZE_IN", mem_size)
                                .define("N", nd)
                                .push_const_block_dyn(&push_constants)
                                .ext(Some(crate::vulkan::shader::ext::SCALAR_BLOCK_LAYOUT))
                                .ext(Some(crate::vulkan::shader::ext::BUFFER_REFERENCE))
                                .ext(Some(crate::vulkan::shader::ext::INT64_TYPES)),
                        )
                        .build(device)
                    },
                )?;

                let push_constants = &push_constants;

                let read_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };
                let points_fg_chunk = if points_fg.metadata.num_chunks() > 0 {
                    Some(
                        ctx.submit(points_fg.chunks.request_gpu(
                            device.id,
                            ChunkIndex(0),
                            read_info,
                        ))
                        .await,
                    )
                } else {
                    None
                };

                let points_bg_chunk = if points_bg.metadata.num_chunks() > 0 {
                    Some(
                        ctx.submit(points_bg.chunks.request_gpu(
                            device.id,
                            ChunkIndex(0),
                            read_info,
                        ))
                        .await,
                    )
                } else {
                    None
                };
                let points_fg_chunk = points_fg_chunk
                    .as_ref()
                    .map(|v| v as &dyn crate::vulkan::pipeline::AsDescriptors)
                    .unwrap_or(&NullBuf);
                let points_bg_chunk = points_bg_chunk
                    .as_ref()
                    .map(|v| v as &dyn crate::vulkan::pipeline::AsDescriptors)
                    .unwrap_or(&NullBuf);

                ctx.run_unordered(positions.into_iter().map(|pos| {
                    async move {
                        let out_info = current_metadata.chunk_info(pos);
                        let nd = upper_metadata.dim().n();

                        let out_begin = out_info.begin.clone();
                        let out_end = out_info.end();

                        let aabb = AABB::new(
                            &out_begin.map(|v| v.raw as f32),
                            &out_end.map(|v| (v.raw - 1) as f32),
                        );

                        let element_out_to_in = upper_result.embedding_data.physical_to_voxel()
                            * &current_ed.voxel_to_physical();
                        let aabb = aabb.transform(&element_out_to_in);

                        let out_begin = aabb.lower().map(|v| v.floor().max(0.0) as u32).global();
                        let out_end = aabb.upper().map(|v| v.ceil() as u32).global();

                        let in_begin_brick = upper_metadata.chunk_pos(&out_begin);
                        let in_end_brick = upper_metadata
                            .chunk_pos(&out_end)
                            .zip(&upper_metadata.dimension_in_chunks(), |l, r| {
                                l.min(r - 1u32)
                            });

                        let in_brick_positions = (0..nd)
                            .into_iter()
                            .map(|i| in_begin_brick[i].raw..=in_end_brick[i].raw)
                            .multi_cartesian_product()
                            .map(|coordinates| {
                                Vector::<D, ChunkCoordinate>::try_from(coordinates).unwrap()
                            })
                            .collect::<Vec<_>>();

                        let intersecting_bricks = ctx
                            .submit(ctx.group(in_brick_positions.iter().map(|pos| {
                                upper_result.chunks.request_gpu(
                                    device.id,
                                    upper_metadata.chunk_index(pos),
                                    DstBarrierInfo {
                                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                        access: vk::AccessFlags2::SHADER_READ,
                                    },
                                )
                            })))
                            .await;

                        let flags = vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::TRANSFER_SRC;

                        let seeds_buf = ctx
                            .submit(device.storage.request_allocate_raw(
                                device,
                                Layout::array::<f32>(out_info.mem_size.hmul()).unwrap(),
                                flags,
                                GpuOnly,
                            ))
                            .await;
                        let seeds_buf = TempRessource::new(device, seeds_buf);

                        let min_max_layout = Layout::array::<f32>(2).unwrap();
                        let min_max_buf = ctx
                            .submit(device.storage.request_allocate_raw(
                                device,
                                min_max_layout,
                                flags,
                                GpuOnly,
                            ))
                            .await;
                        let min_max_buf = TempRessource::new(device, min_max_buf);

                        // Init min/max
                        device.with_cmd_buffer(|cmd| {
                            unsafe {
                                device.functions().cmd_update_buffer(
                                    cmd.raw(),
                                    min_max_buf.buffer,
                                    0,
                                    bytemuck::cast_slice(&[1.0f32, 0.0]),
                                )
                            };
                        });

                        let page_table = device
                            .storage
                            .get_page_table(
                                *ctx,
                                device,
                                upper_result.chunks.operator_descriptor(),
                                DstBarrierInfo {
                                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                    access: vk::AccessFlags2::SHADER_READ,
                                },
                            )
                            .await;
                        let page_table_addr = page_table.root();

                        // Make page_table initialization visible
                        ctx.submit(device.barrier(
                            SrcBarrierInfo {
                                stage: vk::PipelineStageFlags2::TRANSFER,
                                access: vk::AccessFlags2::TRANSFER_WRITE,
                            },
                            DstBarrierInfo {
                                stage: vk::PipelineStageFlags2::TRANSFER,
                                access: vk::AccessFlags2::TRANSFER_WRITE,
                            },
                        ))
                        .await;

                        let pos_and_chunk = intersecting_bricks
                            .into_iter()
                            .zip(in_brick_positions.into_iter())
                            .map(|(gpu_brick_in, in_brick_pos)| {
                                let brick_pos_linear = crate::data::to_linear(
                                    &in_brick_pos,
                                    &upper_metadata.dimension_in_chunks(),
                                );
                                (ChunkIndex(brick_pos_linear as u64), gpu_brick_in)
                            })
                            .collect();
                        page_table.insert(*ctx, pos_and_chunk).await;

                        // Make writes to the index visible
                        ctx.submit(device.barrier(
                            SrcBarrierInfo {
                                stage: vk::PipelineStageFlags2::TRANSFER,
                                access: vk::AccessFlags2::TRANSFER_WRITE,
                            },
                            DstBarrierInfo {
                                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                access: vk::AccessFlags2::SHADER_READ,
                            },
                        ))
                        .await;

                        let descriptor_config = DescriptorConfig::new([
                            &*seeds_buf,
                            &*min_max_buf,
                            points_fg_chunk,
                            points_bg_chunk,
                        ]);

                        let grid_to_grid_scale =
                            element_out_to_in.diagonal().to_non_homogeneous_coord();
                        let out_tensor_size = current_metadata.base.dimensions.raw();
                        let out_chunk_size_memory = current_metadata.mem_size().raw();
                        let out_chunk_size_logical = out_info.logical_size().raw();
                        let out_begin = out_info.begin.raw();
                        let in_dimensions = upper_metadata.dimensions.raw();
                        let in_chunk_size = upper_metadata.chunk_size.raw();
                        let num_points_fg = *points_fg.metadata.dimensions.raw();
                        let num_points_bg = *points_bg.metadata.dimensions.raw();

                        device.with_cmd_buffer(|cmd| unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            pipeline.push_constant_dyn(push_constants, |consts| {
                                consts.scalar(page_table_addr.0)?;
                                consts.vec(&grid_to_grid_scale)?;
                                consts.vec(&out_tensor_size)?;
                                consts.vec(&out_chunk_size_memory)?;
                                consts.vec(&out_chunk_size_logical)?;
                                consts.vec(&out_begin)?;
                                consts.vec(&in_dimensions)?;
                                consts.vec(&in_chunk_size)?;
                                consts.scalar(num_points_fg)?;
                                consts.scalar(num_points_bg)?;
                                Ok(())
                            });

                            pipeline.write_descriptor_set(0, descriptor_config);
                            pipeline.dispatch_dyn(device, out_chunk_size_memory);
                        });

                        // Make seeds buffer writes visible
                        ctx.submit(device.barrier(
                            SrcBarrierInfo {
                                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                access: vk::AccessFlags2::SHADER_WRITE,
                            },
                            DstBarrierInfo {
                                stage: vk::PipelineStageFlags2::COMPUTE_SHADER
                                    | vk::PipelineStageFlags2::TRANSFER,
                                access: vk::AccessFlags2::SHADER_READ
                                    | vk::AccessFlags2::TRANSFER_READ,
                            },
                        ))
                        .await;

                        let mut min_max_cpu = [0.0f32; 2];
                        let min_max_cpu_bytes: &mut [u8] =
                            bytemuck::cast_slice_mut(min_max_cpu.as_mut_slice());
                        unsafe {
                            crate::vulkan::memory::copy_to_cpu(
                                *ctx,
                                device,
                                min_max_buf.buffer,
                                min_max_layout,
                                min_max_cpu_bytes.as_mut_ptr().cast(),
                            )
                            .await
                        };

                        let [min, max] = min_max_cpu;
                        let dt = 0.1;

                        let skip = min > 0.5 + dt || max < 0.5 - dt;

                        if skip {
                            let access_info = DstBarrierInfo {
                                stage: vk::PipelineStageFlags2::TRANSFER,
                                access: vk::AccessFlags2::TRANSFER_READ,
                            };
                            let inplace_res = ctx
                                .submit(init_values.inner.request_inplace_gpu(
                                    device.id,
                                    pos,
                                    ctx.current_op_desc().unwrap(),
                                    init_values.inner.dtype().into(),
                                    init_values.metadata.mem_size().hmul(),
                                    access_info,
                                ))
                                .await;

                            let inplace_res = ctx.submit(inplace_res.alloc()).await;
                            if let InplaceHandle::New(r, w) = &inplace_res {
                                device.with_cmd_buffer(|cmd| {
                                    let copy_info =
                                        vk::BufferCopy::default().size(r.layout.size() as _);
                                    unsafe {
                                        device.functions().cmd_copy_buffer(
                                            cmd.raw(),
                                            r.buffer,
                                            w.buffer,
                                            &[copy_info],
                                        );
                                    }
                                });
                            }

                            unsafe {
                                inplace_res.initialized(
                                    *ctx,
                                    SrcBarrierInfo {
                                        stage: vk::PipelineStageFlags2::TRANSFER,
                                        access: vk::AccessFlags2::TRANSFER_WRITE,
                                    },
                                )
                            };
                            Ok(())
                        } else {
                            // Actual rw solving:

                            let chunk_info_weights = weights.metadata.chunk_info(pos);
                            let chunk_info_init = init_values.metadata.chunk_info(pos);
                            assert_eq!(
                                out_info.logical_size(),
                                chunk_info_weights.logical_size().pop_dim_small()
                            );
                            assert_eq!(out_info.logical_size(), chunk_info_init.logical_size());
                            assert_eq!(
                                out_info.mem_size,
                                chunk_info_weights.mem_size.pop_dim_small(),
                            );
                            assert_eq!(out_info.mem_size, chunk_info_init.mem_size,);

                            let virtual_chunk_info = TensorMetaData {
                                dimensions: out_info.logical_size().global(),
                                chunk_size: out_info.mem_size,
                            };
                            random_walker_on_chunk(
                                ctx,
                                device,
                                &weights.inner,
                                &*seeds_buf,
                                Some(&init_values.inner),
                                pos,
                                **cfg,
                                virtual_chunk_info,
                            )
                            .await
                        }
                    }
                    .into()
                }))
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;

                Ok(())
            }
            .into()
        },
    );
    ExpandedChunkOperator { inner, metadata }
}

fn shrink<D: DynDimension>(
    input: ExpandedChunkOperator<D, StaticElementType<f32>>,
) -> TensorOperator<D, StaticElementType<f32>> {
    TensorOperator::with_state(
        op_descriptor!(),
        Default::default(),
        input.metadata.base.clone(),
        input,
        |ctx, positions, loc, input| {
            async move {
                let device = ctx.preferred_device(loc);

                let m_in = &input.metadata;
                let m_out = &m_in.base;
                let nd = m_out.dim().n();

                let out_chunk_size = &m_out.chunk_size;
                let in_chunk_size = m_in.mem_size();

                let pipeline = device.request_state(&in_chunk_size, |device, chunk_size| {
                    ChunkCopyPipeline::new(device, ScalarType::F32.into(), chunk_size.clone())
                })?;

                let _ = ctx
                    .run_unordered(positions.into_iter().map(|pos| {
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
                                .submit(ctx.alloc_slot_gpu(device, pos, &out_chunk_size))
                                .await;

                            let chunk_info = m_in.chunk_info(pos);
                            let out_size = &m_out.chunk_size;

                            unsafe {
                                pipeline.run(
                                    device,
                                    &in_chunk,
                                    &gpu_chunk_out,
                                    &chunk_info.size_before,
                                    &Vector::fill_with_len(0, nd).local(),
                                    &out_size,
                                    &out_size,
                                )
                            };

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

fn level_step<D: DynDimension + LargerDim>(
    current_level_weights: EmbeddedTensorOperator<D::Larger, StaticElementType<f32>>,
    upper_result: EmbeddedTensorOperator<D, StaticElementType<f32>>,
    points_fg: TensorOperator<D1, DType>,
    points_bg: TensorOperator<D1, DType>,
    cfg: SolverConfig,
) -> EmbeddedTensorOperator<D, StaticElementType<f32>> {
    let level_md = current_level_weights.metadata.clone().pop_dim_small();
    let level_ed = current_level_weights.embedding_data.clone().pop_dim_small();
    let expansion_by = level_md
        .chunk_size
        .map(|s| (((s.raw as f32 * 0.125) as u32).max(1)).into());
    let expanded_weights = expand(
        current_level_weights.inner,
        expansion_by.push_dim_small(0.into()),
    );

    let current_to_upper =
        upper_result.embedding_data.physical_to_voxel() * &level_ed.voxel_to_physical();

    let upper_resampled = resample_transform(
        upper_result.inner.clone(),
        level_md.clone(),
        current_to_upper,
        crate::operators::conv::BorderHandling::Repeat,
    );
    let upper_expanded = expand(upper_resampled, expansion_by.clone());

    let world_to_grid = level_ed.physical_to_voxel();
    let points_fg = crate::operators::geometry::transform(points_fg, world_to_grid.clone());
    let points_bg = crate::operators::geometry::transform(points_bg, world_to_grid);

    let expanded_result = run_rw(
        expanded_weights,
        upper_expanded,
        upper_result,
        points_fg,
        points_bg,
        level_ed.clone(),
        cfg,
    );
    shrink(expanded_result).embedded(level_ed)
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
