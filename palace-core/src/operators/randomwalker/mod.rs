use std::alloc::Layout;

use ash::vk;

use super::tensor::{EmbeddedTensorOperator, LODTensorOperator, TensorOperator};
use crate::{
    array::{ChunkIndex, TensorEmbeddingData, TensorMetaData},
    dim::{DynDimension, LargerDim, D1, D3},
    dtypes::{DType, ScalarType, StaticElementType},
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    task::OpaqueTaskContext,
    vulkan::{
        pipeline::{
            AsBufferDescriptor, ComputePipelineBuilder, DescriptorConfig, DynPushConstants, NullBuf,
        },
        shader::Shader,
        DeviceContext, DstBarrierInfo, SrcBarrierInfo,
    },
};

mod hierarchical;
mod solver;
mod weights;

pub use hierarchical::*;
pub use solver::*;
pub use weights::*;

pub fn rasterize_seed_points<D: DynDimension + LargerDim>(
    points_fg: TensorOperator<D1, DType>,
    points_bg: TensorOperator<D1, DType>,
    md: TensorMetaData<D>,
    ed: TensorEmbeddingData<D>,
) -> EmbeddedTensorOperator<D, StaticElementType<f32>> {
    assert_eq!(md.dim(), ed.dim());
    let in_dtype = points_fg.dtype();
    let nd = md.dim().n();
    assert_eq!(in_dtype, points_bg.dtype());
    assert_eq!(in_dtype.size, nd as u32);
    assert_eq!(in_dtype.scalar, ScalarType::F32);
    assert!(md.is_single_chunk());

    TensorOperator::unbatched(
        op_descriptor!(),
        Default::default(),
        md.clone(),
        (points_fg, points_bg, DataParam(md), DataParam(ed.clone())),
        |ctx, _pos, _, (points_fg, points_bg, md, ed)| {
            async move {
                let device = ctx.preferred_device();

                let nd = md.dim().n();

                let push_constants = DynPushConstants::new()
                    .mat::<f32>(nd + 1, "to_grid")
                    .vec::<u32>(nd, "tensor_dim_memory")
                    .vec::<u32>(nd, "tensor_dim_logical")
                    .scalar::<u32>("num_points_fg")
                    .scalar::<u32>("num_points_bg");

                let to_grid = ed.physical_to_voxel();

                let pipeline = device.request_state(
                    (&push_constants, md.chunk_size.hmul(), nd),
                    |device, (push_constants, mem_size, nd)| {
                        ComputePipelineBuilder::new(
                            Shader::new(include_str!("rasterize_points.glsl"))
                                .push_const_block_dyn(push_constants)
                                .define("BRICK_MEM_SIZE", mem_size)
                                .define("ND", nd),
                        )
                        .build(device)
                    },
                )?;

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

                let out_chunk = ctx
                    .submit(ctx.alloc_slot_gpu(&device, ChunkIndex(0), &md.chunk_size))
                    .await;

                let global_size = md.chunk_size.raw();

                let descriptor_config =
                    DescriptorConfig::new([points_fg_chunk, points_bg_chunk, &out_chunk]);

                device.with_cmd_buffer(|cmd| unsafe {
                    let mut pipeline = pipeline.bind(cmd);

                    pipeline.push_constant_dyn(&push_constants, |consts| {
                        consts.mat(&to_grid)?;
                        consts.vec(&md.chunk_size.raw())?;
                        consts.vec(&md.dimensions.raw())?;
                        consts.scalar(points_fg.metadata.dimensions[0].raw)?;
                        consts.scalar(points_bg.metadata.dimensions[0].raw)?;
                        Ok(())
                    });
                    pipeline.write_descriptor_set(0, descriptor_config);
                    pipeline.dispatch_dyn(device, global_size);
                });

                unsafe {
                    out_chunk.initialized(
                        *ctx,
                        SrcBarrierInfo {
                            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            access: vk::AccessFlags2::SHADER_WRITE,
                        },
                    )
                };

                Ok(())
            }
            .into()
        },
    )
    .embedded(ed)
}

pub fn random_walker(
    tensor: TensorOperator<D3, StaticElementType<f32>>,
    seeds: TensorOperator<D3, StaticElementType<f32>>,
    weight_function: WeightFunction,
    min_edge_weight: f32,
    cfg: SolverConfig,
) -> TensorOperator<D3, StaticElementType<f32>> {
    let weights = random_walker_weights(tensor, weight_function, min_edge_weight);
    random_walker_single_chunk(weights, seeds, cfg)
}

pub fn hierarchical_random_walker(
    tensor: LODTensorOperator<D3, StaticElementType<f32>>,
    points_fg: TensorOperator<D1, DType>,
    points_bg: TensorOperator<D1, DType>,
    weight_function: WeightFunction,
    min_edge_weight: f32,
    cfg: SolverConfig,
) -> LODTensorOperator<D3, StaticElementType<f32>> {
    let weights = random_walker_weights_lod(tensor, weight_function, min_edge_weight);
    hierarchical_random_walker_solver(weights, points_fg, points_bg, cfg)
}

#[allow(unused)] // Very useful for debugging
async fn download<'a, 'b, T: crate::storage::Element + Default>(
    ctx: OpaqueTaskContext<'a, 'b>,
    device: &DeviceContext,
    x: &impl AsBufferDescriptor,
) -> Vec<T> {
    let src = SrcBarrierInfo {
        stage: vk::PipelineStageFlags2::ALL_COMMANDS,
        access: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
    };
    let dst = DstBarrierInfo {
        stage: vk::PipelineStageFlags2::TRANSFER,
        access: vk::AccessFlags2::TRANSFER_READ,
    };
    ctx.submit(device.barrier(src, dst)).await;

    let info = x.gen_buffer_info();
    let size = info.range as usize / std::mem::size_of::<T>();
    let out = vec![T::default(); size];
    let out_ptr = out.as_ptr() as _;
    let layout = Layout::array::<T>(size).unwrap();
    unsafe { crate::vulkan::memory::copy_to_cpu(ctx, device, info.buffer, layout, out_ptr).await };
    out
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        operators::array::from_vec,
        test_util::compare_tensor_approx,
        vec::{LocalVoxelPosition, Vector, VoxelPosition},
    };

    #[test]
    fn simple() {
        let s = [5, 5, 5];
        let size = VoxelPosition::from(s);
        let brick_size = LocalVoxelPosition::from([8, 8, 8]);

        let vol = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            if v.x().raw < size.x().raw / 2 {
                0.1
            } else {
                0.9
            }
        });

        let seeds = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            if v == VoxelPosition::fill(0.into()) {
                0.0
            } else if v == VoxelPosition::from(s) - VoxelPosition::fill(1.into()) {
                1.0
            } else {
                -2.0
            }
        });

        let expected = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            if v.x().raw < size.x().raw / 2 {
                0.0
            } else {
                1.0
            }
        });

        let cfg = Default::default();

        let v = random_walker(vol, seeds, WeightFunction::Grady { beta: 100.0 }, 1e-5, cfg);

        compare_tensor_approx(v, expected, cfg.max_residuum_norm);
    }

    #[test]
    fn chunked_weights() {
        let size = VoxelPosition::fill(8.into());
        let brick_size_full = size.local();
        let brick_size = LocalVoxelPosition::fill(2.into());

        let vol = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            if v.x().raw < size.x().raw / 2 {
                0.1
            } else {
                0.9
            }
        });

        let seeds = crate::operators::rasterize_function::voxel(size, brick_size_full, move |v| {
            if v == VoxelPosition::fill(0.into()) {
                0.0
            } else if v == size - VoxelPosition::fill(1.into()) {
                1.0
            } else {
                -2.0
            }
        });

        let expected =
            crate::operators::rasterize_function::voxel(size, brick_size_full, move |v| {
                if v.x().raw < size.x().raw / 2 {
                    0.0
                } else {
                    1.0
                }
            });

        let cfg = Default::default();

        let weights = random_walker_weights(vol, WeightFunction::Grady { beta: 100.0 }, 1e-5);
        let weights = crate::operators::rechunk::rechunk(
            weights,
            Vector::fill(crate::operators::rechunk::ChunkSize::Full),
        );
        let v = random_walker_single_chunk(weights, seeds, cfg);

        compare_tensor_approx(v, expected, cfg.max_residuum_norm);
    }

    #[test]
    fn hierarchical_rw() {
        let s = 10;
        let size = VoxelPosition::fill(s.into());
        let brick_size = LocalVoxelPosition::fill(4.into());

        let vol = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            if v.x().raw < size.x().raw / 2 {
                0.1
            } else {
                0.9
            }
        })
        .embedded(TensorEmbeddingData {
            spacing: Vector::fill(1.0),
        });
        let vol = crate::operators::resample::create_lod(
            vol,
            Vector::fill(crate::operators::resample::DownsampleStep::Synchronized(
                2.0,
            )),
        );

        let background = from_vec(vec![Vector::<D3, f32>::fill(0.0)])
            .try_into()
            .unwrap();
        let foreground = from_vec(vec![Vector::<D3, f32>::fill((s - 1) as f32)])
            .try_into()
            .unwrap();

        let cfg = Default::default();

        let v = hierarchical_random_walker(
            vol,
            foreground,
            background,
            WeightFunction::Grady { beta: 1000.0 },
            1e-6,
            cfg,
        );

        //for level in v.levels.into_iter().rev() {
        let level = v.levels[0].clone();
        {
            let size = level.metadata.dimensions;
            let expected = crate::operators::rasterize_function::voxel(
                size,
                level.metadata.chunk_size,
                move |v| {
                    if v.x().raw < size.x().raw / 2 {
                        0.0
                    } else {
                        1.0
                    }
                },
            );

            // TODO: This threshold is _very_ forgiving...
            compare_tensor_approx(level.inner, expected, 0.6);
        }
    }
}
