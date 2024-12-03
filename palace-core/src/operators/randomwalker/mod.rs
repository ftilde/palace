use ash::vk;

use super::tensor::{EmbeddedTensorOperator, TensorOperator};
use crate::{
    array::{ChunkIndex, TensorEmbeddingData, TensorMetaData},
    dim::{DynDimension, D1, D3},
    dtypes::{DType, ScalarType, StaticElementType},
    mat::Matrix,
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    vulkan::{
        pipeline::{ComputePipelineBuilder, DescriptorConfig, DynPushConstants},
        shader::Shader,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

mod hierarchical;
mod solver;
mod weights;

pub use hierarchical::*;
pub use solver::*;
pub use weights::*;

pub fn rasterize_seed_points(
    points_fg: TensorOperator<D1, DType>,
    points_bg: TensorOperator<D1, DType>,
    md: TensorMetaData<D3>,
    ed: TensorEmbeddingData<D3>,
) -> EmbeddedTensorOperator<D3, StaticElementType<f32>> {
    let in_dtype = points_fg.dtype();
    assert_eq!(in_dtype, points_bg.dtype());
    assert_eq!(in_dtype.size, 3);
    assert_eq!(in_dtype.scalar, ScalarType::F32);
    assert!(md.is_single_chunk());

    TensorOperator::unbatched(
        op_descriptor!(),
        Default::default(),
        md,
        (points_fg, points_bg, DataParam(md), DataParam(ed)),
        |ctx, _pos, _, (points_fg, points_bg, md, ed)| {
            async move {
                let device = ctx.preferred_device();

                let nd = md.dim().n();

                let in_size = md.dimensions;

                let push_constants = DynPushConstants::new()
                    .mat::<f32>(nd + 1, "to_grid")
                    .vec::<u32>(nd, "tensor_dim");

                let to_grid = Matrix::from_scale(&ed.spacing.map(|v| 1.0 / v)).to_homogeneous();

                let pipeline = device.request_state(
                    (
                        &push_constants,
                        in_size.hmul(),
                        points_fg.metadata.dimensions[0].raw,
                        points_bg.metadata.dimensions[0].raw,
                        nd,
                    ),
                    |device, (push_constants, mem_size, n_points_fg, n_points_bg, nd)| {
                        ComputePipelineBuilder::new(
                            Shader::new(include_str!("rasterize_points.glsl"))
                                .push_const_block_dyn(push_constants)
                                .define("BRICK_MEM_SIZE", mem_size)
                                .define("NUM_POINTS_FG", n_points_fg)
                                .define("NUM_POINTS_BG", n_points_bg)
                                .define("ND", nd),
                        )
                        .build(device)
                    },
                )?;

                let read_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };
                let points_fg = ctx
                    .submit(
                        points_fg
                            .chunks
                            .request_gpu(device.id, ChunkIndex(0), read_info),
                    )
                    .await;
                let points_bg = ctx
                    .submit(
                        points_bg
                            .chunks
                            .request_gpu(device.id, ChunkIndex(0), read_info),
                    )
                    .await;

                let out_chunk = ctx
                    .submit(ctx.alloc_slot_gpu(&device, ChunkIndex(0), md.dimensions.hmul()))
                    .await;

                let global_size = in_size.raw();

                let descriptor_config = DescriptorConfig::new([&points_fg, &points_bg, &out_chunk]);

                device.with_cmd_buffer(|cmd| unsafe {
                    let mut pipeline = pipeline.bind(cmd);

                    pipeline.push_constant_dyn(&push_constants, |consts| {
                        consts.mat(&to_grid)?;
                        consts.vec(&in_size.raw())?;
                        Ok(())
                    });
                    pipeline.write_descriptor_set(0, descriptor_config);
                    pipeline.dispatch3d(global_size);
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
    .try_into_static()
    .unwrap()
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
        let s = [4, 4, 4];
        let size = VoxelPosition::from(s);
        let brick_size = LocalVoxelPosition::from(s);

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
        let weights = crate::operators::volume_gpu::rechunk(
            weights,
            Vector::fill(crate::operators::volume::ChunkSize::Full),
        );
        let v = random_walker_single_chunk(weights, seeds, cfg);

        compare_tensor_approx(v, expected, cfg.max_residuum_norm);
    }

    #[test]
    fn hierarchical() {
        let s = 20;
        let size = VoxelPosition::fill(s.into());
        let brick_size = LocalVoxelPosition::fill(8.into());

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
        let vol = crate::operators::resample::create_lod(vol, 2.0);

        let foreground = from_vec(vec![Vector::<D3, f32>::fill(0.0)])
            .try_into()
            .unwrap();
        let background = from_vec(vec![Vector::<D3, f32>::fill((s - 1) as f32)])
            .try_into()
            .unwrap();

        let cfg = Default::default();

        let v = hierchical_random_walker(
            vol,
            foreground,
            background,
            WeightFunction::Grady { beta: 100.0 },
            1e-5,
            cfg,
        );

        for level in v.levels.into_iter().rev() {
            let size = level.metadata.dimensions;
            let expected = crate::operators::rasterize_function::voxel(
                dbg!(size),
                dbg!(level.metadata.chunk_size),
                move |v| {
                    if v.x().raw < size.x().raw / 2 {
                        0.0
                    } else {
                        1.0
                    }
                },
            );

            compare_tensor_approx(level.inner, expected, cfg.max_residuum_norm);
        }
    }
}
