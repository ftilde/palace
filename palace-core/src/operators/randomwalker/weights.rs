use ash::vk;
use id::Identify;

use crate::{
    chunk_utils::ChunkNeighborhood,
    data::ChunkCoordinate,
    dim::{DynDimension, LargerDim},
    dtypes::{DType, ElementType, ScalarType, StaticElementType},
    jit::{self, JitTensorOperator},
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    operators::{
        aggregation::SampleMethod,
        conv::BorderHandling,
        scalar::ScalarOperator,
        tensor::{LODTensorOperator, TensorOperator},
    },
    vec::Vector,
    vulkan::{
        pipeline::{ComputePipelineBuilder, DescriptorConfig, DynPushConstants},
        shader::Shader,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

pub fn random_walker_weights<D: DynDimension + LargerDim>(
    tensor: TensorOperator<D, StaticElementType<f32>>,
    weight_function: WeightFunction,
    min_edge_weight: f32,
) -> TensorOperator<<D as LargerDim>::Larger, StaticElementType<f32>> {
    let nd = tensor.dim().n();
    match weight_function {
        WeightFunction::Grady { beta } => {
            random_walker_weights_grady(tensor, beta, min_edge_weight)
        }
        WeightFunction::BianMean { extent } => {
            random_walker_weights_bian(tensor, Vector::fill_with_len(extent, nd), min_edge_weight)
        }
        WeightFunction::BhattacharyyaVarGaussian { extent } => {
            random_walker_weights_bhattacharyya_var_gaussian(
                tensor,
                Vector::fill_with_len(extent, nd),
                min_edge_weight,
            )
        }
        WeightFunction::TTest { extent } => {
            random_walker_weights_ttest(tensor, Vector::fill_with_len(extent, nd), min_edge_weight)
        }
    }
}

pub fn random_walker_weights_lod<D: DynDimension + LargerDim>(
    tensor: LODTensorOperator<D, StaticElementType<f32>>,
    weight_function: WeightFunction,
    min_edge_weight: f32,
) -> LODTensorOperator<<D as LargerDim>::Larger, StaticElementType<f32>> {
    tensor.map(|level| {
        random_walker_weights(level.inner, weight_function, min_edge_weight).embedded(
            crate::array::TensorEmbeddingData {
                spacing: level.embedding_data.spacing.push_dim_small(1.0),
            },
        )
    })
}

#[derive(Copy, Clone, Identify)]
pub enum WeightFunction {
    Grady { beta: f32 },
    BianMean { extent: u32 },
    BhattacharyyaVarGaussian { extent: u32 },
    TTest { extent: u32 },
}

pub fn random_walker_weight_pairs<D: DynDimension + LargerDim, T: ElementType>(
    tensor: TensorOperator<D, T>,
) -> TensorOperator<<D as LargerDim>::Larger, DType> {
    let nd = tensor.metadata.dim().n();

    let out_md = tensor
        .metadata
        .clone()
        .push_dim_small((nd as u32).into(), (nd as u32).into());

    let in_dtype: DType = tensor.dtype().into();
    let out_dtype = in_dtype.vectorize(2);

    TensorOperator::with_state(
        op_descriptor!(),
        out_dtype,
        out_md.clone(),
        (tensor, DataParam(out_md)),
        |ctx, mut positions, loc, (tensor, out_md)| {
            async move {
                let device = ctx.preferred_device(loc);
                let nd = tensor.metadata.dim().n();

                let md = &tensor.metadata;
                let dtype: DType = tensor.dtype().into();

                let push_constants = DynPushConstants::new()
                    .vec::<u32>(nd, "tensor_dim_in")
                    .vec::<u32>(nd, "chunk_dim_in")
                    .vec::<u32>(nd, "chunk_begin")
                    .scalar::<u32>("dim");

                let pipeline = device.request_state(
                    (md.chunk_size.hmul(), nd, dtype, &push_constants),
                    |device, (mem_size, nd, dtype, push_constants)| {
                        ComputePipelineBuilder::new(
                            Shader::new(include_str!("randomwalker_weight_pairs.glsl"))
                                .push_const_block_dyn(push_constants)
                                .define("BRICK_MEM_SIZE", mem_size)
                                .define("ND", nd)
                                .define("T", dtype.glsl_type_force_vec())
                                .ext(dtype.glsl_ext()),
                        )
                        .build(device)
                    },
                )?;

                let read_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };

                positions.sort();
                let push_constants = &push_constants;

                let _ = ctx
                    .run_unordered(positions.into_iter().map(|pos| {
                        async move {
                            let input = ctx
                                .submit(tensor.chunks.request_gpu(device.id, pos, read_info))
                                .await;

                            let neighbor_requests = (0..nd).into_iter().map(|dim| {
                                let pos_nd = tensor.metadata.chunk_pos_from_index(pos);
                                let neighbor_nd = pos_nd.map_element(dim, |e: ChunkCoordinate| {
                                    (e + 1u32)
                                        .min(tensor.metadata.dimension_in_chunks()[dim] - 1u32)
                                });
                                tensor.chunks.request_gpu(
                                    device.id,
                                    tensor.metadata.chunk_index(&neighbor_nd),
                                    read_info,
                                )
                            });

                            let neighbors = ctx.submit(ctx.group(neighbor_requests)).await;

                            let chunk_info = md.chunk_info(pos);

                            let out_chunk = ctx
                                .submit(ctx.alloc_slot_gpu(&device, pos, &out_md.chunk_size))
                                .await;
                            for dim in 0..nd {
                                let neighbor = &neighbors[dim];

                                let global_size = md.chunk_size.raw();

                                let descriptor_config =
                                    DescriptorConfig::new([&input, neighbor, &out_chunk]);

                                device.with_cmd_buffer(|cmd| unsafe {
                                    let mut pipeline = pipeline.bind(cmd);

                                    pipeline.push_constant_dyn(&push_constants, |consts| {
                                        consts.vec(&md.dimensions.raw())?;
                                        consts.vec(&md.chunk_size.raw())?;
                                        consts.vec(&chunk_info.begin().raw())?;
                                        consts.scalar(dim as u32)?;
                                        Ok(())
                                    });
                                    pipeline.write_descriptor_set(0, descriptor_config);
                                    pipeline.dispatch_dyn(device, global_size);
                                });
                            }

                            unsafe {
                                out_chunk.initialized(
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

pub fn random_walker_weights_grady<D: DynDimension + LargerDim>(
    tensor: TensorOperator<D, StaticElementType<f32>>,
    beta: f32,
    min_edge_weight: f32,
) -> TensorOperator<<D as LargerDim>::Larger, StaticElementType<f32>> {
    let nd = tensor.metadata.dim().n();

    let out_md = tensor
        .metadata
        .clone()
        .push_dim_small((nd as u32).into(), (nd as u32).into());

    TensorOperator::with_state(
        op_descriptor!(),
        Default::default(),
        out_md.clone(),
        (
            tensor,
            DataParam(beta),
            DataParam(min_edge_weight),
            DataParam(out_md),
        ),
        |ctx, mut positions, loc, (tensor, beta, min_edge_weight, out_md)| {
            async move {
                let device = ctx.preferred_device(loc);
                let nd = tensor.metadata.dim().n();

                let md = &tensor.metadata;

                let push_constants = DynPushConstants::new()
                    .vec::<u32>(nd, "tensor_dim_in")
                    .vec::<u32>(nd, "chunk_dim_in")
                    .vec::<u32>(nd, "chunk_begin")
                    .scalar::<u32>("dim")
                    .scalar::<f32>("min_edge_weight")
                    .scalar::<f32>("grady_beta");

                let pipeline = device.request_state(
                    (md.chunk_size.hmul(), nd, &push_constants),
                    |device, (mem_size, nd, push_constants)| {
                        ComputePipelineBuilder::new(
                            Shader::new(include_str!("randomwalker_weights.glsl"))
                                .push_const_block_dyn(push_constants)
                                .define("BRICK_MEM_SIZE", mem_size)
                                .define("ND", nd)
                                .define("WEIGHT_FUNCTION_GRADY", 1),
                        )
                        .build(device)
                    },
                )?;

                let read_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };

                positions.sort();
                let push_constants = &push_constants;

                let _ = ctx
                    .run_unordered(positions.into_iter().map(|pos| {
                        async move {
                            let input = ctx
                                .submit(tensor.chunks.request_gpu(device.id, pos, read_info))
                                .await;

                            let neighbor_requests = (0..nd).into_iter().map(|dim| {
                                let pos_nd = tensor.metadata.chunk_pos_from_index(pos);
                                let neighbor_nd = pos_nd.map_element(dim, |e: ChunkCoordinate| {
                                    (e + 1u32)
                                        .min(tensor.metadata.dimension_in_chunks()[dim] - 1u32)
                                });
                                tensor.chunks.request_gpu(
                                    device.id,
                                    tensor.metadata.chunk_index(&neighbor_nd),
                                    read_info,
                                )
                            });

                            let neighbors = ctx.submit(ctx.group(neighbor_requests)).await;

                            let chunk_info = md.chunk_info(pos);

                            let out_chunk = ctx
                                .submit(ctx.alloc_slot_gpu(&device, pos, &out_md.chunk_size))
                                .await;
                            for dim in 0..nd {
                                let neighbor = &neighbors[dim];

                                let global_size = md.chunk_size.raw();

                                let descriptor_config =
                                    DescriptorConfig::new([&input, neighbor, &out_chunk]);

                                device.with_cmd_buffer(|cmd| unsafe {
                                    let mut pipeline = pipeline.bind(cmd);

                                    pipeline.push_constant_dyn(&push_constants, |consts| {
                                        consts.vec(&md.dimensions.raw())?;
                                        consts.vec(&md.chunk_size.raw())?;
                                        consts.vec(&chunk_info.begin().raw())?;
                                        consts.scalar(dim as u32)?;
                                        consts.scalar(**min_edge_weight)?;
                                        consts.scalar(**beta)?;
                                        Ok(())
                                    });
                                    pipeline.write_descriptor_set(0, descriptor_config);
                                    pipeline.dispatch_dyn(device, global_size);
                                });
                            }

                            unsafe {
                                out_chunk.initialized(
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

fn num_filter_elements<D: DynDimension>(extent: Vector<D, u32>) -> JitTensorOperator<D> {
    let dim = extent.dim();
    let nd = dim.n();
    let pos = jit::position(dim);
    let one = jit::scalar(1u32).splat(nd as _);
    let end = jit::dimensions(dim).sub(one.clone()).unwrap();
    let extent = jit::const_vec(extent.into_dyn());
    let filter_size_minus = pos.clone().min(extent.clone()).unwrap();
    let filter_size_plus = end.sub(pos).unwrap().min(extent).unwrap();
    let filter_size = filter_size_minus
        .add(one.add(filter_size_plus).unwrap())
        .unwrap();
    filter_size.hmul()
}

fn mean_filter<D: DynDimension>(
    t: TensorOperator<D, StaticElementType<f32>>,
    extent: Vector<D, u32>,
) -> JitTensorOperator<D> {
    let size = extent.map(|e| 2 * e + 1);
    let kernels = size
        .clone()
        .into_iter()
        .map(|s| crate::operators::array::from_vec(vec![1.0; s as usize]))
        .collect::<Vec<_>>();
    let kernels = Vector::try_from_fn_and_len(size.len(), |i| &kernels[i]).unwrap();

    let sum = crate::operators::conv::separable_convolution(t, kernels, BorderHandling::Pad0);
    let num_elements = num_filter_elements(extent);
    jit::jit(sum.into())
        .div(num_elements.cast(ScalarType::F32.into()).unwrap())
        .unwrap()
}

fn variances<D: DynDimension>(
    t: JitTensorOperator<D>,
    t_mean: JitTensorOperator<D>,
    extent: Vector<D, u32>,
) -> JitTensorOperator<D> {
    let size = extent.map(|e| 2 * e + 1);
    let kernels = size
        .clone()
        .into_iter()
        .map(|s| crate::operators::array::from_vec(vec![1.0; s as usize]))
        .collect::<Vec<_>>();
    let kernels = Vector::try_from_fn_and_len(size.len(), |i| &kernels[i]).unwrap();

    let diff = t.clone().sub(t_mean.clone()).unwrap();
    let sqrd = diff.clone().mul(diff).unwrap();

    // TODO: This is not actually the correct estimate as in Angs paper.
    // We actually want
    // diff := t_mean[center] - t[pos]
    // instead of (what we have now)
    // diff := t_mean[pos] - t[pos]

    let diff_sum = crate::operators::conv::separable_convolution(
        sqrd.compile().unwrap().into(),
        kernels,
        BorderHandling::Pad0,
    );

    let num_elements = num_filter_elements(extent);
    let diff_sum = jit::jit(diff_sum.into());
    diff_sum
        .div(
            num_elements
                .sub(jit::scalar(1u32))
                .unwrap()
                .cast(ScalarType::F32.into())
                .unwrap(),
        )
        .unwrap()
}

fn variance<D: DynDimension>(
    t: TensorOperator<D, StaticElementType<f32>>,
    extent: Vector<D, u32>,
) -> ScalarOperator<StaticElementType<f32>> {
    let t_mean = mean_filter(t.clone(), extent.clone());

    let t = jit::jit(t.into());

    let diff = t.sub(t_mean).unwrap();
    let diff_sq = diff
        .clone()
        .mul(diff)
        .unwrap()
        .cast(ScalarType::F32.into())
        .unwrap();

    let uncorrected_variance = crate::operators::aggregation::mean(
        diff_sq.compile().unwrap().try_into().unwrap(),
        SampleMethod::Subset(10),
    );

    let size = extent.map(|e| 2 * e + 1);
    let num_neighborhood_voxels = size.hmul();
    let neighborhood_factor =
        (num_neighborhood_voxels as f32) / (num_neighborhood_voxels - 1) as f32;

    uncorrected_variance.map(neighborhood_factor, |neighborhood_factor: f32, v| {
        v * neighborhood_factor
    })
}

#[derive(Copy, Clone, Identify)]
enum VarGaussianWeightFunction {
    Bhattacharyya,
    TTest,
}

pub fn best_centers_variable_gaussian<D: DynDimension + LargerDim>(
    tensor: TensorOperator<D, StaticElementType<f32>>,
    extent: Vector<D, u32>,
) -> TensorOperator<D::Larger, StaticElementType<i8>> {
    //TODO: The border handling is not quite right here.
    let t_mean = mean_filter(tensor.clone(), extent.clone());
    let t = jit::jit(tensor.clone().into());
    let t_var = variances(t.clone(), t_mean.clone(), extent.clone());

    let add = t_var.clone().log().unwrap().mul(0.5.into()).unwrap();
    let mul = jit::scalar(0.5).div(t_var).unwrap();

    let mean_mul_add = t_mean
        .concat(mul)
        .unwrap()
        .concat(add)
        .unwrap()
        .compile()
        .unwrap();

    let nd = tensor.dim().n();

    let out_md = tensor
        .metadata
        .clone()
        .push_dim_small((nd as u32).into(), (nd as u32).into());

    TensorOperator::with_state(
        op_descriptor!(),
        Default::default(),
        out_md.clone(),
        (tensor, mean_mul_add, DataParam(out_md), DataParam(extent)),
        |ctx, mut positions, loc, (tensor, mean_mul_add, out_md, extent)| {
            async move {
                let device = ctx.preferred_device(loc);

                let md = &tensor.metadata;
                let nd = md.dim().n();

                let push_constants = DynPushConstants::new()
                    .vec::<u32>(nd, "dimensions")
                    .vec::<u32>(nd, "chunk_size")
                    .vec::<u32>(nd, "first_chunk_pos")
                    .vec::<u32>(nd, "neighbor_chunks")
                    .vec::<u32>(nd, "center_chunk_offset")
                    .vec::<u32>(nd, "extent");
                let num_neighbors_max = 3u32.pow(nd as u32);

                let pipeline = device.request_state(
                    (md.chunk_size.hmul(), nd, num_neighbors_max, &push_constants),
                    |device, (mem_size, nd, num_neighbors_max, push_constants)| {
                        ComputePipelineBuilder::new(
                            Shader::new(include_str!("best_centers.glsl"))
                                .push_const_block_dyn(push_constants)
                                .define("BRICK_MEM_SIZE", mem_size)
                                .define("ND", nd)
                                .define("NUM_NEIGHBORS", num_neighbors_max),
                        )
                        .build(device)
                    },
                )?;

                let read_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };

                positions.sort();

                let push_constants = &push_constants;

                let _ = ctx
                    .run_unordered(positions.into_iter().map(|pos| {
                        async move {
                            let input = ctx
                                .submit(tensor.chunks.request_gpu(device.id, pos, read_info))
                                .await;

                            let extent_vec = extent.global();

                            let chunk_neighbors =
                                ChunkNeighborhood::around(&md, pos, extent_vec.clone(), extent_vec);

                            let neighbor_chunks = chunk_neighbors.dim_in_chunks();
                            let first_chunk_pos = chunk_neighbors.begin_chunk.raw();

                            let neighbor_requests = chunk_neighbors
                                .chunk_indices_linear()
                                .map(|p| mean_mul_add.chunks.request_gpu(device.id, p, read_info));

                            let neighbors = ctx.submit(ctx.group(neighbor_requests)).await;
                            assert_eq!(neighbors.len(), neighbor_chunks.hmul());
                            assert!(neighbors.len() <= num_neighbors_max as _);
                            let neighbor_refs = neighbors
                                .iter()
                                .chain(std::iter::repeat(&neighbors[0]))
                                .take(num_neighbors_max as usize)
                                .collect::<Vec<_>>();
                            assert_eq!(neighbor_refs.len(), num_neighbors_max as usize);

                            let chunk_info = md.chunk_info(pos);

                            let out_chunk = ctx
                                .submit(ctx.alloc_slot_gpu(&device, pos, &out_md.chunk_size))
                                .await;

                            device.with_cmd_buffer(|cmd| {
                                let descriptor_config = DescriptorConfig::new([
                                    &neighbor_refs.as_slice(),
                                    &input,
                                    &out_chunk,
                                ]);

                                unsafe {
                                    let mut pipeline = pipeline.bind(cmd);

                                    pipeline.push_constant_dyn(&push_constants, |w| {
                                        w.vec(&md.dimensions.raw())?;
                                        w.vec(&md.chunk_size.raw())?;
                                        w.vec(&first_chunk_pos)?;
                                        w.vec(&neighbor_chunks.raw())?;
                                        w.vec(&chunk_info.begin().raw())?;
                                        w.vec(&*extent)?;

                                        Ok(())
                                    });
                                    pipeline.write_descriptor_set(0, descriptor_config);
                                    pipeline.dispatch(device, chunk_info.mem_elements());
                                }
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
pub fn random_walker_weights_bhattacharyya_var_gaussian<D: DynDimension + LargerDim>(
    tensor: TensorOperator<D, StaticElementType<f32>>,
    extent: Vector<D, u32>,
    min_edge_weight: f32,
) -> TensorOperator<<D as LargerDim>::Larger, StaticElementType<f32>> {
    random_walker_weights_variable_gaussian(
        tensor,
        extent,
        min_edge_weight,
        VarGaussianWeightFunction::Bhattacharyya,
    )
}

pub fn random_walker_weights_ttest<D: DynDimension + LargerDim>(
    tensor: TensorOperator<D, StaticElementType<f32>>,
    extent: Vector<D, u32>,
    min_edge_weight: f32,
) -> TensorOperator<<D as LargerDim>::Larger, StaticElementType<f32>> {
    random_walker_weights_variable_gaussian(
        tensor,
        extent,
        min_edge_weight,
        VarGaussianWeightFunction::TTest,
    )
}

fn random_walker_weights_variable_gaussian<D: DynDimension + LargerDim>(
    tensor: TensorOperator<D, StaticElementType<f32>>,
    extent: Vector<D, u32>,
    min_edge_weight: f32,
    method: VarGaussianWeightFunction,
) -> TensorOperator<<D as LargerDim>::Larger, StaticElementType<f32>> {
    let best_centers = best_centers_variable_gaussian(tensor.clone(), extent.clone());

    let nd = tensor.metadata.dim().n();

    let out_md = tensor
        .metadata
        .clone()
        .push_dim_small((nd as u32).into(), (nd as u32).into());

    for d in 0..nd {
        assert!(
            tensor.metadata.chunk_size[d].raw > 2 * extent[d] + 1,
            "{} (chunksize) <= {} (2*extent+1) in dim {}",
            tensor.metadata.chunk_size[d].raw,
            2 * extent[d] + 1,
            d
        );
    }

    TensorOperator::with_state(
        op_descriptor!(),
        Default::default(),
        out_md.clone(),
        (
            tensor,
            best_centers,
            DataParam(out_md),
            DataParam(extent),
            DataParam(min_edge_weight),
            DataParam(method),
        ),
        |ctx,
         mut positions,
         loc,
         (tensor, best_centers, out_md, extent, min_edge_weight, method)| {
            async move {
                let device = ctx.preferred_device(loc);

                let md = &tensor.metadata;
                let nd = md.dim().n();

                let push_constants = DynPushConstants::new()
                    .vec::<u32>(nd, "dimensions")
                    .vec::<u32>(nd, "chunk_size")
                    .vec::<u32>(nd, "first_chunk_pos")
                    .vec::<u32>(nd, "neighbor_chunks")
                    .vec::<u32>(nd, "center_chunk_offset")
                    .vec::<u32>(nd, "extent")
                    .scalar::<u32>("dim")
                    .scalar::<f32>("min_edge_weight");

                let num_neighbors_max = 3u32.pow(nd as u32);

                let pipeline = device.request_state(
                    (
                        md.chunk_size.hmul(),
                        nd,
                        num_neighbors_max,
                        method,
                        &push_constants,
                    ),
                    |device, (mem_size, nd, num_neighbors_max, method, push_constants)| {
                        let method_string = match **method {
                            VarGaussianWeightFunction::Bhattacharyya => {
                                "WEIGHT_FUNCTION_BHATTACHARYYA_VAR_GAUSSIAN"
                            }
                            VarGaussianWeightFunction::TTest => "WEIGHT_FUNCTION_TTEST",
                        };
                        ComputePipelineBuilder::new(
                            Shader::new(include_str!("randomwalker_weights_centers.glsl"))
                                .push_const_block_dyn(push_constants)
                                .define("BRICK_MEM_SIZE", mem_size)
                                .define("ND", nd)
                                .define("NUM_NEIGHBORS", num_neighbors_max)
                                .define(method_string, 1),
                        )
                        .build(device)
                    },
                )?;

                let read_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };

                positions.sort();

                let push_constants = &push_constants;

                let _ = ctx
                    .run_unordered(positions.into_iter().map(|pos| {
                        async move {
                            let input_centers = ctx
                                .submit(best_centers.chunks.request_gpu(device.id, pos, read_info))
                                .await;

                            let center_neighbor_requests = (0..nd).into_iter().map(|dim| {
                                let pos_nd = best_centers.metadata.chunk_pos_from_index(pos);
                                let neighbor_nd = pos_nd.map_element(dim, |e: ChunkCoordinate| {
                                    (e + 1u32).min(
                                        best_centers.metadata.dimension_in_chunks()[dim] - 1u32,
                                    )
                                });
                                best_centers.chunks.request_gpu(
                                    device.id,
                                    best_centers.metadata.chunk_index(&neighbor_nd),
                                    read_info,
                                )
                            });

                            let center_neighbors =
                                ctx.submit(ctx.group(center_neighbor_requests)).await;

                            let extent_vec = extent.global();
                            let chunk_neighbors =
                                ChunkNeighborhood::around(&md, pos, extent_vec.clone(), extent_vec);

                            let neighbor_chunks = chunk_neighbors.dim_in_chunks();
                            let first_chunk_pos = chunk_neighbors.begin_chunk.raw();

                            let neighbor_requests = chunk_neighbors
                                .chunk_indices_linear()
                                .map(|p| tensor.chunks.request_gpu(device.id, p, read_info));

                            let neighbors = ctx.submit(ctx.group(neighbor_requests)).await;
                            assert_eq!(neighbors.len(), neighbor_chunks.hmul());
                            assert!(neighbors.len() <= num_neighbors_max as _);
                            let neighbor_refs = neighbors
                                .iter()
                                .chain(std::iter::repeat(&neighbors[0]))
                                .take(num_neighbors_max as usize)
                                .collect::<Vec<_>>();
                            assert_eq!(neighbor_refs.len(), num_neighbors_max as usize);

                            let chunk_info = md.chunk_info(pos);

                            let out_chunk = ctx
                                .submit(ctx.alloc_slot_gpu(&device, pos, &out_md.chunk_size))
                                .await;

                            device.with_cmd_buffer(|cmd| {
                                for dim in 0..nd {
                                    let center_neighbor = &center_neighbors[dim];

                                    let descriptor_config = DescriptorConfig::new([
                                        &neighbor_refs.as_slice(),
                                        &input_centers,
                                        center_neighbor,
                                        &out_chunk,
                                    ]);

                                    unsafe {
                                        let mut pipeline = pipeline.bind(cmd);

                                        pipeline.push_constant_dyn(&push_constants, |w| {
                                            w.vec(&md.dimensions.raw())?;
                                            w.vec(&md.chunk_size.raw())?;
                                            w.vec(&first_chunk_pos)?;
                                            w.vec(&neighbor_chunks.raw())?;
                                            w.vec(&chunk_info.begin().raw())?;
                                            w.vec(&*extent)?;
                                            w.scalar(dim as u32)?;
                                            w.scalar(**min_edge_weight)?;

                                            Ok(())
                                        });
                                        pipeline.write_descriptor_set(0, descriptor_config);
                                        pipeline.dispatch(device, chunk_info.mem_elements());
                                    }
                                }
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

pub fn random_walker_weights_bian<D: DynDimension + LargerDim>(
    tensor: TensorOperator<D, StaticElementType<f32>>,
    extent: Vector<D, u32>,
    min_edge_weight: f32,
) -> TensorOperator<<D as LargerDim>::Larger, StaticElementType<f32>> {
    let t_mean = mean_filter(tensor.clone(), extent.clone())
        .compile()
        .unwrap();
    let variance = variance(tensor.clone(), extent.clone());

    let nd = tensor.metadata.dim().n();

    let out_md = tensor
        .metadata
        .clone()
        .push_dim_small((nd as u32).into(), (nd as u32).into());

    TensorOperator::with_state(
        op_descriptor!(),
        Default::default(),
        out_md.clone(),
        (
            t_mean,
            variance,
            DataParam(out_md),
            DataParam(extent),
            DataParam(min_edge_weight),
        ),
        |ctx, mut positions, loc, (t_mean, variance, out_md, extent, min_edge_weight)| {
            async move {
                let device = ctx.preferred_device(loc);

                let md = &t_mean.metadata;
                let nd = md.dim().n();

                let push_constants = DynPushConstants::new()
                    .vec::<u32>(nd, "tensor_dim_in")
                    .vec::<u32>(nd, "chunk_dim_in")
                    .vec::<u32>(nd, "chunk_begin")
                    .scalar::<u32>("dim")
                    .scalar::<f32>("min_edge_weight")
                    .scalar::<f32>("diff_variance_inv");

                let pipeline = device.request_state(
                    (md.chunk_size.hmul(), nd, &push_constants),
                    |device, (mem_size, nd, push_constants)| {
                        ComputePipelineBuilder::new(
                            Shader::new(include_str!("randomwalker_weights.glsl"))
                                .push_const_block_dyn(push_constants)
                                .define("BRICK_MEM_SIZE", mem_size)
                                .define("ND", nd)
                                .define("WEIGHT_FUNCTION_BIAN_MEAN", 1),
                        )
                        .build(device)
                    },
                )?;

                let read_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };

                positions.sort();

                let push_constants = &push_constants;

                let _ = ctx
                    .run_unordered(positions.into_iter().map(|pos| {
                        async move {
                            let input = ctx
                                .submit(t_mean.chunks.request_gpu(device.id, pos, read_info))
                                .await;

                            let variance = ctx.submit(variance.request_scalar()).await;

                            let size = extent.map(|e| 2 * e + 1);
                            let kernel_size = size.hmul();
                            let variance_correction_factor =
                                2.0 / (kernel_size.pow(nd as u32 + 1) as f32);
                            let diff_variance =
                                (variance * variance_correction_factor).max(std::f32::MIN_POSITIVE);
                            let diff_variance_inv = 1.0 / diff_variance;

                            let neighbor_requests = (0..nd).into_iter().map(|dim| {
                                let pos_nd = t_mean.metadata.chunk_pos_from_index(pos);
                                let neighbor_nd = pos_nd.map_element(dim, |e: ChunkCoordinate| {
                                    (e + 1u32)
                                        .min(t_mean.metadata.dimension_in_chunks()[dim] - 1u32)
                                });
                                t_mean.chunks.request_gpu(
                                    device.id,
                                    t_mean.metadata.chunk_index(&neighbor_nd),
                                    read_info,
                                )
                            });

                            let neighbors = ctx.submit(ctx.group(neighbor_requests)).await;

                            let chunk_info = md.chunk_info(pos);

                            let out_chunk = ctx
                                .submit(ctx.alloc_slot_gpu(&device, pos, &out_md.chunk_size))
                                .await;

                            for dim in 0..nd {
                                let neighbor = &neighbors[dim];
                                let global_size = md.chunk_size.raw();

                                let descriptor_config =
                                    DescriptorConfig::new([&input, neighbor, &out_chunk]);

                                device.with_cmd_buffer(|cmd| unsafe {
                                    let mut pipeline = pipeline.bind(cmd);

                                    pipeline.push_constant_dyn(&push_constants, |consts| {
                                        consts.vec(&md.dimensions.raw())?;
                                        consts.vec(&md.chunk_size.raw())?;
                                        consts.vec(&chunk_info.begin().raw())?;
                                        consts.scalar(dim as u32)?;
                                        consts.scalar(**min_edge_weight)?;
                                        consts.scalar(diff_variance_inv)?;
                                        Ok(())
                                    });
                                    pipeline.write_descriptor_set(0, descriptor_config);
                                    pipeline.dispatch_dyn(device, global_size);
                                });
                            }

                            unsafe {
                                out_chunk.initialized(
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
