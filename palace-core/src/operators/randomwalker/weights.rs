use ash::vk;
use id::Identify;

use crate::{
    array::{ChunkIndex, TensorMetaData},
    dim::{DynDimension, LargerDim, D3},
    dtypes::{ScalarType, StaticElementType},
    jit::{self},
    operator::OperatorDescriptor,
    operators::{scalar::ScalarOperator, tensor::TensorOperator},
    vec::Vector,
    vulkan::{
        pipeline::{ComputePipelineBuilder, DescriptorConfig, DynPushConstants},
        shader::Shader,
        state::ResourceId,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

pub fn random_walker_weights(
    tensor: TensorOperator<D3, StaticElementType<f32>>,
    weight_function: WeightFunction,
    min_edge_weight: f32,
) -> TensorOperator<<D3 as LargerDim>::Larger, StaticElementType<f32>> {
    match weight_function {
        WeightFunction::Grady { beta } => {
            random_walker_weights_grady(tensor, beta, min_edge_weight)
        }
        WeightFunction::BianMean { extent } => {
            random_walker_weights_bian(tensor, extent, min_edge_weight)
        }
    }
}

#[derive(Copy, Clone, Identify)]
pub enum WeightFunction {
    Grady { beta: f32 },
    BianMean { extent: usize },
}

pub fn random_walker_weights_grady(
    tensor: TensorOperator<D3, StaticElementType<f32>>,
    beta: f32,
    min_edge_weight: f32,
) -> TensorOperator<<D3 as LargerDim>::Larger, StaticElementType<f32>> {
    assert_eq!(
        tensor.metadata.dimensions.raw(),
        tensor.metadata.chunk_size.raw()
    );

    let nd = tensor.metadata.dim().n();
    let out_size = tensor
        .metadata
        .dimensions
        .push_dim_small((nd as u32).into());

    let out_md = TensorMetaData::single_chunk(out_size);

    TensorOperator::unbatched(
        OperatorDescriptor::new("random_walker_weights_grady")
            .dependent_on(&tensor)
            .dependent_on_data(&min_edge_weight)
            .dependent_on_data(&beta),
        Default::default(),
        out_md,
        tensor,
        move |ctx, _pos, _, tensor| {
            async move {
                let device = ctx.preferred_device();

                let in_size = tensor.metadata.dimensions;

                let push_constants = DynPushConstants::new()
                    .vec::<u32>(nd, "tensor_dim_in")
                    .scalar::<f32>("min_edge_weight")
                    .scalar::<f32>("grady_beta");

                let pipeline = device.request_state(
                    ResourceId::new("randomwalker_weights_grady").dependent_on(&in_size),
                    || {
                        ComputePipelineBuilder::new(
                            Shader::new(include_str!("randomwalker_weights.glsl"))
                                .push_const_block_dyn(&push_constants)
                                .define("BRICK_MEM_SIZE", in_size.hmul())
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
                let input = ctx
                    .submit(
                        tensor
                            .chunks
                            .request_gpu(device.id, ChunkIndex(0), read_info),
                    )
                    .await;

                let out_chunk = ctx
                    .submit(ctx.alloc_slot_gpu(&device, ChunkIndex(0), out_md.dimensions.hmul()))
                    .await;

                let global_size = in_size.raw();

                let descriptor_config = DescriptorConfig::new([&input, &out_chunk]);

                device.with_cmd_buffer(|cmd| unsafe {
                    let mut pipeline = pipeline.bind(cmd);

                    pipeline.push_constant_dyn(&push_constants, |consts| {
                        consts.vec(&in_size.raw())?;
                        consts.scalar(min_edge_weight)?;
                        consts.scalar(beta)?;
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
}

fn mean_filter<D: DynDimension>(
    t: TensorOperator<D, StaticElementType<f32>>,
    extent: usize,
) -> TensorOperator<D, StaticElementType<f32>> {
    let size = 2 * extent + 1;
    let kernel = crate::operators::array::from_vec(vec![1.0 / size as f32; size]);
    let kernels = Vector::fill_with_len(&kernel, t.dim().n());
    crate::operators::volume_gpu::separable_convolution(t, kernels)
}

fn variance(
    t: TensorOperator<D3, StaticElementType<f32>>,
    extent: usize,
) -> ScalarOperator<StaticElementType<f32>> {
    let nd = t.dim().n();
    let t_mean = mean_filter(t.clone(), extent);

    let t = jit::jit(t.into());
    let t_mean = jit::jit(t_mean.into());

    let diff = t.sub(t_mean).unwrap();
    let diff_sq = diff
        .clone()
        .mul(diff)
        .unwrap()
        .cast(ScalarType::F32.into())
        .unwrap();

    let uncorrected_variance =
        crate::operators::volume_gpu::mean(diff_sq.compile().unwrap().try_into().unwrap());

    let size = 2 * extent + 1;
    let num_neighborhood_voxels = size.pow(nd as u32);
    let neighborhood_factor =
        (num_neighborhood_voxels as f32) / (num_neighborhood_voxels - 1) as f32;

    uncorrected_variance.map(neighborhood_factor, |neighborhood_factor: f32, v| {
        v * neighborhood_factor
    })
}

pub fn random_walker_weights_bian(
    t: TensorOperator<D3, StaticElementType<f32>>,
    extent: usize,
    min_edge_weight: f32,
) -> TensorOperator<<D3 as LargerDim>::Larger, StaticElementType<f32>> {
    let t_mean = mean_filter(t.clone(), extent);
    let variance = variance(t.clone(), extent);

    let nd = t.metadata.dim().n();
    let out_size = t.metadata.dimensions.push_dim_small((nd as u32).into());

    let out_md = TensorMetaData::single_chunk(out_size);

    TensorOperator::unbatched(
        OperatorDescriptor::new("random_walker_weights_bian")
            .dependent_on(&t)
            .dependent_on_data(&min_edge_weight)
            .dependent_on_data(&extent),
        Default::default(),
        out_md,
        (t, t_mean, variance),
        move |ctx, _pos, _, (t, t_mean, variance)| {
            async move {
                let device = ctx.preferred_device();

                let in_size = t.metadata.dimensions;

                let push_constants = DynPushConstants::new()
                    .vec::<u32>(nd, "tensor_dim_in")
                    .scalar::<f32>("min_edge_weight")
                    .scalar::<f32>("diff_variance_inv");

                let pipeline = device.request_state(
                    ResourceId::new("randomwalker_weights_bian").dependent_on(&in_size),
                    || {
                        ComputePipelineBuilder::new(
                            Shader::new(include_str!("randomwalker_weights.glsl"))
                                .push_const_block_dyn(&push_constants)
                                .define("BRICK_MEM_SIZE", in_size.hmul())
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
                let input = ctx
                    .submit(
                        t_mean
                            .chunks
                            .request_gpu(device.id, ChunkIndex(0), read_info),
                    )
                    .await;

                let variance = ctx.submit(variance.request_scalar()).await;

                let kernel_size = 2 * extent + 1;
                let variance_correction_factor = 2.0 / (kernel_size.pow(nd as u32 + 1) as f32);
                let diff_variance =
                    (variance * variance_correction_factor).max(std::f32::MIN_POSITIVE);
                let diff_variance_inv = 1.0 / diff_variance;

                let out_chunk = ctx
                    .submit(ctx.alloc_slot_gpu(&device, ChunkIndex(0), out_md.dimensions.hmul()))
                    .await;

                let global_size = in_size.raw();

                let descriptor_config = DescriptorConfig::new([&input, &out_chunk]);

                device.with_cmd_buffer(|cmd| unsafe {
                    let mut pipeline = pipeline.bind(cmd);

                    pipeline.push_constant_dyn(&push_constants, |consts| {
                        consts.vec(&in_size.raw())?;
                        consts.scalar(min_edge_weight)?;
                        consts.scalar(diff_variance_inv)?;
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
}
