use std::{alloc::Layout, mem::MaybeUninit};

// TODO: We probably do not want NUM_ROWS to be a constant, because it will change everytime we
// add/remove seeds

use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};
use gpu_allocator::MemoryLocation::GpuOnly;
use id::Identify;

use super::tensor::TensorOperator;
use crate::{
    array::{ChunkIndex, TensorMetaData},
    coordinate::GlobalCoordinate,
    dim::{DynDimension, LargerDim, D3},
    dtypes::StaticElementType,
    operator::OperatorDescriptor,
    storage::gpu::Allocation,
    task::OpaqueTaskContext,
    vec::Vector,
    vulkan::{
        memory::TempRessource,
        pipeline::{AsBufferDescriptor, ComputePipeline, DescriptorConfig, DynPushConstants},
        shader::ShaderDefines,
        state::{RessourceId, VulkanState},
        DeviceContext, DstBarrierInfo, SrcBarrierInfo,
    },
};

struct SparseMatrix {
    values: Allocation,
    index: Allocation,
    max_entries_per_row: u32,
    num_rows: u32,
}

impl VulkanState for SparseMatrix {
    unsafe fn deinitialize(&mut self, context: &crate::vulkan::DeviceContext) {
        self.values.deinitialize(context);
        self.index.deinitialize(context);
    }
}

async fn tensor_to_rows_table<'a, 'req, 'inv>(
    ctx: OpaqueTaskContext<'req, 'inv>,
    device: &'a DeviceContext,
    seeds: &impl AsBufferDescriptor,
    tensor_size: usize,
) -> Result<(TempRessource<'a, Allocation>, u32), crate::Error> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstantsStep {
        s: u32,
    }

    let local_size = device
        .physical_device_properties()
        .limits
        .max_compute_work_group_size[0]
        .min(tensor_size as u32);

    let pipeline_init = device.request_state(
        RessourceId::new("tensor_vec_table_init")
            .dependent_on(&tensor_size)
            .dependent_on(&local_size),
        || {
            ComputePipeline::new(
                device,
                (
                    include_str!("tensor_vec_table_init.glsl"),
                    ShaderDefines::new()
                        .add("BRICK_MEM_SIZE", tensor_size)
                        .add("LOCAL_SIZE", local_size),
                ),
                false,
            )
        },
    )?;

    let pipeline_step = device.request_state(
        RessourceId::new("tensor_vec_table_step")
            .dependent_on(&tensor_size)
            .dependent_on(&local_size),
        || {
            ComputePipeline::new(
                device,
                (
                    include_str!("tensor_vec_table_step.glsl"),
                    ShaderDefines::new()
                        .push_const_block::<PushConstantsStep>()
                        .add("BRICK_MEM_SIZE", tensor_size)
                        .add("LOCAL_SIZE", local_size),
                ),
                false,
            )
        },
    )?;

    let pipeline_finish = device.request_state(
        RessourceId::new("tensor_vec_table_finish")
            .dependent_on(&tensor_size)
            .dependent_on(&local_size),
        || {
            ComputePipeline::new(
                device,
                (
                    include_str!("tensor_vec_table_finish.glsl"),
                    ShaderDefines::new()
                        .add("BRICK_MEM_SIZE", tensor_size)
                        .add("LOCAL_SIZE", local_size),
                ),
                false,
            )
        },
    )?;

    let flags = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::TRANSFER_SRC;

    let (table, num_rows) = futures::join!(
        ctx.submit(device.storage.request_allocate_raw(
            device,
            Layout::array::<u32>(tensor_size).unwrap(),
            flags,
            GpuOnly,
        )),
        ctx.submit(device.storage.request_allocate_raw(
            device,
            Layout::new::<u32>(),
            flags,
            GpuOnly,
        )),
    );

    let table = TempRessource::new(device, table);
    let num_rows = TempRessource::new(device, num_rows);

    // Init table:
    let descriptor_config = DescriptorConfig::new([seeds, &*table]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline_init.bind(cmd);

        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(tensor_size as _);
    });

    //dbg!(download::<u32>(ctx, device, &*table).await);

    // Global reduce
    let mut s = local_size as u32;
    while (s as usize) < tensor_size {
        ctx.submit(device.barrier(
            SrcBarrierInfo {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            },
            DstBarrierInfo {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            },
        ))
        .await;

        let descriptor_config = DescriptorConfig::new([&*table]);

        device.with_cmd_buffer(|cmd| unsafe {
            let mut pipeline = pipeline_step.bind(cmd);

            pipeline.push_constant(PushConstantsStep { s });
            pipeline.write_descriptor_set(0, descriptor_config);
            pipeline.dispatch(tensor_size as _);
        });

        //dbg!(download::<u32>(ctx, device, &*table).await);

        s *= 2;
    }

    ctx.submit(device.barrier(
        SrcBarrierInfo {
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        },
        DstBarrierInfo {
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        },
    ))
    .await;

    // Finish
    let descriptor_config = DescriptorConfig::new([seeds, &*table, &*num_rows]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline_finish.bind(cmd);

        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(tensor_size as _);
    });

    //dbg!(download::<u32>(ctx, device, &*table).await);

    ctx.submit(device.barrier(
        SrcBarrierInfo {
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        },
        DstBarrierInfo {
            stage: vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_READ
                | vk::AccessFlags2::SHADER_WRITE
                | vk::AccessFlags2::TRANSFER_READ,
        },
    ))
    .await;

    let num_rows = read_scalar(ctx, device, &*num_rows).await;

    Ok((table, num_rows))
}
pub fn random_walker_weights(
    tensor: TensorOperator<D3, StaticElementType<f32>>,
    weight_function: WeightFunction,
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
        OperatorDescriptor::new("random_walker_weights")
            .dependent_on(&tensor)
            .dependent_on_data(&weight_function),
        Default::default(),
        out_md,
        tensor,
        move |ctx, _pos, _, tensor| {
            async move {
                let device = ctx.preferred_device();

                let in_size = tensor.metadata.dimensions;

                let push_constants = DynPushConstants::new()
                    .vec::<u32>(nd, "tensor_dim_in")
                    .scalar::<f32>("grady_beta");

                let pipeline = device.request_state(
                    RessourceId::new("randomwalker_weights").dependent_on(&in_size),
                    || {
                        ComputePipeline::new(
                            device,
                            (
                                include_str!("randomwalker_weights.glsl"),
                                ShaderDefines::new()
                                    .push_const_block_dyn(&push_constants)
                                    .add("BRICK_MEM_SIZE", in_size.hmul())
                                    .add("ND", nd)
                                    .add(weight_function.define_name(), 1),
                            ),
                            false,
                        )
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
                        consts.scalar(match weight_function {
                            WeightFunction::Grady { beta } => beta,
                        })?;
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
pub fn random_walker(
    tensor: TensorOperator<D3, StaticElementType<f32>>,
    seeds: TensorOperator<D3, StaticElementType<f32>>,
    weight_function: WeightFunction,
    cfg: SolverConfig,
) -> TensorOperator<D3, StaticElementType<f32>> {
    let weights = random_walker_weights(tensor, weight_function);
    random_walker_inner(weights, seeds, cfg)
}

pub fn random_walker_inner(
    weights: TensorOperator<<D3 as LargerDim>::Larger, StaticElementType<f32>>,
    seeds: TensorOperator<D3, StaticElementType<f32>>,
    cfg: SolverConfig,
) -> TensorOperator<D3, StaticElementType<f32>> {
    assert_eq!(
        weights.metadata.dimensions.raw(),
        weights.metadata.chunk_size.raw()
    );
    assert_eq!(
        seeds.metadata.dimensions.raw(),
        seeds.metadata.chunk_size.raw()
    );
    assert_eq!(
        weights.metadata.dimensions.pop_dim_small(),
        seeds.metadata.dimensions
    );

    TensorOperator::unbatched(
        OperatorDescriptor::new("random_walker")
            .dependent_on(&weights)
            .dependent_on(&seeds)
            .dependent_on_data(&cfg),
        Default::default(),
        seeds.metadata.clone(),
        (weights, seeds, cfg),
        move |ctx, _pos, _, (weights, seeds, cfg)| {
            async move {
                let device = ctx.preferred_device();

                let tensor_size = seeds.metadata.dimensions;
                let tensor_elements = tensor_size.hmul();

                if tensor_elements > u32::MAX as usize {
                    return Err(format!(
                        "Tensor cannot have more than 2^32 elements, but it has {}",
                        tensor_elements,
                    )
                    .into());
                }

                let read_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };
                let (weights, seeds) = futures::join!(
                    ctx.submit(
                        weights
                            .chunks
                            .request_gpu(device.id, ChunkIndex(0), read_info)
                    ),
                    ctx.submit(
                        seeds
                            .chunks
                            .request_gpu(device.id, ChunkIndex(0), read_info),
                    ),
                );

                let (tensor_to_rows_table, num_rows) =
                    tensor_to_rows_table(*ctx, device, &seeds, tensor_elements).await?;

                if num_rows == 0 {
                    return Err(format!("Tensor has 0 unseeded elements",).into());
                }

                let (mat, vec) = mat_setup(
                    *ctx,
                    &device,
                    &weights,
                    &seeds,
                    &tensor_to_rows_table,
                    tensor_size,
                    num_rows,
                )
                .await?;

                let mat = TempRessource::new(device, mat);
                let vec = TempRessource::new(device, vec);

                let flags = vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC;

                let result = ctx
                    .submit(device.storage.request_allocate_raw(
                        device,
                        Layout::array::<f32>(num_rows as usize).unwrap(),
                        flags,
                        GpuOnly,
                    ))
                    .await;
                let result = TempRessource::new(device, result);

                device.with_cmd_buffer(|cmd| unsafe {
                    cmd.functions().cmd_fill_buffer(
                        cmd.raw(),
                        result.buffer,
                        0,
                        vk::WHOLE_SIZE,
                        f32::to_bits(0.5),
                    );
                });

                //TODO: Try to reduce barriers here. We have one in init_weights already
                //Make result and weights initialization visible
                ctx.submit(device.barrier(
                    SrcBarrierInfo {
                        stage: vk::PipelineStageFlags2::TRANSFER
                            | vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_WRITE,
                    },
                    DstBarrierInfo {
                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                    },
                ))
                .await;

                conjugate_gradient(*ctx, device, &mat, &vec, &result, cfg).await?;

                let out_chunk = ctx
                    .submit(ctx.alloc_slot_gpu(&device, ChunkIndex(0), tensor_elements))
                    .await;

                results_to_tensor(
                    device,
                    &seeds,
                    &tensor_to_rows_table,
                    &result,
                    &out_chunk,
                    tensor_size,
                    num_rows,
                )
                .await?;

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

async fn results_to_tensor<'req, 'inv>(
    device: &DeviceContext,
    seeds: &impl AsBufferDescriptor,
    tensor_to_rows_table: &Allocation,
    result_vec: &Allocation,
    out_allocation: &impl AsBufferDescriptor,
    tensor_size: Vector<D3, GlobalCoordinate>,
    num_rows: u32,
) -> Result<(), crate::Error> {
    let nd = tensor_size.dim().n();
    let tensor_elements = tensor_size.hmul();

    let push_constants = DynPushConstants::new().vec::<u32>(nd, "tensor_dim");

    let pipeline = device
        .request_state(
            RessourceId::new("results_to_tensor")
                .dependent_on(&tensor_size)
                .dependent_on(&num_rows),
            || {
                ComputePipeline::new(
                    device,
                    (
                        include_str!("results_to_tensor.glsl"),
                        ShaderDefines::new()
                            .push_const_block_dyn(&push_constants)
                            .add("BRICK_MEM_SIZE", tensor_elements)
                            .add("NUM_ROWS", num_rows)
                            .add("ND", nd),
                    ),
                    false,
                )
            },
        )
        .unwrap();
    let global_size = tensor_size.raw();

    let descriptor_config =
        DescriptorConfig::new([seeds, tensor_to_rows_table, result_vec, out_allocation]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.push_constant_dyn(&push_constants, |consts| {
            consts.vec(&tensor_size.raw())?;
            Ok(())
        });
        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch3d(global_size);
    });

    Ok(())
}

async fn mat_setup<'req, 'inv>(
    ctx: OpaqueTaskContext<'req, 'inv>,
    device: &DeviceContext,
    weights: &impl AsBufferDescriptor,
    seeds: &impl AsBufferDescriptor,
    tensor_to_rows_table: &Allocation,
    tensor_size: Vector<D3, GlobalCoordinate>,
    num_rows: u32,
) -> Result<(SparseMatrix, Allocation), crate::Error> {
    let nd = tensor_size.dim().n();

    let push_constants = DynPushConstants::new().vec::<u32>(nd, "tensor_dim_in");

    let tensor_elements = tensor_size.hmul() as usize;
    let max_entries = nd * 2 + 1;
    let mat_size = num_rows as usize * max_entries;

    let pipeline = device.request_state(
        RessourceId::new("randomwalker_mat_setup")
            .dependent_on(&tensor_size)
            .dependent_on(&num_rows),
        || {
            ComputePipeline::new(
                device,
                (
                    include_str!("randomwalker_mat_setup.glsl"),
                    ShaderDefines::new()
                        .push_const_block_dyn(&push_constants)
                        .add("BRICK_MEM_SIZE", tensor_elements)
                        .add("NUM_ROWS", num_rows)
                        .add("ND", nd),
                ),
                false,
            )
        },
    )?;

    let flags = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::TRANSFER_SRC;

    let (values, index, vec) = futures::join!(
        ctx.submit(device.storage.request_allocate_raw(
            device,
            Layout::array::<f32>(mat_size).unwrap(),
            flags,
            GpuOnly,
        )),
        ctx.submit(device.storage.request_allocate_raw(
            device,
            Layout::array::<u32>(mat_size).unwrap(),
            flags,
            GpuOnly,
        )),
        ctx.submit(device.storage.request_allocate_raw(
            device,
            Layout::array::<f32>(num_rows as usize).unwrap(),
            flags,
            GpuOnly,
        ))
    );

    //dbg!(download::<u32>(ctx, device, &*tensor_to_rows_table).await);
    //dbg!(download::<f32>(ctx, device, &*weights).await);

    device.with_cmd_buffer(|cmd| unsafe {
        cmd.functions()
            .cmd_fill_buffer(cmd.raw(), index.buffer, 0, vk::WHOLE_SIZE, 0xffffffff);
        cmd.functions()
            .cmd_fill_buffer(cmd.raw(), values.buffer, 0, vk::WHOLE_SIZE, 0);
        cmd.functions()
            .cmd_fill_buffer(cmd.raw(), vec.buffer, 0, vk::WHOLE_SIZE, 0);
    });

    ctx.submit(device.barrier(
        SrcBarrierInfo {
            stage: vk::PipelineStageFlags2::TRANSFER,
            access: vk::AccessFlags2::TRANSFER_WRITE,
        },
        DstBarrierInfo {
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        },
    ))
    .await;

    let global_size = tensor_size.raw();

    let descriptor_config =
        DescriptorConfig::new([weights, seeds, tensor_to_rows_table, &values, &index, &vec]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.push_constant_dyn(&push_constants, |consts| {
            consts.vec(&tensor_size.raw())?;
            Ok(())
        });
        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch3d(global_size);
    });

    Ok((
        SparseMatrix {
            values,
            index,
            max_entries_per_row: max_entries as _,
            num_rows: num_rows as _,
        },
        vec,
    ))
}

#[derive(Copy, Clone, Identify)]
pub enum WeightFunction {
    Grady { beta: f32 },
}

impl WeightFunction {
    fn define_name(&self) -> &'static str {
        match self {
            WeightFunction::Grady { .. } => "WEIGHT_FUNCTION_GRADY",
        }
    }
}

#[derive(Identify, Copy, Clone)]
pub struct SolverConfig {
    pub max_residuum_norm: f32,
    pub residuum_check_period: usize,
    pub max_iterations: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            max_residuum_norm: 1.0e-3,
            residuum_check_period: 2,
            max_iterations: 1000,
        }
    }
}

async fn conjugate_gradient<'req, 'inv>(
    ctx: OpaqueTaskContext<'req, 'inv>,
    device: &DeviceContext,
    mat: &SparseMatrix,
    vec: &Allocation,
    result: &Allocation,
    cfg: &SolverConfig,
) -> Result<(), crate::Error> {
    let num_rows = vec.size as usize / std::mem::size_of::<f32>();

    let srw_src = SrcBarrierInfo {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
    };
    let srw_dst = DstBarrierInfo {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
    };

    //TODO
    let flags = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::TRANSFER_SRC;

    let mut alloc_requests = Vec::new();
    for _ in 0..5 {
        alloc_requests.push(device.storage.request_allocate_raw(
            device,
            Layout::array::<f32>(num_rows).unwrap(),
            flags,
            GpuOnly,
        ))
    }
    for _ in 0..3 {
        alloc_requests.push(device.storage.request_allocate_raw(
            device,
            Layout::new::<f32>(),
            flags,
            GpuOnly,
        ))
    }
    let mut allocs = ctx.submit(ctx.group(alloc_requests)).await.into_iter();

    let x = result;
    let b = vec;
    let r = TempRessource::new(device, allocs.next().unwrap());
    let d = TempRessource::new(device, allocs.next().unwrap());
    let z = TempRessource::new(device, allocs.next().unwrap());
    let c = TempRessource::new(device, allocs.next().unwrap());
    let h = TempRessource::new(device, allocs.next().unwrap());

    let o = TempRessource::new(device, allocs.next().unwrap());
    let u = TempRessource::new(device, allocs.next().unwrap());
    let r_norm_sq_buf = TempRessource::new(device, allocs.next().unwrap());

    // Initialization

    //dbg!(download::<u32>(ctx, device, &mat.index).await);
    //dbg!(download::<f32>(ctx, device, &mat.values).await);
    //dbg!(download::<f32>(ctx, device, x).await);
    //dbg!(download::<f32>(ctx, device, b).await);

    // For jacobi preconditioning
    extract_inv_diag(device, &mat, &c)?;

    // r_tmp = A*x_0
    sparse_mat_prod(device, &mat, x, &r)?;

    // Make r_tmp visible (also also c)
    ctx.submit(device.barrier(srw_src, srw_dst)).await;
    // r (aka r_0) = b - r_tmp (aka A*x_0)
    scale_and_sum(device, -1.0, &r, &b, &r)?;

    // Make r visible (also also c)
    ctx.submit(device.barrier(srw_src, srw_dst)).await;

    // For jacobi preconditioning
    // d := C * r;
    point_wise_mul(device, &c, &r, &d)?;
    // h := C * r
    point_wise_mul(device, &c, &r, &h)?;

    for iteration in 0..cfg.max_iterations {
        // Make d and h visible
        ctx.submit(device.barrier(srw_src, srw_dst)).await;
        // z = A * d;
        sparse_mat_prod(device, &mat, &d, &z)?;

        fill(device, &o, 0.0);
        fill(device, &u, 0.0);
        // Make z visible
        ctx.submit(device.barrier(srw_src, srw_dst)).await;
        // o := dot(r, h)
        dot_product_add(device, &r, &h, &o)?;
        // u := dot(d, z)
        dot_product_add(device, &d, &z, &u)?;

        // Make o and u visible
        ctx.submit(device.barrier(srw_src, srw_dst)).await;
        // x_n := (o/u) * d + x
        scale_and_sum_quotient(device, 1.0, &o, &u, &d, x, x)?;
        // r_n := -1 * (o/u) * z + r
        scale_and_sum_quotient(device, -1.0, &o, &u, &z, &r, &r)?;

        // Make r visible
        ctx.submit(device.barrier(srw_src, srw_dst)).await;
        // h_n := C * r_n
        point_wise_mul(device, &c, &r, &h)?;

        fill(device, &u, 0.0);
        fill(device, &r_norm_sq_buf, 0.0);
        // Make h visible
        ctx.submit(device.barrier(srw_src, srw_dst)).await;
        // o_n := dot(r_n, h_n)
        dot_product_add(device, &r, &h, &u)?;

        if iteration % cfg.residuum_check_period == 0 {
            dot_product_add(device, &r, &r, &r_norm_sq_buf)?;
        }

        // Make u (i.e., o_n) visible
        ctx.submit(device.barrier(
            srw_src,
            DstBarrierInfo {
                stage: vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::SHADER_READ,
            },
        ))
        .await;

        // read ||r_n||_2^2
        if iteration % cfg.residuum_check_period == 0 {
            let r_norm_sq = read_scalar::<f32>(ctx, device, &r_norm_sq_buf).await;

            if r_norm_sq.sqrt() < cfg.max_residuum_norm {
                println!("Break after {} it", iteration);
                break;
            }
        }

        // d_n := 1.0 * (o_n/o) * d + h_n
        scale_and_sum_quotient(device, 1.0, &u, &o, &d, &h, &d)?;
    }
    Ok(())
}

// Note: buf's value must be visible for transfers already
async fn read_scalar<'req, 'inv, T: Copy + Default>(
    ctx: OpaqueTaskContext<'req, 'inv>,
    device: &DeviceContext,
    buf: &Allocation,
) -> T {
    let mut out = T::default();
    unsafe {
        crate::vulkan::memory::copy_to_cpu(
            ctx,
            device,
            buf.buffer,
            Layout::new::<T>(),
            &mut out as *mut T as *mut MaybeUninit<u8>,
        )
        .await
    };
    out
}

//async fn download<'a, 'b, T: crate::storage::Element + Default>(
//    ctx: OpaqueTaskContext<'a, 'b>,
//    device: &DeviceContext,
//    x: &impl AsBufferDescriptor,
//) -> Vec<T> {
//    let src = SrcBarrierInfo {
//        stage: vk::PipelineStageFlags2::ALL_COMMANDS,
//        access: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
//    };
//    let dst = DstBarrierInfo {
//        stage: vk::PipelineStageFlags2::TRANSFER,
//        access: vk::AccessFlags2::TRANSFER_READ,
//    };
//    ctx.submit(device.barrier(src, dst)).await;
//
//    let info = x.gen_buffer_info();
//    let size = info.range as usize / std::mem::size_of::<T>();
//    let out = vec![T::default(); size];
//    let out_ptr = out.as_ptr() as _;
//    let layout = Layout::array::<T>(size).unwrap();
//    unsafe { crate::vulkan::memory::copy_to_cpu(ctx, device, info.buffer, layout, out_ptr).await };
//    out
//}

fn fill(device: &DeviceContext, buf: &Allocation, value: f32) {
    device.with_cmd_buffer(|cmd| unsafe {
        cmd.functions().cmd_fill_buffer(
            cmd.raw(),
            buf.buffer,
            0,
            vk::WHOLE_SIZE,
            f32::to_bits(value),
        );
    });
}

fn dot_product_add(
    device: &DeviceContext,
    x: &Allocation,
    y: &Allocation,
    result: &Allocation,
) -> Result<(), crate::Error> {
    let x_info = x.gen_buffer_info();
    let y_info = y.gen_buffer_info();

    let size = x_info.range;
    assert_eq!(size, y_info.range);

    let num_rows = size as usize / std::mem::size_of::<f32>();

    let pipeline = device.request_state(
        RessourceId::new("dot_product").dependent_on(&num_rows),
        || {
            ComputePipeline::new(
                device,
                (
                    include_str!("dot_product.glsl"),
                    ShaderDefines::new().add("NUM_ROWS", num_rows),
                ),
                false,
            )
        },
    )?;

    let descriptor_config = DescriptorConfig::new([&*x, &*y, &*result]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(num_rows as _);
    });

    Ok(())
}

fn extract_inv_diag(
    device: &DeviceContext,
    mat: &SparseMatrix,
    result: &Allocation,
) -> Result<(), crate::Error> {
    let pipeline = device.request_state(
        RessourceId::new("extract_inv_diag").dependent_on(&mat.num_rows),
        || {
            ComputePipeline::new(
                device,
                (
                    include_str!("extract_inv_diag.glsl"),
                    ShaderDefines::new()
                        .add("NUM_ROWS", mat.num_rows)
                        .add("MAX_ENTRIES_PER_ROW", mat.max_entries_per_row),
                ),
                false,
            )
        },
    )?;

    let descriptor_config = DescriptorConfig::new([&mat.values, &mat.index, result]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(mat.num_rows as _);
    });

    Ok(())
}

fn point_wise_mul(
    device: &DeviceContext,
    x: &Allocation,
    y: &Allocation,
    result: &Allocation,
) -> Result<(), crate::Error> {
    let x_info = x.gen_buffer_info();
    let y_info = y.gen_buffer_info();
    let result_info = result.gen_buffer_info();

    let size = x_info.range;
    assert_eq!(size, y_info.range);
    assert_eq!(size, result_info.range);

    let num_rows = size as usize / std::mem::size_of::<f32>();

    let pipeline = device.request_state(
        RessourceId::new("point_wise_mul").dependent_on(&num_rows),
        || {
            ComputePipeline::new(
                device,
                (
                    include_str!("point_wise_mul.glsl"),
                    ShaderDefines::new().add("NUM_ROWS", num_rows),
                ),
                false,
            )
        },
    )?;

    let descriptor_config = DescriptorConfig::new([&*x, &*y, &*result]);
    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(num_rows as _);
    });

    Ok(())
}

fn sparse_mat_prod(
    device: &DeviceContext,
    mat: &SparseMatrix,
    x: &Allocation,
    result: &Allocation,
) -> Result<(), crate::Error> {
    let x_info = x.gen_buffer_info();
    let result_info = result.gen_buffer_info();

    let size = x_info.range;
    assert_eq!(size, result_info.range);

    let num_rows = mat.num_rows;

    let pipeline = device.request_state(
        RessourceId::new("sparse_mat_prod")
            .dependent_on(&num_rows)
            .dependent_on(&mat.max_entries_per_row),
        || {
            ComputePipeline::new(
                device,
                (
                    include_str!("sparse_mat_prod.glsl"),
                    ShaderDefines::new()
                        .add("NUM_ROWS", num_rows)
                        .add("MAX_ENTRIES_PER_ROW", mat.max_entries_per_row),
                ),
                false,
            )
        },
    )?;

    let descriptor_config = DescriptorConfig::new([&mat.values, &mat.index, &*x, &*result]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(num_rows as _);
    });

    Ok(())
}
// result := alpha * (o/u) *x + y
fn scale_and_sum_quotient(
    device: &DeviceContext,
    alpha: f32,
    o: &Allocation,
    u: &Allocation,
    x: &Allocation,
    y: &Allocation,
    result: &Allocation,
) -> Result<(), crate::Error> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        alpha: f32,
    }

    let x_info = x.gen_buffer_info();
    let y_info = y.gen_buffer_info();
    let result_info = result.gen_buffer_info();

    let size = x_info.range;
    assert_eq!(size, y_info.range);
    assert_eq!(size, result_info.range);

    let num_rows = size as usize / std::mem::size_of::<f32>();

    let pipeline = device.request_state(
        RessourceId::new("scale_and_sum_quotient").dependent_on(&num_rows),
        || {
            ComputePipeline::new(
                device,
                (
                    include_str!("scale_and_sum_quotient.glsl"),
                    ShaderDefines::new()
                        .push_const_block::<PushConstants>()
                        .add("NUM_ROWS", num_rows),
                ),
                false,
            )
        },
    )?;

    let descriptor_config = DescriptorConfig::new([&*o, &*u, &*x, &*y, &*result]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.push_constant(PushConstants { alpha });
        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(num_rows as _);
    });

    Ok(())
}

// result := alpha * x + y
fn scale_and_sum(
    device: &DeviceContext,
    alpha: f32,
    x: &Allocation,
    y: &Allocation,
    result: &Allocation,
) -> Result<(), crate::Error> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        alpha: f32,
    }

    let x_info = x.gen_buffer_info();
    let y_info = y.gen_buffer_info();
    let result_info = result.gen_buffer_info();

    let size = x_info.range;
    assert_eq!(size, y_info.range);
    assert_eq!(size, result_info.range);

    let num_rows = size as usize / std::mem::size_of::<f32>();

    let pipeline = device.request_state(
        RessourceId::new("scale_and_sum").dependent_on(&num_rows),
        || {
            ComputePipeline::new(
                device,
                (
                    include_str!("scale_and_sum.glsl"),
                    ShaderDefines::new()
                        .push_const_block::<PushConstants>()
                        .add("NUM_ROWS", num_rows),
                ),
                false,
            )
        },
    )?;

    let descriptor_config = DescriptorConfig::new([&*x, &*y, &*result]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.push_constant(PushConstants { alpha });
        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(num_rows as _);
    });

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        test_util::compare_tensor_approx,
        vec::{LocalVoxelPosition, VoxelPosition},
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

        let v = random_walker(vol, seeds, WeightFunction::Grady { beta: 100.0 }, cfg);

        compare_tensor_approx(v, expected, cfg.max_residuum_norm);
    }
}
