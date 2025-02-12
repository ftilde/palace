use std::{alloc::Layout, mem::MaybeUninit};

use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};
use gpu_allocator::MemoryLocation::GpuOnly;
use id::Identify;

use crate::{
    array::{ChunkIndex, TensorMetaData},
    dim::{DynDimension, LargerDim},
    dtypes::StaticElementType,
    operator::{op_descriptor, DataParam, Operator, OperatorDescriptor},
    operators::tensor::TensorOperator,
    storage::gpu::{Allocation, ReadHandle},
    task::{OpaqueTaskContext, TaskContext},
    vulkan::{
        memory::TempRessource,
        pipeline::{
            AsBufferDescriptor, ComputePipelineBuilder, DescriptorConfig, DynPushConstants,
            LocalSizeConfig,
        },
        shader::Shader,
        state::VulkanState,
        DeviceContext, DstBarrierInfo, SrcBarrierInfo,
    },
};

pub async fn random_walker_on_chunk<'req, 'inv, D: DynDimension>(
    ctx: TaskContext<'req, 'inv, StaticElementType<f32>>,
    weights: &'inv Operator<StaticElementType<f32>>,
    seeds: &impl AsBufferDescriptor,
    init_values: Option<&'inv Operator<StaticElementType<f32>>>,
    pos: ChunkIndex,
    cfg: SolverConfig,
    tensor_md: TensorMetaData<D>,
) -> Result<(), crate::Error> {
    assert!(tensor_md.is_single_chunk());
    let device = ctx.preferred_device();

    if tensor_md.num_chunk_elements() > u32::MAX as usize {
        return Err(format!(
            "Tensor cannot have more than 2^32 elements, but it has {}",
            tensor_md.num_chunk_elements()
        )
        .into());
    }

    let read_info = DstBarrierInfo {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_READ,
    };
    let init_values = if let Some(init_values) = init_values {
        Some(
            ctx.submit(init_values.request_gpu(device.id, pos, read_info))
                .await,
        )
    } else {
        None
    };

    let weights = ctx
        .submit(weights.request_gpu(device.id, pos, read_info))
        .await;

    let start = std::time::Instant::now();

    //dbg!(&super::download::<f32>(*ctx, device, &seeds).await[..]);

    let (tensor_to_rows_table, num_rows) =
        tensor_to_rows_table(*ctx, device, seeds, tensor_md.clone()).await?;

    let (result_vec, num_iterations) = if num_rows > 0 {
        assert!(
            num_rows as usize <= tensor_md.num_tensor_elements(),
            "invalid num_rows: {}, too large for tensor_elements: {}",
            num_rows,
            tensor_md.num_tensor_elements()
        );

        let (mat, vec, result_vec) = mat_setup(
            *ctx,
            &device,
            &weights,
            seeds,
            init_values.as_ref(),
            &tensor_to_rows_table,
            tensor_md.clone(),
            num_rows,
        )
        .await?;
        std::mem::drop(weights);

        let mat = TempRessource::new(device, mat);
        let vec = TempRessource::new(device, vec);
        let result_vec = TempRessource::new(device, result_vec);

        //TODO: Try to reduce barriers here. We have one in init_weights already
        //Make result and weights initialization visible
        ctx.submit(device.barrier(
            SrcBarrierInfo {
                stage: vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_WRITE,
            },
            DstBarrierInfo {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            },
        ))
        .await;

        let num_iterations =
            conjugate_gradient(*ctx, device, &mat, &vec, &result_vec, &cfg).await?;
        (Some(result_vec), num_iterations)
    } else {
        (None, 0)
    };

    let out_chunk = ctx
        .submit(ctx.alloc_slot_gpu(&device, pos, &tensor_md.chunk_size))
        .await;

    results_to_tensor(
        device,
        seeds,
        &tensor_to_rows_table,
        result_vec.as_deref(),
        &out_chunk,
        tensor_md.clone(),
        num_rows,
    )
    .await?;

    let ms = start.elapsed().as_millis();
    let ms_per_iter = ms as f32 / num_iterations as f32;

    println!(
        "RW took {}ms, {} iter \t ({} ms/iter)| \t {}",
        ms,
        num_iterations,
        ms_per_iter,
        tensor_md.chunk_size.hmul(),
    );

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

pub fn random_walker_single_chunk<D: DynDimension + LargerDim>(
    weights: TensorOperator<D::Larger, StaticElementType<f32>>,
    seeds: TensorOperator<D, StaticElementType<f32>>,
    cfg: SolverConfig,
) -> TensorOperator<D, StaticElementType<f32>> {
    assert_eq!(weights.metadata.dimension_in_chunks().hmul(), 1);
    assert_eq!(seeds.metadata.dimension_in_chunks().hmul(), 1);
    assert_eq!(weights.metadata.clone().pop_dim_small(), seeds.metadata);

    TensorOperator::unbatched(
        op_descriptor!(),
        Default::default(),
        seeds.metadata.clone(),
        (weights, seeds, DataParam(cfg)),
        move |ctx, _pos, _, (weights, seeds, cfg)| {
            async move {
                let device = ctx.preferred_device();
                let read_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };
                let metadata = seeds.metadata.clone();
                let pos = ChunkIndex(0);
                let seeds = ctx
                    .submit(seeds.chunks.request_gpu(device.id, pos, read_info))
                    .await;
                random_walker_on_chunk::<D>(
                    ctx,
                    &weights.chunks,
                    &seeds,
                    None,
                    pos,
                    **cfg,
                    metadata,
                )
                .await
            }
            .into()
        },
    )
}

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

async fn tensor_to_rows_table<'a, 'req, 'inv, D: DynDimension>(
    ctx: OpaqueTaskContext<'req, 'inv>,
    device: &'a DeviceContext,
    seeds: &impl AsBufferDescriptor,
    tensor_md: TensorMetaData<D>,
) -> Result<(TempRessource<'a, Allocation>, u32), crate::Error> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstantsStep {
        s: u32,
    }

    let nd = tensor_md.dim().n();
    let push_constants_init = DynPushConstants::new()
        .vec::<u32>(nd, "tensor_size_memory")
        .vec::<u32>(nd, "tensor_size_logical");

    let push_constants_finish = DynPushConstants::new()
        .vec::<u32>(nd, "tensor_size_memory")
        .vec::<u32>(nd, "tensor_size_logical");

    let tensor_elements_mem = tensor_md.num_chunk_elements();

    let pipeline_init = device.request_state(
        (&push_constants_init, tensor_elements_mem, nd),
        |device, (push_constants_init, tensor_elements, nd)| {
            ComputePipelineBuilder::new(
                Shader::new(include_str!("tensor_vec_table_init.glsl"))
                    .push_const_block_dyn(push_constants_init)
                    .define("BRICK_MEM_SIZE", tensor_elements)
                    .define("ND", nd),
            )
            .local_size(LocalSizeConfig::Large)
            .build(device)
        },
    )?;

    let pipeline_step = device.request_state(tensor_elements_mem, |device, tensor_elements| {
        ComputePipelineBuilder::new(
            Shader::new(include_str!("tensor_vec_table_step.glsl"))
                .define("BRICK_MEM_SIZE", tensor_elements)
                .push_const_block::<PushConstantsStep>(),
        )
        .build(device)
    })?;

    let pipeline_finish = device.request_state(
        (&push_constants_finish, tensor_elements_mem, nd),
        |device, (push_constants_finish, tensor_elements, nd)| {
            ComputePipelineBuilder::new(
                Shader::new(include_str!("tensor_vec_table_finish.glsl"))
                    .push_const_block_dyn(push_constants_finish)
                    .define("BRICK_MEM_SIZE", tensor_elements)
                    .define("N", nd),
            )
            .build(device)
        },
    )?;

    let flags = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::TRANSFER_SRC;

    let (table, num_rows) = futures::join!(
        ctx.submit(device.storage.request_allocate_raw(
            device,
            Layout::array::<u32>(tensor_elements_mem).unwrap(),
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

        pipeline.push_constant_dyn(&push_constants_init, |consts| {
            consts.vec(&tensor_md.chunk_size.raw())?;
            consts.vec(&tensor_md.dimensions.raw())?;
            Ok(())
        });

        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(device, tensor_elements_mem as _);
    });

    //dbg!(&super::download::<u32>(ctx, device, &*table).await[..]);

    // Global reduce
    let mut s = pipeline_init.local_size().x();
    while (s as usize) < tensor_elements_mem {
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
            pipeline.dispatch(device, tensor_elements_mem.div_ceil(2) as _);
        });

        //dbg!(&download::<u32>(ctx, device, &*table).await[..]);

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

        pipeline.push_constant_dyn(&push_constants_finish, |consts| {
            consts.vec(&tensor_md.chunk_size.raw())?;
            consts.vec(&tensor_md.dimensions.raw())?;
            Ok(())
        });

        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(device, tensor_elements_mem as _);
    });

    //dbg!(&download::<u32>(ctx, device, &*table).await[..]);

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

async fn results_to_tensor<'req, 'inv, D: DynDimension>(
    device: &DeviceContext,
    seeds: &impl AsBufferDescriptor,
    tensor_to_rows_table: &Allocation,
    result_vec: Option<&Allocation>,
    out_allocation: &impl AsBufferDescriptor,
    tensor_md: TensorMetaData<D>,
    num_rows: u32,
) -> Result<(), crate::Error> {
    let nd = tensor_md.dim().n();
    let chunk_elements = tensor_md.num_chunk_elements();

    let pipeline = device
        .request_state((chunk_elements, nd), |device, (chunk_elements, nd)| {
            ComputePipelineBuilder::new(
                Shader::new(include_str!("results_to_tensor.glsl"))
                    .define("BRICK_MEM_SIZE", chunk_elements)
                    .define("ND", nd),
            )
            .build(device)
        })
        .unwrap();
    let global_size = tensor_md.chunk_size.raw();

    // Note: We do not have a result_vec exactly iff we do not have any non-seeded values. In that
    // case we do not read from the result_vec in the shader, so we can just set any other buffer
    // as the argument to make vulkan happy.
    assert_eq!(result_vec.is_none(), num_rows == 0);
    let results_vec_arg = result_vec.unwrap_or(tensor_to_rows_table);

    let descriptor_config =
        DescriptorConfig::new([seeds, tensor_to_rows_table, results_vec_arg, out_allocation]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch_dyn(device, global_size);
    });

    Ok(())
}

async fn mat_setup<'req, 'inv, D: DynDimension>(
    ctx: OpaqueTaskContext<'req, 'inv>,
    device: &DeviceContext,
    weights: &impl AsBufferDescriptor,
    seeds: &impl AsBufferDescriptor,
    init_values: Option<&ReadHandle<'req>>,
    tensor_to_rows_table: &Allocation,
    tensor_md: TensorMetaData<D>,
    num_rows: u32,
) -> Result<(SparseMatrix, Allocation, Allocation), crate::Error> {
    assert!(tensor_md.is_single_chunk());
    let nd = tensor_md.dim().n();

    let push_constants = DynPushConstants::new()
        .vec::<u32>(nd, "tensor_size_memory")
        .vec::<u32>(nd, "tensor_size_logical");

    let chunk_elements = tensor_md.num_chunk_elements();
    let max_entries = nd * 2 + 1;
    let mat_size = num_rows as usize * max_entries;

    let pipeline = device.request_state(
        (&push_constants, chunk_elements, nd, init_values.is_some()),
        |device, (push_constants, chunk_elements, nd, with_init)| {
            ComputePipelineBuilder::new(
                Shader::new(include_str!("randomwalker_mat_setup.glsl"))
                    .push_const_block_dyn(&push_constants)
                    .define("BRICK_MEM_SIZE", chunk_elements)
                    .define("ND", nd)
                    .define("WITH_INIT_VALUES", with_init as usize),
            )
            .build(device)
        },
    )?;

    let flags = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::TRANSFER_SRC;

    let (values, index, vec, results_vec) = futures::join!(
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
        )),
        ctx.submit(device.storage.request_allocate_raw(
            device,
            Layout::array::<f32>(num_rows as usize).unwrap(),
            flags,
            GpuOnly,
        ))
    );

    //dbg!(super::download::<u32>(ctx, device, &*tensor_to_rows_table).await);
    //dbg!(super::download::<f32>(ctx, device, &*weights).await);

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

    let global_size = tensor_md.dimensions.raw();

    let mut descriptor_config: Vec<&dyn crate::vulkan::pipeline::AsDescriptors> = vec![
        weights,
        seeds,
        tensor_to_rows_table,
        &values,
        &index,
        &vec,
        &results_vec,
    ];

    if let Some(init_values) = init_values {
        descriptor_config.push(init_values);
    }
    let descriptor_config = DescriptorConfig::from_vec(descriptor_config);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.push_constant_dyn(&push_constants, |consts| {
            consts.vec(&tensor_md.chunk_size.raw())?;
            consts.vec(&tensor_md.dimensions.raw())?;
            Ok(())
        });
        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch_dyn(device, global_size);
    });

    Ok((
        SparseMatrix {
            values,
            index,
            max_entries_per_row: max_entries as _,
            num_rows: num_rows as _,
        },
        vec,
        results_vec,
    ))
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
            residuum_check_period: 4,
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
) -> Result<usize, crate::Error> {
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

    let sum_buffer_size = num_rows.div_ceil(
        device
            .physical_device_properties()
            .limits
            .max_compute_work_group_size[0] as usize,
    );
    for _ in 0..4 {
        alloc_requests.push(device.storage.request_allocate_raw(
            device,
            Layout::array::<f32>(sum_buffer_size).unwrap(),
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

    let mut rth = TempRessource::new(device, allocs.next().unwrap());
    let mut rth_p1 = TempRessource::new(device, allocs.next().unwrap());
    let dtz = TempRessource::new(device, allocs.next().unwrap());
    let r_norm_sq_buf = TempRessource::new(device, allocs.next().unwrap());

    // Initialization

    //dbg!(super::download::<u32>(ctx, device, &mat.index).await);
    //dbg!(super::download::<f32>(ctx, device, &mat.values).await);
    //dbg!(super::download::<f32>(ctx, device, x).await);
    //dbg!(super::download::<f32>(ctx, device, b).await);

    ctx.submit(device.barrier(srw_src, srw_dst)).await;
    cg_init(device, mat, x, b, &*c, &*r, &*h, &*d, &*rth)?;
    dot_product_finish(ctx, device, &rth).await?;

    let mut total_it = cfg.max_iterations;
    for iteration in 0..cfg.max_iterations {
        ctx.submit(device.barrier(srw_src, srw_dst)).await;

        cg_alpha(device, mat, &d, &z, &dtz)?;
        dot_product_finish(ctx, device, &dtz).await?;

        ctx.submit(device.barrier(srw_src, srw_dst)).await;

        cg_beta(
            device,
            mat.num_rows,
            &z,
            &d,
            &c,
            &rth,
            &dtz,
            &x,
            &r,
            &h,
            &rth_p1,
        )?;
        dot_product_finish(ctx, device, &rth_p1).await?;

        ctx.submit(device.barrier(srw_src, srw_dst)).await;

        // read ||r_n||_2^2
        if iteration % cfg.residuum_check_period == 0 {
            dot_product_init(device, &r, &r, &r_norm_sq_buf)?;
            dot_product_finish(ctx, device, &r_norm_sq_buf).await?;

            // Make u (i.e., o_n) visible
            ctx.submit(device.barrier(
                srw_src,
                DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::TRANSFER,
                    access: vk::AccessFlags2::TRANSFER_READ,
                },
            ))
            .await;

            let r_norm_sq = read_scalar::<f32>(ctx, device, &r_norm_sq_buf).await;

            if r_norm_sq.is_nan() {
                panic!("Norm nan after {} it!", iteration);
            }
            if r_norm_sq.sqrt() < cfg.max_residuum_norm {
                total_it = iteration + 1;
                break;
            }
        }

        scale_and_sum_quotient(device, &rth_p1, &rth, &d, &h, &d)?;

        std::mem::swap(&mut rth, &mut rth_p1);
    }
    Ok(total_it)
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

#[derive(Copy, Clone, AsStd140, GlslStruct)]
struct PushConstsNumRows {
    num_rows: u32,
}

fn cg_init(
    device: &DeviceContext,
    // Input
    a: &SparseMatrix,
    x0: &Allocation,
    b: &Allocation,
    // Results
    c: &Allocation,
    r: &Allocation,
    h: &Allocation,
    d: &Allocation,
    rth: &Allocation,
) -> Result<(), crate::Error> {
    let num_rows = a.num_rows;

    let pipeline = device.request_state(a.max_entries_per_row, |device, max_entries_per_row| {
        ComputePipelineBuilder::new(
            Shader::new(include_str!("cg_init.glsl"))
                .define("MAX_ENTRIES_PER_ROW", max_entries_per_row)
                .push_const_block::<PushConstsNumRows>(),
        )
        .local_size(LocalSizeConfig::Large)
        .build(device)
    })?;

    let descriptor_config = DescriptorConfig::new([&a.values, &a.index, x0, b, c, r, h, d, rth]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.push_constant(PushConstsNumRows { num_rows });
        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(device, num_rows as _);
    });

    Ok(())
}

fn cg_alpha(
    device: &DeviceContext,
    // Input
    a: &SparseMatrix,
    d: &Allocation,
    // Results
    z: &Allocation,
    dtz: &Allocation,
) -> Result<(), crate::Error> {
    let num_rows = a.num_rows;

    let pipeline = device.request_state(a.max_entries_per_row, |device, max_entries_per_row| {
        ComputePipelineBuilder::new(
            Shader::new(include_str!("cg_alpha.glsl"))
                .define("MAX_ENTRIES_PER_ROW", max_entries_per_row)
                .push_const_block::<PushConstsNumRows>(),
        )
        .local_size(LocalSizeConfig::Large)
        .build(device)
    })?;

    let descriptor_config = DescriptorConfig::new([&a.values, &a.index, d, z, dtz]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.push_constant(PushConstsNumRows { num_rows });
        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(device, num_rows as _);
    });

    Ok(())
}

fn cg_beta(
    device: &DeviceContext,
    num_rows: u32,
    // Input
    z: &Allocation,
    d: &Allocation,
    c: &Allocation,
    rth: &Allocation,
    dtz: &Allocation,
    // Input/Output
    x: &Allocation,
    r: &Allocation,
    // Results
    h: &Allocation,
    rth_p1: &Allocation,
) -> Result<(), crate::Error> {
    let pipeline = device.request_state((), |device, ()| {
        ComputePipelineBuilder::new(
            Shader::new(include_str!("cg_beta.glsl")).push_const_block::<PushConstsNumRows>(),
        )
        .local_size(LocalSizeConfig::Large)
        .build(device)
    })?;

    let descriptor_config = DescriptorConfig::new([z, d, c, rth, dtz, x, r, h, rth_p1]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.push_constant(PushConstsNumRows { num_rows });
        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(device, num_rows as _);
    });

    Ok(())
}

async fn dot_product_finish<'req, 'inv>(
    ctx: OpaqueTaskContext<'req, 'inv>,
    device: &DeviceContext,
    x: &Allocation,
) -> Result<(), crate::Error> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstantsStep {
        stride: u32,
        num_values: u32,
    }

    let x_info = x.gen_buffer_info();

    let size = x_info.range;

    let num_values = (size as usize / std::mem::size_of::<f32>()) as u32;

    let pipeline = device.request_state((), |device, ()| {
        ComputePipelineBuilder::new(
            Shader::new(include_str!("dot_product_step.glsl"))
                .push_const_block::<PushConstantsStep>(),
        )
        .local_size(LocalSizeConfig::Large)
        .build(device)
    })?;

    let mut stride = 1u32;

    while stride < num_values {
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

        let descriptor_config = DescriptorConfig::new([&*x]);
        device.with_cmd_buffer(|cmd| unsafe {
            let mut pipeline = pipeline.bind(cmd);

            pipeline.push_constant(PushConstantsStep { stride, num_values });
            pipeline.write_descriptor_set(0, descriptor_config);
            pipeline.dispatch(device, (num_values / stride) as _);
        });

        stride *= pipeline.local_size().x();
    }

    Ok(())
}

fn dot_product_init(
    device: &DeviceContext,
    x: &Allocation,
    y: &Allocation,
    result: &Allocation,
) -> Result<(), crate::Error> {
    let x_info = x.gen_buffer_info();
    let y_info = y.gen_buffer_info();

    let size = x_info.range;
    assert_eq!(size, y_info.range);

    let num_rows = (size as usize / std::mem::size_of::<f32>()) as u32;

    let pipeline = device.request_state((), |device, ()| {
        ComputePipelineBuilder::new(
            Shader::new(include_str!("dot_product.glsl")).push_const_block::<PushConstsNumRows>(),
        )
        .local_size(LocalSizeConfig::Large)
        .build(device)
    })?;

    let descriptor_config = DescriptorConfig::new([&*x, &*y, &*result]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.push_constant(PushConstsNumRows { num_rows });

        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(device, num_rows as _);
    });

    Ok(())
}

// result := (o/u) *x + y
fn scale_and_sum_quotient(
    device: &DeviceContext,
    o: &Allocation,
    u: &Allocation,
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

    let num_rows = (size as usize / std::mem::size_of::<f32>()) as u32;

    let pipeline = device.request_state((), |device, ()| {
        ComputePipelineBuilder::new(
            Shader::new(include_str!("scale_and_sum_quotient.glsl"))
                .push_const_block::<PushConstsNumRows>(),
        )
        .build(device)
    })?;

    let descriptor_config = DescriptorConfig::new([&*o, &*u, &*x, &*y, &*result]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.push_constant(PushConstsNumRows { num_rows });

        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(device, num_rows as _);
    });

    Ok(())
}
