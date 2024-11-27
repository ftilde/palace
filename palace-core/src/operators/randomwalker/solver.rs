use std::{alloc::Layout, mem::MaybeUninit};

// TODO: We probably do not want NUM_ROWS to be a constant, because it will change everytime we
// add/remove seeds

use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};
use gpu_allocator::MemoryLocation::GpuOnly;
use id::Identify;

use crate::{
    array::ChunkIndex,
    coordinate::GlobalCoordinate,
    dim::{DynDimension, LargerDim, D3},
    dtypes::StaticElementType,
    operator::{op_descriptor, DataParam, OperatorDescriptor},
    operators::tensor::TensorOperator,
    storage::gpu::Allocation,
    task::OpaqueTaskContext,
    vec::Vector,
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
        op_descriptor!(),
        Default::default(),
        seeds.metadata.clone(),
        (weights, seeds, DataParam(cfg)),
        move |ctx, _pos, _, (weights, seeds, cfg)| {
            async move {
                let device = ctx.preferred_device();

                let tensor_size = seeds.metadata.dimensions;
                let tensor_elements = tensor_size.hmul();

                let start = std::time::Instant::now();

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

                assert!(
                    num_rows as usize <= tensor_elements,
                    "invalid num_rows: {}, too large for tensor_elements: {}",
                    num_rows,
                    tensor_elements
                );

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

                let num_iterations =
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

                println!(
                    "RW took {}ms, {} iter",
                    start.elapsed().as_millis(),
                    num_iterations
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

    let pipeline_init = device.request_state(tensor_size, |device, tensor_size| {
        ComputePipelineBuilder::new(
            Shader::new(include_str!("tensor_vec_table_init.glsl"))
                .define("BRICK_MEM_SIZE", tensor_size),
        )
        .local_size(LocalSizeConfig::Large)
        .build(device)
    })?;

    let pipeline_step = device.request_state(tensor_size, |device, tensor_size| {
        ComputePipelineBuilder::new(
            Shader::new(include_str!("tensor_vec_table_step.glsl"))
                .define("BRICK_MEM_SIZE", tensor_size)
                .push_const_block::<PushConstantsStep>(),
        )
        .build(device)
    })?;

    let pipeline_finish = device.request_state(tensor_size, |device, tensor_size| {
        ComputePipelineBuilder::new(
            Shader::new(include_str!("tensor_vec_table_finish.glsl"))
                .define("BRICK_MEM_SIZE", tensor_size),
        )
        .build(device)
    })?;

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
        pipeline.dispatch(device, tensor_size as _);
    });

    //dbg!(&download::<u32>(ctx, device, &*table).await[..]);

    // Global reduce
    let mut s = pipeline_init.local_size().x();
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
            pipeline.dispatch(device, tensor_size.div_ceil(2) as _);
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

        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(device, tensor_size as _);
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

    let pipeline = device
        .request_state(
            (tensor_elements, num_rows, nd),
            |device, (tensor_elements, num_rows, nd)| {
                ComputePipelineBuilder::new(
                    Shader::new(include_str!("results_to_tensor.glsl"))
                        .define("BRICK_MEM_SIZE", tensor_elements)
                        .define("NUM_ROWS", num_rows)
                        .define("ND", nd),
                )
                .build(device)
            },
        )
        .unwrap();
    let global_size = tensor_size.raw();

    let descriptor_config =
        DescriptorConfig::new([seeds, tensor_to_rows_table, result_vec, out_allocation]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

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
        (&push_constants, tensor_elements, num_rows, nd),
        |device, (push_constants, tensor_elements, num_rows, nd)| {
            ComputePipelineBuilder::new(
                Shader::new(include_str!("randomwalker_mat_setup.glsl"))
                    .push_const_block_dyn(&push_constants)
                    .define("BRICK_MEM_SIZE", tensor_elements)
                    .define("NUM_ROWS", num_rows)
                    .define("ND", nd),
            )
            .build(device)
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

    //dbg!(download::<u32>(ctx, device, &mat.index).await);
    //dbg!(download::<f32>(ctx, device, &mat.values).await);
    //dbg!(download::<f32>(ctx, device, x).await);
    //dbg!(download::<f32>(ctx, device, b).await);

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

    let pipeline = device.request_state(
        (num_rows, a.max_entries_per_row),
        |device, (num_rows, max_entries_per_row)| {
            ComputePipelineBuilder::new(
                Shader::new(include_str!("cg_init.glsl"))
                    .define("NUM_ROWS", num_rows)
                    .define("MAX_ENTRIES_PER_ROW", max_entries_per_row),
            )
            .local_size(LocalSizeConfig::Large)
            .build(device)
        },
    )?;

    let descriptor_config = DescriptorConfig::new([&a.values, &a.index, x0, b, c, r, h, d, rth]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

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

    let pipeline = device.request_state(
        (num_rows, a.max_entries_per_row),
        |device, (num_rows, max_entries_per_row)| {
            ComputePipelineBuilder::new(
                Shader::new(include_str!("cg_alpha.glsl"))
                    .define("NUM_ROWS", num_rows)
                    .define("MAX_ENTRIES_PER_ROW", max_entries_per_row),
            )
            .local_size(LocalSizeConfig::Large)
            .build(device)
        },
    )?;

    let descriptor_config = DescriptorConfig::new([&a.values, &a.index, d, z, dtz]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

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
    let pipeline = device.request_state(num_rows, |device, num_rows| {
        ComputePipelineBuilder::new(
            Shader::new(include_str!("cg_beta.glsl")).define("NUM_ROWS", num_rows),
        )
        .local_size(LocalSizeConfig::Large)
        .build(device)
    })?;

    let descriptor_config = DescriptorConfig::new([z, d, c, rth, dtz, x, r, h, rth_p1]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

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
    }

    let x_info = x.gen_buffer_info();

    let size = x_info.range;

    let num_values = (size as usize / std::mem::size_of::<f32>()) as u32;

    let pipeline = device.request_state(num_values, |device, num_values| {
        ComputePipelineBuilder::new(
            Shader::new(include_str!("dot_product_step.glsl"))
                .define("NUM_VALUES", num_values)
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

            pipeline.push_constant(PushConstantsStep { stride });
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

    let num_rows = size as usize / std::mem::size_of::<f32>();

    let pipeline = device.request_state(num_rows, |device, num_rows| {
        ComputePipelineBuilder::new(
            Shader::new(include_str!("dot_product.glsl")).define("NUM_ROWS", num_rows),
        )
        .local_size(LocalSizeConfig::Large)
        .build(device)
    })?;

    let descriptor_config = DescriptorConfig::new([&*x, &*y, &*result]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

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

    let num_rows = size as usize / std::mem::size_of::<f32>();

    let pipeline = device.request_state(num_rows, |device, num_rows| {
        ComputePipelineBuilder::new(
            Shader::new(include_str!("scale_and_sum_quotient.glsl")).define("NUM_ROWS", num_rows),
        )
        .build(device)
    })?;

    let descriptor_config = DescriptorConfig::new([&*o, &*u, &*x, &*y, &*result]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch(device, num_rows as _);
    });

    Ok(())
}
