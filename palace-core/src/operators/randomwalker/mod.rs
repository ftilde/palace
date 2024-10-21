use std::alloc::Layout;

// NO_PUSH_main: TODO: We probably do not want NUM_ROWS to be a constant, because it will change everytime we
// NO_PUSH_main: TODO: Oh god, free those allocations
// add/remove seeds

use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};
use gpu_allocator::MemoryLocation::GpuOnly;

use super::{
    scalar::{scalar, ScalarOperator},
    tensor::TensorOperator,
};
use crate::{
    array::{ChunkIndex, TensorMetaData},
    dim::{DynDimension, D1, D3},
    dtypes::StaticElementType,
    operator::OperatorDescriptor,
    storage::gpu::Allocation,
    task::OpaqueTaskContext,
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

const UNSEEDED: f32 = -2.0;

fn volume_to_rows_table(
    seeds: TensorOperator<D3, StaticElementType<f32>>,
) -> TensorOperator<D1, StaticElementType<u32>> {
    let size = seeds.metadata.dimensions.hmul();
    TensorOperator::unbatched(
        OperatorDescriptor::new("volume_to_rows_table").dependent_on(&seeds),
        Default::default(),
        TensorMetaData::single_chunk([size as u32].into()),
        seeds,
        move |ctx, _pos, _, seeds| {
            async move {
                let mut out = ctx.submit(ctx.alloc_slot(ChunkIndex(0), size)).await;
                let s = ctx.submit(seeds.chunks.request(ChunkIndex(0))).await;

                let mut row_counter = 0;
                for (i, o) in s.iter().zip(out.iter_mut()) {
                    if *i == UNSEEDED {
                        o.write(row_counter);
                        row_counter += 1;
                    } else {
                        o.write(0xffffffff);
                    }
                }

                unsafe { out.initialized(*ctx) };

                Ok(())
            }
            .into()
        },
    )
}

fn num_seeds(
    tensor_to_rows_table: TensorOperator<D1, StaticElementType<u32>>,
) -> ScalarOperator<StaticElementType<u32>> {
    scalar(
        OperatorDescriptor::new("num_seeds").dependent_on(&tensor_to_rows_table),
        tensor_to_rows_table,
        move |ctx, tensor_to_rows_table| {
            async move {
                let s = ctx
                    .submit(tensor_to_rows_table.chunks.request(ChunkIndex(0)))
                    .await;

                let mut max_row = 0;
                for i in s.iter() {
                    if *i != 0xffffffff {
                        max_row = max_row.max(*i);
                    }
                }

                let num_seeds = s.len() as u32 - max_row - 1;

                ctx.submit(ctx.write_scalar(num_seeds)).await;

                Ok(())
            }
            .into()
        },
    )
}
pub fn random_walker(
    tensor: TensorOperator<D3, StaticElementType<f32>>,
    seeds: TensorOperator<D3, StaticElementType<f32>>,
) -> TensorOperator<D3, StaticElementType<f32>> {
    let v2rt = volume_to_rows_table(seeds.clone());
    random_walker_inner(tensor, seeds.clone(), v2rt.clone(), num_seeds(v2rt))
}

pub fn random_walker_inner(
    tensor: TensorOperator<D3, StaticElementType<f32>>,
    seeds: TensorOperator<D3, StaticElementType<f32>>,
    tensor_to_rows_table: TensorOperator<D1, StaticElementType<u32>>,
    num_seeds: ScalarOperator<StaticElementType<u32>>,
) -> TensorOperator<D3, StaticElementType<f32>> {
    TensorOperator::unbatched(
        OperatorDescriptor::new("random_walker")
            .dependent_on(&tensor)
            .dependent_on(&seeds),
        Default::default(),
        tensor.metadata.clone(),
        (tensor, seeds, tensor_to_rows_table, num_seeds),
        move |ctx, _pos, _, (tensor, seeds, tensor_to_rows_table, num_seeds)| {
            async move {
                let device = ctx.preferred_device();

                let (mat, vec) = init_weights(
                    *ctx,
                    &device,
                    tensor,
                    seeds,
                    tensor_to_rows_table,
                    num_seeds,
                )
                .await?;

                let mat = TempRessource::new(device, mat);
                let vec = TempRessource::new(device, vec);

                let num_rows = vec.size as usize / std::mem::size_of::<f32>();

                let flags = vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC;

                let result = ctx
                    .submit(device.storage.request_allocate_raw(
                        device,
                        Layout::array::<f32>(num_rows).unwrap(),
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

                //TODO: Make result and weights initialization visible
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

                conjugate_gradient(*ctx, device, &mat, &vec, &result).await?;

                let out_chunk = ctx
                    .submit(ctx.alloc_slot_gpu(
                        &device,
                        ChunkIndex(0),
                        tensor.metadata.dimensions.hmul(),
                    ))
                    .await;

                results_to_tensor(
                    *ctx,
                    device,
                    seeds,
                    tensor_to_rows_table,
                    &result,
                    &out_chunk,
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
    ctx: OpaqueTaskContext<'req, 'inv>,
    device: &DeviceContext,
    seeds: &'inv TensorOperator<D3, StaticElementType<f32>>,
    tensor_to_rows_table: &'inv TensorOperator<D1, StaticElementType<u32>>,
    result_vec: &Allocation,
    out_allocation: &impl AsBufferDescriptor,
) -> Result<(), crate::Error> {
    let nd = seeds.dim().n();

    assert_eq!(
        seeds.metadata.dimensions.raw(),
        seeds.metadata.chunk_size.raw()
    );
    assert_eq!(
        tensor_to_rows_table.metadata.dimensions,
        tensor_to_rows_table.metadata.dimensions
    );

    let num_rows = result_vec.size as usize / std::mem::size_of::<f32>();
    let tensor_size = seeds.metadata.dimensions.hmul();

    let push_constants = DynPushConstants::new().vec::<u32>(nd, "tensor_dim");

    let pipeline = device
        .request_state(
            RessourceId::new("results_to_tensor")
                .dependent_on(&seeds.metadata.chunk_size)
                .dependent_on(&num_rows)
                .dependent_on(&nd),
            || {
                ComputePipeline::new(
                    device,
                    (
                        include_str!("results_to_tensor.glsl"),
                        ShaderDefines::new()
                            .push_const_block_dyn(&push_constants)
                            .add("BRICK_MEM_SIZE", tensor_size)
                            .add("NUM_ROWS", num_rows)
                            .add("ND", nd),
                    ),
                    false,
                )
            },
        )
        .unwrap();
    let global_size = seeds.metadata.chunk_size.raw();
    let tensor_dim = seeds.metadata.chunk_size.raw();

    let read_info = DstBarrierInfo {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_READ,
    };

    let (seeds, tensor_to_rows_table) = futures::join!(
        ctx.submit(
            seeds
                .chunks
                .request_gpu(device.id, ChunkIndex(0), read_info),
        ),
        ctx.submit(
            tensor_to_rows_table
                .chunks
                .request_gpu(device.id, ChunkIndex(0), read_info),
        ),
    );

    let descriptor_config =
        DescriptorConfig::new([&seeds, &tensor_to_rows_table, result_vec, out_allocation]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.push_constant_dyn(&push_constants, |consts| {
            consts.vec(&tensor_dim)?;
            Ok(())
        });
        pipeline.write_descriptor_set(0, descriptor_config);
        pipeline.dispatch3d(global_size);
    });

    Ok(())
}

async fn init_weights<'req, 'inv>(
    ctx: OpaqueTaskContext<'req, 'inv>,
    device: &DeviceContext,
    t: &'inv TensorOperator<D3, StaticElementType<f32>>,
    seeds: &'inv TensorOperator<D3, StaticElementType<f32>>,
    tensor_to_rows_table: &'inv TensorOperator<D1, StaticElementType<u32>>,
    num_seeds: &'inv ScalarOperator<StaticElementType<u32>>,
) -> Result<(SparseMatrix, Allocation), crate::Error> {
    let nd = t.dim().n();

    let num_seeds = ctx.submit(num_seeds.request_scalar()).await;

    assert_eq!(t.metadata.dimensions.raw(), t.metadata.chunk_size.raw());
    assert_eq!(
        seeds.metadata.dimensions.raw(),
        seeds.metadata.chunk_size.raw()
    );
    assert_eq!(t.metadata.dimensions, seeds.metadata.dimensions);

    let push_constants = DynPushConstants::new().vec::<u32>(nd, "tensor_dim_in");

    let tensor_size = t.metadata.chunk_size.hmul();
    let mat_rows = tensor_size - num_seeds as usize;
    let max_entries = nd * 2 + 1;
    let mat_size = mat_rows * max_entries;

    let pipeline = device.request_state(
        RessourceId::new("init_rw_weights")
            .dependent_on(&t.metadata.chunk_size)
            .dependent_on(&mat_rows)
            .dependent_on(&nd),
        || {
            ComputePipeline::new(
                device,
                (
                    include_str!("init_rw_weights.glsl"),
                    ShaderDefines::new()
                        .push_const_block_dyn(&push_constants)
                        .add("BRICK_MEM_SIZE", tensor_size)
                        .add("NUM_ROWS", mat_rows)
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
            Layout::array::<f32>(mat_rows).unwrap(),
            flags,
            GpuOnly,
        ))
    );

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

    let global_size = t.metadata.chunk_size.raw();

    let read_info = DstBarrierInfo {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_READ,
    };

    let (input, seeds, tensor_to_rows_table) = futures::join!(
        ctx.submit(t.chunks.request_gpu(device.id, ChunkIndex(0), read_info)),
        ctx.submit(
            seeds
                .chunks
                .request_gpu(device.id, ChunkIndex(0), read_info),
        ),
        ctx.submit(
            tensor_to_rows_table
                .chunks
                .request_gpu(device.id, ChunkIndex(0), read_info),
        ),
    );

    let descriptor_config =
        DescriptorConfig::new([&input, &seeds, &tensor_to_rows_table, &values, &index, &vec]);

    device.with_cmd_buffer(|cmd| unsafe {
        let mut pipeline = pipeline.bind(cmd);

        pipeline.push_constant_dyn(&push_constants, |consts| {
            consts.vec(&t.metadata.chunk_size.raw())?;
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
            num_rows: mat_rows as _,
        },
        vec,
    ))
}

async fn conjugate_gradient<'req, 'inv>(
    ctx: OpaqueTaskContext<'req, 'inv>,
    device: &DeviceContext,
    mat: &SparseMatrix,
    vec: &Allocation,
    result: &Allocation,
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
    for _ in 0..2 {
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

    // For jacobi preconditioning
    // d := C * r;
    point_wise_mul(device, &c, &r, &d)?;
    // h := C * r
    point_wise_mul(device, &c, &r, &h)?;

    for _ in 0..100 {
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
        // Make h visible
        ctx.submit(device.barrier(srw_src, srw_dst)).await;
        // o_n := dot(r_n, h_n)
        dot_product_add(device, &r, &h, &u)?;

        // READ: ||r_n||_2
        // TODO

        // Make u (i.e., o_n) visible
        ctx.submit(device.barrier(srw_src, srw_dst)).await;

        // d_n := 1.0 * (o_n/o) * d + h_n
        scale_and_sum_quotient(device, 1.0, &u, &o, &d, &h, &d)?;
    }
    Ok(())
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
    fn tiny() {
        let s = [5, 5, 5];
        let size = VoxelPosition::from(s);
        let brick_size = LocalVoxelPosition::from(s);

        let vol = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            if v.x().raw <= size.x().raw / 2 {
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
                UNSEEDED
            }
        });

        let expected = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            if v.x().raw <= size.x().raw / 2 {
                0.0
            } else {
                1.0
            }
        });

        let v = random_walker(vol, seeds);

        compare_tensor_approx(v, expected, 0.001);
    }
}
