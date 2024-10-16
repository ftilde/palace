use std::alloc::Layout;

use ash::vk;

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
        pipeline::{ComputePipeline, DescriptorConfig, DynPushConstants},
        shader::ShaderDefines,
        state::{RessourceId, VulkanState},
        DstBarrierInfo, SrcBarrierInfo,
    },
};

struct SparseMatrix {
    values: Allocation,
    index: Allocation,
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

                let mut num_rows = 0;
                for i in s.iter() {
                    if *i != 0xffffffff {
                        num_rows = num_rows.max(*i);
                    }
                }

                ctx.submit(ctx.write_scalar(num_rows)).await;

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
                let _device = ctx.preferred_device();

                let (_matrix, _vec) =
                    init_weights(*ctx, tensor, seeds, tensor_to_rows_table, num_seeds).await?;

                todo!();
            }
            .into()
        },
    )
}

async fn init_weights<'req, 'inv>(
    ctx: OpaqueTaskContext<'req, 'inv>,
    t: &'inv TensorOperator<D3, StaticElementType<f32>>,
    seeds: &'inv TensorOperator<D3, StaticElementType<f32>>,
    tensor_to_rows_table: &'inv TensorOperator<D1, StaticElementType<u32>>,
    num_seeds: &'inv ScalarOperator<StaticElementType<u32>>,
) -> Result<(SparseMatrix, Allocation), crate::Error> {
    let device = ctx.preferred_device();

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
                        .add("MAT_ROWS", mat_rows)
                        .add("ND", nd),
                ),
                false,
            )
        },
    )?;

    let flags = vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::TRANSFER_SRC;
    let buf_type = gpu_allocator::MemoryLocation::GpuOnly;

    let (values, index, vec) = futures::join!(
        ctx.submit(device.storage.request_allocate_raw(
            device,
            Layout::array::<f32>(mat_size).unwrap(),
            flags,
            buf_type,
        )),
        ctx.submit(device.storage.request_allocate_raw(
            device,
            Layout::array::<u32>(mat_size).unwrap(),
            flags,
            buf_type,
        )),
        ctx.submit(device.storage.request_allocate_raw(
            device,
            Layout::array::<f32>(tensor_size).unwrap(),
            flags,
            buf_type,
        ))
    );

    device.with_cmd_buffer(|cmd| unsafe {
        cmd.functions()
            .cmd_fill_buffer(cmd.raw(), index.buffer, 0, vk::WHOLE_SIZE, 0xffffffff);
        cmd.functions()
            .cmd_fill_buffer(cmd.raw(), values.buffer, 0, vk::WHOLE_SIZE, 0);
    });

    ctx.submit(device.barrier(
        SrcBarrierInfo {
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access: vk::AccessFlags2::SHADER_WRITE,
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

    Ok((SparseMatrix { values, index }, vec))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        test_util::compare_tensor,
        vec::{LocalVoxelPosition, VoxelPosition},
    };

    #[test]
    fn tiny() {
        let size = VoxelPosition::fill(2.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let vol = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            if v.x().raw == 0 {
                0.1
            } else {
                0.9
            }
        });

        let seeds = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            if v == VoxelPosition::fill(0.into()) {
                0.1
            } else if v == VoxelPosition::fill(1.into()) {
                0.9
            } else {
                UNSEEDED
            }
        });

        let expected = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            if v.x().raw == 0 {
                0.001
            } else {
                0.999
            }
        });

        let v = random_walker(vol, seeds);

        compare_tensor(v, expected);
    }
}
