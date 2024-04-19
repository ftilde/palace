use std::alloc::Layout;

use ash::vk;

use crate::{
    dim::Dimension,
    dtypes::StaticElementType,
    operators::tensor::TensorOperator,
    storage::{
        gpu::{Allocation, IndexHandle},
        Element,
    },
    task::{OpaqueTaskContext, Request},
    vulkan::{state::VulkanState, CommandBuffer, DeviceContext, DstBarrierInfo},
};

pub struct ChunkRequestTable {
    num_elements: usize,
    allocation: Allocation,
    layout: Layout,
}

type RTElement = u32;

impl ChunkRequestTable {
    // Note: a barrier is needed after initialization to make values visible
    pub fn new<'a, 'inv>(
        num_elements: usize,
        device: &'a DeviceContext,
    ) -> Request<'a, 'inv, Self> {
        let request_table_buffer_layout = Layout::array::<RTElement>(num_elements).unwrap();
        let flags = vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER;
        let buf_type = gpu_allocator::MemoryLocation::GpuOnly;
        device
            .storage
            .request_allocate_raw(device, request_table_buffer_layout, flags, buf_type)
            .map(move |allocation| {
                let ret = ChunkRequestTable {
                    num_elements,
                    allocation,
                    layout: request_table_buffer_layout,
                };

                device.with_cmd_buffer(|cmd| ret.clear(cmd));

                ret
            })
    }

    // Note: a barrier is needed after clearing to make values visible
    pub fn clear(&self, cmd: &mut CommandBuffer) {
        unsafe {
            cmd.functions().cmd_fill_buffer(
                cmd.raw(),
                self.allocation.buffer,
                0,
                vk::WHOLE_SIZE,
                0xffffffff,
            )
        };
    }

    pub fn buffer(&self) -> &Allocation {
        &self.allocation
    }

    // Note: any changes to the buffer have to be made visible to the cpu side via a barrier first
    pub async fn download_requested<'cref, 'inv>(
        &self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        device: &'cref DeviceContext,
    ) -> Vec<RTElement> {
        let mut request_table_cpu = vec![0u32; self.num_elements];
        let request_table_cpu_bytes: &mut [u8] =
            bytemuck::cast_slice_mut(request_table_cpu.as_mut_slice());
        unsafe {
            crate::vulkan::memory::copy_to_cpu(
                ctx,
                device,
                self.allocation.buffer,
                self.layout,
                request_table_cpu_bytes.as_mut_ptr().cast(),
            )
            .await
        };

        let to_request_linear = request_table_cpu
            .into_iter()
            .filter(|v| *v != RTElement::max_value())
            .collect::<Vec<RTElement>>();

        to_request_linear
    }
}

impl VulkanState for ChunkRequestTable {
    unsafe fn deinitialize(&mut self, context: &DeviceContext) {
        self.allocation.deinitialize(context);
    }
}

pub struct Timeout;

pub async fn request_to_index_with_timeout<'cref, 'inv, D: Dimension, E: Element>(
    ctx: &OpaqueTaskContext<'cref, 'inv>,
    device: &DeviceContext,
    to_request_linear: &mut [RTElement],
    vol: &'inv TensorOperator<D, StaticElementType<E>>,
    index: &IndexHandle<'_>,
) -> Result<(), Timeout> {
    let dim_in_bricks = vol.metadata.dimension_in_chunks();
    let num_bricks = dim_in_bricks.hmul();

    // Sort to get at least some benefit from spatial neighborhood
    to_request_linear.sort_unstable();

    // Fulfill requests
    let mut batch_size = 1;
    let mut to_request_linear = &to_request_linear[..];
    while !to_request_linear.is_empty() {
        let batch;
        (batch, to_request_linear) =
            to_request_linear.split_at(batch_size.min(to_request_linear.len()));

        let to_request = batch.iter().map(|v| {
            assert!(*v < num_bricks as _);
            vol.chunks.request_gpu(
                device.id,
                crate::vec::from_linear(*v as usize, dim_in_bricks),
                DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                },
            )
        });
        let requested_bricks = ctx.submit(ctx.group(to_request)).await;

        for (brick, brick_linear_pos) in requested_bricks.into_iter().zip(batch.into_iter()) {
            index.insert(*brick_linear_pos as u64, brick);
        }

        if ctx.past_deadline() {
            return Err(Timeout);
        }

        batch_size *= 2;
    }

    Ok(())
}
