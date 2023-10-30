use std::alloc::Layout;

use ash::vk;

use crate::{
    storage::gpu::Allocation,
    task::OpaqueTaskContext,
    vulkan::{state::VulkanState, CommandBuffer, DeviceContext},
};

pub struct ChunkRequestTable {
    num_elements: usize,
    allocation: Allocation,
    layout: Layout,
}

type RTElement = u32;

impl ChunkRequestTable {
    // Note: a barrier is needed after initialization to make values visible
    pub fn new(num_elements: usize, device: &DeviceContext) -> Self {
        let request_table_buffer_layout = Layout::array::<RTElement>(num_elements).unwrap();
        let flags = vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER;
        let buf_type = gpu_allocator::MemoryLocation::GpuOnly;
        let allocation =
            device
                .storage
                .allocate(device, request_table_buffer_layout, flags, buf_type);

        let ret = ChunkRequestTable {
            num_elements,
            allocation,
            layout: request_table_buffer_layout,
        };

        device.with_cmd_buffer(|cmd| ret.clear(cmd));

        ret
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
        let request_table_cpu_bytes = bytemuck::cast_slice_mut(request_table_cpu.as_mut_slice());
        unsafe {
            crate::vulkan::memory::copy_to_cpu(
                ctx,
                device,
                self.allocation.buffer,
                self.layout,
                request_table_cpu_bytes.as_mut_ptr(),
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
