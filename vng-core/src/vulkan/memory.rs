use ash::vk;
use std::{
    alloc::Layout,
    cell::{Cell, RefCell},
    collections::{BTreeMap, HashMap},
};

use crate::{
    operator::OperatorId,
    storage::gpu::Allocation,
    task::{OpaqueTaskContext, Task},
    task_graph::TaskId,
};

use super::{CmdBufferEpoch, DeviceContext};

pub struct CachedAllocation {
    inner: Allocation,
    requested_layout: Layout, //May differ from layout of the allocation, at least in size
}

impl std::ops::Deref for CachedAllocation {
    type Target = Allocation;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct TempBuffer {
    pub allocation: Allocation,
    pub size: u64,
}

#[derive(Default)]
pub struct TempBuffers {
    returns: RefCell<BTreeMap<CmdBufferEpoch, Vec<Allocation>>>,
}

impl TempBuffers {
    pub fn request(&self, device: &DeviceContext, layout: Layout) -> TempBuffer {
        let flags = vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER;
        let buf_type = gpu_allocator::MemoryLocation::GpuOnly;
        TempBuffer {
            allocation: device.storage.allocate(device, layout, flags, buf_type),
            size: layout.size() as u64,
        }
    }

    /// Safety: The buffer must have previously been allocated from tmp buffer manager
    pub unsafe fn return_buf(&self, device: &DeviceContext, buf: TempBuffer) {
        let mut returns = self.returns.borrow_mut();
        returns
            .entry(device.current_epoch())
            .or_default()
            .push(buf.allocation);
    }

    pub fn collect_returns(&self, device: &DeviceContext) {
        let mut returns = self.returns.borrow_mut();
        while returns
            .first_key_value()
            .map(|(epoch, _)| *epoch <= device.oldest_finished_epoch())
            .unwrap_or(false)
        {
            let (_, allocs) = returns.pop_first().unwrap();
            for alloc in allocs {
                unsafe { device.storage.deallocate(alloc) };
            }
        }
    }
}

pub struct BufferStash {
    buffers: RefCell<HashMap<Layout, Vec<Allocation>>>,
    buf_type: gpu_allocator::MemoryLocation,
    flags: vk::BufferUsageFlags,
}

impl BufferStash {
    pub fn new(buf_type: gpu_allocator::MemoryLocation, flags: vk::BufferUsageFlags) -> Self {
        Self {
            buffers: Default::default(),
            buf_type,
            flags,
        }
    }
    fn request(&self, device: &DeviceContext, layout: Layout) -> CachedAllocation {
        let mut buffers = self.buffers.borrow_mut();
        let buffers = buffers.entry(layout).or_default();
        let inner = buffers.pop().unwrap_or_else(|| {
            device
                .storage
                .allocate(device, layout, self.flags, self.buf_type)
        });
        CachedAllocation {
            inner,
            requested_layout: layout,
        }
    }
    /// Safety: The buffer must have previously been allocated from this stash
    unsafe fn return_buf(&self, allocation: CachedAllocation) {
        let mut buffers = self.buffers.borrow_mut();
        buffers
            .get_mut(&allocation.requested_layout)
            .unwrap()
            .push(allocation.inner);
    }

    /// Safety: The device must be the same that was used for all `request`s.
    pub unsafe fn deinitialize(&self, device: &DeviceContext) {
        let mut buffers = self.buffers.borrow_mut();
        for (_, b) in buffers.drain() {
            for b in b {
                device.storage.deallocate(b);
            }
        }
    }
}

pub struct TransferManager {
    transfer_count: Cell<usize>,
    op_id: OperatorId,
}

impl Default for TransferManager {
    fn default() -> Self {
        Self {
            transfer_count: Cell::new(0),
            op_id: OperatorId::new("builtin::TransferManager"),
        }
    }
}

struct SendPointer<T>(*mut T);
impl<T> SendPointer<T> {
    // Safety: You may only do thread-safe things in other threads after unpacking
    unsafe fn pack(p: *mut T) -> Self {
        Self(p)
    }
    fn unpack(self) -> *mut T {
        self.0
    }
}
unsafe impl<T> Send for SendPointer<T> {}

impl TransferManager {
    pub fn next_id(&self) -> TaskId {
        let count = self.transfer_count.get();
        self.transfer_count.set(count + 1);
        TaskId::new(self.op_id, count)
    }
    pub fn transfer_to_gpu<'cref, 'inv>(
        &self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        device: &'cref DeviceContext,
        access: crate::storage::ram::AccessToken<'cref>,
    ) -> Task<'cref> {
        async move {
            let storage = ctx.storage;
            let key = access.id;
            let Ok(input_buf) = storage.read_raw(access) else {
                panic!("Data should already be in ram");
            };
            let layout = input_buf.info.layout;
            let staging_buf = device.staging_to_gpu.request(&device, layout);
            let out_ptr =
                unsafe { SendPointer::pack(staging_buf.mapped_ptr().unwrap().as_ptr().cast()) };
            let in_ptr = unsafe { SendPointer::pack(input_buf.info.data) };

            // Safety: Both buffers contain plain bytes, are of the same size and do not overlap.
            ctx.submit(ctx.spawn_compute(|| {
                unsafe {
                    std::ptr::copy_nonoverlapping(in_ptr.unpack(), out_ptr.unpack(), layout.size())
                };
            }))
            .await;

            let gpu_buf_out = device.storage.alloc_slot_raw(device, key, layout)?;
            device.with_cmd_buffer(|cmd| {
                let copy_info = vk::BufferCopy::builder().size(layout.size() as _);
                unsafe {
                    device.functions().cmd_copy_buffer(
                        cmd.raw(),
                        staging_buf.buffer,
                        gpu_buf_out.buffer,
                        &[*copy_info],
                    );
                }

                Ok::<(), crate::Error>(())
            })?;
            unsafe {
                gpu_buf_out.initialized(super::SrcBarrierInfo {
                    stage: vk::PipelineStageFlags2::TRANSFER,
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                })
            };

            ctx.submit(device.wait_for_current_cmd_buffer_completion())
                .await;

            // Safety: We have waited for cmd_buffer completion. Thus the staging_buf is not used
            // in copying anymore and can be freed.
            unsafe {
                device.staging_to_gpu.return_buf(staging_buf);
            }
            Ok(())
        }
        .into()
    }

    pub fn transfer_to_cpu<'cref, 'inv>(
        &self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        device: &'cref DeviceContext,
        access: crate::storage::gpu::AccessToken<'cref>,
    ) -> Task<'cref> {
        async move {
            let key = access.id;
            let storage = ctx.storage;
            let dst_info = super::DstBarrierInfo {
                stage: vk::PipelineStageFlags2::TRANSFER,
                access: vk::AccessFlags2::TRANSFER_READ,
            };
            if let Err(src_info) = device.storage.is_visible(access.id, dst_info) {
                ctx.submit(device.barrier(src_info, dst_info)).await;
            };
            let Ok(gpu_buf_in) = device.storage.read(access, dst_info) else {
                panic!("Data should already be in vram and visible");
            };
            let layout = gpu_buf_in.layout;

            let staging_buf = device.staging_to_cpu.request(&device, layout);

            device.with_cmd_buffer(|cmd| {
                let copy_info = vk::BufferCopy::builder().size(layout.size() as _);
                unsafe {
                    device.functions().cmd_copy_buffer(
                        cmd.raw(),
                        gpu_buf_in.buffer,
                        staging_buf.buffer,
                        &[*copy_info],
                    );
                }

                Ok::<(), crate::Error>(())
            })?;

            ctx.submit(device.wait_for_current_cmd_buffer_completion())
                .await;

            let out_buf = storage.alloc_slot_raw(key, layout).unwrap();

            // Safety: Both buffers have `layout` as their layout. Staging buf data is now valid
            // since we have waited for the command buffer to finish. This content has also reached
            // the mapped region of the staging_buf allocation since it was allocated with
            // HOST_VISIBLE and HOST_COHERENT.
            let in_ptr =
                unsafe { SendPointer::pack(staging_buf.mapped_ptr().unwrap().as_ptr().cast()) };
            let out_ptr = unsafe { SendPointer::pack(out_buf.data) };
            ctx.submit(ctx.spawn_compute(|| unsafe {
                std::ptr::copy_nonoverlapping(in_ptr.unpack(), out_ptr.unpack(), layout.size())
            }))
            .await;

            // Safety: We have just written the complete buffer using a memcpy
            unsafe {
                out_buf.initialized();
            }

            // Safety: This is exactly the buffer that we requested above and it is no longer used
            // in the compute pipeline since we have waited for the command buffer to finish.
            unsafe {
                device.staging_to_cpu.return_buf(staging_buf);
            }

            Ok(())
        }
        .into()
    }
}
