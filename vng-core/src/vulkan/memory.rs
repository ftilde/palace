use ash::vk;
use std::{
    alloc::Layout,
    cell::{Cell, RefCell},
    collections::BTreeMap,
    mem::MaybeUninit,
    rc::Rc,
};

use crate::{
    operator::{DataDescriptor, DataId, OperatorId},
    storage::{cpu::CpuAllocator, gpu::Allocation, CpuDataLocation, DataLocation},
    task::{OpaqueTaskContext, Request, Task},
    task_graph::TaskId,
    util::Map,
};

use super::{state::VulkanState, CmdBufferEpoch, DeviceContext};

// Since we are using a direct mapping of repr(c) structs to scalar layout glsl structs in a bunch
// of places, we need to ensure that the vulkan/spirv rule* of size == align for scalars is true on
// the host system as well:
// * See https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#interfaces-resources-layout
const fn _scalar_align_check<T>() {
    assert!(std::mem::size_of::<T>() == std::mem::align_of::<T>());
}
const _: () = _scalar_align_check::<u8>();
const _: () = _scalar_align_check::<i8>();
const _: () = _scalar_align_check::<u16>();
const _: () = _scalar_align_check::<i16>();
const _: () = _scalar_align_check::<u32>();
const _: () = _scalar_align_check::<i32>();
const _: () = _scalar_align_check::<u64>();
const _: () = _scalar_align_check::<i64>();
const _: () = _scalar_align_check::<f32>();
const _: () = _scalar_align_check::<f64>();

pub struct CachedAllocation {
    pub inner: Allocation,
    //requested_layout: Layout, //May differ from layout of the allocation, at least in size
}

impl std::ops::Deref for CachedAllocation {
    type Target = Allocation;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct TempRessource<'a, R: VulkanState> {
    // Always init except after drop is called
    pub ressource: MaybeUninit<R>,
    device: &'a DeviceContext,
}

impl<'a, R: VulkanState> TempRessource<'a, R> {
    pub fn new(device: &'a DeviceContext, ressource: R) -> Self {
        TempRessource {
            ressource: MaybeUninit::new(ressource),
            device,
        }
    }
}

impl<'a, R: VulkanState> std::ops::Deref for TempRessource<'a, R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        unsafe { self.ressource.assume_init_ref() }
    }
}

impl<'a, R: VulkanState> std::ops::DerefMut for TempRessource<'a, R> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ressource.assume_init_mut() }
    }
}

impl<R: VulkanState> Drop for TempRessource<'_, R> {
    fn drop(&mut self) {
        unsafe {
            self.device.tmp_states.release(
                self.device,
                std::mem::replace(&mut self.ressource, MaybeUninit::uninit()).assume_init(),
            )
        };
    }
}

#[derive(Default)]
pub struct TempStates {
    returns: RefCell<BTreeMap<CmdBufferEpoch, Vec<Box<dyn VulkanState>>>>,
}

impl TempStates {
    /// Release a VulkanState for deinitialization and freeing after the current epoch has finished.
    ///
    /// Safety: The state must have previously been initialized from the device of these TempStates
    pub unsafe fn release(&self, device: &DeviceContext, state: impl VulkanState) {
        let mut returns = self.returns.borrow_mut();
        returns
            .entry(device.current_epoch())
            .or_default()
            .push(Box::new(state));
    }

    pub fn collect_returns(&self, device: &DeviceContext) {
        let mut returns = self.returns.borrow_mut();
        while returns
            .first_key_value()
            .map(|(epoch, _)| *epoch <= device.oldest_finished_epoch())
            .unwrap_or(false)
        {
            let (_, allocs) = returns.pop_first().unwrap();
            for mut alloc in allocs {
                unsafe {
                    alloc.deinitialize(device);
                }
            }
        }
    }
}

pub struct BufferStash {
    //buffers: RefCell<HashMap<Layout, Vec<Allocation>>>,
    buf_type: gpu_allocator::MemoryLocation,
    flags: vk::BufferUsageFlags,
}

impl BufferStash {
    pub fn new(buf_type: gpu_allocator::MemoryLocation, flags: vk::BufferUsageFlags) -> Self {
        Self {
            //buffers: Default::default(),
            buf_type,
            flags,
        }
    }
    pub fn request<'a, 'inv>(
        &self,
        device: &'a DeviceContext,
        layout: Layout,
    ) -> Request<'a, 'inv, CachedAllocation> {
        //let mut buffers = self.buffers.borrow_mut();
        //let buffers = buffers.entry(layout).or_default();
        //let inner = if let Some(b) = buffers.pop() {
        //    Request::ready(b)
        //} else {
        //    device
        //        .storage
        //        .request_allocate_raw(device, layout, self.flags, self.buf_type)
        //};
        let inner = device
            .storage
            .request_allocate_raw(device, layout, self.flags, self.buf_type);
        inner.map(move |inner| CachedAllocation {
            inner,
            //requested_layout: layout,
        })
    }
    /// Safety: The buffer must have previously been allocated from this stash
    pub unsafe fn return_buf(&self, device: &DeviceContext, allocation: CachedAllocation) {
        //let mut buffers = self.buffers.borrow_mut();
        //let entry = buffers.get_mut(&allocation.requested_layout).unwrap();
        //entry.push(allocation.inner);
        let _ = TempRessource::new(device, allocation.inner);
    }

    /// Safety: The device must be the same that was used for all `request`s.
    pub unsafe fn deinitialize(&self, _device: &DeviceContext) {
        //let mut buffers = self.buffers.borrow_mut();
        //for (_, b) in buffers.drain() {
        //    for b in b {
        //        device.storage.deallocate(b);
        //    }
        //}
    }
}

pub struct TransferManager {
    transfer_count: Cell<usize>,
    op_id: OperatorId,
    running: Rc<RefCell<Map<(DataId, DataLocation), TaskId>>>,
}

impl Default for TransferManager {
    fn default() -> Self {
        Self {
            transfer_count: Cell::new(0),
            op_id: OperatorId::new("builtin::TransferManager"),
            running: Default::default(),
        }
    }
}

struct SendPointerMut<T>(*mut T);
impl<T> SendPointerMut<T> {
    // Safety: You may only do thread-safe things in other threads after unpacking
    unsafe fn pack(p: *mut T) -> Self {
        Self(p)
    }
    fn unpack(self) -> *mut T {
        self.0
    }
}
unsafe impl<T> Send for SendPointerMut<T> {}

struct SendPointer<T>(*const T);
impl<T> SendPointer<T> {
    // Safety: You may only do thread-safe things in other threads after unpacking
    unsafe fn pack(p: *const T) -> Self {
        Self(p)
    }
    fn unpack(self) -> *const T {
        self.0
    }
}
unsafe impl<T> Send for SendPointer<T> {}

pub enum TransferTaskResult<'cref> {
    New(Task<'cref>),
    Existing(TaskId),
}

impl TransferManager {
    pub fn next_id(&self) -> TaskId {
        let count = self.transfer_count.get();
        self.transfer_count.set(count + 1);
        TaskId::new(self.op_id, count)
    }

    fn supply_transfer_task<'cref, 'inv, F: futures::Future + 'cref>(
        &self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        transfer_task_key: (DataId, DataLocation),
        task: F,
    ) -> TransferTaskResult<'cref> {
        let running = Rc::clone(&self.running);
        let maybe_existing = { self.running.borrow().get(&transfer_task_key).cloned() };
        if let Some(id) = maybe_existing {
            TransferTaskResult::Existing(id)
        } else {
            let prev = running
                .borrow_mut()
                .insert(transfer_task_key, ctx.current_task);
            assert!(prev.is_none());
            TransferTaskResult::New(
                async move {
                    task.await;
                    running.borrow_mut().remove(&transfer_task_key).unwrap();

                    Ok(())
                }
                .into(),
            )
        }
    }

    pub fn transfer_cpu_to_gpu<'cref, 'inv, A: CpuAllocator>(
        &self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        device: &'cref DeviceContext,
        source: crate::storage::cpu::RawReadHandle<'cref, A>,
    ) -> TransferTaskResult<'cref> {
        let transfer_task_key = (source.id(), DataLocation::GPU(device.id));
        self.supply_transfer_task(ctx, transfer_task_key, async move {
            let key = source.id();

            let layout = source.info.layout;
            let desc = DataDescriptor {
                id: key,
                longevity: source.info.data_longevity,
            };

            let gpu_buf_out = ctx.submit(ctx.alloc_raw_gpu(device, desc, layout)).await;

            unsafe { copy_to_gpu(ctx, device, source.info.data, layout, gpu_buf_out.buffer).await };

            unsafe {
                gpu_buf_out.initialized(
                    ctx,
                    super::SrcBarrierInfo {
                        stage: vk::PipelineStageFlags2::TRANSFER,
                        access: vk::AccessFlags2::TRANSFER_WRITE,
                    },
                )
            };
        })
    }

    pub fn transfer_gpu_to_cpu<'cref, 'inv>(
        &self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        device: &'cref DeviceContext,
        access: crate::storage::gpu::AccessToken<'cref>,
        dst: CpuDataLocation,
    ) -> TransferTaskResult<'cref> {
        let transfer_task_key = (access.id, DataLocation::CPU(dst));
        self.supply_transfer_task(ctx, transfer_task_key, async move {
            let key = access.id;
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

            let desc = DataDescriptor {
                id: key,
                longevity: gpu_buf_in.data_longevity,
            };
            match dst {
                CpuDataLocation::Ram => {
                    let out_buf = ctx.submit(ctx.alloc_raw(desc, layout)).await;

                    unsafe {
                        copy_to_cpu(ctx, device, gpu_buf_in.buffer, layout, out_buf.data).await
                    };

                    // Safety: We have just written the complete buffer using a memcpy
                    unsafe {
                        out_buf.initialized(ctx);
                    }
                }
                CpuDataLocation::Disk => {
                    let out_buf = ctx.submit(ctx.alloc_raw_disk(desc, layout)).await;

                    unsafe {
                        copy_to_cpu(ctx, device, gpu_buf_in.buffer, layout, out_buf.data).await
                    };

                    // Safety: We have just written the complete buffer using a memcpy
                    unsafe {
                        out_buf.initialized(ctx);
                    }
                }
            };
        })
    }
    pub fn transfer_cpu_to_cpu<'cref, 'inv, A: CpuAllocator>(
        &self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        source: crate::storage::cpu::RawReadHandle<'cref, A>,
        dst: CpuDataLocation,
    ) -> TransferTaskResult<'cref> {
        assert_ne!(A::LOCATION, dst, "src and dst must differ for transfer");
        let transfer_task_key = (source.id(), DataLocation::CPU(dst));

        self.supply_transfer_task(ctx, transfer_task_key, async move {
            let key = source.id();
            let layout = source.info.layout;
            let desc = DataDescriptor {
                id: key,
                longevity: source.info.data_longevity,
            };
            match dst {
                CpuDataLocation::Ram => {
                    let out_buf = ctx.submit(ctx.alloc_raw(desc, layout)).await;

                    // Safety: layout is taken from input, output was constructed with the very
                    // layout
                    unsafe { copy_cpu_to_cpu(ctx, source.info.data, layout, out_buf.data).await };

                    // Safety: We have just written the complete buffer using a memcpy
                    unsafe {
                        out_buf.initialized(ctx);
                    }
                }
                CpuDataLocation::Disk => {
                    let out_buf = ctx.submit(ctx.alloc_raw_disk(desc, layout)).await;

                    // Safety: layout is taken from input, output was constructed with the very
                    // layout
                    unsafe { copy_cpu_to_cpu(ctx, source.info.data, layout, out_buf.data).await };

                    // Safety: We have just written the complete buffer using a memcpy
                    unsafe {
                        out_buf.initialized(ctx);
                    }
                }
            };
        })
    }
}

// Safety: `in_buf` and `out_buf` must have the layout `layout`
pub async unsafe fn copy_cpu_to_cpu<'cref, 'inv>(
    ctx: OpaqueTaskContext<'cref, 'inv>,
    in_buf: *const MaybeUninit<u8>,
    layout: Layout,
    out_buf: *mut MaybeUninit<u8>,
) {
    let in_ptr = unsafe { SendPointer::pack(in_buf) };
    let out_ptr = unsafe { SendPointerMut::pack(out_buf) };

    ctx.submit(ctx.spawn_compute(|| {
        unsafe { std::ptr::copy_nonoverlapping(in_ptr.unpack(), out_ptr.unpack(), layout.size()) };
    }))
    .await;
}

/// Copy buffer from cpu to gpu. Both buffers must have the same layout.
pub async unsafe fn copy_to_gpu<'cref, 'inv>(
    ctx: OpaqueTaskContext<'cref, 'inv>,
    device: &'cref DeviceContext,
    in_buf: *const MaybeUninit<u8>,
    layout: Layout,
    buffer_out: ash::vk::Buffer,
) {
    let staging_buf = ctx
        .submit(device.staging_to_gpu.request(&device, layout))
        .await;

    let out_ptr =
        unsafe { SendPointerMut::pack(staging_buf.mapped_ptr().unwrap().as_ptr().cast()) };
    let in_ptr = unsafe { SendPointer::pack(in_buf) };

    // Safety: Both buffers contain plain bytes, are of the same size and do not overlap.
    ctx.submit(ctx.spawn_compute(|| {
        unsafe { std::ptr::copy_nonoverlapping(in_ptr.unpack(), out_ptr.unpack(), layout.size()) };
    }))
    .await;

    device.with_cmd_buffer(|cmd| {
        let copy_info = vk::BufferCopy::builder().size(layout.size() as _);
        unsafe {
            device.functions().cmd_copy_buffer(
                cmd.raw(),
                staging_buf.buffer,
                buffer_out,
                &[*copy_info],
            );
        }
    });

    //TODO: We might be able to speed this up by returning the buf with an cmdbuf epoch after which
    //it can be used again (similar to tmpressources). This may a actually provide a benefit for
    //automatic GPU<->CPU transfers
    //ctx.submit(device.wait_for_current_cmd_buffer_completion())
    //    .await;

    // Safety: We have waited for cmd_buffer completion. Thus the staging_buf is not used
    // in copying anymore and can be freed.
    unsafe {
        device.staging_to_gpu.return_buf(device, staging_buf);
    }
}

/// Copy buffer from gpu to cpu. Both buffers must have the same layout.
/// All bytes of out_buf are written to in this call.
pub async unsafe fn copy_to_cpu<'cref, 'inv>(
    ctx: OpaqueTaskContext<'cref, 'inv>,
    device: &'cref DeviceContext,
    buffer_in: ash::vk::Buffer,
    layout: Layout,
    out_buf: *mut MaybeUninit<u8>,
) {
    let staging_buf = ctx
        .submit(device.staging_to_cpu.request(&device, layout))
        .await;

    device.with_cmd_buffer(|cmd| {
        let copy_info = vk::BufferCopy::builder().size(layout.size() as _);
        unsafe {
            device.functions().cmd_copy_buffer(
                cmd.raw(),
                buffer_in,
                staging_buf.buffer,
                &[*copy_info],
            );
        }
    });

    ctx.submit(device.wait_for_current_cmd_buffer_completion())
        .await;

    // Safety: Both buffers have `layout` as their layout. Staging buf data is now valid
    // since we have waited for the command buffer to finish. This content has also reached
    // the mapped region of the staging_buf allocation since it was allocated with
    // HOST_VISIBLE and HOST_COHERENT.
    let in_ptr = unsafe { SendPointerMut::pack(staging_buf.mapped_ptr().unwrap().as_ptr().cast()) };
    let out_ptr = unsafe { SendPointerMut::pack(out_buf) };
    ctx.submit(ctx.spawn_compute(|| unsafe {
        std::ptr::copy_nonoverlapping(in_ptr.unpack(), out_ptr.unpack(), layout.size())
    }))
    .await;

    // Safety: This is exactly the buffer that we requested above and it is no longer used
    // in the compute pipeline since we have waited for the command buffer to finish.
    unsafe {
        device.staging_to_cpu.return_buf(device, staging_buf);
    }
}
