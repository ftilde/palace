use crate::id::Id;
use crate::operator::OperatorId;
use crate::storage::gpu::Allocation;
use crate::storage::gpu::Allocator;
use crate::storage::gpu::MemoryLocation;
use crate::task::OpaqueTaskContext;
use crate::task::Request;
use crate::task::Task;
use crate::task_graph::TaskId;
use crate::Error;
use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::PushDescriptor;
use ash::vk;
use std::alloc::Layout;
use std::any::Any;
use std::borrow::Cow;
use std::cell::Cell;
use std::cell::RefCell;
use std::cell::UnsafeCell;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::time::Duration;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct RessourceId(Id);
impl RessourceId {
    pub fn new(name: &'static str) -> Self {
        let id = Id::from_data(name.as_bytes());
        RessourceId(id)
    }
    pub fn of(self, op: OperatorId) -> Self {
        RessourceId(Id::combine(&[self.0, op.inner()]))
    }
    pub fn dependent_on(self, id: impl Into<Id>) -> Self {
        RessourceId(Id::combine(&[self.0, id.into()]))
    }
    pub fn inner(&self) -> Id {
        self.0
    }
}

#[derive(Default)]
struct Cache {
    values: RefCell<BTreeMap<RessourceId, UnsafeCell<Box<dyn VulkanState>>>>,
}

impl Cache {
    fn get<'a, V: VulkanState, F: FnOnce() -> V>(&'a self, id: RessourceId, generate: F) -> &'a V {
        let mut m = self.values.borrow_mut();
        let raw = m.entry(id).or_insert_with(|| {
            let v = generate();
            UnsafeCell::new(Box::new(v))
        });
        // Safety: We only ever hand out immutable references (thus no conflict with mutability)
        // and never allow removal of elements.
        unsafe { (*raw.get()).downcast_ref().unwrap() }
    }

    fn drain(&mut self) -> impl Iterator<Item = Box<dyn VulkanState>> {
        std::mem::take(self.values.get_mut())
            .into_values()
            .map(|v| v.into_inner())
    }
}

const REQUIRED_EXTENSION_NAMES: &[*const std::ffi::c_char] = &[DebugUtils::name().as_ptr()];
const REQUIRED_DEVICE_EXTENSION_NAMES: &[*const std::ffi::c_char] =
    &[PushDescriptor::name().as_ptr()];

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "[Vulkan Debug Callback --- {:?} / {:?} --- {} / {} ]\n\t{}\n",
        message_type,
        message_severity,
        message_id_name,
        &message_id_number.to_string(),
        message
    );

    vk::FALSE
}

#[allow(dead_code)]
pub struct VulkanManager {
    entry: ash::Entry,

    instance: ash::Instance,
    debug_utils_loader: DebugUtils,
    debug_callback: vk::DebugUtilsMessengerEXT,

    device_contexts: Vec<DeviceContext>,
}

impl VulkanManager {
    pub fn new() -> Result<Self, Error> {
        unsafe {
            let entry = ash::Entry::load()?;

            // Create instance
            let application_name = cstr::cstr!("voreen-ng");
            let application_info = vk::ApplicationInfo::builder()
                .application_name(application_name)
                .engine_name(application_name)
                .api_version(vk::API_VERSION_1_3);

            let layer_names = [cstr::cstr!("VK_LAYER_KHRONOS_validation")];
            let layer_names_raw: Vec<*const c_char> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&application_info)
                .enabled_layer_names(&layer_names_raw)
                .enabled_extension_names(REQUIRED_EXTENSION_NAMES);
            let instance = entry
                .create_instance(&create_info, None)
                .expect("Instance creation failed.");

            // Register debug callback
            let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING, // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));
            let debug_utils_loader = DebugUtils::new(&entry, &instance);
            let debug_callback = debug_utils_loader
                .create_debug_utils_messenger(&create_info, None)
                .unwrap();

            // Create device contexts
            let physical_devices = instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices.");
            let device_contexts: Vec<DeviceContext> = physical_devices
                .iter()
                .enumerate()
                .filter_map(|(device_num, physical_device)| {
                    instance
                        .get_physical_device_queue_family_properties(*physical_device)
                        .iter()
                        .enumerate()
                        .find_map(|(index, info)| {
                            if info
                                .queue_flags
                                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                            {
                                DeviceContext::new(
                                    device_num,
                                    &instance,
                                    *physical_device,
                                    index as u32,
                                    info.queue_count,
                                )
                                .ok()
                            } else {
                                None
                            }
                        })
                })
                .collect();
            assert!(
                !device_contexts.is_empty(),
                "Unable to find suitable physical devices."
            );

            // Return VulkanManager
            println!("Finished initializing Vulkan!");

            Ok(VulkanManager {
                entry,

                instance,
                debug_utils_loader,
                debug_callback,

                device_contexts,
            })
        }
    }

    pub fn device_contexts(&self) -> &[DeviceContext] {
        self.device_contexts.as_slice()
    }
}

impl Drop for VulkanManager {
    fn drop(&mut self) {
        unsafe {
            for device_context in self.device_contexts.drain(..) {
                std::mem::drop(device_context);
            }

            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_callback, None);
            self.instance.destroy_instance(None);
        }
    }
}

pub struct StagingBufferStash {
    buffers: RefCell<HashMap<Layout, Vec<Allocation>>>,
    buf_type: gpu_allocator::MemoryLocation,
    flags: vk::BufferUsageFlags,
}

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

impl StagingBufferStash {
    fn new(buf_type: gpu_allocator::MemoryLocation, flags: vk::BufferUsageFlags) -> Self {
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
    unsafe fn deinitialize(&self, device: &DeviceContext) {
        let mut buffers = self.buffers.borrow_mut();
        for (_, b) in buffers.drain() {
            for b in b {
                device.storage.deallocate(b);
            }
        }
    }
}

unsafe fn strcmp(v1: *const std::ffi::c_char, v2: *const std::ffi::c_char) -> bool {
    CStr::from_ptr(v1) == CStr::from_ptr(v2)
}

#[allow(unused)]
pub struct DeviceContext {
    physical_device: vk::PhysicalDevice,
    physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    queue_family_index: u32,
    queue_count: u32,

    pub device: ash::Device,
    pub push_descriptor_ext: PushDescriptor,
    queues: Vec<vk::Queue>,

    command_pool: vk::CommandPool,
    available_command_buffers: RefCell<Vec<RawCommandBuffer>>,
    waiting_command_buffers: RefCell<BTreeMap<CmdBufferEpoch, RawCommandBuffer>>,
    current_command_buffer: RefCell<CommandBuffer>,

    pub id: DeviceId,
    submission_count: Cell<usize>,

    vulkan_states: Cache,
    pub storage: crate::storage::gpu::Storage,
    staging_to_gpu: StagingBufferStash,
    staging_to_cpu: StagingBufferStash,
}

pub struct RawCommandBuffer {
    buffer: vk::CommandBuffer,
    fence: vk::Fence,
}
pub struct CommandBuffer {
    buffer: vk::CommandBuffer,
    fence: vk::Fence,
    id: CmdBufferSubmissionId,
    device: ash::Device,
    used: Cell<bool>,
}

impl CommandBuffer {
    pub unsafe fn raw(&self) -> vk::CommandBuffer {
        self.buffer
    }

    pub fn id(&self) -> CmdBufferSubmissionId {
        self.id
    }

    pub fn pipeline_barrier(&self, barrier_info: &vk::DependencyInfo) {
        unsafe {
            self.device
                .cmd_pipeline_barrier2(self.buffer, &barrier_info)
        };
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
            let out_ptr = staging_buf.mapped_ptr().unwrap();

            // Safety: Both buffers contain plain bytes, are of the same size and do not overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    input_buf.info.data,
                    out_ptr.as_ptr().cast(),
                    layout.size(),
                )
            };

            let gpu_buf_out = device.storage.alloc_slot_raw(device, key, layout)?;
            device.with_cmd_buffer(|cmd| {
                let copy_info = vk::BufferCopy::builder().size(layout.size() as _);
                unsafe {
                    device.device.cmd_copy_buffer(
                        cmd.raw(),
                        staging_buf.buffer,
                        gpu_buf_out.buffer,
                        &[*copy_info],
                    );
                }

                Ok::<(), crate::Error>(())
            })?;
            unsafe { gpu_buf_out.initialized() };

            ctx.submit(device.wait_for_cmd_buffer_completion()).await;

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
            let Ok(gpu_buf_in) = device.storage.read(access) else {
                panic!("Data should already be in vram");
            };
            let layout = gpu_buf_in.layout;

            let staging_buf = device.staging_to_cpu.request(&device, layout);

            device.with_cmd_buffer(|cmd| {
                let copy_info = vk::BufferCopy::builder().size(layout.size() as _);
                unsafe {
                    device.device.cmd_copy_buffer(
                        cmd.raw(),
                        gpu_buf_in.buffer,
                        staging_buf.buffer,
                        &[*copy_info],
                    );
                }

                Ok::<(), crate::Error>(())
            })?;

            ctx.submit(device.wait_for_cmd_buffer_completion()).await;

            let out_buf = storage.alloc_slot_raw(key, layout).unwrap();

            // Safety: Both buffers have `layout` as their layout. Staging buf data is now valid
            // since we have waited for the command buffer to finish. This content has also reached
            // the mapped region of the staging_buf allocation since it was allocated with
            // HOST_VISIBLE and HOST_COHERENT.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    staging_buf.mapped_ptr().unwrap().as_ptr().cast(),
                    out_buf.data,
                    layout.size(),
                )
            };

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

pub type DeviceId = usize;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CmdBufferEpoch(usize);

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CmdBufferSubmissionId {
    pub device: DeviceId,
    pub epoch: CmdBufferEpoch,
}

impl DeviceContext {
    pub fn new(
        id: DeviceId,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
        queue_count: u32,
    ) -> Result<Self, Error> {
        unsafe {
            let device_extension_props = instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap();

            for ext in REQUIRED_DEVICE_EXTENSION_NAMES {
                if device_extension_props
                    .iter()
                    .find(|p| strcmp(*ext, p.extension_name.as_ptr()))
                    .is_none()
                {
                    return Err(format!(
                        "Device does not support extension {}",
                        CStr::from_ptr(*ext).to_string_lossy()
                    )
                    .into());
                }
            }

            // Create logical device
            let queue_priorities = vec![0 as f32; queue_count as usize];
            let queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities);
            let mut enabled_features_13 = vk::PhysicalDeviceVulkan13Features::builder()
                .synchronization2(true)
                .build();
            let enabled_features = vk::PhysicalDeviceFeatures::builder();
            let create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(std::slice::from_ref(&queue_create_info))
                .enabled_extension_names(REQUIRED_DEVICE_EXTENSION_NAMES)
                .enabled_features(&enabled_features)
                .push_next(&mut enabled_features_13);
            let device = instance
                .create_device(physical_device, &create_info, None)
                .expect("Device creation failed.");

            let push_descriptor_ext = PushDescriptor::new(instance, &device);

            // Get device queues
            let queues: Vec<vk::Queue> = (0..queue_count)
                .map(|index| device.get_device_queue(queue_family_index, index))
                .collect();

            // Create command pool
            let create_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);
            let command_pool = device
                .create_command_pool(&create_info, None)
                .expect("Command pool creation failed.");
            let command_buffers = RefCell::new(Vec::new());

            let vulkan_states = Cache::default();

            let allocator = Allocator::new(instance.clone(), device.clone(), physical_device);

            let staging_to_cpu = StagingBufferStash::new(
                MemoryLocation::GpuToCpu,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            );
            let staging_to_gpu = StagingBufferStash::new(
                MemoryLocation::CpuToGpu,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            );

            let physical_device_memory_properties =
                instance.get_physical_device_memory_properties(physical_device);

            let submission_count = Cell::new(0);

            let current_command_buffer = Self::create_command_buffer(command_pool, &device);
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(current_command_buffer.buffer, &begin_info)
                .expect("Failed to begin command buffer.");

            let current_command_buffer = RefCell::new(Self::pack_cmd_buffer(
                device.clone(),
                current_command_buffer,
                id,
                &submission_count,
            ));

            let storage = crate::storage::gpu::Storage::new(id, allocator);

            Ok(DeviceContext {
                physical_device,
                physical_device_memory_properties,
                queue_family_index,
                queue_count,

                device,
                queues,

                command_pool,
                available_command_buffers: command_buffers,
                waiting_command_buffers: RefCell::new(Default::default()),
                current_command_buffer,

                id,
                submission_count,

                push_descriptor_ext,

                vulkan_states,
                storage,
                staging_to_cpu,
                staging_to_gpu,
            })
        }
    }

    pub fn request_state<'a, T: VulkanState + 'static>(
        &'a self,
        identifier: RessourceId,
        init: impl FnOnce() -> T + 'a,
    ) -> &'a T {
        self.vulkan_states.get(identifier, || init())
    }

    pub fn find_memory_type_index(
        &self,
        memory_requirements: &vk::MemoryRequirements,
        memory_property_flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        self.physical_device_memory_properties.memory_types
            [..self.physical_device_memory_properties.memory_type_count as _]
            .iter()
            .enumerate()
            .find(|(index, memory_type)| {
                (0x1 << index) & memory_requirements.memory_type_bits != 0
                    && memory_type.property_flags & memory_property_flags == memory_property_flags
            })
            .map(|(index, _memory_type)| index as _)
    }

    pub fn cmd_buffer_completed(&self, epoch: CmdBufferEpoch) -> bool {
        self.waiting_command_buffers
            .borrow()
            .first_key_value()
            .map(|(k, _)| k > &epoch)
            .unwrap_or_else(|| self.current_command_buffer.borrow().id.epoch > epoch)
    }

    pub fn current_epoch(&self) -> CmdBufferEpoch {
        self.current_command_buffer.borrow().id.epoch
    }

    fn create_command_buffer(
        command_pool: vk::CommandPool,
        device: &ash::Device,
    ) -> RawCommandBuffer {
        // Create command buffer
        let create_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .command_buffer_count(1);
        let command_buffer = unsafe {
            *device
                .allocate_command_buffers(&create_info)
                .expect("Failed to allocate command buffer.")
                .first()
                .unwrap()
        };

        // Create fence
        let create_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            device
                .create_fence(&create_info, None)
                .expect("Failed to create fence.")
        };

        RawCommandBuffer {
            buffer: command_buffer,
            fence,
        }
    }

    fn pack_cmd_buffer(
        device: ash::Device,
        RawCommandBuffer { buffer, fence }: RawCommandBuffer,
        id: DeviceId,
        submission_count: &Cell<usize>,
    ) -> CommandBuffer {
        let submission_id = submission_count.get();
        submission_count.set(submission_id + 1);

        let id = CmdBufferSubmissionId {
            device: id,
            epoch: CmdBufferEpoch(submission_id),
        };

        CommandBuffer {
            buffer,
            fence,
            id,
            device,
            used: Cell::new(false),
        }
    }

    pub(crate) fn try_submit_and_cycle_command_buffer(&self) {
        let mut current = self.current_command_buffer.borrow_mut();
        if !current.used.get() {
            return;
        }

        // Create new
        let next_cmd_buffer = self
            .available_command_buffers
            .borrow_mut()
            .pop()
            .unwrap_or_else(|| Self::create_command_buffer(self.command_pool, &self.device));

        let next_cmd_buffer = Self::pack_cmd_buffer(
            self.device.clone(),
            next_cmd_buffer,
            self.id,
            &self.submission_count,
        );

        assert_eq!(
            unsafe { self.device.get_fence_status(next_cmd_buffer.fence) },
            Ok(false)
        );

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(next_cmd_buffer.buffer, &begin_info)
                .expect("Failed to begin command buffer.")
        };

        // Swap current
        let prev_cmd_buffer = std::mem::replace(&mut *current, next_cmd_buffer);

        // Submit current
        unsafe {
            self.device
                .end_command_buffer(prev_cmd_buffer.buffer)
                .unwrap()
        };
        let bufs = [prev_cmd_buffer.buffer];
        let submit = vk::SubmitInfo::builder().command_buffers(&bufs);
        unsafe {
            self.device
                .queue_submit(
                    *self.queues.first().unwrap(),
                    &[submit.build()],
                    prev_cmd_buffer.fence,
                )
                .expect("Failed to submit command buffers to queue.")
        };

        // Stow away prev one as waiting
        let CommandBuffer {
            buffer, fence, id, ..
        } = prev_cmd_buffer;
        let CmdBufferSubmissionId { epoch, .. } = id;
        self.waiting_command_buffers
            .borrow_mut()
            .insert(epoch, RawCommandBuffer { buffer, fence });
    }

    #[must_use]
    pub fn wait_for_cmd_buffer_completion<'req, 'irrelevant>(
        &'req self,
    ) -> Request<'req, 'irrelevant, ()> {
        let current = self.current_command_buffer.borrow();
        let id = current.id;
        Request {
            type_: crate::task::RequestType::CmdBufferCompletion(id),
            gen_poll: Box::new(move |ctx| {
                Box::new(move || {
                    if ctx.device_contexts[id.device].cmd_buffer_completed(id.epoch) {
                        Some(())
                    } else {
                        None
                    }
                })
            }),
            _marker: Default::default(),
        }
    }

    pub(crate) fn wait_for_cmd_buffers(&self, timeout: Duration) -> Vec<CmdBufferSubmissionId> {
        let mut result = Vec::new();

        let mut waiting_ref = self.waiting_command_buffers.borrow_mut();
        if waiting_ref.is_empty() {
            return result;
        }
        let waiting_fences = waiting_ref.iter().map(|v| v.1.fence).collect::<Vec<_>>();

        let wait_nanos = timeout.as_nanos().min(u64::max_value() as _) as u64;
        match unsafe {
            self.device
                .wait_for_fences(&waiting_fences[..], true, wait_nanos)
        } {
            Ok(()) => {}
            Err(vk::Result::TIMEOUT) => return result,
            Err(o) => panic!("Wait for fences failed {}", o),
        }

        let waiting = std::mem::take(&mut *waiting_ref);
        // TODO: replace with BTreeMap::drain_filter once stable
        let new_waiting = waiting
            .into_iter()
            .filter_map(|(epoch, command_buffer)| {
                if unsafe {
                    self.device
                        .get_fence_status(command_buffer.fence)
                        .unwrap_or(false)
                } {
                    unsafe {
                        self.device
                            .reset_fences(&[command_buffer.fence])
                            .expect("Failed to reset fence.");
                        self.device
                            .reset_command_buffer(
                                command_buffer.buffer,
                                vk::CommandBufferResetFlags::empty(),
                            )
                            .expect("Failed to reset command buffer.");
                    }

                    self.available_command_buffers
                        .borrow_mut()
                        .push(command_buffer);
                    result.push(CmdBufferSubmissionId {
                        device: self.id,
                        epoch,
                    });
                    None
                } else {
                    Some((epoch, command_buffer))
                }
            })
            .collect();

        *waiting_ref = new_waiting;
        result
    }

    pub fn with_cmd_buffer<R, F: FnOnce(&CommandBuffer) -> R>(&self, f: F) -> R {
        let cmd_buf = self.current_command_buffer.borrow();
        cmd_buf.used.set(true);

        f(&cmd_buf)
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        for mut vulkan_state in self.vulkan_states.drain() {
            unsafe { vulkan_state.deinitialize(self) };
        }

        // Safety: Using the exact same allocator (the only one of this device)
        unsafe {
            self.staging_to_gpu.deinitialize(&self);
            self.staging_to_cpu.deinitialize(&self);
        }

        self.storage.deinitialize();

        assert!(self.waiting_command_buffers.get_mut().is_empty());

        unsafe {
            for cb in self.available_command_buffers.get_mut().drain(..) {
                self.device.destroy_fence(cb.fence, None);
            }
            self.device
                .destroy_fence(self.current_command_buffer.borrow().fence, None);

            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
        }
    }
}

pub trait VulkanState: Any {
    unsafe fn deinitialize(&mut self, context: &DeviceContext);
}

impl dyn VulkanState {
    // This is required as long as trait upcasting is still unstable:
    // https://github.com/rust-lang/rust/issues/65991
    fn downcast_ref<T: VulkanState>(&self) -> Option<&T> {
        if self.type_id() == std::any::TypeId::of::<T>() {
            Some(unsafe { &*(self as *const Self as *const T) })
        } else {
            None
        }
    }
}
