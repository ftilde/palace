use crate::storage::gpu::Allocator;
use crate::storage::gpu::MemoryLocation;
use crate::task::Request;
use crate::Error;
use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::PushDescriptor;
use ash::extensions::khr::Surface as SurfaceExt;
use ash::extensions::khr::Swapchain;

#[cfg(target_family = "unix")]
use ash::extensions::khr::{WaylandSurface, XlibSurface};

#[cfg(target_family = "windows")]
use ash::extensions::khr::Win32Surface;

use ash::vk;
use std::borrow::Cow;
use std::cell::Cell;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::{c_char, CStr};
use std::time::Duration;

use self::state::RessourceId;
use self::state::VulkanState;

pub mod memory;
pub mod pipeline;
pub mod shader;
pub mod state;
pub mod window;

const REQUIRED_EXTENSION_NAMES: &[*const std::ffi::c_char] = &[
    DebugUtils::name().as_ptr(),
    SurfaceExt::name().as_ptr(),
    #[cfg(target_family = "unix")]
    XlibSurface::name().as_ptr(),
    #[cfg(target_family = "unix")]
    WaylandSurface::name().as_ptr(),
    #[cfg(target_family = "windows")]
    Win32Surface::name().as_ptr(),
];
const REQUIRED_DEVICE_EXTENSION_NAMES: &[*const std::ffi::c_char] =
    &[PushDescriptor::name().as_ptr(), Swapchain::name().as_ptr()];

#[cfg(debug_assertions)]
const DEFAULT_LAYER_NAMES: &[&CStr] = &[cstr::cstr!("VK_LAYER_KHRONOS_validation")];

#[cfg(not(debug_assertions))]
const DEFAULT_LAYER_NAMES: &[&CStr] = &[];

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

#[derive(Clone)]
pub struct GlobalFunctions {
    debug_utils_ext: DebugUtils,
    surface_ext: SurfaceExt,
}

#[allow(dead_code)]
pub struct VulkanContext {
    entry: ash::Entry,

    instance: ash::Instance,
    debug_callback: vk::DebugUtilsMessengerEXT,
    functions: GlobalFunctions,

    device_contexts: Vec<DeviceContext>,
}

impl VulkanContext {
    pub fn new(gpu_mem_capacity: u64) -> Result<Self, Error> {
        unsafe {
            let entry = ash::Entry::load()?;

            // Create instance
            let application_name = cstr::cstr!("voreen-ng");
            let application_info = vk::ApplicationInfo::builder()
                .application_name(application_name)
                .engine_name(application_name)
                .api_version(vk::API_VERSION_1_3);

            let layer_names_raw: Vec<*const c_char> = DEFAULT_LAYER_NAMES
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
            let debug_utils_ext = DebugUtils::new(&entry, &instance);
            let surface_ext = SurfaceExt::new(&entry, &instance);
            let functions = GlobalFunctions {
                surface_ext,
                debug_utils_ext,
            };
            let debug_callback = functions
                .debug_utils_ext
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
                                    gpu_mem_capacity,
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
            println!(
                "Finished initializing Vulkan! ({} devices)",
                device_contexts.len()
            );

            Ok(VulkanContext {
                entry,

                instance,
                debug_callback,
                functions,

                device_contexts,
            })
        }
    }

    pub fn device_contexts(&self) -> &[DeviceContext] {
        self.device_contexts.as_slice()
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            for device_context in self.device_contexts.drain(..) {
                std::mem::drop(device_context);
            }

            self.functions
                .debug_utils_ext
                .destroy_debug_utils_messenger(self.debug_callback, None);
            self.instance.destroy_instance(None);
        }
    }
}

unsafe fn strcmp(v1: *const std::ffi::c_char, v2: *const std::ffi::c_char) -> bool {
    CStr::from_ptr(v1) == CStr::from_ptr(v2)
}

#[derive(Clone)]
pub struct DeviceFunctions {
    pub device: ash::Device,
    pub push_descriptor_ext: PushDescriptor,
    pub swap_chain_ext: Swapchain, //TODO: Make optional?
}

impl std::ops::Deref for DeviceFunctions {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

pub struct RawCommandBuffer {
    buffer: vk::CommandBuffer,
    fence: vk::Fence,
}
pub struct CommandBuffer {
    buffer: vk::CommandBuffer,
    fence: vk::Fence,
    id: CmdBufferSubmissionId,
    functions: DeviceFunctions,
    oldest_finished_epoch: CmdBufferEpoch,
    used_since: Cell<Option<std::time::Instant>>,
    wait_semaphores: RefCell<Vec<vk::SemaphoreSubmitInfo>>,
    signal_semaphores: RefCell<Vec<vk::SemaphoreSubmitInfo>>,
}

impl CommandBuffer {
    pub unsafe fn raw(&self) -> vk::CommandBuffer {
        self.buffer
    }

    pub fn id(&self) -> CmdBufferSubmissionId {
        self.id
    }

    pub fn functions(&self) -> &DeviceFunctions {
        &self.functions
    }

    pub fn wait_semaphore(&mut self, s: impl std::ops::Deref<Target = vk::SemaphoreSubmitInfo>) {
        self.wait_semaphores.borrow_mut().push(*s);
    }

    pub fn signal_semaphore(&mut self, s: impl std::ops::Deref<Target = vk::SemaphoreSubmitInfo>) {
        self.signal_semaphores.borrow_mut().push(*s);
    }

    pub fn age(&self) -> std::time::Duration {
        self.used_since
            .get()
            .map(|b| b.elapsed())
            .unwrap_or(std::time::Duration::from_secs(0))
    }
}

pub type DeviceId = usize;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CmdBufferEpoch(usize);

impl CmdBufferEpoch {
    /// An epoch that definitely lies in the past
    fn ancient() -> Self {
        // Since works because we start the first commandbuffer with epoch 1
        Self(0)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CmdBufferSubmissionId {
    pub device: DeviceId,
    pub epoch: CmdBufferEpoch,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SrcBarrierInfo {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DstBarrierInfo {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BarrierInfo {
    pub src: SrcBarrierInfo,
    pub dst: DstBarrierInfo,
    pub device: DeviceId,
}

#[allow(unused)]
pub struct DeviceContext {
    physical_device: vk::PhysicalDevice,
    physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    queue_family_index: u32,
    queue_count: u32,

    functions: DeviceFunctions,
    queues: Vec<vk::Queue>,

    command_pool: vk::CommandPool,
    available_command_buffers: RefCell<Vec<RawCommandBuffer>>,
    waiting_command_buffers: RefCell<BTreeMap<CmdBufferEpoch, RawCommandBuffer>>,
    current_command_buffer: RefCell<CommandBuffer>,
    oldest_finished: Cell<CmdBufferEpoch>,

    pub id: DeviceId,
    submission_count: Cell<usize>,

    vulkan_states: state::Cache,
    pub storage: crate::storage::gpu::Storage,
    staging_to_gpu: memory::BufferStash,
    staging_to_cpu: memory::BufferStash,
    pub tmp_states: memory::TempStates,
}

impl DeviceContext {
    pub fn new(
        id: DeviceId,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
        queue_count: u32,
        mem_capacity: u64,
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
            let mut enabled_features_12 = vk::PhysicalDeviceVulkan12Features::builder()
                .shader_buffer_int64_atomics(true)
                .buffer_device_address(true)
                .scalar_block_layout(true)
                .runtime_descriptor_array(true)
                .storage_buffer8_bit_access(true)
                .shader_int8(true)
                .build();
            let enabled_features = vk::PhysicalDeviceFeatures::builder().shader_int64(true);
            let create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(std::slice::from_ref(&queue_create_info))
                .enabled_extension_names(REQUIRED_DEVICE_EXTENSION_NAMES)
                .enabled_features(&enabled_features)
                .push_next(&mut enabled_features_12)
                .push_next(&mut enabled_features_13);
            let device = instance
                .create_device(physical_device, &create_info, None)
                .expect("Device creation failed.");

            let push_descriptor_ext = PushDescriptor::new(instance, &device);
            let swap_chain_ext = Swapchain::new(instance, &device);

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

            let vulkan_states = state::Cache::default();

            let allocator = Allocator::new(
                instance.clone(),
                device.clone(),
                physical_device,
                mem_capacity,
            );

            let staging_to_cpu = memory::BufferStash::new(
                MemoryLocation::GpuToCpu,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            );
            let staging_to_gpu = memory::BufferStash::new(
                MemoryLocation::CpuToGpu,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            );

            let physical_device_memory_properties =
                instance.get_physical_device_memory_properties(physical_device);

            // We start epochs at one so that oldest_finished (with value 0) means that none are
            // finished, yet.
            let oldest_finished = Cell::new(CmdBufferEpoch::ancient());
            let submission_count = Cell::new(1);

            let functions = DeviceFunctions {
                device,
                push_descriptor_ext,
                swap_chain_ext,
            };

            let current_command_buffer = Self::create_command_buffer(command_pool, &functions);
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            functions
                .begin_command_buffer(current_command_buffer.buffer, &begin_info)
                .expect("Failed to begin command buffer.");

            let current_command_buffer = RefCell::new(Self::pack_cmd_buffer(
                functions.clone(),
                current_command_buffer,
                id,
                &submission_count,
                oldest_finished.get(),
            ));

            let storage = crate::storage::gpu::Storage::new(id, allocator);

            Ok(DeviceContext {
                physical_device,
                physical_device_memory_properties,
                queue_family_index,
                queue_count,

                functions,
                queues,

                command_pool,
                available_command_buffers: command_buffers,
                waiting_command_buffers: RefCell::new(Default::default()),
                current_command_buffer,
                oldest_finished,

                id,
                submission_count,

                vulkan_states,
                storage,
                staging_to_cpu,
                staging_to_gpu,
                tmp_states: Default::default(),
            })
        }
    }

    pub fn functions(&self) -> &DeviceFunctions {
        &self.functions
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

    pub fn oldest_finished_epoch(&self) -> CmdBufferEpoch {
        self.oldest_finished.get()
    }

    fn create_command_buffer(
        command_pool: vk::CommandPool,
        functions: &DeviceFunctions,
    ) -> RawCommandBuffer {
        // Create command buffer
        let create_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .command_buffer_count(1);
        let command_buffer = unsafe {
            *functions
                .allocate_command_buffers(&create_info)
                .expect("Failed to allocate command buffer.")
                .first()
                .unwrap()
        };

        // Create fence
        let create_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            functions
                .create_fence(&create_info, None)
                .expect("Failed to create fence.")
        };

        RawCommandBuffer {
            buffer: command_buffer,
            fence,
        }
    }

    fn pack_cmd_buffer(
        functions: DeviceFunctions,
        RawCommandBuffer { buffer, fence }: RawCommandBuffer,
        id: DeviceId,
        submission_count: &Cell<usize>,
        oldest_finished_epoch: CmdBufferEpoch,
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
            functions,
            oldest_finished_epoch,
            used_since: Cell::new(None),
            wait_semaphores: Default::default(),
            signal_semaphores: Default::default(),
        }
    }

    pub(crate) fn try_submit_and_cycle_command_buffer(&self) -> Option<CmdBufferSubmissionId> {
        let mut current = self.current_command_buffer.borrow_mut();
        if current.used_since.get().is_none() {
            return None;
        }

        // Create new
        let next_cmd_buffer = self
            .available_command_buffers
            .borrow_mut()
            .pop()
            .unwrap_or_else(|| Self::create_command_buffer(self.command_pool, &self.functions));

        let next_cmd_buffer = Self::pack_cmd_buffer(
            self.functions.clone(),
            next_cmd_buffer,
            self.id,
            &self.submission_count,
            self.oldest_finished_epoch(),
        );

        assert_eq!(
            unsafe { self.functions.get_fence_status(next_cmd_buffer.fence) },
            Ok(false)
        );

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.functions
                .begin_command_buffer(next_cmd_buffer.buffer, &begin_info)
                .expect("Failed to begin command buffer.")
        };

        // Swap current
        let prev_cmd_buffer = std::mem::replace(&mut *current, next_cmd_buffer);

        // Submit current
        unsafe {
            self.functions
                .end_command_buffer(prev_cmd_buffer.buffer)
                .unwrap()
        };
        let wait_semaphores = prev_cmd_buffer.wait_semaphores.borrow();
        let signal_semaphores = prev_cmd_buffer.signal_semaphores.borrow();

        let cmd_infos = [vk::CommandBufferSubmitInfo::builder()
            .command_buffer(prev_cmd_buffer.buffer)
            .build()];
        let submit = vk::SubmitInfo2::builder()
            .command_buffer_infos(&cmd_infos)
            .wait_semaphore_infos(&wait_semaphores)
            .signal_semaphore_infos(&signal_semaphores);
        let submits = [submit.build()];
        unsafe {
            self.functions
                .queue_submit2(
                    *self.queues.first().unwrap(),
                    &submits,
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
        Some(id)
    }

    #[must_use]
    pub fn wait_for_current_cmd_buffer_submission<'req, 'irrelevant>(
        &'req self,
    ) -> Request<'req, 'irrelevant, ()> {
        let current = self.current_command_buffer.borrow();
        let id = current.id;
        Request {
            type_: crate::task::RequestType::CmdBufferSubmission(id),
            gen_poll: Box::new(move |ctx| {
                Box::new(move || {
                    if ctx.device_contexts[id.device].current_epoch() > id.epoch {
                        Some(())
                    } else {
                        None
                    }
                })
            }),
            _marker: Default::default(),
        }
    }

    #[must_use]
    pub fn wait_for_cmd_buffer_completion<'req, 'irrelevant>(
        &'req self,
        epoch: CmdBufferEpoch,
    ) -> Request<'req, 'irrelevant, ()> {
        let id = CmdBufferSubmissionId {
            device: self.id,
            epoch,
        };
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

    #[must_use]
    pub fn wait_for_current_cmd_buffer_completion<'req, 'irrelevant>(
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

    #[must_use]
    pub fn barrier<'req, 'irrelevant>(
        &'req self,
        src: SrcBarrierInfo,
        dst: DstBarrierInfo,
    ) -> Request<'req, 'irrelevant, ()> {
        let barrier_info = BarrierInfo {
            src,
            dst,
            device: self.id,
        };
        let initial_epoch = self.storage.barrier_manager.current_epoch();
        Request {
            type_: crate::task::RequestType::Barrier(barrier_info),
            gen_poll: Box::new(move |_ctx| {
                Box::new(move || {
                    if self
                        .storage
                        .barrier_manager
                        .is_visible(src, dst, initial_epoch)
                    {
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
        let oldest_waiting_epoch = *waiting_ref.first_key_value().unwrap().0;
        let waiting_fences = waiting_ref.iter().map(|v| v.1.fence).collect::<Vec<_>>();

        let wait_nanos = timeout.as_nanos().min(u64::max_value() as _) as u64;
        match unsafe {
            self.functions
                .wait_for_fences(&waiting_fences[..], true, wait_nanos)
        } {
            Ok(()) => {}
            Err(vk::Result::TIMEOUT) => return result,
            Err(o) => panic!("Wait for fences failed {}", o),
        }

        let waiting = std::mem::take(&mut *waiting_ref);
        // TODO: replace with Map::drain_filter once stable
        let new_waiting = waiting
            .into_iter()
            .filter_map(|(epoch, command_buffer)| {
                if unsafe {
                    self.functions
                        .get_fence_status(command_buffer.fence)
                        .unwrap_or(false)
                } {
                    unsafe {
                        self.functions
                            .reset_fences(&[command_buffer.fence])
                            .expect("Failed to reset fence.");
                        self.functions
                            .reset_command_buffer(
                                command_buffer.buffer,
                                vk::CommandBufferResetFlags::empty(),
                            )
                            .expect("Failed to reset command buffer.");
                    }

                    // TODO: This is not perfect (since self.oldest_finished is only set to the
                    // oldest waiting even though newer ones may be done as well), but still
                    // ensures that the oldest_waiting_epoch is updated eventually.
                    if epoch == oldest_waiting_epoch {
                        self.oldest_finished.set(epoch);
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

        // Now that we have marked some cmd buffers as finished, try to collect garbage tmp
        // buffers.
        self.tmp_states.collect_returns(self);

        result
    }

    pub fn cmd_buffer_age(&self) -> std::time::Duration {
        self.current_command_buffer.borrow().age()
    }
    pub fn with_cmd_buffer<R, F: FnOnce(&mut CommandBuffer) -> R>(&self, f: F) -> R {
        let mut cmd_buf = self.current_command_buffer.borrow_mut();
        cmd_buf.used_since.set(
            cmd_buf
                .used_since
                .get()
                .or_else(|| Some(std::time::Instant::now())),
        );

        f(&mut cmd_buf)
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        // Try to return tmp buffers a final time
        self.tmp_states.collect_returns(self);

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
                self.functions.destroy_fence(cb.fence, None);
            }
            self.functions
                .destroy_fence(self.current_command_buffer.borrow().fence, None);

            self.functions.destroy_command_pool(self.command_pool, None);
            self.functions.destroy_device(None);
        }
    }
}
