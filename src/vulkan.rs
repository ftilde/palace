use crate::id::Id;
use crate::task::Request;
use crate::Error;
use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::PushDescriptor;
use ash::vk;
use gpu_allocator::vulkan::AllocationScheme;
use gpu_allocator::MemoryLocation;
use std::alloc::Layout;
use std::any::Any;
use std::borrow::Cow;
use std::cell::Cell;
use std::cell::RefCell;
use std::cell::UnsafeCell;
use std::collections::BTreeMap;
use std::ffi::{c_char, CStr};
use std::ops::Deref;
use std::time::Duration;

#[derive(Default)]
struct Cache {
    values: RefCell<BTreeMap<Id, UnsafeCell<Box<dyn VulkanState>>>>,
}

impl Cache {
    fn get<'a, V: VulkanState, F: FnOnce() -> V>(
        &'a self,
        key: &'static str,
        generate: F,
    ) -> &'a V {
        let t_id = crate::id::func_id::<V>();
        let key = Id::combine(&[Id::from_data(key.as_bytes()), t_id]);

        let mut m = self.values.borrow_mut();
        let raw = m.entry(key).or_insert_with(|| {
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

pub struct Allocation {
    pub allocation: gpu_allocator::vulkan::Allocation,
    pub buffer: vk::Buffer,
}

pub struct Allocator {
    allocator: RefCell<Option<gpu_allocator::vulkan::Allocator>>,
    device: ash::Device,
}

impl Allocator {
    pub fn new(
        instance: ash::Instance,
        device: ash::Device,
        physical_device: vk::PhysicalDevice,
    ) -> Self {
        let allocator = RefCell::new(Some(
            gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
                instance,
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false, // TODO: check the BufferDeviceAddressFeatures struct.
            })
            .unwrap(),
        ));
        Self { allocator, device }
    }
    pub fn allocate(
        &self,
        layout: Layout,
        use_flags: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Allocation {
        // Setup vulkan info
        let vk_info = vk::BufferCreateInfo::builder()
            .size(layout.size() as u64)
            .usage(use_flags);

        let buffer = unsafe { self.device.create_buffer(&vk_info, None) }.unwrap();
        let mut requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        requirements.alignment = requirements.alignment.max(layout.align() as u64);

        let mut allocator = self.allocator.borrow_mut();
        let allocator = allocator.as_mut().unwrap();
        let allocation = allocator
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "some allocation",
                requirements,
                location,     // TODO: Try to choose something more specific
                linear: true, // Buffers are always linear
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .unwrap();

        // Bind memory to the buffer
        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap()
        };

        Allocation { allocation, buffer }
    }

    pub fn deallocate(&self, allocation: Allocation) {
        let mut allocator = self.allocator.borrow_mut();
        let allocator = allocator.as_mut().unwrap();
        allocator.free(allocation.allocation).unwrap();
        unsafe { self.device.destroy_buffer(allocation.buffer, None) };
    }

    pub fn deinitialize(&mut self) {
        let mut a = self.allocator.borrow_mut();
        let mut tmp = None;
        std::mem::swap(&mut *a, &mut tmp);
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
    available_command_buffers: RefCell<Vec<(vk::CommandBuffer, vk::Fence)>>,
    waiting_command_buffers: RefCell<BTreeMap<CmdBufferSubmissionId, CommandBuffer>>,

    id: DeviceId,
    submission_count: Cell<usize>,

    vulkan_states: Cache,
    allocator: Allocator,
}

pub struct CommandBuffer {
    buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl Deref for CommandBuffer {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

pub type DeviceId = usize;
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CmdBufferSubmissionId {
    device: DeviceId,
    num: usize,
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

            let physical_device_memory_properties =
                instance.get_physical_device_memory_properties(physical_device);

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

                id,
                submission_count: Cell::new(0),

                push_descriptor_ext,

                vulkan_states,
                allocator,
            })
        }
    }

    pub fn allocator(&self) -> &Allocator {
        &self.allocator
    }

    pub fn request_state<'a, T: VulkanState + 'static>(
        &'a self,
        identifier: &'static str,
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

    pub fn begin_command_buffer(&self) -> CommandBuffer {
        unsafe {
            let (buffer, fence) = self
                .available_command_buffers
                .borrow_mut()
                .pop()
                .unwrap_or_else(|| {
                    // Create command buffer
                    let create_info = vk::CommandBufferAllocateInfo::builder()
                        .command_pool(self.command_pool)
                        .command_buffer_count(1);
                    let command_buffer = *self
                        .device
                        .allocate_command_buffers(&create_info)
                        .expect("Failed to allocate command buffer.")
                        .first()
                        .unwrap();

                    // Create fence
                    let create_info = vk::FenceCreateInfo::default();
                    let fence = self
                        .device
                        .create_fence(&create_info, None)
                        .expect("Failed to create fence.");

                    (command_buffer, fence)
                });

            assert_eq!(self.device.get_fence_status(fence), Ok(false));

            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(buffer, &begin_info)
                .expect("Failed to begin command buffer.");
            return CommandBuffer { buffer, fence };
        }
    }

    #[must_use]
    pub fn submit_command_buffer<'req, 'irrelevant>(
        &'req self,
        command_buffer: CommandBuffer,
    ) -> Request<'req, 'irrelevant, ()> {
        unsafe {
            self.device.end_command_buffer(*command_buffer).unwrap();
            let submits = [vk::SubmitInfo::builder()
                .command_buffers(&[*command_buffer])
                .build()];
            self.device
                .queue_submit(
                    *self.queues.first().unwrap(),
                    &submits,
                    command_buffer.fence,
                )
                .expect("Failed to submit command buffers to queue.");

            //self.device
            //    .wait_for_fences(&[command_buffer.fence], true, u64::max_value())
            //    .expect("Failed to wait for fence.");

            let submission_id = self.submission_count.get();
            self.submission_count.set(submission_id + 1);

            let id = CmdBufferSubmissionId {
                device: self.id,
                num: submission_id,
            };
            let fence = command_buffer.fence;
            self.waiting_command_buffers
                .borrow_mut()
                .insert(id, command_buffer);
            Request {
                type_: crate::task::RequestType::CmdBufferCompletion(id),
                poll: Box::new(move |_ctx| {
                    if self.device.get_fence_status(fence).unwrap() {
                        self.recover_finished_cmd_buffer(id);
                        Some(())
                    } else {
                        None
                    }
                }),
                _marker: Default::default(),
            }
        }
    }

    fn recover_finished_cmd_buffer(&self, id: CmdBufferSubmissionId) {
        let command_buffer = self
            .waiting_command_buffers
            .borrow_mut()
            .remove(&id)
            .unwrap();
        unsafe {
            assert_eq!(self.device.get_fence_status(command_buffer.fence), Ok(true));

            self.device
                .reset_fences(&[command_buffer.fence])
                .expect("Failed to reset fence.");
            self.device
                .reset_command_buffer(*command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("Failed to reset command buffer.");
        }

        self.available_command_buffers
            .borrow_mut()
            .push((command_buffer.buffer, command_buffer.fence));
    }

    pub(crate) fn wait_for_cmd_buffers(&self, timeout: Duration) -> Vec<CmdBufferSubmissionId> {
        let mut result = Vec::new();

        let waiting = self.waiting_command_buffers.borrow();
        if waiting.is_empty() {
            return result;
        }
        let waiting_fences = waiting.iter().map(|v| v.1.fence).collect::<Vec<_>>();

        let wait_nanos = timeout.as_nanos().min(u64::max_value() as _) as u64;
        match unsafe {
            self.device
                .wait_for_fences(&waiting_fences[..], true, wait_nanos)
        } {
            Ok(()) => {}
            Err(vk::Result::TIMEOUT) => return result,
            Err(o) => panic!("Wait for fences failed {}", o),
        }

        for (id, cb) in waiting.iter() {
            if unsafe { self.device.get_fence_status(cb.fence).unwrap_or(false) } {
                result.push(*id);
            }
        }

        result
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        for mut vulkan_state in self.vulkan_states.drain() {
            unsafe { vulkan_state.deinitialize(self) };
        }

        self.allocator.deinitialize();

        assert!(self.waiting_command_buffers.get_mut().is_empty());

        unsafe {
            for (_buf, fence) in self.available_command_buffers.get_mut().drain(..) {
                self.device.destroy_fence(fence, None);
            }

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
