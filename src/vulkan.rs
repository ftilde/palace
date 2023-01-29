use crate::Error;
use std::any::Any;
use std::cell::Ref;
use std::cell::RefCell;
use std::borrow::{Cow, Borrow};
use std::collections::HashMap;
use std::ffi::{CStr, c_char};
use ash::vk::{self, DeviceGroupDeviceCreateInfo};
use ash::extensions::ext::DebugUtils;
use ndarray::IndexLonger;

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

    println!("[Vulkan Debug Callback --- {:?} / {:?} --- {} / {} ]\n\t{}\n",
        message_type, message_severity, message_id_name, &message_id_number.to_string(), message);

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
            let application_name = CStr::from_bytes_with_nul_unchecked(b"voreen-ng\0");
            let application_info = vk::ApplicationInfo::builder()
                .application_name(application_name)
                .engine_name(application_name)
                .api_version(vk::make_api_version(0, 1, 3, 0));

            let layer_names = [CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0")];
            let layer_names_raw: Vec<*const c_char> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();
            
            let extension_names_raw = vec![DebugUtils::name().as_ptr()];

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&application_info)
                .enabled_layer_names(&layer_names_raw)
                .enabled_extension_names(&extension_names_raw);
            let instance = entry.create_instance(&create_info, None).expect("Instance creation failed.");

            // Register debug callback
            let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                )
                .pfn_user_callback(Some(vulkan_debug_callback));
            let debug_utils_loader = DebugUtils::new(&entry, &instance);
            let debug_callback = debug_utils_loader
                .create_debug_utils_messenger(&create_info, None)
                .unwrap();

            // Create device contexts
            let physical_devices = instance.enumerate_physical_devices().expect("Failed to enumerate physical devices.");
            let device_contexts: Vec<DeviceContext> = physical_devices.iter().filter_map(|physical_device| {
                instance.get_physical_device_queue_family_properties(*physical_device).iter().enumerate().find_map(|(index, info)| {
                    if info.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE) {
                        DeviceContext::new(&instance, *physical_device, index as u32, info.queue_count).ok()
                    } else {
                        None
                    }
                })
            }).collect();
            assert!(!device_contexts.is_empty(), "Unable to find suitable physical devices.");

            // Return VulkanManager
            println!("Finished initializing Vulkan!");

            Ok(VulkanManager {
                entry,

                instance,
                debug_utils_loader,
                debug_callback,

                device_contexts
            })
        }
    }
}

impl Drop for VulkanManager {
    fn drop(&mut self) {
        unsafe {
            for device_context in &self.device_contexts {
                device_context.deinitialize();
            }

            self.debug_utils_loader.destroy_debug_utils_messenger(self.debug_callback, None);
            self.instance.destroy_instance(None);
        }
    }
}

pub struct DeviceContext {
    physical_device: vk::PhysicalDevice,
    physical_device_memory_properties : vk::PhysicalDeviceMemoryProperties,
    queue_family_index: u32,
    queue_count: u32,
    
    pub device: ash::Device,
    queues: Vec<vk::Queue>,

    command_pool: vk::CommandPool,
    command_buffers: RefCell<Vec<(vk::CommandBuffer, vk::Fence)>>,

    vulkan_states: RefCell<HashMap<String, Box<dyn Any>>>
}

impl DeviceContext {
    pub fn new(instance: &ash::Instance, physical_device: vk::PhysicalDevice, queue_family_index: u32, queue_count: u32) -> Result<Self, Error> {
        unsafe {
            let physical_device_memory_properties = instance.get_physical_device_memory_properties(physical_device);

            // Create logical device
            let queue_priorities = vec![0 as f32; queue_count as usize];
            let queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities);
            let enabled_features = vk::PhysicalDeviceFeatures::builder();
            let create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(std::slice::from_ref(&queue_create_info))
                .enabled_features(&enabled_features);
            let device = instance.create_device(physical_device, &create_info, None).expect("Device creation failed.");

            // Get device queues
            let queues: Vec<vk::Queue> = (0..queue_count).map(|index|{
                device.get_device_queue(queue_family_index, index)
            }).collect();

            // Create command pool
            let create_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);
            let command_pool = device.create_command_pool(&create_info, None).expect("Command pool creation failed.");
            let command_buffers = RefCell::new(Vec::new());

            let vulkan_states = RefCell::new(HashMap::new());

            Ok(DeviceContext {
                physical_device,
                physical_device_memory_properties,
                queue_family_index,
                queue_count,

                device,
                queues,

                command_pool,
                command_buffers,

                vulkan_states
            })
        }
    }

    pub fn request_state<T: VulkanState + Default + 'static>(&self, identifier: &String) -> Ref<'_, Box<T>> {
        if !self.vulkan_states.borrow().contains_key(identifier) {
            let state = Box::new(T::default());
            state.initialize(self);
            self.vulkan_states.borrow_mut().insert(identifier.clone(), state);
        }
        
        Ref::map(self.vulkan_states.borrow(), |state| state.get(identifier).unwrap().downcast_ref::<Box<T>>().expect("wrong type"))
    }

    pub fn find_memory_type_index(&self,
        memory_requirements : &vk::MemoryRequirements,
        memory_property_flags: vk::MemoryPropertyFlags 
    ) -> Option<u32> {
        self.physical_device_memory_properties.memory_types[..self.physical_device_memory_properties.memory_type_count as _]
            .iter()
            .enumerate()
            .find(|(index, memory_type)| {
                (0x1 << index) & memory_requirements.memory_type_bits != 0 && memory_type.property_flags & memory_property_flags == memory_property_flags
            }).map(|(index, _memory_type)| index as _)
    }


    pub fn begin_command_buffer(&self) -> vk::CommandBuffer {
        unsafe {
            for (command_buffer, fence) in self.command_buffers.borrow().iter() {
                if self.device.get_fence_status(*fence).unwrap_or(false) {
                    let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                    self.device.begin_command_buffer(*command_buffer, &begin_info).expect("Failed to begin command buffer.");
                    return *command_buffer
                }
            }

            // Create command buffer
            let create_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.command_pool)
                .command_buffer_count(1);
            let command_buffer = *self.device.allocate_command_buffers(&create_info).expect("Failed to allocate command buffer.").first().unwrap();

            // Create fence
            let create_info = vk::FenceCreateInfo::default();
            let fence = self.device.create_fence(&create_info, None).expect("Failed to create fence.");

            self.command_buffers.borrow_mut().push((command_buffer, fence));

            let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(command_buffer, &begin_info).expect("Failed to begin command buffer.");
            return command_buffer
        }
    }
    pub fn submit_command_buffer(&self, command_buffer: vk::CommandBuffer) -> vk::Fence {        
        unsafe {
            for (other, fence) in self.command_buffers.borrow().iter() {
                if command_buffer == *other {
                    let submits = [vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&command_buffer)).build()];
                    self.device.queue_submit(*self.queues.first().unwrap(), &submits, *fence).expect("Failed to submit command buffers to queue.");

                    self.device.wait_for_fences(std::slice::from_ref(fence), true, u64::max_value()).expect("Failed to wait for fence.");
                    self.device.reset_fences(std::slice::from_ref(fence)).expect("Failed to reset fence.");
                    self.device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty()).expect("Failed to reset command buffer.");
        
                    return *fence        
                }
            }
            panic!("Tried to submit unknown command buffer.");
        }
    }

    pub fn deinitialize(&self) {
        unsafe {
            for (_identifier, vulkan_state) in self.vulkan_states.borrow().iter() {
                vulkan_state.downcast_ref::<Box<dyn VulkanState>>().unwrap().deinitialize(self);
            }

            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
        }
    }
}

pub trait VulkanState {
    fn initialize(&self, context: &DeviceContext);
    fn deinitialize(&self, context: &DeviceContext);
}