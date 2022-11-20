use crate::Error;
use std::borrow::Cow;
use std::ffi::{CStr, c_char};
use ash::vk;
use ash::extensions::ext::DebugUtils;

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

    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    queue_count: u32,

    device : ash::Device
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

            // Find suitable physical device
            let physical_devices = instance.enumerate_physical_devices().expect("Failed to enumerate physical devices.");
            let (physical_device, queue_family_index, queue_count) = physical_devices.iter().find_map(|physical_device| {
                instance.get_physical_device_queue_family_properties(*physical_device).iter().enumerate().find_map(|(index, info)| {
                    if info.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE) {
                        Some((*physical_device, index as u32, info.queue_count))
                    } else {
                        None
                    }
                })
            }).expect("Failed to find suitable physical device.");

            // Create device
            let queue_priorities = vec![0 as f32; queue_count as usize];
            let queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities);
            let enabled_features = vk::PhysicalDeviceFeatures::builder();
            let create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(std::slice::from_ref(&queue_create_info))
                .enabled_features(&enabled_features);
            let device = instance.create_device(physical_device, &create_info, None).expect("Device creation failed.");

            // Return VulkanManager
            println!("Finished initializing Vulkan!");

            Ok(VulkanManager {
                entry,

                instance,
                debug_utils_loader,
                debug_callback,

                physical_device,
                queue_family_index,
                queue_count,

                device
            })
        }
    }
}

impl Drop for VulkanManager {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.debug_utils_loader.destroy_debug_utils_messenger(self.debug_callback, None);
            self.instance.destroy_instance(None);
        }
    }
}