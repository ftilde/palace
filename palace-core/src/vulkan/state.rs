use ash::vk;
use std::{
    any::Any,
    cell::{RefCell, UnsafeCell},
};

use crate::util::Map;
use id::{Id, Identify};

use super::DeviceContext;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct ResourceId(Id);
impl ResourceId {
    #[track_caller]
    pub fn new() -> Self {
        let caller = std::panic::Location::caller();

        let id = Id::combine(&[
            Id::from_data(caller.file().as_bytes()),
            Id::hash(&caller.line()),
            Id::hash(&caller.column()),
        ]);

        ResourceId(id)
    }

    pub fn with_name(name: &'static str) -> Self {
        let id = Id::from_data(name.as_bytes());
        ResourceId(id)
    }
}

#[derive(Default)]
pub struct Cache {
    values: RefCell<Map<ResourceId, UnsafeCell<Box<dyn VulkanState>>>>,
}

impl Cache {
    pub fn get<'a, V: VulkanState, D: Identify>(
        &'a self,
        id: ResourceId,
        device: &DeviceContext,
        data: D,
        generate: fn(&DeviceContext, D) -> Result<V, crate::Error>,
    ) -> Result<&'a V, crate::Error> {
        let id = ResourceId(Id::combine(&[id.0, data.id()]));
        let mut m = self.values.borrow_mut();
        let raw = match m.entry(id) {
            std::collections::hash_map::Entry::Occupied(o) => o.into_mut(),
            std::collections::hash_map::Entry::Vacant(o) => {
                let v = generate(device, data)?;
                o.insert(UnsafeCell::new(Box::new(v)))
            }
        };
        // Safety: We only ever hand out immutable references (thus no conflict with mutability)
        // and never allow removal of elements.
        Ok(unsafe { (*raw.get()).downcast_ref().unwrap() })
    }

    pub fn drain(&mut self) -> impl Iterator<Item = Box<dyn VulkanState>> {
        std::mem::take(self.values.get_mut())
            .into_values()
            .map(|v| v.into_inner())
    }
}

pub trait VulkanState: Any {
    /// Deinitialize the object which marks the end of its lifetime.
    ///
    /// Safety: Callers must ensure that the object is not used in any way anymore after this
    /// method is called. In particular, this method may not be called twice.
    ///
    /// Implementors may use this guarantee, but should note that a drop implementation (if it
    /// exists) may still be called after deinitilization.
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

impl VulkanState for vk::Framebuffer {
    unsafe fn deinitialize(&mut self, context: &DeviceContext) {
        unsafe { context.functions().destroy_framebuffer(*self, None) };
    }
}

impl VulkanState for vk::ImageView {
    unsafe fn deinitialize(&mut self, context: &DeviceContext) {
        unsafe { context.functions().destroy_image_view(*self, None) };
    }
}

impl VulkanState for vk::RenderPass {
    unsafe fn deinitialize(&mut self, context: &crate::vulkan::DeviceContext) {
        unsafe { context.functions().destroy_render_pass(*self, None) };
    }
}

impl VulkanState for vk::Sampler {
    unsafe fn deinitialize(&mut self, context: &crate::vulkan::DeviceContext) {
        unsafe { context.functions().destroy_sampler(*self, None) };
    }
}

impl<V1: VulkanState, V2: VulkanState> VulkanState for (V1, V2) {
    unsafe fn deinitialize(&mut self, context: &crate::vulkan::DeviceContext) {
        unsafe { self.0.deinitialize(context) };
        unsafe { self.1.deinitialize(context) };
    }
}
