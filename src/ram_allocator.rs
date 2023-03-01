use std::{alloc::Layout, cell::RefCell, pin::Pin};

use crate::Error;

pub struct Allocator {
    alloc: RefCell<Pin<Box<good_memory_allocator::Allocator>>>,
    buffer: *mut u8,
    storage_layout: Layout,
}

impl Allocator {
    pub fn new(size: usize) -> Result<Self, Error> {
        let alignment = 4096;
        let storage_layout = Layout::from_size_align(size, alignment).unwrap();

        assert!(size > 0, "invalid storage size");

        // Safety: size is > 0
        let buffer = unsafe { std::alloc::alloc(storage_layout) };
        if buffer.is_null() {
            return Err("Failed to allocate memory buffer. Is it too large?".into());
        }

        let mut alloc = Box::pin(good_memory_allocator::Allocator::empty());

        // Safety: The allocator is pinned and will thus not move. The memory region is only used
        // by the allocator.
        unsafe { alloc.init(buffer as usize, size) };

        let alloc = RefCell::new(alloc);

        Ok(Self {
            buffer,
            alloc,
            storage_layout,
        })
    }

    pub fn alloc(&self, layout: Layout) -> Result<*mut u8, Error> {
        let mut alloc = self.alloc.borrow_mut();

        assert!(layout.size() > 0);
        // Safety: We ensure that layout.size() > 0
        let ret = unsafe { alloc.alloc(layout) };
        if ret.is_null() {
            Err("Out of memory".into())
        } else {
            Ok(ret)
        }
    }

    /// Safety: `ptr` must have been allocated with this allocator and must not have been
    /// deallocated already.
    pub unsafe fn dealloc(&self, ptr: *mut u8) {
        let mut alloc = self.alloc.borrow_mut();
        unsafe { alloc.dealloc(ptr) };
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        // Safety: buffer was allocated with exactly this layout, see new()
        unsafe { std::alloc::dealloc(self.buffer, self.storage_layout) }
    }
}
