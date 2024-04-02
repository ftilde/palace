use std::{alloc::Layout, cell::RefCell, mem::MaybeUninit, pin::Pin};

use crate::Error;

use super::cpu::{CpuAllocator, OOMError};

pub type Storage = super::cpu::Storage<RamAllocator>;
pub type ReadHandle<'a, T> = super::cpu::ReadHandle<'a, T, RamAllocator>;
pub type WriteHandleInit<'a, T> = super::cpu::WriteHandleInit<'a, T, RamAllocator>;
pub type WriteHandleUninit<'a, T> = super::cpu::WriteHandleUninit<'a, T, RamAllocator>;
pub type RawWriteHandleInit<'a> = super::cpu::RawWriteHandleInit<'a, RamAllocator>;
pub type RawWriteHandleUninit<'a> = super::cpu::RawWriteHandleUninit<'a, RamAllocator>;
pub type AccessToken<'a> = super::cpu::AccessToken<'a, RamAllocator>;
pub type InplaceResult<'a, 'inv, T> = super::cpu::InplaceResult<'a, 'inv, T, RamAllocator>;
pub type InplaceHandle<'a, T> = super::cpu::InplaceHandle<'a, T, RamAllocator>;

pub struct RamAllocator {
    alloc: RefCell<Pin<Box<good_memory_allocator::Allocator>>>,
    buffer: *mut u8,
    storage_layout: Layout,
}

impl RamAllocator {
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
}

impl CpuAllocator for RamAllocator {
    const LOCATION: super::CpuDataLocation = super::CpuDataLocation::Ram;

    fn alloc(&self, layout: Layout) -> Result<*mut MaybeUninit<u8>, OOMError> {
        let mut alloc = self.alloc.borrow_mut();

        assert!(layout.size() > 0);
        // Safety: We ensure that layout.size() > 0
        let ret = unsafe { alloc.alloc(layout) };
        if ret.is_null() {
            Err(OOMError {
                //requested: layout.size(),
            })
        } else {
            // Casting from *mut u8 to *mut MaybeUninit<u8> is always fine
            Ok(ret.cast())
        }
    }

    /// Safety: `ptr` must have been allocated with this allocator and must not have been
    /// deallocated already.
    unsafe fn dealloc(&self, ptr: *mut MaybeUninit<u8>) {
        let mut alloc = self.alloc.borrow_mut();

        // Assuming the allocator does not read the bytes (why would it?) it is fine to cast
        // from *mut MaybeUninit<u8> to *mut u8 here.
        unsafe { alloc.dealloc(ptr.cast()) };
    }

    fn size(&self) -> usize {
        self.storage_layout.size()
    }
}

impl Drop for RamAllocator {
    fn drop(&mut self) {
        // Safety: buffer was allocated with exactly this layout, see new()
        unsafe { std::alloc::dealloc(self.buffer, self.storage_layout) }
    }
}
