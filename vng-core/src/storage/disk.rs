use std::{alloc::Layout, cell::RefCell, mem::MaybeUninit, pin::Pin};

use crate::Error;

use super::cpu::{CpuAllocator, OOMError};

pub type Storage = super::cpu::Storage<MmapAllocator>;
pub type ReadHandle<'a, T> = super::cpu::ReadHandle<'a, T, MmapAllocator>;
pub type WriteHandleInit<'a, T> = super::cpu::WriteHandleInit<'a, T, MmapAllocator>;
pub type WriteHandleUninit<'a, T> = super::cpu::WriteHandleUninit<'a, T, MmapAllocator>;
pub type RawWriteHandleInit<'a> = super::cpu::RawWriteHandleInit<'a, MmapAllocator>;
pub type RawWriteHandleUninit<'a> = super::cpu::RawWriteHandleUninit<'a, MmapAllocator>;
pub type AccessToken<'a> = super::cpu::AccessToken<'a, MmapAllocator>;
pub type InplaceResult<'a, 'inv, T> = super::cpu::InplaceResult<'a, 'inv, T, MmapAllocator>;
pub type InplaceHandle<'a, T> = super::cpu::InplaceHandle<'a, T, MmapAllocator>;

pub struct MmapAllocator {
    alloc: RefCell<Pin<Box<good_memory_allocator::Allocator>>>,
    _file: std::fs::File,
    memmap: memmap::MmapMut,
}

impl MmapAllocator {
    pub fn new(path: &std::path::Path, size: usize) -> Result<Self, Error> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        file.set_len(size as _).unwrap();

        let mut memmap = unsafe { memmap::MmapMut::map_mut(&file) }.unwrap();

        assert!(size > 0, "invalid storage size");

        // Safety: size is > 0
        let buffer = memmap.as_mut_ptr();
        if buffer.is_null() {
            return Err("Failed to allocate memory buffer. Is it too large?".into());
        }

        let mut alloc = Box::pin(good_memory_allocator::Allocator::empty());

        // Safety: The allocator is pinned and will thus not move. The memory region is only used
        // by the allocator.
        unsafe { alloc.init(buffer as usize, size) };

        let alloc = RefCell::new(alloc);

        Ok(Self {
            alloc,
            _file: file,
            memmap,
        })
    }
}

impl CpuAllocator for MmapAllocator {
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
        self.memmap.len()
    }
}

//impl Drop for MmapAllocator {
//    fn drop(&mut self) {
//        // Safety: buffer was allocated with exactly this layout, see new()
//        unsafe { std::alloc::dealloc(self.buffer, self.storage_layout) }
//    }
//}
