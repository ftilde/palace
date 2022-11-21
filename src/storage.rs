use std::{alloc::Layout, cell::RefCell, collections::BTreeMap, mem::MaybeUninit};

use crate::{task::TaskId, Error};

struct StorageEntry {
    offset: usize,
}

pub struct Storage {
    index: RefCell<BTreeMap<TaskId, StorageEntry>>,
    buffer: BumpAllocator,
}

impl Storage {
    pub fn new(size: usize) -> Self {
        let buffer = BumpAllocator::new(size);
        Self {
            index: RefCell::new(BTreeMap::new()),
            buffer,
        }
    }

    fn alloc<T>(&self, key: TaskId) -> Result<&mut MaybeUninit<T>, Error> {
        let layout = Layout::new::<T>();
        let offset = self.buffer.alloc(layout)?;

        let entry = StorageEntry { offset };

        // Safety: We have just obtained this offset from alloc
        let ptr = unsafe { self.buffer.buffer.offset(offset as _) };
        let t_ptr = ptr.cast::<MaybeUninit<T>>();

        let prev = self.index.borrow_mut().insert(key, entry);
        assert!(prev.is_none());

        // Safety: We constructed the pointer with the required layout
        let t_ref = unsafe { &mut *t_ptr as _ };
        Ok(t_ref)
    }

    fn alloc_slice<T>(&self, key: TaskId, size: usize) -> Result<&mut [MaybeUninit<T>], Error> {
        let layout = Layout::array::<T>(size).unwrap();
        let offset = self.buffer.alloc(layout)?;

        let entry = StorageEntry { offset };

        // Safety: We have just obtained this offset from alloc
        let ptr = unsafe { self.buffer.buffer.offset(offset as _) };
        let t_ptr = ptr.cast::<MaybeUninit<T>>();

        let prev = self.index.borrow_mut().insert(key, entry);
        assert!(prev.is_none());

        // Safety: We constructed the pointer with the required layout
        let t_ref = unsafe { std::slice::from_raw_parts_mut(t_ptr, size) };
        Ok(t_ref)
    }

    // Safety: the MaybeUninit needs to be written to in f (i.e., made valid).
    pub unsafe fn with_ram_slot<T, F: FnOnce(&mut MaybeUninit<T>) -> Result<(), Error>>(
        &self,
        key: TaskId,
        f: F,
    ) -> Result<(), Error> {
        let mut slot = self.alloc(key)?;
        f(&mut slot)?;

        Ok(())
    }

    // Safety: the MaybeUninit needs to be written to in f (i.e., made valid).
    pub unsafe fn with_ram_slot_slice<T, F: FnOnce(&mut [MaybeUninit<T>]) -> Result<(), Error>>(
        &self,
        key: TaskId,
        size: usize,
        f: F,
    ) -> Result<(), Error> {
        let mut slot = self.alloc_slice(key, size)?;
        f(&mut slot)?;

        Ok(())
    }

    // Safety: The initial allocation for the TaskId must have happened with the same type.
    pub unsafe fn read_ram<T>(&self, key: TaskId) -> Option<&T> {
        let index = self.index.borrow();
        let entry = index.get(&key)?;

        let ptr = unsafe { self.buffer.buffer.offset(entry.offset as _) };
        let t_ptr = ptr.cast::<T>();

        // Safety: Must be upheld by caller
        let t_ref = unsafe { &mut *t_ptr as _ };
        Some(t_ref)
    }

    // Safety: The initial allocation for the TaskId must have happened with the same type and the
    // size must match the initial allocation
    pub unsafe fn read_ram_slice<T>(&self, key: TaskId, size: usize) -> Option<&[T]> {
        let index = self.index.borrow();
        let entry = index.get(&key)?;

        let ptr = unsafe { self.buffer.buffer.offset(entry.offset as _) };
        let t_ptr = ptr.cast::<T>();

        // Safety: Must be upheld by caller
        let t_ref = unsafe { std::slice::from_raw_parts(t_ptr, size) };
        Some(t_ref)
    }
}

struct BumpAllocator {
    buffer: *mut u8,
    next_alloc_offset: std::cell::Cell<usize>,
    size: usize,
}

impl BumpAllocator {
    pub fn new(size: usize) -> Self {
        let alignment = 4096;
        let storage_layout = Layout::from_size_align(size, alignment).unwrap();

        assert!(size > 0, "invalid storage size");

        // Safety: size is > 0
        let buffer = unsafe { std::alloc::alloc(storage_layout) };
        Self {
            buffer,
            next_alloc_offset: std::cell::Cell::new(0),
            size,
        }
    }

    pub fn alloc(&self, layout: Layout) -> Result<usize, Error> {
        let next_alloc_offset = self.next_alloc_offset.get();
        if next_alloc_offset + layout.align() + layout.size() > self.size {
            return Err("Out of memory".into());
        }

        // Safety: Offset does not exceed allocation size
        let next_ptr = unsafe { self.buffer.offset(next_alloc_offset.try_into().unwrap()) };
        let offset = next_ptr.align_offset(layout.align());
        assert_ne!(offset, usize::MAX, "unable to align pointer");

        let new_alloc_start = next_alloc_offset + offset;
        let new_alloc_end = new_alloc_start + layout.size();

        // Safety: Offset does not exceed allocation size
        self.next_alloc_offset.set(new_alloc_end);

        return Ok(new_alloc_start);
    }
}
