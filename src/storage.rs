use std::{alloc::Layout, cell::RefCell, collections::BTreeMap, mem::MaybeUninit};

use bytemuck::AnyBitPattern;

use crate::{task::TaskId, Error};

enum StorageEntryState {
    Uninitialized,
    Initialized,
}

impl StorageEntryState {
    fn initialized(&self) -> bool {
        matches!(self, StorageEntryState::Initialized)
    }
}

struct StorageEntry {
    state: StorageEntryState,
    num_readers: usize,
    offset: usize,
}

pub struct ReadHandle<'a, T: ?Sized> {
    storage: &'a Storage,
    id: TaskId,
    data: &'a T,
}
impl<'a, T: ?Sized> ReadHandle<'a, T> {
    fn new(storage: &'a Storage, id: TaskId, data: &'a T) -> Self {
        storage.index.borrow_mut().get_mut(&id).unwrap().num_readers += 1;

        Self { storage, id, data }
    }
}
impl<T: ?Sized> std::ops::Deref for ReadHandle<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}
impl<T: ?Sized> Drop for ReadHandle<'_, T> {
    fn drop(&mut self) {
        self.storage
            .index
            .borrow_mut()
            .get_mut(&self.id)
            .unwrap()
            .num_readers -= 1;
    }
}

pub struct WriteHandle<'a, T: ?Sized> {
    storage: &'a Storage,
    id: TaskId,
    data: &'a mut T,
}
impl<'a, T: ?Sized> WriteHandle<'a, T> {
    fn new(storage: &'a Storage, id: TaskId, data: &'a mut T) -> Self {
        Self { storage, id, data }
    }

    /// Safety: The corresponding slot has to have been completely written to.
    pub unsafe fn mark_initialized(self) {
        self.storage
            .index
            .borrow_mut()
            .get_mut(&self.id)
            .unwrap()
            .state = StorageEntryState::Initialized;

        // Avoid running the destructor of RamSlotToken (which always panics)
        std::mem::forget(self);
    }
}
impl<T: ?Sized> std::ops::Deref for WriteHandle<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}
impl<T: ?Sized> std::ops::DerefMut for WriteHandle<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}
impl<T: ?Sized> Drop for WriteHandle<'_, T> {
    fn drop(&mut self) {
        panic!("The WriteHandle MUST be consumed by calling mark_initialized!");
    }
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

    pub fn try_free(&self, key: TaskId) -> Result<(), ()> {
        let mut index = self.index.borrow_mut();
        if index.get(&key).unwrap().num_readers == 0 {
            index.remove(&key);
            Ok(())
        } else {
            Err(())
        }
    }

    fn alloc(&self, key: TaskId, layout: Layout) -> Result<*mut u8, Error> {
        let offset = self.buffer.alloc(layout)?;

        let entry = StorageEntry {
            state: StorageEntryState::Uninitialized,
            num_readers: 0,
            offset,
        };

        // Safety: We have just obtained this offset from alloc
        let ptr = unsafe { self.buffer.buffer.offset(offset as _) };

        let prev = self.index.borrow_mut().insert(key, entry);
        assert!(prev.is_none());

        Ok(ptr)
    }

    pub fn alloc_ram_slot<T: AnyBitPattern>(
        &self,
        key: TaskId,
    ) -> Result<WriteHandle<MaybeUninit<T>>, Error> {
        let layout = Layout::new::<T>();
        let ptr = self.alloc(key, layout)?;

        let t_ptr = ptr.cast::<MaybeUninit<T>>();

        // Safety: We constructed the pointer with the required layout
        let t_ref = unsafe { &mut *t_ptr as _ };
        Ok(WriteHandle::new(self, key, t_ref))
    }

    pub fn alloc_ram_slot_slice<T: AnyBitPattern>(
        &self,
        key: TaskId,
        size: usize,
    ) -> Result<WriteHandle<[MaybeUninit<T>]>, Error> {
        let layout = Layout::array::<T>(size).unwrap();
        let ptr = self.alloc(key, layout)?;

        let t_ptr = ptr.cast::<MaybeUninit<T>>();

        // Safety: We constructed the pointer with the required layout
        let t_ref = unsafe { std::slice::from_raw_parts_mut(t_ptr, size) };
        Ok(WriteHandle::new(self, key, t_ref))
    }

    pub fn write_to_ram<T: AnyBitPattern>(&self, id: TaskId, value: T) -> Result<(), Error> {
        unsafe {
            self.with_ram_slot(id, |v| {
                v.write(value);
                Ok(())
            })
        }
    }

    /// Safety: the MaybeUninit needs to be written to in f (i.e., made valid).
    pub unsafe fn with_ram_slot<
        T: AnyBitPattern,
        F: FnOnce(&mut MaybeUninit<T>) -> Result<(), Error>,
    >(
        &self,
        key: TaskId,
        f: F,
    ) -> Result<(), Error> {
        let mut slot = self.alloc_ram_slot(key)?;
        f(&mut slot)?;
        slot.mark_initialized();

        Ok(())
    }

    /// Safety: the MaybeUninit needs to be written to in f (i.e., made valid).
    pub unsafe fn with_ram_slot_slice<
        T: AnyBitPattern,
        F: FnOnce(&mut [MaybeUninit<T>]) -> Result<(), Error>,
    >(
        &self,
        key: TaskId,
        size: usize,
        f: F,
    ) -> Result<(), Error> {
        let mut slot = self.alloc_ram_slot_slice(key, size)?;
        f(&mut slot)?;
        slot.mark_initialized();

        Ok(())
    }

    // Safety: The initial allocation for the TaskId must have happened with the same type.
    pub unsafe fn read_ram<'a, T>(&'a self, key: TaskId) -> Option<ReadHandle<'a, T>> {
        let t_ref = {
            let index = self.index.borrow();
            let entry = index.get(&key)?;
            assert!(entry.state.initialized());

            let ptr = unsafe { self.buffer.buffer.offset(entry.offset as _) };
            let t_ptr = ptr.cast::<T>();

            // Safety: Must be upheld by caller
            unsafe { &mut *t_ptr as _ }
        };
        Some(ReadHandle::new(self, key, t_ref))
    }

    // Safety: The initial allocation for the TaskId must have happened with the same type and the
    // size must match the initial allocation
    pub unsafe fn read_ram_slice<'a, T>(
        &'a self,
        key: TaskId,
        size: usize,
    ) -> Option<ReadHandle<'a, [T]>> {
        let t_ref = {
            let index = self.index.borrow();
            let entry = index.get(&key)?;

            let ptr = unsafe { self.buffer.buffer.offset(entry.offset as _) };
            let t_ptr = ptr.cast::<T>();

            // Safety: Must be upheld by caller
            unsafe { std::slice::from_raw_parts(t_ptr, size) }
        };
        Some(ReadHandle::new(self, key, t_ref))
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
