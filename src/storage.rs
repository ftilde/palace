use std::{
    alloc::Layout,
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    mem::MaybeUninit,
    pin::Pin,
};

use crate::{operator::DataId, util::num_elms_in_array, Error};

#[derive(Debug, Eq, PartialEq)]
enum StorageEntryState {
    Initializing,
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
    data: *mut u8,
    size: usize,
}

impl StorageEntry {
    fn safe_to_delete(&self) -> bool {
        self.num_readers == 0 && matches!(self.state, StorageEntryState::Initialized)
    }
}

pub struct ReadHandle<'a, T: ?Sized> {
    storage: &'a Storage,
    id: DataId,
    data: &'a T,
}
impl<'a, T: ?Sized> ReadHandle<'a, T> {
    fn new(storage: &'a Storage, id: DataId, data: &'a T) -> Self {
        storage.index.borrow_mut().get_mut(&id).unwrap().num_readers += 1;

        Self { storage, id, data }
    }

    pub fn map<O>(self, f: impl FnOnce(&'a T) -> &'a O) -> ReadHandle<'a, O> {
        let ret = ReadHandle {
            storage: self.storage,
            id: self.id,
            data: f(&self.data),
        };
        // Avoid running destructor which would signify that we are done reading.
        std::mem::forget(self);
        ret
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

pub struct DropError<'a> {
    storage: &'a Storage,
    id: DataId,
}
impl Drop for DropError<'_> {
    fn drop(&mut self) {
        panic!("The WriteHandle was not marked initialized!");
    }
}
pub struct DropMarkInitialized<'a> {
    storage: &'a Storage,
    id: DataId,
}
impl Drop for DropMarkInitialized<'_> {
    fn drop(&mut self) {
        self.storage.mark_initialized(self.id);
    }
}

pub struct WriteHandle<'a, T: ?Sized, DropHandler> {
    data: &'a mut T,
    drop_handler: DropHandler,
}

impl<'a, T: ?Sized, DropHandler> WriteHandle<'a, T, DropHandler> {
    pub fn map<O>(self, f: impl FnOnce(&'a mut T) -> &'a mut O) -> WriteHandle<'a, O, DropHandler> {
        let WriteHandle { data, drop_handler } = self;
        WriteHandle {
            drop_handler,
            data: f(data),
        }
    }
}
impl<T: ?Sized, D> std::ops::Deref for WriteHandle<'_, T, D> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}
impl<T: ?Sized, D> std::ops::DerefMut for WriteHandle<'_, T, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}

pub type WriteHandleUninit<'a, T> = WriteHandle<'a, T, DropError<'a>>;
impl<'a, T: ?Sized> WriteHandleUninit<'a, T> {
    pub fn new(storage: &'a Storage, id: DataId, data: &'a mut T) -> Self {
        WriteHandle {
            drop_handler: DropError { id, storage },
            data,
        }
    }
    /// Safety: The corresponding slot has to have been completely written to.
    pub unsafe fn initialized(self) -> WriteHandleInit<'a, T> {
        let WriteHandle { data, drop_handler } = self;
        let ret = WriteHandle {
            drop_handler: DropMarkInitialized {
                id: drop_handler.id,
                storage: drop_handler.storage,
            },
            data,
        };

        // Avoid running destructor which would panic
        std::mem::forget(drop_handler);

        ret
    }
}
pub type WriteHandleInit<'a, T> = WriteHandle<'a, T, DropMarkInitialized<'a>>;
impl<'a, T: ?Sized> WriteHandleInit<'a, T> {
    pub fn new(storage: &'a Storage, id: DataId, data: &'a mut T) -> Self {
        WriteHandle {
            drop_handler: DropMarkInitialized { id, storage },
            data,
        }
    }
}

pub struct Storage {
    index: RefCell<BTreeMap<DataId, StorageEntry>>,
    new_data: RefCell<BTreeSet<DataId>>,
    allocator: Allocator,
}

pub type InplaceResult<'a, T> = Result<
    WriteHandleInit<'a, [T]>,
    (
        ReadHandle<'a, [T]>,
        Result<WriteHandleUninit<'a, [MaybeUninit<T>]>, Error>,
    ),
>;

impl Storage {
    pub fn new(size: usize) -> Self {
        let allocator = Allocator::new(size);
        Self {
            index: RefCell::new(BTreeMap::new()),
            new_data: RefCell::new(BTreeSet::new()),
            allocator,
        }
    }

    pub fn try_free(&self, key: DataId) -> Result<(), ()> {
        let mut index = self.index.borrow_mut();
        if index.get(&key).unwrap().safe_to_delete() {
            let entry = index.remove(&key).unwrap();
            // Safety: all data ptrs in the index have been allocated with the allocator.
            // Deallocation only happens exactly here where the entry is also removed from the
            // index.
            unsafe { self.allocator.dealloc(entry.data) };
            self.new_data.borrow_mut().remove(&key);
            Ok(())
        } else {
            Err(())
        }
    }
    pub fn mark_initialized(&self, key: DataId) {
        self.new_data.borrow_mut().insert(key);
        let mut binding = self.index.borrow_mut();
        let state = &mut binding.get_mut(&key).unwrap().state;
        assert_ne!(*state, StorageEntryState::Initialized);
        *state = StorageEntryState::Initialized;
    }
    pub(crate) fn newest_data(&self) -> impl Iterator<Item = DataId> {
        let mut place_holder = BTreeSet::new();
        let mut d = self.new_data.borrow_mut();
        std::mem::swap(&mut *d, &mut place_holder);
        place_holder.into_iter()
    }

    fn alloc(&self, key: DataId, layout: Layout) -> Result<*mut u8, Error> {
        let data = self.allocator.alloc(layout)?;

        let entry = StorageEntry {
            state: StorageEntryState::Initializing,
            num_readers: 0,
            data,
            size: layout.size(),
        };

        let prev = self.index.borrow_mut().insert(key, entry);
        assert!(prev.is_none());

        Ok(data)
    }

    pub fn alloc_ram_slot<T: Copy>(
        &self,
        key: DataId,
        size: usize,
    ) -> Result<WriteHandleUninit<[MaybeUninit<T>]>, Error> {
        let layout = Layout::array::<T>(size).unwrap();
        let ptr = self.alloc(key, layout)?;

        let t_ptr = ptr.cast::<MaybeUninit<T>>();

        // Safety: We constructed the pointer with the required layout
        let t_ref = unsafe { std::slice::from_raw_parts_mut(t_ptr, size) };
        Ok(WriteHandleUninit::new(self, key, t_ref))
    }

    /// Safety: The initial allocation for the TaskId must have happened with the same type
    pub unsafe fn read_ram<'a, T: Copy>(&'a self, key: DataId) -> Option<ReadHandle<'a, [T]>> {
        let t_ref = {
            let index = self.index.borrow();
            let entry = index.get(&key)?;
            assert!(entry.state.initialized());

            let ptr = entry.data;
            let t_ptr = ptr.cast::<T>();

            let num_elements = num_elms_in_array::<T>(entry.size);

            // Safety: Type matches as per contract upheld by caller. There are also no mutable
            // references to the slot since it has already been initialized.
            unsafe { std::slice::from_raw_parts(t_ptr, num_elements) }
        };
        Some(ReadHandle::new(self, key, t_ref))
    }

    /// Safety: The initial allocation for the TaskId must have happened with the same type and the
    /// size must match the initial allocation
    pub unsafe fn try_update_inplace<'a, T: Copy>(
        &'a self,
        old_key: DataId,
        new_key: DataId,
    ) -> Option<InplaceResult<'a, T>> {
        let mut index = self.index.borrow_mut();
        let entry = index.get(&old_key)?;
        assert!(entry.state.initialized());

        let num_elements = num_elms_in_array::<T>(entry.size);

        let ptr = entry.data;
        let t_ptr = ptr.cast::<T>();

        let in_place_possible = entry.safe_to_delete();
        Some(if in_place_possible {
            let mut entry = index.remove(&old_key).unwrap();
            entry.state = StorageEntryState::Initializing;

            let prev = index.insert(new_key, entry);
            assert!(prev.is_none());

            // Safety: Type matches as per contract upheld by caller. There are also no other
            // references to the slot since it has (1.) been already initialized and (2.) there are
            // no readers. In other words: safe_to_delete also implies no other references.
            let t_ref = unsafe { std::slice::from_raw_parts_mut(t_ptr, num_elements) };

            Ok(WriteHandleInit::new(self, new_key, t_ref))
        } else {
            std::mem::drop(index); // Release borrow for alloc

            // Safety: Type matches as per contract upheld by caller. There are also no mutable
            // references to the slot since it has already been initialized.
            let t_ref = unsafe { std::slice::from_raw_parts(t_ptr, num_elements) };

            let w = self.alloc_ram_slot(new_key, num_elements);
            let r = ReadHandle::new(self, old_key, t_ref);
            Err((r, w))
        })
    }
}

struct Allocator {
    alloc: RefCell<Pin<Box<good_memory_allocator::Allocator>>>,
    buffer: *mut u8,
    storage_layout: Layout,
}

impl Allocator {
    pub fn new(size: usize) -> Self {
        let alignment = 4096;
        let storage_layout = Layout::from_size_align(size, alignment).unwrap();

        assert!(size > 0, "invalid storage size");

        // Safety: size is > 0
        let buffer = unsafe { std::alloc::alloc(storage_layout) };

        let mut alloc = Box::pin(good_memory_allocator::Allocator::empty());

        // Safety: The allocator is pinned and will thus not move. The memory region is only used
        // by the allocator.
        unsafe { alloc.init(buffer as usize, size) };

        let alloc = RefCell::new(alloc);

        Self {
            buffer,
            alloc,
            storage_layout,
        }
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
