use std::{alloc::Layout, cell::RefCell, collections::BTreeMap, mem::MaybeUninit};

use bytemuck::AnyBitPattern;

use crate::{task::TaskId, util::num_elms_in_array, Error};

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
    offset: usize,
    size: usize,
}

impl StorageEntry {
    fn safe_to_delete(&self) -> bool {
        self.num_readers == 0 && matches!(self.state, StorageEntryState::Initialized)
    }
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
    id: TaskId,
}
impl Drop for DropError<'_> {
    fn drop(&mut self) {
        panic!("The WriteHandle was not marked initialized!");
    }
}
pub struct DropMarkInitialized<'a> {
    storage: &'a Storage,
    id: TaskId,
}
impl Drop for DropMarkInitialized<'_> {
    fn drop(&mut self) {
        self.storage
            .index
            .borrow_mut()
            .get_mut(&self.id)
            .unwrap()
            .state = StorageEntryState::Initialized;
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
    pub fn new(storage: &'a Storage, id: TaskId, data: &'a mut T) -> Self {
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
    pub fn new(storage: &'a Storage, id: TaskId, data: &'a mut T) -> Self {
        WriteHandle {
            drop_handler: DropMarkInitialized { id, storage },
            data,
        }
    }
}

pub struct Storage {
    index: RefCell<BTreeMap<TaskId, StorageEntry>>,
    buffer: BumpAllocator,
}

pub type InplaceResultSlice<'a, T> = Result<
    WriteHandleInit<'a, [T]>,
    (
        ReadHandle<'a, [T]>,
        Result<WriteHandleUninit<'a, [MaybeUninit<T>]>, Error>,
    ),
>;

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
        if index.get(&key).unwrap().safe_to_delete() {
            index.remove(&key);
            Ok(())
        } else {
            Err(())
        }
    }

    fn alloc(&self, key: TaskId, layout: Layout) -> Result<*mut u8, Error> {
        let offset = self.buffer.alloc(layout)?;

        let entry = StorageEntry {
            state: StorageEntryState::Initializing,
            num_readers: 0,
            offset,
            size: layout.size(),
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
    ) -> Result<WriteHandleUninit<MaybeUninit<T>>, Error> {
        self.alloc_ram_slot_slice(key, 1)
            .map(|v| v.map(|a| &mut a[0]))
    }

    pub fn alloc_ram_slot_slice<T: AnyBitPattern>(
        &self,
        key: TaskId,
        size: usize,
    ) -> Result<WriteHandleUninit<[MaybeUninit<T>]>, Error> {
        let layout = Layout::array::<T>(size).unwrap();
        let ptr = self.alloc(key, layout)?;

        let t_ptr = ptr.cast::<MaybeUninit<T>>();

        // Safety: We constructed the pointer with the required layout
        let t_ref = unsafe { std::slice::from_raw_parts_mut(t_ptr, size) };
        Ok(WriteHandleUninit::new(self, key, t_ref))
    }

    pub fn write_to_ram<T: AnyBitPattern>(&self, id: TaskId, value: T) -> Result<(), Error> {
        let mut slot = self.alloc_ram_slot(id)?;
        slot.write(value);
        unsafe { slot.initialized() };

        Ok(())
    }

    /// Safety: The initial allocation for the TaskId must have happened with the same type.
    pub unsafe fn read_ram<'a, T: AnyBitPattern>(
        &'a self,
        key: TaskId,
    ) -> Option<ReadHandle<'a, T>> {
        self.read_ram_slice(key).map(|v| v.map(|a| &a[0]))
    }

    /// Safety: The initial allocation for the TaskId must have happened with the same type
    pub unsafe fn read_ram_slice<'a, T: AnyBitPattern>(
        &'a self,
        key: TaskId,
    ) -> Option<ReadHandle<'a, [T]>> {
        let t_ref = {
            let index = self.index.borrow();
            let entry = index.get(&key)?;
            assert!(entry.state.initialized());

            let ptr = unsafe { self.buffer.buffer.offset(entry.offset as _) };
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
    pub unsafe fn try_update_inplace_slice<'a, T: AnyBitPattern>(
        &'a self,
        old_key: TaskId,
        new_key: TaskId,
    ) -> Option<InplaceResultSlice<'a, T>> {
        let mut index = self.index.borrow_mut();
        let entry = index.get(&old_key)?;
        assert!(entry.state.initialized());

        let num_elements = num_elms_in_array::<T>(entry.size);

        let ptr = unsafe { self.buffer.buffer.offset(entry.offset as _) };
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

            let w = self.alloc_ram_slot_slice(new_key, num_elements);
            let r = ReadHandle::new(self, old_key, t_ref);
            Err((r, w))
        })
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
