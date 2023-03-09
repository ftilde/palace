use std::{alloc::Layout, cell::RefCell, collections::BTreeMap, mem::MaybeUninit, pin::Pin};

use crate::{operator::DataId, task_graph::LocatedDataId, util::num_elms_in_array, Error};

use super::LRUIndex;

#[derive(Debug, Eq, PartialEq)]
enum RamAccessState {
    Some(usize),
    None(LRUIndex),
}

#[derive(Copy, Clone, Debug)]
pub struct RamStorageInfo {
    pub data: *mut u8,
    pub layout: Layout,
}

#[derive(Copy, Clone, Debug)]
enum RamStorageEntryState {
    Registered,
    Initializing(RamStorageInfo),
    Initialized(RamStorageInfo),
}

struct RamEntry {
    state: RamStorageEntryState,
    access: RamAccessState,
}

impl RamEntry {
    fn safe_to_delete(&self) -> bool {
        matches!(self.access, RamAccessState::None(_))
    }

    fn lru_index(&self) -> Option<LRUIndex> {
        if let RamAccessState::None(id) = self.access {
            Some(id)
        } else {
            None
        }
    }
}

pub struct RamAccessToken<'a> {
    storage: &'a Storage,
    pub id: DataId,
}
impl<'a> RamAccessToken<'a> {
    fn new(storage: &'a Storage, id: DataId) -> Self {
        let mut index = storage.index.borrow_mut();
        let ram_entry = index.get_mut(&id).unwrap();

        ram_entry.access = match ram_entry.access {
            RamAccessState::Some(n) => RamAccessState::Some(n + 1),
            RamAccessState::None(id) => {
                storage.lru_manager.borrow_mut().remove(id);
                RamAccessState::Some(1)
            }
        };

        Self { storage, id }
    }
}
impl Drop for RamAccessToken<'_> {
    fn drop(&mut self) {
        let mut index = self.storage.index.borrow_mut();
        let ram_entry = index.get_mut(&self.id).unwrap();

        ram_entry.access = match ram_entry.access {
            RamAccessState::Some(1) => {
                let lru_id = self.storage.lru_manager.borrow_mut().add(self.id);
                RamAccessState::None(lru_id)
            }
            RamAccessState::Some(n) => RamAccessState::Some(n - 1),
            RamAccessState::None(_id) => {
                panic!("Invalid state");
            }
        };
    }
}

pub struct ReadHandle<'a, T: ?Sized> {
    access: RamAccessToken<'a>,
    data: &'a T,
}
impl<'a, T: ?Sized> ReadHandle<'a, T> {
    pub fn map<O>(self, f: impl FnOnce(&'a T) -> &'a O) -> ReadHandle<'a, O> {
        let ret = ReadHandle {
            access: self.access,
            data: f(&self.data),
        };
        ret
    }

    pub fn into_thread_handle(self) -> ThreadReadHandle<'a, T>
    where
        T: Send,
    {
        let ret = ThreadReadHandle {
            id: self.access.id,
            data: self.data,
            panic_handle: Default::default(),
        };
        //Avoid running destructor
        std::mem::forget(self.access);

        ret
    }
}
impl<T: ?Sized> std::ops::Deref for ReadHandle<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}
pub struct RawReadHandle<'a> {
    pub info: RamStorageInfo,
    #[allow(unused)]
    access: RamAccessToken<'a>,
}

pub struct ThreadReadHandle<'a, T: ?Sized + Send> {
    id: DataId,
    data: &'a T,
    panic_handle: ThreadHandleDropPanic,
}
impl<T: ?Sized + Send> std::ops::Deref for ThreadReadHandle<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}
impl<'a, T: ?Sized + Send> ThreadReadHandle<'a, T> {
    pub fn into_main_handle(self, storage: &'a Storage) -> ReadHandle<'a, T> {
        self.panic_handle.dismiss();
        ReadHandle {
            access: RamAccessToken {
                storage,
                id: self.id,
            },
            data: self.data,
        }
    }
}

pub struct DropError<'a> {
    access: RamAccessToken<'a>,
}
impl<'a> DropError<'a> {
    fn into_mark_initialized(self) -> DropMarkInitialized<'a> {
        let id = self.access.id;
        let storage = self.access.storage;
        // Avoid running destructor which would panic
        std::mem::forget(self);
        DropMarkInitialized {
            access: RamAccessToken { storage, id },
        }
    }
}
impl Drop for DropError<'_> {
    fn drop(&mut self) {
        panic!("The WriteHandle was not marked initialized!");
    }
}
pub struct DropMarkInitialized<'a> {
    access: RamAccessToken<'a>,
}
impl Drop for DropMarkInitialized<'_> {
    fn drop(&mut self) {
        self.access.storage.new_data.add(self.access.id);
        {
            let mut binding = self.access.storage.index.borrow_mut();
            let state = &mut binding.get_mut(&self.access.id).unwrap().state;
            *state = match state {
                RamStorageEntryState::Registered => {
                    panic!("Entry should be in state Initializing, but is in Registered");
                }
                RamStorageEntryState::Initialized(_) => {
                    panic!("Entry should be in state Initializing, but is in Initialized");
                }
                RamStorageEntryState::Initializing(info) => {
                    RamStorageEntryState::Initialized(*info)
                }
            };
        }
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

pub struct RawWriteHandle<DropHandler> {
    pub data: *mut u8,
    pub layout: Layout,
    drop_handler: DropHandler,
}

#[derive(Default)]
struct ThreadHandleDropPanic;
impl Drop for ThreadHandleDropPanic {
    fn drop(&mut self) {
        panic!("ThreadHandles must be returned to main thread before being dropped!");
    }
}
impl ThreadHandleDropPanic {
    fn dismiss(self) {
        std::mem::forget(self);
    }
}
pub struct ThreadMarkerInitialized;
pub struct ThreadMarkerUninitialized;
pub struct ThreadWriteHandle<'a, T: ?Sized + Send, D: Send> {
    id: DataId,
    data: &'a mut T,
    _marker: D,
    _panic_handle: ThreadHandleDropPanic,
}
impl<T: ?Sized + Send, D: Send> std::ops::Deref for ThreadWriteHandle<'_, T, D> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}
impl<T: ?Sized + Send, D: Send> std::ops::DerefMut for ThreadWriteHandle<'_, T, D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}
impl<'a, T: ?Sized + Send> ThreadWriteHandle<'a, T, ThreadMarkerInitialized> {
    pub fn into_main_handle(self, storage: &'a Storage) -> WriteHandleInit<'a, T> {
        self._panic_handle.dismiss();
        WriteHandle {
            drop_handler: DropMarkInitialized {
                access: RamAccessToken {
                    storage,
                    id: self.id,
                },
            },
            data: self.data,
        }
    }
}
impl<'a, T: ?Sized + Send> ThreadWriteHandle<'a, T, ThreadMarkerUninitialized> {
    pub fn into_main_handle(self, storage: &'a Storage) -> WriteHandleUninit<'a, T> {
        self._panic_handle.dismiss();
        WriteHandle {
            drop_handler: DropError {
                access: RamAccessToken {
                    storage,
                    id: self.id,
                },
            },
            data: self.data,
        }
    }
}

pub type WriteHandleUninit<'a, T> = WriteHandle<'a, T, DropError<'a>>;
pub type RawWriteHandleUninit<'a> = RawWriteHandle<DropError<'a>>;
pub type ThreadWriteHandleUninit<'a, T> = ThreadWriteHandle<'a, T, ThreadMarkerUninitialized>;

impl<'a, T: ?Sized> WriteHandleUninit<'a, T> {
    /// Safety: The corresponding slot has to have been completely written to.
    pub unsafe fn initialized(self) -> WriteHandleInit<'a, T> {
        WriteHandle {
            drop_handler: self.drop_handler.into_mark_initialized(),
            data: self.data,
        }
    }
    pub fn into_thread_handle(self) -> ThreadWriteHandleUninit<'a, T>
    where
        T: Send,
    {
        let id = self.drop_handler.access.id;
        std::mem::forget(self.drop_handler);
        ThreadWriteHandle {
            id,
            data: self.data,
            _marker: ThreadMarkerUninitialized,
            _panic_handle: Default::default(),
        }
    }
}
impl<'a> RawWriteHandleUninit<'a> {
    /// Safety: The corresponding slot has to have been completely written to.
    pub unsafe fn initialized(self) -> RawWriteHandleInit<'a> {
        RawWriteHandle {
            drop_handler: self.drop_handler.into_mark_initialized(),
            data: self.data,
            layout: self.layout,
        }
    }
}

pub type WriteHandleInit<'a, T> = WriteHandle<'a, T, DropMarkInitialized<'a>>;
pub type RawWriteHandleInit<'a> = RawWriteHandle<DropMarkInitialized<'a>>;
pub type ThreadWriteHandleInit<'a, T> = ThreadWriteHandle<'a, T, ThreadMarkerInitialized>;
impl<'a, T: ?Sized> WriteHandleInit<'a, T> {
    pub fn into_thread_handle(self) -> ThreadWriteHandleInit<'a, T>
    where
        T: Send,
    {
        let id = self.drop_handler.access.id;
        std::mem::forget(self.drop_handler);
        ThreadWriteHandle {
            id,
            data: self.data,
            _marker: ThreadMarkerInitialized,
            _panic_handle: Default::default(),
        }
    }
}

pub enum InplaceResult<'a, T> {
    Inplace(WriteHandleInit<'a, [T]>),
    New(ReadHandle<'a, [T]>, WriteHandleUninit<'a, [MaybeUninit<T>]>),
}

impl<'a, T: Send> InplaceResult<'a, T> {
    pub fn into_thread_handle(self) -> ThreadInplaceResult<'a, T> {
        match self {
            InplaceResult::Inplace(rw) => ThreadInplaceResult::Inplace(rw.into_thread_handle()),
            InplaceResult::New(r, w) => {
                ThreadInplaceResult::New(r.into_thread_handle(), w.into_thread_handle())
            }
        }
    }
}

pub enum ThreadInplaceResult<'a, T: Send> {
    Inplace(ThreadWriteHandleInit<'a, [T]>),
    New(
        ThreadReadHandle<'a, [T]>,
        ThreadWriteHandleUninit<'a, [MaybeUninit<T>]>,
    ),
}

impl<'a, T: Send> ThreadInplaceResult<'a, T> {
    pub fn into_main_handle(self, storage: &'a Storage) -> InplaceResult<'a, T> {
        match self {
            ThreadInplaceResult::Inplace(rw) => {
                InplaceResult::Inplace(rw.into_main_handle(storage))
            }
            ThreadInplaceResult::New(r, w) => {
                InplaceResult::New(r.into_main_handle(storage), w.into_main_handle(storage))
            }
        }
    }
}

pub struct Storage {
    index: RefCell<BTreeMap<DataId, RamEntry>>,
    lru_manager: RefCell<super::LRUManager>,
    allocator: Allocator,
    new_data: super::NewDataManager,
}

impl Storage {
    /// Safety: The same combination of `state`, `ram` and `vram` must be used to construct a
    /// `Storage` at all times. No modifications outside of `Storage` must be done to either of
    /// these structs in the meantime.
    pub fn new(size: usize) -> Result<Self, Error> {
        let allocator = Allocator::new(size)?;
        Ok(Self {
            index: Default::default(),
            lru_manager: Default::default(),
            new_data: Default::default(),
            allocator,
        })
    }

    pub fn try_garbage_collect(&self, mut goal_in_bytes: usize) {
        let mut lru = self.lru_manager.borrow_mut();
        let mut index = self.index.borrow_mut();
        let mut collected = 0;
        for key in lru.drain_lru().into_iter() {
            let entry = index.get_mut(&key).unwrap();
            let info = match entry.state {
                RamStorageEntryState::Registered => panic!("Should not be in LRU list"),
                RamStorageEntryState::Initializing(info)
                | RamStorageEntryState::Initialized(info) => info,
            };
            assert!(matches!(entry.access, RamAccessState::None(_)));

            index.remove(&key).unwrap();

            // Safety: all data ptrs in the index have been allocated with the allocator.
            // Deallocation only happens exactly here where the entry is also removed from the
            // index.
            unsafe { self.allocator.dealloc(info.data) };
            self.new_data.remove(key);

            let size = info.layout.size();
            collected += size;
            let Some(rest) = goal_in_bytes.checked_sub(size) else {
                break;
            };
            goal_in_bytes = rest;
        }
        println!("Garbage collect: {}B", collected);
    }

    pub fn is_readable(&self, id: DataId) -> bool {
        self.index
            .borrow()
            .get(&id)
            .map(|e| matches!(e.state, RamStorageEntryState::Initialized(_)))
            .unwrap_or(false)
    }

    pub fn try_free_ram(&self, key: DataId) -> Result<(), ()> {
        let mut index = self.index.borrow_mut();
        if index.get(&key).unwrap().safe_to_delete() {
            let entry = index.get_mut(&key).unwrap();

            let info = match entry.state {
                RamStorageEntryState::Registered => return Err(()),
                RamStorageEntryState::Initializing(info)
                | RamStorageEntryState::Initialized(info) => info,
            };

            let lru_index = entry.lru_index().unwrap();
            self.lru_manager.borrow_mut().remove(lru_index);
            index.remove(&key).unwrap();

            // Safety: all data ptrs in the index have been allocated with the allocator.
            // Deallocation only happens exactly here where the entry is also removed from the
            // index.
            unsafe { self.allocator.dealloc(info.data) };
            self.new_data.remove(key);
            Ok(())
        } else {
            Err(())
        }
    }

    pub(crate) fn newest_data(&self) -> impl Iterator<Item = LocatedDataId> {
        self.new_data
            .drain()
            .map(|d| d.in_location(super::DataLocation::Ram))
    }

    pub fn register_ram_access(&self, id: DataId) -> RamAccessToken {
        {
            let mut index = self.index.borrow_mut();
            index.entry(id).or_insert_with(|| RamEntry {
                state: RamStorageEntryState::Registered,
                access: RamAccessState::Some(0), // Will be overwritten immediately when generating
                                                 // the RamToken
            });
        }
        RamAccessToken::new(self, id)
    }

    fn alloc_ram(&self, key: DataId, layout: Layout) -> Result<(*mut u8, RamAccessToken), Error> {
        let data = {
            let data = match self.allocator.alloc(layout) {
                Ok(d) => d,
                Err(_e) => {
                    // Always try to free half of available ram
                    // TODO: Other solutions may be better.
                    let garbage_collect_goal = self.allocator.size() / 2;
                    self.try_garbage_collect(garbage_collect_goal);
                    self.allocator.alloc(layout)?
                }
            };

            let mut index = self.index.borrow_mut();

            let entry = index.entry(key).or_insert_with(|| RamEntry {
                state: RamStorageEntryState::Registered,
                access: RamAccessState::Some(0), // Will be overwritten immediately when generating
                                                 // the RamToken
            });

            let info = RamStorageInfo { data, layout };

            entry.state = RamStorageEntryState::Initializing(info);

            data
        };

        Ok((data, RamAccessToken::new(self, key)))
    }

    pub fn alloc_ram_slot_raw(
        &self,
        key: DataId,
        layout: Layout,
    ) -> Result<RawWriteHandleUninit, Error> {
        let (ptr, access) = self.alloc_ram(key, layout)?;

        Ok(RawWriteHandleUninit {
            data: ptr,
            layout,
            drop_handler: DropError { access },
        })
    }

    pub fn alloc_ram_slot<T: Copy>(
        &self,
        key: DataId,
        size: usize,
    ) -> Result<WriteHandleUninit<[MaybeUninit<T>]>, Error> {
        let layout = Layout::array::<T>(size).unwrap();
        let (ptr, access) = self.alloc_ram(key, layout)?;

        let t_ptr = ptr.cast::<MaybeUninit<T>>();

        // Safety: We constructed the pointer with the required layout
        let t_ref = unsafe { std::slice::from_raw_parts_mut(t_ptr, size) };
        Ok(WriteHandleUninit {
            data: t_ref,
            drop_handler: DropError { access },
        })
    }

    /// Safety: The initial allocation for the TaskId must have happened with the same type
    pub fn read_ram_raw<'b, 't: 'b>(
        &'b self,
        access: RamAccessToken<'t>,
    ) -> Result<RawReadHandle<'b>, RamAccessToken<'t>> {
        let info = {
            let index = self.index.borrow();
            let Some(entry) = index.get(&access.id) else {
                return Err(access);
            };
            let RamStorageEntryState::Initialized(info) = entry.state else {
                return Err(access);
            };

            info
        };
        Ok(RawReadHandle { access, info })
    }

    /// Safety: The initial allocation for the TaskId must have happened with the same type
    pub unsafe fn read_ram<'b, 't: 'b, T: Copy>(
        &'b self,
        access: RamAccessToken<'t>,
    ) -> Result<ReadHandle<'b, [T]>, RamAccessToken<'t>> {
        let t_ref = {
            let index = self.index.borrow();
            let Some(entry) = index.get(&access.id) else {
                return Err(access);
            };

            let RamStorageEntryState::Initialized(info) = entry.state else {
                return Err(access);
            };

            let ptr = info.data;
            let t_ptr = ptr.cast::<T>();

            let num_elements = num_elms_in_array::<T>(info.layout.size());

            // Safety: Type matches as per contract upheld by caller. There are also no mutable
            // references to the slot since it has already been initialized.
            unsafe { std::slice::from_raw_parts(t_ptr, num_elements) }
        };
        Ok(ReadHandle {
            access,
            data: t_ref,
        })
    }

    /// Safety: The initial allocation for the TaskId must have happened with the same type and the
    /// size must match the initial allocation
    pub unsafe fn try_update_inplace<'b, 't: 'b, T: Copy>(
        &'b self,
        old_access: RamAccessToken<'t>,
        new_key: DataId,
    ) -> Result<Result<InplaceResult<'b, T>, Error>, RamAccessToken<'t>> {
        let old_key = old_access.id;

        let mut index = self.index.borrow_mut();
        let Some(entry) = index.get(&old_access.id) else {
                return Err(old_access);
            };

        let RamStorageEntryState::Initialized(info) = entry.state else {
            return Err(old_access)
        };

        let num_elements = num_elms_in_array::<T>(info.layout.size());

        let ptr = info.data;
        let t_ptr = ptr.cast::<T>();

        // Only allow inplace if we are EXACTLY the one reader
        let in_place_possible = matches!(entry.access, RamAccessState::Some(1));

        Ok(Ok(if in_place_possible {
            // Repurpose access key for the read/write handle
            let mut new_access = old_access;
            new_access.id = new_key;

            let _old_entry = index.remove(&old_key).unwrap();
            let new_entry = index.entry(new_key).or_insert_with(|| RamEntry {
                state: RamStorageEntryState::Registered,
                access: RamAccessState::Some(0),
            });

            new_entry.state = RamStorageEntryState::Initializing(info);

            new_entry.access = match new_entry.access {
                RamAccessState::Some(n) => RamAccessState::Some(n + 1),
                RamAccessState::None(_) => panic!("If present, entry should have accessors"),
            };

            // Safety: Type matches as per contract upheld by caller. There are also no other
            // references to the slot since it has (1.) been already initialized and (2.) there are
            // no readers. In other words: safe_to_delete also implies no other references.
            let t_ref = unsafe { std::slice::from_raw_parts_mut(t_ptr, num_elements) };

            InplaceResult::Inplace(WriteHandleInit {
                data: t_ref,
                drop_handler: DropMarkInitialized { access: new_access },
            })
        } else {
            std::mem::drop(index); // Release borrow for alloc

            // Safety: Type matches as per contract upheld by caller. There are also no mutable
            // references to the slot since it has already been initialized.
            let t_ref = unsafe { std::slice::from_raw_parts(t_ptr, num_elements) };

            let w = match self.alloc_ram_slot(new_key, num_elements) {
                Ok(w) => w,
                Err(e) => return Ok(Err(e)),
            };
            let r = ReadHandle {
                data: t_ref,
                access: RamAccessToken::new(self, old_key),
            };
            InplaceResult::New(r, w)
        }))
    }
}

pub struct Allocator {
    alloc: RefCell<Pin<Box<good_memory_allocator::Allocator>>>,
    buffer: *mut u8,
    storage_layout: Layout,
}

impl Allocator {
    fn new(size: usize) -> Result<Self, Error> {
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

    fn alloc(&self, layout: Layout) -> Result<*mut u8, Error> {
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
    unsafe fn dealloc(&self, ptr: *mut u8) {
        let mut alloc = self.alloc.borrow_mut();
        unsafe { alloc.dealloc(ptr) };
    }

    fn size(&self) -> usize {
        self.storage_layout.size()
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        // Safety: buffer was allocated with exactly this layout, see new()
        unsafe { std::alloc::dealloc(self.buffer, self.storage_layout) }
    }
}
