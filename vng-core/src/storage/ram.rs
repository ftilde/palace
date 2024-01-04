use std::{alloc::Layout, cell::RefCell, mem::MaybeUninit, pin::Pin};

use crate::{
    operator::{DataDescriptor, DataId},
    runtime::FrameNumber,
    task::OpaqueTaskContext,
    task_graph::TaskId,
    util::{num_elms_in_array, Map, Set},
    Error,
};

use super::{DataLocation, DataLongevity, DataVersion, DataVersionType, Element, LRUIndex};

#[derive(Debug, Eq, PartialEq)]
enum AccessState {
    Some(usize),
    None(Option<LRUIndex>),
}

#[derive(Copy, Clone, Debug)]
pub struct StorageInfo {
    pub data: *mut MaybeUninit<u8>,
    pub layout: Layout,
    pub data_longevity: DataLongevity,
}

#[derive(Copy, Clone, Debug)]
enum StorageEntryState {
    Registered,
    Initializing(StorageInfo),
    Initialized(StorageInfo, DataVersion),
}

struct Entry {
    state: StorageEntryState,
    access: AccessState,
}

impl Entry {
    fn safe_to_delete(&self) -> bool {
        matches!(self.access, AccessState::None(_))
    }

    fn lru_index(&self) -> Option<LRUIndex> {
        if let AccessState::None(id) = self.access {
            id
        } else {
            None
        }
    }
}

pub struct AccessToken<'a> {
    storage: &'a Storage,
    pub id: DataId,
}
impl<'a> AccessToken<'a> {
    fn new(storage: &'a Storage, id: DataId) -> Self {
        let mut index = storage.index.borrow_mut();
        let ram_entry = index.get_mut(&id).unwrap();

        ram_entry.access = match ram_entry.access {
            AccessState::Some(n) => AccessState::Some(n + 1),
            AccessState::None(id) => {
                if let Some(id) = id {
                    storage.lru_manager.borrow_mut().remove(id);
                }
                AccessState::Some(1)
            }
        };

        Self { storage, id }
    }
}
impl Drop for AccessToken<'_> {
    fn drop(&mut self) {
        let mut index = self.storage.index.borrow_mut();
        let ram_entry = index.get_mut(&self.id).unwrap();
        ram_entry.access = match ram_entry.access {
            AccessState::Some(1) => {
                let lru_id = if let StorageEntryState::Initialized(si, _)
                | StorageEntryState::Initializing(si) = &ram_entry.state
                {
                    let lru_id = self
                        .storage
                        .lru_manager
                        .borrow_mut()
                        .add(self.id, si.data_longevity);
                    Some(lru_id)
                } else {
                    None
                };
                AccessState::None(lru_id)
            }
            AccessState::Some(n) => AccessState::Some(n - 1),
            AccessState::None(_id) => {
                panic!("Invalid state");
            }
        };
    }
}

pub struct ReadHandle<'a, T: ?Sized> {
    access: AccessToken<'a>,
    data_longevity: DataLongevity,
    data: &'a T,
}
impl<'a, T: ?Sized> ReadHandle<'a, T> {
    pub fn map<O>(self, f: impl FnOnce(&'a T) -> &'a O) -> ReadHandle<'a, O> {
        let ret = ReadHandle {
            access: self.access,
            data_longevity: self.data_longevity,
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
            data_longevity: self.data_longevity,
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
    pub info: StorageInfo,
    #[allow(unused)]
    access: AccessToken<'a>,
}

pub struct ThreadReadHandle<'a, T: ?Sized + Send> {
    id: DataId,
    data: &'a T,
    panic_handle: ThreadHandleDropPanic,
    data_longevity: DataLongevity,
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
            access: AccessToken {
                storage,
                id: self.id,
            },
            data: self.data,
            data_longevity: self.data_longevity,
        }
    }
}

pub struct DropError<'a> {
    access: AccessToken<'a>,
}
impl<'a> DropError<'a> {
    fn into_mark_initialized<'inv>(
        self,
        ctx: OpaqueTaskContext<'a, 'inv>,
        version: Option<DataVersionType>,
    ) -> DropMarkInitialized<'a> {
        let id = self.access.id;
        let storage = self.access.storage;
        // Avoid running destructor which would panic
        std::mem::forget(self);
        DropMarkInitialized {
            access: AccessToken { storage, id },
            predicted_preview_tasks: ctx.predicted_preview_tasks,
            current_frame: ctx.current_frame,
            current_task: ctx.current_task,
            version,
        }
    }
}
impl Drop for DropError<'_> {
    fn drop(&mut self) {
        panic!("The WriteHandle was not marked initialized!");
    }
}
pub struct DropMarkInitialized<'a> {
    access: AccessToken<'a>,
    version: Option<DataVersionType>,
    predicted_preview_tasks: &'a RefCell<Set<TaskId>>,
    current_frame: FrameNumber,
    current_task: TaskId,
}
impl Drop for DropMarkInitialized<'_> {
    fn drop(&mut self) {
        {
            let version_type = self.version.unwrap_or_else(|| {
                if self
                    .predicted_preview_tasks
                    .borrow()
                    .contains(&self.current_task)
                {
                    DataVersionType::Preview
                } else {
                    DataVersionType::Final
                }
            });
            let version = match version_type {
                DataVersionType::Final => DataVersion::Final,
                DataVersionType::Preview => DataVersion::Preview(self.current_frame),
            };

            let mut binding = self.access.storage.index.borrow_mut();
            let state = &mut binding.get_mut(&self.access.id).unwrap().state;

            self.access
                .storage
                .new_data
                .add(self.access.id, version_type);

            *state = match state {
                StorageEntryState::Registered => {
                    panic!("Entry should be in state Initializing, but is in Registered");
                }
                StorageEntryState::Initialized(..) => {
                    panic!("Entry should be in state Initializing, but is in Initialized");
                }
                StorageEntryState::Initializing(info) => {
                    StorageEntryState::Initialized(*info, version)
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
    pub data: *mut MaybeUninit<u8>,
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
pub struct ThreadMarkerInitialized {
    version: Option<DataVersionType>,
}
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
    pub fn into_main_handle<'inv>(
        self,
        ctx: OpaqueTaskContext<'a, 'inv>,
    ) -> WriteHandleInit<'a, T> {
        self._panic_handle.dismiss();
        WriteHandle {
            drop_handler: DropMarkInitialized {
                access: AccessToken {
                    storage: ctx.storage,
                    id: self.id,
                },
                current_frame: ctx.current_frame,
                current_task: ctx.current_task,
                predicted_preview_tasks: ctx.predicted_preview_tasks,
                version: self._marker.version,
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
                access: AccessToken {
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
    /// Safety: The corresponding slot has to have been completely written to, i.e. all
    /// (non-padding) bytes of T must have been written.
    pub unsafe fn initialized<'inv>(
        self,
        ctx: OpaqueTaskContext<'a, 'inv>,
    ) -> WriteHandleInit<'a, T> {
        WriteHandle {
            drop_handler: self.drop_handler.into_mark_initialized(ctx, None),
            data: self.data,
        }
    }

    /// Safety: The corresponding slot has to have been completely written to, i.e. all
    /// (non-padding) bytes of T must have been written.
    pub unsafe fn initialized_version<'inv>(
        self,
        ctx: OpaqueTaskContext<'a, 'inv>,
        version: DataVersionType,
    ) -> WriteHandleInit<'a, T> {
        WriteHandle {
            drop_handler: self.drop_handler.into_mark_initialized(ctx, Some(version)),
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
    pub unsafe fn initialized<'inv>(
        self,
        ctx: OpaqueTaskContext<'a, 'inv>,
    ) -> RawWriteHandleInit<'a> {
        RawWriteHandle {
            drop_handler: self.drop_handler.into_mark_initialized(ctx, None),
            data: self.data,
            layout: self.layout,
        }
    }

    pub fn transmute<T: Element>(self, size: usize) -> WriteHandleUninit<'a, [MaybeUninit<T>]> {
        let layout = Layout::array::<T>(size).unwrap();
        assert_eq!(layout, self.layout);

        let t_ptr = self.data.cast::<MaybeUninit<T>>();

        // Safety: We constructed the pointer with the required layout
        let t_ref = unsafe { std::slice::from_raw_parts_mut(t_ptr, size) };
        WriteHandleUninit {
            data: t_ref,
            drop_handler: self.drop_handler,
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
        let version = self.drop_handler.version;
        std::mem::forget(self.drop_handler);
        ThreadWriteHandle {
            id,
            data: self.data,
            _marker: ThreadMarkerInitialized { version },
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
    pub fn into_main_handle<'inv>(self, ctx: OpaqueTaskContext<'a, 'inv>) -> InplaceResult<'a, T> {
        match self {
            ThreadInplaceResult::Inplace(rw) => InplaceResult::Inplace(rw.into_main_handle(ctx)),
            ThreadInplaceResult::New(r, w) => InplaceResult::New(
                r.into_main_handle(&ctx.storage),
                w.into_main_handle(&ctx.storage),
            ),
        }
    }
}

pub struct Storage {
    index: RefCell<Map<DataId, Entry>>,
    lru_manager: RefCell<super::LRUManager<DataId>>,
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

        for (longevity, inner_lru) in lru.inner_mut() {
            let mut collected = 0;
            for key in inner_lru.drain_lru().into_iter() {
                let entry = index.get_mut(&key).unwrap();
                let info = match entry.state {
                    StorageEntryState::Registered => panic!("Should not be in LRU list"),
                    StorageEntryState::Initializing(info)
                    | StorageEntryState::Initialized(info, _) => info,
                };
                assert!(matches!(entry.access, AccessState::None(_)));

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
            println!("Garbage collect RAM ({:?}): {}B", longevity, collected);
        }
    }

    pub fn is_readable(&self, id: DataId) -> bool {
        self.index
            .borrow()
            .get(&id)
            .map(|e| matches!(e.state, StorageEntryState::Initialized(..)))
            .unwrap_or(false)
    }

    pub fn try_free(&self, key: DataId) -> Result<(), ()> {
        let mut index = self.index.borrow_mut();
        if index.get(&key).unwrap().safe_to_delete() {
            let entry = index.get_mut(&key).unwrap();

            let info = match entry.state {
                StorageEntryState::Registered => return Err(()),
                StorageEntryState::Initializing(info) | StorageEntryState::Initialized(info, _) => {
                    info
                }
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

    pub(crate) fn newest_data(
        &self,
    ) -> impl Iterator<Item = (DataId, DataLocation, DataVersionType)> {
        self.new_data
            .drain()
            .map(|(d, v)| (d, DataLocation::Ram, v))
    }

    pub fn register_access(&self, id: DataId) -> AccessToken {
        {
            let mut index = self.index.borrow_mut();
            index.entry(id).or_insert_with(|| Entry {
                state: StorageEntryState::Registered,
                access: AccessState::Some(0), // Will be overwritten immediately when generating
                                              // the RamToken
            });
        }
        AccessToken::new(self, id)
    }

    pub(crate) fn alloc(
        &self,
        descriptor: DataDescriptor,
        layout: Layout,
    ) -> (*mut MaybeUninit<u8>, AccessToken) {
        let key = descriptor.id;

        let data = {
            let data = match self.allocator.alloc(layout) {
                Ok(d) => d,
                Err(_e) => {
                    // Always try to free half of available ram
                    // TODO: Other solutions may be better.
                    let garbage_collect_goal =
                        self.allocator.size() / super::GARBAGE_COLLECT_GOAL_FRACTION as usize;
                    self.try_garbage_collect(garbage_collect_goal);
                    self.allocator
                        .alloc(layout)
                        .expect("Out of memory and nothing we can do about it.")
                }
            };

            let mut index = self.index.borrow_mut();

            let entry = index.entry(key).or_insert_with(|| Entry {
                state: StorageEntryState::Registered,
                access: AccessState::Some(0), // Will be overwritten immediately when generating
                                              // the RamToken
            });

            let info = StorageInfo {
                data,
                layout,
                data_longevity: descriptor.longevity,
            };

            entry.state = StorageEntryState::Initializing(info);

            data
        };

        (data, AccessToken::new(self, descriptor.id))
    }

    pub fn access_initializing<'a>(
        &self,
        access: AccessToken<'a>,
    ) -> Result<RawWriteHandleUninit<'a>, AccessToken<'a>> {
        let index = self.index.borrow_mut();
        let entry = index.get(&access.id).unwrap();

        if let StorageEntryState::Initializing(info) = entry.state {
            Ok(RawWriteHandleUninit {
                data: info.data,
                layout: info.layout,
                drop_handler: DropError { access },
            })
        } else {
            Err(access)
        }
    }

    pub fn alloc_slot_raw(&self, key: DataDescriptor, layout: Layout) -> RawWriteHandleUninit {
        let (ptr, access) = self.alloc(key, layout);

        RawWriteHandleUninit {
            data: ptr,
            layout,
            drop_handler: DropError { access },
        }
    }

    pub fn alloc_slot<T: Element>(
        &self,
        key: DataDescriptor,
        size: usize,
    ) -> WriteHandleUninit<[MaybeUninit<T>]> {
        let layout = Layout::array::<T>(size).unwrap();
        let (ptr, access) = self.alloc(key, layout);

        let t_ptr = ptr.cast::<MaybeUninit<T>>();

        // Safety: We constructed the pointer with the required layout
        let t_ref = unsafe { std::slice::from_raw_parts_mut(t_ptr, size) };
        WriteHandleUninit {
            data: t_ref,
            drop_handler: DropError { access },
        }
    }

    /// Safety: The initial allocation for the TaskId must have happened with the same type
    pub fn read_raw<'b, 't: 'b>(
        &'b self,
        access: AccessToken<'t>,
    ) -> Result<RawReadHandle<'b>, AccessToken<'t>> {
        let info = {
            let index = self.index.borrow();
            let Some(entry) = index.get(&access.id) else {
                return Err(access);
            };
            let StorageEntryState::Initialized(info, _) = entry.state else {
                return Err(access);
            };

            info
        };
        Ok(RawReadHandle { access, info })
    }

    /// Safety: The initial allocation for the TaskId must have happened with the same type
    pub unsafe fn read<'b, 't: 'b, T: Element>(
        &'b self,
        access: AccessToken<'t>,
    ) -> Result<ReadHandle<'b, [T]>, AccessToken<'t>> {
        let (t_ref, data_longevity) = {
            let index = self.index.borrow();
            let Some(entry) = index.get(&access.id) else {
                return Err(access);
            };

            let StorageEntryState::Initialized(info, _) = entry.state else {
                return Err(access);
            };

            let ptr = info.data;
            let t_ptr = ptr.cast::<T>();

            let num_elements = num_elms_in_array::<T>(info.layout.size());

            // Safety: Type matches as per contract upheld by caller. There are also no mutable
            // references to the slot since it has already been initialized.
            (
                unsafe { std::slice::from_raw_parts(t_ptr, num_elements) },
                info.data_longevity,
            )
        };
        Ok(ReadHandle {
            access,
            data: t_ref,
            data_longevity,
        })
    }

    /// Safety: The initial allocation for the TaskId must have happened with the same type and the
    /// size must match the initial allocation
    pub unsafe fn try_update_inplace<'b, 't: 'b, 'inv, T: Element>(
        &'b self,
        ctx: OpaqueTaskContext<'t, 'inv>,
        old_access: AccessToken<'t>,
        new_desc: DataDescriptor,
    ) -> Result<InplaceResult<'b, T>, AccessToken<'t>> {
        let old_key = old_access.id;
        let new_key = new_desc.id;

        let mut index = self.index.borrow_mut();
        let Some(entry) = index.get(&old_access.id) else {
            return Err(old_access);
        };

        let StorageEntryState::Initialized(info, _) = entry.state else {
            return Err(old_access);
        };

        let num_elements = num_elms_in_array::<T>(info.layout.size());

        let ptr = info.data;
        let t_ptr = ptr.cast::<T>();

        // Only allow inplace if we are EXACTLY the one reader
        let in_place_possible = matches!(entry.access, AccessState::Some(1));

        Ok(if in_place_possible {
            // Repurpose access key for the read/write handle
            let mut new_access = old_access;
            new_access.id = new_key;

            let _old_entry = index.remove(&old_key).unwrap();
            let new_entry = index.entry(new_key).or_insert_with(|| Entry {
                state: StorageEntryState::Registered,
                access: AccessState::Some(0),
            });

            new_entry.state = StorageEntryState::Initializing(info);

            new_entry.access = match new_entry.access {
                AccessState::Some(n) => AccessState::Some(n + 1),
                AccessState::None(_) => panic!("If present, entry should have accessors"),
            };

            // Safety: Type matches as per contract upheld by caller. There are also no other
            // references to the slot since it has (1.) been already initialized and (2.) there are
            // no readers. In other words: safe_to_delete also implies no other references.
            let t_ref = unsafe { std::slice::from_raw_parts_mut(t_ptr, num_elements) };

            InplaceResult::Inplace(WriteHandleInit {
                data: t_ref,
                drop_handler: DropMarkInitialized {
                    access: new_access,
                    version: None,
                    predicted_preview_tasks: ctx.predicted_preview_tasks,
                    current_frame: ctx.current_frame,
                    current_task: ctx.current_task,
                },
            })
        } else {
            std::mem::drop(index); // Release borrow for alloc

            // Safety: Type matches as per contract upheld by caller. There are also no mutable
            // references to the slot since it has already been initialized.
            let t_ref = unsafe { std::slice::from_raw_parts(t_ptr, num_elements) };

            let w = self.alloc_slot(new_desc, num_elements);

            let r = ReadHandle {
                data: t_ref,
                access: AccessToken::new(self, old_key),
                data_longevity: info.data_longevity,
            };
            InplaceResult::New(r, w)
        })
    }
}

pub struct Allocator {
    alloc: RefCell<Pin<Box<good_memory_allocator::Allocator>>>,
    buffer: *mut u8,
    storage_layout: Layout,
}

#[derive(Debug, Copy, Clone)]
pub struct OOMError {
    //requested: usize,
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

impl Drop for Allocator {
    fn drop(&mut self) {
        // Safety: buffer was allocated with exactly this layout, see new()
        unsafe { std::alloc::dealloc(self.buffer, self.storage_layout) }
    }
}
