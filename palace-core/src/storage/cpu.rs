use std::{alloc::Layout, cell::RefCell, mem::MaybeUninit};

use crate::{
    operator::{DataDescriptor, DataId},
    runtime::FrameNumber,
    task::{AllocationId, AllocationRequest, OpaqueTaskContext, Request, RequestType},
    task_graph::TaskId,
    util::{num_elms_in_array, IdGenerator, Map, Set},
};

use super::{
    DataLocation, DataLongevity, DataVersion, DataVersionType, Element, GarbageCollectId, LRUIndex,
    ThreadHandleDropPanic,
};

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

impl StorageInfo {
    //unsafe fn as_mut_slice_of<'a, 'b, T>(&'a mut self) -> &'b mut [T] {
    //    let t_ptr = self.data.cast::<T>();

    //    let num_elements = num_elms_in_array::<T>(self.layout.size());

    //    // Safety: Type matches as per contract upheld by caller. There are also no mutable
    //    // references to the slot since it has already been initialized.
    //    unsafe { std::slice::from_raw_parts_mut(t_ptr, num_elements) }
    //}
    unsafe fn as_slice_of<'a, 'b, T>(&'a self) -> &'b [T] {
        let t_ptr = self.data.cast::<T>();

        let num_elements = num_elms_in_array::<T>(self.layout.size());

        // Safety: Type matches as per contract upheld by caller. There are also no mutable
        // references to the slot since it has already been initialized.
        unsafe { std::slice::from_raw_parts(t_ptr, num_elements) }
    }
}

#[derive(Copy, Clone, Debug)]
enum WriteAccessCount {
    One,
    Zero,
}

#[derive(Copy, Clone, Debug)]
enum StorageEntryState {
    Registered,
    Initializing(StorageInfo, WriteAccessCount),
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

pub struct AccessToken<'a, Allocator> {
    storage: &'a Storage<Allocator>,
    pub id: DataId,
}
impl<'a, Allocator> AccessToken<'a, Allocator> {
    fn new(storage: &'a Storage<Allocator>, id: DataId) -> Self {
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
impl<Allocator> Drop for AccessToken<'_, Allocator> {
    fn drop(&mut self) {
        let mut index = self.storage.index.borrow_mut();
        let ram_entry = index.get_mut(&self.id).unwrap();
        ram_entry.access = match ram_entry.access {
            AccessState::Some(1) => {
                let lru_id = if let StorageEntryState::Initialized(si, _)
                | StorageEntryState::Initializing(si, WriteAccessCount::One) =
                    &ram_entry.state
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

pub struct ReadHandle<'a, T: ?Sized, Allocator> {
    access: AccessToken<'a, Allocator>,
    data_longevity: DataLongevity,
    pub version: DataVersionType,
    data: &'a T,
}
impl<'a, T: ?Sized, Allocator> ReadHandle<'a, T, Allocator> {
    pub fn map<O>(self, f: impl FnOnce(&'a T) -> &'a O) -> ReadHandle<'a, O, Allocator> {
        let ret = ReadHandle {
            access: self.access,
            data_longevity: self.data_longevity,
            version: self.version,
            data: f(&self.data),
        };
        ret
    }
}

impl<'a, T: ?Sized> ReadHandle<'a, T, super::ram::RamAllocator> {
    pub fn into_thread_handle(self) -> ThreadReadHandle<'a, T>
    where
        T: Send,
    {
        let ret = ThreadReadHandle {
            id: self.access.id,
            data: self.data,
            panic_handle: Default::default(),
            data_longevity: self.data_longevity,
            version: self.version,
        };
        //Avoid running destructor
        std::mem::forget(self.access);

        ret
    }
}
impl<T: ?Sized, Allocator> std::ops::Deref for ReadHandle<'_, T, Allocator> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

pub struct ThreadReadHandle<'a, T: ?Sized + Send> {
    id: DataId,
    data: &'a T,
    panic_handle: ThreadHandleDropPanic,
    data_longevity: DataLongevity,
    version: DataVersionType,
}
impl<T: ?Sized + Send> std::ops::Deref for ThreadReadHandle<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}
impl<'a, T: ?Sized + Send> ThreadReadHandle<'a, T> {
    pub fn into_main_handle<Allocator>(
        self,
        storage: &'a Storage<Allocator>,
    ) -> ReadHandle<'a, T, Allocator> {
        self.panic_handle.dismiss();
        ReadHandle {
            access: AccessToken {
                storage,
                id: self.id,
            },
            version: self.version,
            data: self.data,
            data_longevity: self.data_longevity,
        }
    }
}

pub struct DropError<'a, Allocator> {
    access: AccessToken<'a, Allocator>,
}
impl<'a, Allocator> DropError<'a, Allocator> {
    fn into_mark_initialized<'inv>(
        self,
        ctx: OpaqueTaskContext<'a, 'inv>,
        version: Option<DataVersionType>,
    ) -> DropMarkInitialized<'a, Allocator> {
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
impl<Allocator> Drop for DropError<'_, Allocator> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            // Avoid additional panics (-> aborts) while already panicking (unwinding)
            panic!("The WriteHandle was not marked initialized!");
        }
    }
}
pub struct DropMarkInitialized<'a, Allocator> {
    access: AccessToken<'a, Allocator>,
    version: Option<DataVersionType>,
    predicted_preview_tasks: &'a RefCell<Set<TaskId>>,
    current_frame: FrameNumber,
    current_task: TaskId,
}
impl<Allocator> Drop for DropMarkInitialized<'_, Allocator> {
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

            self.access.storage.new_data.add(self.access.id);

            *state = match state {
                StorageEntryState::Registered => {
                    panic!("Entry should be in state Initializing, but is in Registered");
                }
                StorageEntryState::Initialized(..) => {
                    panic!("Entry should be in state Initializing, but is in Initialized");
                }
                StorageEntryState::Initializing(info, WriteAccessCount::One) => {
                    StorageEntryState::Initialized(*info, version)
                }
                StorageEntryState::Initializing(_info, WriteAccessCount::Zero) => {
                    panic!("Invalid state");
                }
            };
        }
    }
}

pub struct RawReadHandle<'a, Allocator> {
    pub info: StorageInfo,
    #[allow(unused)]
    access: AccessToken<'a, Allocator>,
    pub version: DataVersionType,
}

impl<'a, Allocator> RawReadHandle<'a, Allocator> {
    pub fn id(&self) -> DataId {
        self.access.id
    }

    pub fn data<'b>(&'b self) -> &'b [u8] {
        // Safety: Since this is a read handle, the slot has already been written to. Since Element
        // is AnyBitPattern we know that all bytes in the slot are in fact initialized.
        unsafe { std::slice::from_raw_parts(self.info.data as *const u8, self.info.layout.size()) }
    }

    pub fn into_thread_handle(self) -> RawThreadReadHandle<'a> {
        let ret = RawThreadReadHandle {
            id: self.access.id,
            info: self.info,
            panic_handle: Default::default(),
            _marker: Default::default(),
            version: self.version,
        };
        //Avoid running destructor
        std::mem::forget(self.access);

        ret
    }
}

pub struct RawThreadReadHandle<'a> {
    id: DataId,
    info: StorageInfo,
    panic_handle: ThreadHandleDropPanic,
    version: DataVersionType,
    _marker: std::marker::PhantomData<&'a ()>,
}

// Safety: The raw pointer is not accessible
unsafe impl Send for RawThreadReadHandle<'_> {}

impl<'a> RawThreadReadHandle<'a> {
    pub fn data<'b>(&'b self) -> &'b [u8] {
        // Safety: Since this is a read handle, the slot has already been written to. Since Element
        // is AnyBitPattern we know that all bytes in the slot are in fact initialized.
        unsafe { std::slice::from_raw_parts(self.info.data as *const u8, self.info.layout.size()) }
    }
    pub fn into_main_handle<Allocator>(
        self,
        storage: &'a Storage<Allocator>,
    ) -> RawReadHandle<'a, Allocator> {
        self.panic_handle.dismiss();
        RawReadHandle {
            info: self.info,
            access: AccessToken {
                storage,
                id: self.id,
            },
            version: self.version,
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
    ) -> WriteHandleInit<'a, T, super::ram::RamAllocator> {
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
    pub fn into_main_handle<Allocator>(
        self,
        storage: &'a Storage<Allocator>,
    ) -> WriteHandleUninit<'a, T, Allocator> {
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

pub struct RawWriteHandle<DropHandler> {
    data: *mut MaybeUninit<u8>,
    pub layout: Layout,
    drop_handler: DropHandler,
}
// Safety: We never expose the data pointer (safely) to the outside
impl<DropHandler> RawWriteHandle<DropHandler> {
    pub unsafe fn data_ptr(&self) -> *mut MaybeUninit<u8> {
        self.data
    }
    pub fn data(&mut self) -> &mut [MaybeUninit<u8>] {
        // Safety: We ensure exclusive access by taking a mutable self reference
        unsafe { std::slice::from_raw_parts_mut(self.data, self.layout.size()) }
    }
}

impl<D: Send> std::ops::Deref for RawWriteHandle<D> {
    type Target = [MaybeUninit<u8>];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.data, self.layout.size()) }
    }
}
impl<D: Send> std::ops::DerefMut for RawWriteHandle<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.layout.size()) }
    }
}

unsafe impl<D: Send> Send for RawThreadWriteHandle<D> {}
impl<D: Send> std::ops::Deref for RawThreadWriteHandle<D> {
    type Target = [MaybeUninit<u8>];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.data, self.layout.size()) }
    }
}
impl<D: Send> std::ops::DerefMut for RawThreadWriteHandle<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.layout.size()) }
    }
}

impl<'a> RawWriteHandleUninit<'a, super::ram::RamAllocator> {
    pub fn into_thread_handle(self) -> RawThreadWriteHandle<ThreadMarkerUninitialized> {
        let id = self.drop_handler.access.id;
        std::mem::forget(self.drop_handler);
        RawThreadWriteHandle {
            id,
            data: self.data,
            layout: self.layout,
            _marker: ThreadMarkerUninitialized,
            _panic_handle: Default::default(),
        }
    }
}
pub struct RawThreadWriteHandle<D: Send> {
    id: DataId,
    pub data: *mut MaybeUninit<u8>,
    pub layout: Layout,
    _marker: D,
    _panic_handle: ThreadHandleDropPanic,
}
impl<DropHandler: Send> RawThreadWriteHandle<DropHandler> {
    pub unsafe fn data_ptr(&self) -> *mut MaybeUninit<u8> {
        self.data
    }
    pub fn data(&self) -> &mut [MaybeUninit<u8>] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.layout.size()) }
    }
}
impl RawThreadWriteHandle<ThreadMarkerInitialized> {
    pub fn into_main_handle<'a, 'inv>(
        self,
        ctx: OpaqueTaskContext<'a, 'inv>,
    ) -> RawWriteHandleInit<'a, super::ram::RamAllocator> {
        self._panic_handle.dismiss();
        RawWriteHandle {
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
            layout: self.layout,
            data: self.data,
        }
    }
}
impl RawThreadWriteHandle<ThreadMarkerUninitialized> {
    pub fn into_main_handle<'a, Allocator>(
        self,
        storage: &'a Storage<Allocator>,
    ) -> RawWriteHandleUninit<'a, Allocator> {
        self._panic_handle.dismiss();
        RawWriteHandle {
            drop_handler: DropError {
                access: AccessToken {
                    storage,
                    id: self.id,
                },
            },
            layout: self.layout,
            data: self.data,
        }
    }
}

pub type WriteHandleUninit<'a, T, Allocator> = WriteHandle<'a, T, DropError<'a, Allocator>>;
pub type RawWriteHandleUninit<'a, Allocator> = RawWriteHandle<DropError<'a, Allocator>>;
pub type ThreadWriteHandleUninit<'a, T> = ThreadWriteHandle<'a, T, ThreadMarkerUninitialized>;

impl<'a, T: AsInit + ?Sized, Allocator> WriteHandleUninit<'a, T, Allocator> {
    /// Safety: The corresponding slot has to have been completely written to, i.e. all
    /// (non-padding) bytes of T must have been written.
    pub unsafe fn initialized<'inv>(
        self,
        ctx: OpaqueTaskContext<'a, 'inv>,
    ) -> WriteHandleInit<'a, T::Init, Allocator> {
        WriteHandle {
            drop_handler: self.drop_handler.into_mark_initialized(ctx, None),
            data: T::assume_init(self.data),
        }
    }

    /// Safety: The corresponding slot has to have been completely written to, i.e. all
    /// (non-padding) bytes of T must have been written.
    pub unsafe fn initialized_version<'inv>(
        self,
        ctx: OpaqueTaskContext<'a, 'inv>,
        version: DataVersionType,
    ) -> WriteHandleInit<'a, T::Init, Allocator> {
        WriteHandle {
            drop_handler: self.drop_handler.into_mark_initialized(ctx, Some(version)),
            data: T::assume_init(self.data),
        }
    }
}

impl<'a, T: ?Sized> WriteHandleUninit<'a, T, super::ram::RamAllocator> {
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
impl<DropHandler> RawWriteHandle<DropHandler> {
    fn transmute<'a, T: bytemuck::AnyBitPattern>(
        self,
        size: usize,
    ) -> WriteHandle<'a, [MaybeUninit<T>], DropHandler> {
        let layout = Layout::array::<T>(size).unwrap();
        assert_eq!(layout, self.layout);

        let t_ptr = self.data.cast::<MaybeUninit<T>>();

        // Safety: We constructed the pointer with the required layout and T is
        // bytemuck::AnyBitPattern
        let t_ref = unsafe { std::slice::from_raw_parts_mut(t_ptr, size) };
        WriteHandle {
            data: t_ref,
            drop_handler: self.drop_handler,
        }
    }
}
impl<'a, Allocator> RawWriteHandleUninit<'a, Allocator> {
    /// Safety: The corresponding slot has to have been completely written to.
    pub unsafe fn initialized<'inv>(
        self,
        ctx: OpaqueTaskContext<'a, 'inv>,
    ) -> RawWriteHandleInit<'a, Allocator> {
        RawWriteHandle {
            drop_handler: self.drop_handler.into_mark_initialized(ctx, None),
            data: self.data,
            layout: self.layout,
        }
    }

    pub unsafe fn initialized_version<'inv>(
        self,
        ctx: OpaqueTaskContext<'a, 'inv>,
        data_version: DataVersionType,
    ) -> RawWriteHandleInit<'a, Allocator> {
        RawWriteHandle {
            drop_handler: self
                .drop_handler
                .into_mark_initialized(ctx, Some(data_version)),
            data: self.data,
            layout: self.layout,
        }
    }
}

pub type WriteHandleInit<'a, T, Allocator> = WriteHandle<'a, T, DropMarkInitialized<'a, Allocator>>;
pub type RawWriteHandleInit<'a, Allocator> = RawWriteHandle<DropMarkInitialized<'a, Allocator>>;
pub type ThreadWriteHandleInit<'a, T> = ThreadWriteHandle<'a, T, ThreadMarkerInitialized>;

impl<'a, T: ?Sized> WriteHandleInit<'a, T, super::ram::RamAllocator> {
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
pub enum InplaceResult<'a, 'inv, T, Allocator> {
    Inplace(WriteHandleInit<'a, [T], Allocator>, DataVersionType),
    New(
        ReadHandle<'a, [T], Allocator>,
        Request<'a, 'inv, WriteHandleUninit<'a, [MaybeUninit<T>], Allocator>>,
    ),
}

impl<'a, 'inv, T, Allocator> InplaceResult<'a, 'inv, T, Allocator> {
    pub fn alloc(self) -> Request<'a, 'inv, InplaceHandle<'a, T, Allocator>> {
        match self {
            InplaceResult::Inplace(a, _) => Request::ready(InplaceHandle::Inplace(a)),
            InplaceResult::New(r, w) => w.map(move |w| InplaceHandle::New(r, w)),
        }
    }

    pub fn version(&self) -> DataVersionType {
        match self {
            InplaceResult::Inplace(_, data_version_type) => *data_version_type,
            InplaceResult::New(read_handle, _) => read_handle.version,
        }
    }
}

pub enum InplaceHandle<'a, T, Allocator> {
    Inplace(WriteHandleInit<'a, [T], Allocator>),
    New(
        ReadHandle<'a, [T], Allocator>,
        WriteHandleUninit<'a, [MaybeUninit<T>], Allocator>,
    ),
}

impl<'a, T: Send> InplaceHandle<'a, T, super::ram::RamAllocator> {
    pub fn into_thread_handle(self) -> ThreadInplaceHandle<'a, T> {
        match self {
            InplaceHandle::Inplace(rw) => ThreadInplaceHandle::Inplace(rw.into_thread_handle()),
            InplaceHandle::New(r, w) => {
                ThreadInplaceHandle::New(r.into_thread_handle(), w.into_thread_handle())
            }
        }
    }
}

pub enum ThreadInplaceHandle<'a, T: Send> {
    Inplace(ThreadWriteHandleInit<'a, [T]>),
    New(
        ThreadReadHandle<'a, [T]>,
        ThreadWriteHandleUninit<'a, [MaybeUninit<T>]>,
    ),
}

impl<'a, T: Send> ThreadInplaceHandle<'a, T> {
    pub fn into_main_handle<'inv>(
        self,
        ctx: OpaqueTaskContext<'a, 'inv>,
    ) -> InplaceHandle<'a, T, super::ram::RamAllocator> {
        match self {
            ThreadInplaceHandle::Inplace(rw) => InplaceHandle::Inplace(rw.into_main_handle(ctx)),
            ThreadInplaceHandle::New(r, w) => InplaceHandle::New(
                r.into_main_handle(&ctx.storage),
                w.into_main_handle(&ctx.storage),
            ),
        }
    }
}

//pub struct StateCacheHandle<'a, T, Allocator> {
//    data: &'a mut MaybeUninit<T>,
//    access: AccessToken<'a, Allocator>,
//}

pub trait AsInit {
    type Init: ?Sized + Send;
    unsafe fn assume_init(&mut self) -> &mut Self::Init;
}

impl<T: Send> AsInit for MaybeUninit<T> {
    type Init = T;
    unsafe fn assume_init(&mut self) -> &mut Self::Init {
        self.assume_init_mut()
    }
}

impl<T: Send> AsInit for [MaybeUninit<T>] {
    type Init = [T];
    unsafe fn assume_init(&mut self) -> &mut Self::Init {
        crate::data::slice_assume_init_mut(self)
    }
}

pub struct DropUnref<'a, Allocator> {
    access: AccessToken<'a, Allocator>,
}
impl<'a, Allocator> DropUnref<'a, Allocator> {
    fn into_error(self) -> DropError<'a, Allocator> {
        let id = self.access.id;
        let storage = self.access.storage;
        // Avoid running destructor
        std::mem::forget(self);
        DropError {
            access: AccessToken { storage, id },
        }
    }
}
impl<Allocator> Drop for DropUnref<'_, Allocator> {
    fn drop(&mut self) {
        let mut binding = self.access.storage.index.borrow_mut();
        let state = &mut binding.get_mut(&self.access.id).unwrap().state;

        *state = match state {
            StorageEntryState::Registered => {
                panic!("Entry should be in state Initializing, but is in Registered");
            }
            StorageEntryState::Initialized(..) => {
                panic!("Entry should be in state Initializing, but is in Initialized");
            }
            StorageEntryState::Initializing(info, WriteAccessCount::One) => {
                StorageEntryState::Initializing(*info, WriteAccessCount::Zero)
            }
            StorageEntryState::Initializing(_info, WriteAccessCount::Zero) => {
                panic!("Invalid state");
            }
        };
    }
}

pub enum StateCacheResult<'a, T: ?Sized, Allocator> {
    New(WriteHandle<'a, T, DropUnref<'a, Allocator>>),
    Existing(WriteHandle<'a, T, DropUnref<'a, Allocator>>),
}

impl<'a, T: AsInit + ?Sized, Allocator> StateCacheResult<'a, T, Allocator> {
    // Safety: The caller must actually initialize all values in the write handle
    pub unsafe fn init(
        self,
        f: impl FnOnce(&mut WriteHandle<'a, T, DropUnref<'a, Allocator>>),
    ) -> WriteHandle<'a, T::Init, DropUnref<'a, Allocator>> {
        let n = match self {
            StateCacheResult::New(mut n) => {
                f(&mut n);
                n
            }
            StateCacheResult::Existing(n) => n,
        };
        WriteHandle {
            data: T::assume_init(n.data),
            drop_handler: n.drop_handler,
        }
    }
}

pub struct Storage<Allocator> {
    index: RefCell<Map<DataId, Entry>>,
    lru_manager: RefCell<super::LRUManager<DataId>>,
    allocator: Allocator,
    new_data: super::NewDataManager,
    garbage_collect_id_gen: IdGenerator<GarbageCollectId>,
}

impl<Allocator: CpuAllocator> Storage<Allocator> {
    /// Safety: The same combination of `state`, `ram` and `vram` must be used to construct a
    /// `Storage` at all times. No modifications outside of `Storage` must be done to either of
    /// these structs in the meantime.
    pub fn new(allocator: Allocator) -> Self {
        Self {
            index: Default::default(),
            lru_manager: Default::default(),
            new_data: Default::default(),
            allocator,
            garbage_collect_id_gen: Default::default(),
        }
    }

    pub fn next_garbage_collect(&self) -> GarbageCollectId {
        self.garbage_collect_id_gen.preview_next()
    }

    pub fn wait_garbage_collect<'a, 'inv>(&self) -> Request<'a, 'inv, ()> {
        Request::garbage_collect(
            DataLocation::CPU(Allocator::LOCATION),
            self.next_garbage_collect(),
        )
    }

    pub fn try_garbage_collect(&self, mut goal_in_bytes: usize) -> usize {
        let mut lru = self.lru_manager.borrow_mut();
        let mut index = self.index.borrow_mut();

        let mut collected_total = 0;
        for (longevity, inner_lru) in lru.inner_mut() {
            let mut collected = 0;
            for key in inner_lru.drain_lru().into_iter() {
                let entry = index.get_mut(&key).unwrap();
                let info = match entry.state {
                    StorageEntryState::Registered => panic!("Should not be in LRU list"),
                    StorageEntryState::Initializing(info, _)
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
            if collected > 0 {
                println!(
                    "Garbage collect RAM ({:?}): {}",
                    longevity,
                    bytesize::to_string(collected as _, true)
                );
            }
            collected_total += collected;
        }

        if collected_total > 0 {
            let _ = self.garbage_collect_id_gen.next();
        }

        collected_total
    }

    pub fn is_readable(&self, id: DataId, requested_version: DataVersion) -> bool {
        self.index
            .borrow()
            .get(&id)
            .map(|e| {
                if let StorageEntryState::Initialized(_, version) = e.state {
                    version >= requested_version
                } else {
                    false
                }
            })
            .unwrap_or(false)
    }

    pub fn try_free(&self, key: DataId) -> Result<(), ()> {
        let mut index = self.index.borrow_mut();
        if index.get(&key).unwrap().safe_to_delete() {
            let entry = index.get_mut(&key).unwrap();

            let info = match entry.state {
                StorageEntryState::Registered => return Err(()),
                StorageEntryState::Initializing(info, _)
                | StorageEntryState::Initialized(info, _) => info,
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

    pub(crate) fn newest_data(&self) -> impl Iterator<Item = (DataId, DataLocation)> {
        self.new_data
            .drain()
            .map(|d| (d, DataLocation::CPU(Allocator::LOCATION)))
    }

    fn ensure_presence<'a>(
        &self,
        current_frame: FrameNumber,
        entry: crate::util::MapEntry<'a, DataId, Entry>,
    ) -> &'a mut Entry {
        match entry {
            crate::util::MapEntry::Occupied(mut e) => {
                if let StorageEntryState::Initialized(_, version) = e.get().state {
                    if version < DataVersion::Preview(current_frame) {
                        let old = e.insert(Entry {
                            state: StorageEntryState::Registered,
                            access: AccessState::Some(0), // Will be overwritten immediately when generating token
                        });
                        match old.access {
                            AccessState::Some(_) => {
                                panic!(
                                    "There should not be any readers left from the previous frame"
                                );
                            }
                            AccessState::None(lru_index) => {
                                let StorageEntryState::Initialized(info, _) = old.state else {
                                    panic!("we just checked that");
                                };
                                if let Some(lru_index) = lru_index {
                                    self.lru_manager.borrow_mut().remove(lru_index);
                                }

                                // Safety: all data ptrs in the index have been allocated with the allocator.
                                // Deallocation only happens exactly here where the entry is also removed from the
                                // index.
                                unsafe { self.allocator.dealloc(info.data) };
                                self.new_data.remove(*e.key());
                            }
                        }
                    }
                }
                e.into_mut()
            }
            crate::util::MapEntry::Vacant(v) => {
                v.insert(Entry {
                    state: StorageEntryState::Registered,
                    access: AccessState::Some(0), // Will be overwritten immediately when generating token
                })
            }
        }
    }

    pub fn register_access(
        &self,
        current_frame: FrameNumber,
        id: DataId,
    ) -> AccessToken<Allocator> {
        {
            let mut index = self.index.borrow_mut();
            let entry = index.entry(id);
            self.ensure_presence(current_frame, entry);
        }
        AccessToken::new(self, id)
    }

    pub fn size(&self) -> usize {
        self.allocator.size()
    }

    pub(crate) fn alloc(
        &self,
        descriptor: DataDescriptor,
        layout: Layout,
    ) -> Result<(*mut MaybeUninit<u8>, AccessToken<Allocator>), ()> {
        let key = descriptor.id;

        let data = {
            let data = self.allocator.alloc(layout).map_err(|_| ())?;
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

            entry.state = StorageEntryState::Initializing(info, WriteAccessCount::Zero);

            data
        };

        Ok((data, AccessToken::new(self, descriptor.id)))
    }

    fn access_initializing<'a>(
        &self,
        access: AccessToken<'a, Allocator>,
    ) -> Result<RawWriteHandle<DropUnref<'a, Allocator>>, AccessToken<'a, Allocator>> {
        let mut index = self.index.borrow_mut();
        let entry = index.get_mut(&access.id).unwrap();

        if let StorageEntryState::Initializing(info, ref mut count) = entry.state {
            if let WriteAccessCount::Zero = *count {
                *count = WriteAccessCount::One;
                Ok(RawWriteHandle {
                    data: info.data,
                    layout: info.layout,
                    drop_handler: DropUnref { access },
                })
            } else {
                panic!("Concurrent write access to initializing datum attempted")
            }
        } else {
            Err(access)
        }
    }

    pub fn alloc_slot_raw(
        &self,
        key: DataDescriptor,
        layout: Layout,
    ) -> Result<RawWriteHandleUninit<Allocator>, ()> {
        let (ptr, access) = self.alloc(key, layout)?;

        Ok(RawWriteHandleUninit {
            data: ptr,
            layout,
            drop_handler: DropError { access },
        })
    }

    pub fn alloc_slot<T: Element>(
        &self,
        key: DataDescriptor,
        size: usize,
    ) -> Result<WriteHandleUninit<[MaybeUninit<T>], Allocator>, ()> {
        let layout = Layout::array::<T>(size).unwrap();
        let (ptr, access) = self.alloc(key, layout)?;

        let t_ptr = ptr.cast::<MaybeUninit<T>>();

        // Safety: We constructed the pointer with the required layout
        let t_ref = unsafe { std::slice::from_raw_parts_mut(t_ptr, size) };
        Ok(WriteHandleUninit {
            data: t_ref,
            drop_handler: DropError { access },
        })
    }

    /// Safety: The initial allocation for the TaskId must have happened with the same type
    pub fn read_raw<'b, 't: 'b>(
        &'b self,
        access: AccessToken<'t, Allocator>,
    ) -> Result<RawReadHandle<'b, Allocator>, AccessToken<'t, Allocator>> {
        let (info, version) = {
            let index = self.index.borrow();
            let Some(entry) = index.get(&access.id) else {
                return Err(access);
            };
            let StorageEntryState::Initialized(info, version) = entry.state else {
                return Err(access);
            };

            (info, version.type_())
        };
        Ok(RawReadHandle {
            access,
            info,
            version,
        })
    }

    /// Safety: The initial allocation for the TaskId must have happened with the same type
    pub unsafe fn read<'b, 't: 'b, T: Element>(
        &'b self,
        access: AccessToken<'t, Allocator>,
    ) -> Result<ReadHandle<'b, [T], Allocator>, AccessToken<'t, Allocator>> {
        let (t_ref, data_longevity, version) = {
            let index = self.index.borrow();
            let Some(entry) = index.get(&access.id) else {
                return Err(access);
            };

            let StorageEntryState::Initialized(info, version) = entry.state else {
                return Err(access);
            };

            // Safety: Type matches as per contract upheld by caller. There are also no mutable
            // references to the slot since it has already been initialized.
            (
                info.as_slice_of::<T>(),
                info.data_longevity,
                version.type_(),
            )
        };
        Ok(ReadHandle {
            access,
            data: t_ref,
            data_longevity,
            version,
        })
    }

    pub fn request_alloc_raw<'req, 'inv>(
        &'req self,
        current_frame: FrameNumber,
        data_descriptor: DataDescriptor,
        layout: Layout,
    ) -> Request<'req, 'inv, RawWriteHandleUninit<'req, Allocator>> {
        let mut access = Some(self.register_access(current_frame, data_descriptor.id));

        Request {
            type_: RequestType::Allocation(
                AllocationId::next(),
                AllocationRequest::Ram(layout, data_descriptor, Allocator::LOCATION),
            ),
            gen_poll: Box::new(move |_ctx| {
                Box::new(move || {
                    access = match self.access_initializing(access.take().unwrap()) {
                        Ok(r) => {
                            return Some(RawWriteHandle {
                                data: r.data,
                                layout: r.layout,
                                drop_handler: r.drop_handler.into_error(),
                            })
                        }
                        Err(acc) => Some(acc),
                    };
                    None
                })
            }),
            _marker: Default::default(),
        }
    }

    pub fn request_alloc_slot<'req, 'inv, T: Element>(
        &'req self,
        current_frame: FrameNumber,
        key: DataDescriptor,
        size: usize,
    ) -> Request<'req, 'inv, WriteHandleUninit<'req, [MaybeUninit<T>], Allocator>> {
        let layout = Layout::array::<T>(size).unwrap();
        self.request_alloc_raw(current_frame, key, layout)
            .map(move |v| v.transmute(size))
    }

    pub fn request_access_state_cache<'req, 'inv, T: bytemuck::AnyBitPattern + Send>(
        &'req self,
        current_frame: FrameNumber,
        id: DataId,
        size: usize,
    ) -> Request<'req, 'inv, StateCacheResult<'req, [MaybeUninit<T>], Allocator>> {
        let access = self.register_access(current_frame, id);
        let layout = Layout::array::<T>(size).unwrap();

        let data_descriptor = DataDescriptor {
            id,
            longevity: DataLongevity::Cache,
        };

        match self.access_initializing(access) {
            Ok(r) => Request::ready(StateCacheResult::Existing(r.transmute(size))),
            Err(access) => {
                let mut access = Some(access);
                Request {
                    type_: RequestType::Allocation(
                        AllocationId::next(),
                        AllocationRequest::Ram(layout, data_descriptor, Allocator::LOCATION),
                    ),
                    gen_poll: Box::new(move |_ctx| {
                        Box::new(move || {
                            access = match self.access_initializing(access.take().unwrap()) {
                                Ok(r) => return Some(StateCacheResult::New(r.transmute(size))),
                                Err(acc) => Some(acc),
                            };
                            None
                        })
                    }),
                    _marker: Default::default(),
                }
            }
        }
    }
}

impl Storage<super::ram::RamAllocator> {
    /// Safety: The initial allocation for the TaskId must have happened with the same type and the
    /// size must match the initial allocation
    pub unsafe fn try_update_inplace<'b, 't: 'b, 'inv, T: Element>(
        &'b self,
        ctx: OpaqueTaskContext<'t, 'inv>,
        old_access: AccessToken<'t, super::ram::RamAllocator>,
        new_desc: DataDescriptor,
    ) -> Result<
        InplaceResult<'b, 'inv, T, super::ram::RamAllocator>,
        AccessToken<'t, super::ram::RamAllocator>,
    > {
        let old_key = old_access.id;
        let new_key = new_desc.id;

        let mut index = self.index.borrow_mut();
        let Some(entry) = index.get(&old_access.id) else {
            return Err(old_access);
        };

        let StorageEntryState::Initialized(info, version) = entry.state else {
            return Err(old_access);
        };
        let version = version.type_();

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

            new_entry.state = StorageEntryState::Initializing(info, WriteAccessCount::One);

            new_entry.access = match new_entry.access {
                AccessState::Some(n) => AccessState::Some(n + 1),
                AccessState::None(_) => panic!("If present, entry should have accessors"),
            };

            // Safety: Type matches as per contract upheld by caller. There are also no other
            // references to the slot since it has (1.) been already initialized and (2.) there are
            // no readers. In other words: safe_to_delete also implies no other references.
            let t_ref = unsafe { std::slice::from_raw_parts_mut(t_ptr, num_elements) };

            InplaceResult::Inplace(
                WriteHandleInit {
                    data: t_ref,
                    drop_handler: DropMarkInitialized {
                        access: new_access,
                        version: None,
                        predicted_preview_tasks: ctx.predicted_preview_tasks,
                        current_frame: ctx.current_frame,
                        current_task: ctx.current_task,
                    },
                },
                version,
            )
        } else {
            std::mem::drop(index); // Release borrow for alloc

            // Safety: Type matches as per contract upheld by caller. There are also no mutable
            // references to the slot since it has already been initialized.
            let t_ref = unsafe { std::slice::from_raw_parts(t_ptr, num_elements) };

            let w = self.request_alloc_slot(ctx.current_frame, new_desc, num_elements);

            let r = ReadHandle {
                data: t_ref,
                access: AccessToken::new(self, old_key),
                data_longevity: info.data_longevity,
                version,
            };
            InplaceResult::New(r, w)
        })
    }
}

#[derive(Debug, Copy, Clone)]
pub struct OOMError {
    //requested: usize,
}

pub trait CpuAllocator {
    const LOCATION: super::CpuDataLocation;

    fn alloc(&self, layout: Layout) -> Result<*mut MaybeUninit<u8>, OOMError>;

    /// Safety: `ptr` must have been allocated with this allocator and must not have been
    /// deallocated already.
    unsafe fn dealloc(&self, ptr: *mut MaybeUninit<u8>);

    fn size(&self) -> usize;
}
