use std::{
    alloc::Layout,
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    mem::MaybeUninit,
};

use ash::vk;

use crate::{
    operator::DataId,
    task_graph::LocatedDataId,
    util::num_elms_in_array,
    vulkan::{CmdBufferEpoch, DeviceContext, DeviceId},
    Error,
};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum DataLocation {
    Ram,
    VRam(DeviceId),
}

#[derive(Debug, Eq, PartialEq)]
enum RamAccessState {
    Some(usize),
    None(LRUIndex),
}

#[derive(Debug, Eq, PartialEq)]
enum VRamAccessState {
    Some(usize),
    None(/*LRUIndex, */ CmdBufferEpoch),
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

enum VRamStorageEntryState {
    Registered,
    Initializing(VRamStorageInfo),
    Initialized(VRamStorageInfo),
}

pub struct VRamStorageInfo {
    pub allocation: crate::vulkan::Allocation,
    pub layout: Layout,
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

struct VRamEntry {
    state: VRamStorageEntryState,
    access: VRamAccessState,
}

struct StorageEntry {
    ram: Option<RamEntry>,
    vram: Vec<Option<VRamEntry>>,
}

impl StorageEntry {
    fn is_present(&self) -> bool {
        self.ram.is_some() || self.vram.iter().any(|v| v.is_some())
    }
}

pub struct RamAccessToken<'a> {
    storage: &'a Storage<'a>,
    pub id: DataId,
}
impl<'a> RamAccessToken<'a> {
    fn new(storage: &'a Storage, id: DataId) -> Self {
        let mut index = storage.state.index.borrow_mut();
        let ram_entry = index.get_mut(&id).unwrap().ram.as_mut().unwrap();

        ram_entry.access = match ram_entry.access {
            RamAccessState::Some(n) => RamAccessState::Some(n + 1),
            RamAccessState::None(id) => {
                storage.state.lru_manager.borrow_mut().remove(id);
                RamAccessState::Some(1)
            }
        };

        Self { storage, id }
    }
}
impl Drop for RamAccessToken<'_> {
    fn drop(&mut self) {
        let mut index = self.storage.state.index.borrow_mut();
        let ram_entry = index.get_mut(&self.id).unwrap().ram.as_mut().unwrap();

        ram_entry.access = match ram_entry.access {
            RamAccessState::Some(1) => {
                let lru_id = self.storage.state.lru_manager.borrow_mut().add(self.id);
                RamAccessState::None(lru_id)
            }
            RamAccessState::Some(n) => RamAccessState::Some(n - 1),
            RamAccessState::None(_id) => {
                panic!("Invalid state");
            }
        };
    }
}

pub struct VRamAccessToken<'a> {
    storage: &'a Storage<'a>,
    device: &'a DeviceContext,
    pub id: DataId,
}
impl<'a> VRamAccessToken<'a> {
    fn new(storage: &'a Storage, device: &'a DeviceContext, id: DataId) -> Self {
        let mut index = storage.state.index.borrow_mut();
        let vram_entry = index.get_mut(&id).unwrap().vram[device.id]
            .as_mut()
            .unwrap();

        vram_entry.access = match vram_entry.access {
            VRamAccessState::Some(n) => VRamAccessState::Some(n + 1),
            VRamAccessState::None(/*id, */ _) => {
                //storage.state.lru_manager.borrow_mut().remove(id);
                VRamAccessState::Some(1)
            }
        };

        Self {
            storage,
            id,
            device,
        }
    }
}
impl Drop for VRamAccessToken<'_> {
    fn drop(&mut self) {
        let mut index = self.storage.state.index.borrow_mut();
        let vram_entry = index.get_mut(&self.id).unwrap().vram[self.device.id]
            .as_mut()
            .unwrap();

        vram_entry.access = match vram_entry.access {
            VRamAccessState::Some(1) => {
                //let lru_id = self.storage.state.lru_manager.borrow_mut().add(self.id);
                VRamAccessState::None(/*lru_id, */ self.device.current_epoch())
            }
            VRamAccessState::Some(n) => VRamAccessState::Some(n - 1),
            VRamAccessState::None(..) => {
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
        self.access
            .storage
            .new_data
            .borrow_mut()
            .insert(LocatedDataId {
                id: self.access.id,
                location: DataLocation::Ram,
            });
        {
            let mut binding = self.access.storage.state.index.borrow_mut();
            let state = &mut binding
                .get_mut(&self.access.id)
                .unwrap()
                .ram
                .as_mut()
                .unwrap()
                .state;
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

pub struct VRamWriteHandle<'a> {
    pub buffer: ash::vk::Buffer,
    pub size: u64,
    drop_handler: DropErrorVram,
    access: VRamAccessToken<'a>,
}

pub struct DropErrorVram;
impl Drop for DropErrorVram {
    fn drop(&mut self) {
        panic!("The WriteHandle was not marked initialized!");
    }
}

impl<'a> VRamWriteHandle<'a> {
    /// Safety: The corresponding slot has to have been completely written to.
    pub unsafe fn initialized(self) {
        let VRamWriteHandle {
            buffer,
            access,
            drop_handler,
            ..
        } = self;

        // Avoid running destructor which would panic
        std::mem::forget(drop_handler);

        // Mark as initialized
        access.storage.new_data.borrow_mut().insert(LocatedDataId {
            id: access.id,
            location: DataLocation::VRam(access.device.id),
        });
        let mut binding = access.storage.state.index.borrow_mut();

        {
            let vram_entry_ref = &mut binding.get_mut(&access.id).unwrap().vram[access.device.id];
            let VRamEntry { mut state, access } = vram_entry_ref.take().unwrap();

            state = match state {
                VRamStorageEntryState::Registered => {
                    panic!("Entry should be in state Initializing, but is in Registered");
                }
                VRamStorageEntryState::Initialized(_) => {
                    panic!("Entry should be in state Initializing, but is in Initialized");
                }
                VRamStorageEntryState::Initializing(info) => {
                    VRamStorageEntryState::Initialized(info)
                }
            };

            *vram_entry_ref = Some(VRamEntry { state, access });
        }

        // Add pipeline barrier
        // TODO: This is probably not especially efficient
        let memory_barriers = &[vk::BufferMemoryBarrier2::builder()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
            .buffer(buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE)
            .build()];
        let barrier_info = vk::DependencyInfo::builder().buffer_memory_barriers(memory_barriers);
        access.device.with_cmd_buffer(|cmd| {
            cmd.pipeline_barrier(&barrier_info);
        });
    }
}
pub struct VRamReadHandle<'a> {
    pub buffer: ash::vk::Buffer,
    pub layout: Layout,
    #[allow(unused)]
    access: VRamAccessToken<'a>,
}

type LRUIndex = u64;

#[derive(Default)]
struct LRUManager {
    list: BTreeMap<LRUIndex, DataId>,
    current: LRUIndex,
}

impl LRUManager {
    fn remove(&mut self, old: LRUIndex) {
        self.list.remove(&old).unwrap();
    }

    #[must_use]
    fn add(&mut self, data: DataId) -> LRUIndex {
        let new = self
            .current
            .checked_add(1)
            .expect("Looks like we need to handle wrapping here...");
        self.current = new;

        self.list.insert(new, data);

        new
    }

    fn drain_lru<'a>(&'a mut self) -> impl Iterator<Item = DataId> + 'a {
        std::iter::from_fn(move || {
            return self.list.pop_first().map(|(_, d)| d);
        })
    }
}

#[derive(Default)]
pub struct StorageState {
    index: RefCell<BTreeMap<DataId, StorageEntry>>,
    lru_manager: RefCell<LRUManager>,
}

pub struct Storage<'a> {
    state: &'a StorageState,
    new_data: RefCell<BTreeSet<LocatedDataId>>,
    ram: &'a crate::ram_allocator::Allocator,
    vram: Vec<&'a crate::vulkan::Allocator>,
}

impl<'a> Storage<'a> {
    /// Safety: The same combination of `state`, `ram` and `vram` must be used to construct a
    /// `Storage` at all times. No modifications outside of `Storage` must be done to either of
    /// these structs in the meantime.
    pub unsafe fn new(
        state: &'a StorageState,
        ram: &'a crate::ram_allocator::Allocator,
        vram: Vec<&'a crate::vulkan::Allocator>,
    ) -> Self {
        Self {
            state,
            new_data: RefCell::new(BTreeSet::new()),
            ram,
            vram,
        }
    }

    /// Safety: Danger zone: The entries cannot be in use anymore! No checking for dangling
    /// references is done!
    pub unsafe fn free_vram(&self) {
        let mut index = self.state.index.borrow_mut();
        for entry in index.values_mut() {
            for (i, vram_entry) in entry.vram.iter_mut().enumerate() {
                if let Some(vram_entry) = vram_entry.take() {
                    match vram_entry.state {
                        VRamStorageEntryState::Registered => {}
                        VRamStorageEntryState::Initializing(info)
                        | VRamStorageEntryState::Initialized(info) => {
                            self.vram[i].deallocate(info.allocation);
                        }
                    }
                }
            }
        }
    }

    pub fn try_garbage_collect(&self, mut goal_in_bytes: usize) {
        let mut lru = self.state.lru_manager.borrow_mut();
        let mut index = self.state.index.borrow_mut();
        let mut collected = 0;
        for key in lru.drain_lru().into_iter() {
            let entry = index.get_mut(&key).unwrap();
            let ram_entry = entry.ram.take().unwrap();
            let info = match ram_entry.state {
                RamStorageEntryState::Registered => panic!("Should not be in LRU list"),
                RamStorageEntryState::Initializing(info)
                | RamStorageEntryState::Initialized(info) => info,
            };
            assert!(matches!(ram_entry.access, RamAccessState::None(_)));

            if !entry.is_present() {
                index.remove(&key).unwrap();
            }
            // Safety: all data ptrs in the index have been allocated with the allocator.
            // Deallocation only happens exactly here where the entry is also removed from the
            // index.
            unsafe { self.ram.dealloc(info.data) };
            let data_key = LocatedDataId {
                id: key,
                location: DataLocation::Ram,
            };
            self.new_data.borrow_mut().remove(&data_key);

            let size = info.layout.size();
            collected += size;
            let Some(rest) = goal_in_bytes.checked_sub(size) else {
                break;
            };
            goal_in_bytes = rest;
        }
        println!("Garbage collect: {}B", collected);
    }

    pub fn try_free_ram(&self, key: DataId) -> Result<(), ()> {
        let mut index = self.state.index.borrow_mut();
        if index
            .get(&key)
            .unwrap()
            .ram
            .as_ref()
            .unwrap()
            .safe_to_delete()
        {
            let entry = index.get_mut(&key).unwrap();
            let ram_entry = entry.ram.take().unwrap();

            let info = match ram_entry.state {
                RamStorageEntryState::Registered => return Err(()),
                RamStorageEntryState::Initializing(info)
                | RamStorageEntryState::Initialized(info) => info,
            };

            if !entry.is_present() {
                index.remove(&key).unwrap();

                let lru_index = ram_entry.lru_index().unwrap();
                self.state.lru_manager.borrow_mut().remove(lru_index);
            }

            // Safety: all data ptrs in the index have been allocated with the allocator.
            // Deallocation only happens exactly here where the entry is also removed from the
            // index.
            unsafe { self.ram.dealloc(info.data) };
            let data_key = LocatedDataId {
                id: key,
                location: DataLocation::Ram,
            };
            self.new_data.borrow_mut().remove(&data_key);
            Ok(())
        } else {
            Err(())
        }
    }

    pub(crate) fn available_locations(&self, id: DataId) -> Vec<DataLocation> {
        let index = self.state.index.borrow();
        index
            .get(&id)
            .map(|e| {
                let mut res = Vec::new();
                if e.ram
                    .as_ref()
                    .map(|e| match e.state {
                        RamStorageEntryState::Initialized(_) => true,
                        RamStorageEntryState::Registered => false,
                        RamStorageEntryState::Initializing(_) => panic!("This should not happen"),
                    })
                    .unwrap_or(false)
                {
                    res.push(DataLocation::Ram);
                }
                for (i, v) in e.vram.iter().enumerate() {
                    if v.as_ref()
                        .map(|e| match e.state {
                            VRamStorageEntryState::Initialized(_) => true,
                            VRamStorageEntryState::Registered => false,
                            VRamStorageEntryState::Initializing(_) => {
                                panic!("This should not happen")
                            }
                        })
                        .unwrap_or(false)
                    {
                        res.push(DataLocation::VRam(i));
                    }
                }
                res
            })
            .unwrap_or(Vec::new())
    }

    pub(crate) fn newest_data(&self) -> impl Iterator<Item = LocatedDataId> {
        let mut place_holder = BTreeSet::new();
        let mut d = self.new_data.borrow_mut();
        std::mem::swap(&mut *d, &mut place_holder);
        place_holder.into_iter()
    }

    pub fn register_ram_access(&self, id: DataId) -> RamAccessToken {
        {
            let mut index = self.state.index.borrow_mut();
            let entry = index
                .entry(id)
                .or_insert_with(|| self.gen_empty_storage_entry());

            if entry.ram.is_none() {
                entry.ram = Some(RamEntry {
                    state: RamStorageEntryState::Registered,
                    access: RamAccessState::Some(0), // Will be overwritten immediately when generating
                                                     // the RamToken
                });
            }
        }
        RamAccessToken::new(self, id)
    }

    pub fn register_vram_access<'b>(
        &'b self,
        device: &'b DeviceContext,
        id: DataId,
    ) -> VRamAccessToken<'b> {
        {
            let mut index = self.state.index.borrow_mut();
            let entry = index
                .entry(id)
                .or_insert_with(|| self.gen_empty_storage_entry());

            let vram_entry = &mut entry.vram[device.id];
            if vram_entry.is_none() {
                *vram_entry = Some(VRamEntry {
                    state: VRamStorageEntryState::Registered,
                    access: VRamAccessState::Some(0), // Will be overwritten immediately when generating token
                });
            }
        }
        VRamAccessToken::new(self, device, id)
    }

    // Allocates a GpuOnly storage buffer
    fn alloc_vram<'b>(
        &'b self,
        device: &'b DeviceContext,
        key: DataId,
        layout: Layout,
    ) -> Result<(ash::vk::Buffer, VRamAccessToken<'b>), Error> {
        let vram = &self.vram[device.id];
        let allocation = vram.allocate(
            layout,
            ash::vk::BufferUsageFlags::STORAGE_BUFFER
                | ash::vk::BufferUsageFlags::TRANSFER_DST
                | ash::vk::BufferUsageFlags::TRANSFER_SRC,
            crate::vulkan::MemoryLocation::GpuOnly,
        );
        let buffer = allocation.buffer;

        {
            let mut index = self.state.index.borrow_mut();
            let entry = index
                .entry(key)
                .or_insert_with(|| self.gen_empty_storage_entry());

            let info = VRamStorageInfo { allocation, layout };
            let vram_entry = &mut entry.vram[device.id];

            let prev = vram_entry.take();
            let new_entry = VRamEntry {
                state: VRamStorageEntryState::Initializing(info),
                access: prev
                    .map(|p| p.access)
                    .unwrap_or_else(|| VRamAccessState::Some(0)),
            };
            *vram_entry = Some(new_entry);
        }

        Ok((buffer, VRamAccessToken::new(self, device, key)))
    }

    pub fn alloc_vram_slot_raw<'b>(
        &'b self,
        device: &'b DeviceContext,
        key: DataId,
        layout: Layout,
    ) -> Result<VRamWriteHandle<'b>, Error> {
        let size = layout.size();
        let (buffer, access) = self.alloc_vram(device, key, layout)?;

        Ok(VRamWriteHandle {
            buffer,
            size: size as u64,
            access,
            drop_handler: DropErrorVram,
        })
    }

    pub fn alloc_vram_slot<'b, T: Copy + crevice::std430::Std430>(
        &'b self,
        device: &'b DeviceContext,
        key: DataId,
        num: usize,
    ) -> Result<VRamWriteHandle<'b>, Error> {
        //TODO: Not sure if this actually works with std430
        let layout = Layout::array::<T>(num).unwrap();
        self.alloc_vram_slot_raw(device, key, layout)
    }

    pub fn read_vram<'b, 't: 'b>(
        &'b self,
        device: &'b DeviceContext,
        access: VRamAccessToken<'t>,
    ) -> Result<VRamReadHandle<'b>, VRamAccessToken<'t>> {
        let index = self.state.index.borrow();
        let Some(entry) = index.get(&access.id) else {
            return Err(access);
        };
        let Some(vram_entry) = entry.vram[device.id].as_ref() else {
            return Err(access);
        };

        let VRamStorageEntryState::Initialized(info) = &vram_entry.state else {
            return Err(access);
        };

        Ok(VRamReadHandle {
            buffer: info.allocation.buffer,
            layout: info.layout,
            access,
        })
    }

    fn gen_empty_storage_entry(&self) -> StorageEntry {
        StorageEntry {
            ram: None,
            vram: self.vram.iter().map(|_| None).collect(),
        }
    }

    fn alloc_ram(&self, key: DataId, layout: Layout) -> Result<(*mut u8, RamAccessToken), Error> {
        let data = {
            let data = match self.ram.alloc(layout) {
                Ok(d) => d,
                Err(_e) => {
                    // Always try to free half of available ram
                    // TODO: Other solutions may be better.
                    let garbage_collect_goal = self.ram.size() / 2;
                    self.try_garbage_collect(garbage_collect_goal);
                    self.ram.alloc(layout)?
                }
            };

            let mut index = self.state.index.borrow_mut();
            let entry = index
                .entry(key)
                .or_insert_with(|| self.gen_empty_storage_entry());

            let info = RamStorageInfo { data, layout };

            let prev = entry.ram.take();
            let new_entry = RamEntry {
                state: RamStorageEntryState::Initializing(info),
                access: prev
                    .map(|p| p.access)
                    .unwrap_or_else(|| RamAccessState::Some(0)),
            };
            entry.ram = Some(new_entry);
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
            let index = self.state.index.borrow();
            let Some(entry) = index.get(&access.id) else {
                return Err(access);
            };
            let Some(ram_entry) = entry.ram.as_ref() else {
                return Err(access);
            };
            let RamStorageEntryState::Initialized(info) = ram_entry.state else {
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
            let index = self.state.index.borrow();
            let Some(entry) = index.get(&access.id) else {
                return Err(access);
            };
            let Some(ram_entry) = entry.ram.as_ref() else {
                return Err(access);
            };

            let RamStorageEntryState::Initialized(info) = ram_entry.state else {
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

        let mut index = self.state.index.borrow_mut();
        let Some(entry) = index.get(&old_access.id) else {
                return Err(old_access);
            };
        let Some(ram_entry) = entry.ram.as_ref() else {
                return Err(old_access);
            };

        let RamStorageEntryState::Initialized(info) = ram_entry.state else {
            return Err(old_access)
        };

        let num_elements = num_elms_in_array::<T>(info.layout.size());

        let ptr = info.data;
        let t_ptr = ptr.cast::<T>();

        // Only allow inplace if we are EXACTLY the one reader
        let in_place_possible = matches!(ram_entry.access, RamAccessState::Some(1));

        Ok(Ok(if in_place_possible {
            // Repurpose access key for the read/write handle
            let mut new_access = old_access;
            new_access.id = new_key;

            let _old_entry = index.remove(&old_key).unwrap();
            let new_entry = index
                .entry(new_key)
                .or_insert_with(|| self.gen_empty_storage_entry());

            let prev = new_entry.ram.take();
            let access = match prev.map(|v| v.access) {
                Some(RamAccessState::Some(n)) => RamAccessState::Some(n + 1),
                Some(RamAccessState::None(_)) => panic!("If present, entry should have accessors"),
                None => RamAccessState::Some(1),
            };
            let new_ram_entry = RamEntry {
                state: RamStorageEntryState::Initializing(info),
                access,
            };
            new_entry.ram = Some(new_ram_entry);

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
