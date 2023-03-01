use std::{
    alloc::Layout,
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    mem::MaybeUninit,
};

use crate::{operator::DataId, util::num_elms_in_array, vulkan::DeviceId, Error};

#[derive(Debug, Eq, PartialEq)]
enum AccessState {
    ReadBy(usize),
    UnRead(LRUIndex),
}

#[derive(Debug, Eq, PartialEq)]
enum RamStorageEntryState {
    Initializing,
    Initialized(AccessState),
}

#[derive(Debug, Eq, PartialEq)]
enum VRamStorageEntryState {
    Initializing,
    Initialized(AccessState),
}

impl RamStorageEntryState {
    fn initialized(&self) -> bool {
        matches!(self, RamStorageEntryState::Initialized(_))
    }

    fn lru_index(&self) -> Option<LRUIndex> {
        if let RamStorageEntryState::Initialized(AccessState::UnRead(id)) = self {
            Some(*id)
        } else {
            None
        }
    }
}

struct RamEntry {
    state: RamStorageEntryState,
    data: *mut u8,
    size: usize,
}

impl RamEntry {
    fn safe_to_delete(&self) -> bool {
        matches!(
            self.state,
            RamStorageEntryState::Initialized(AccessState::UnRead(_))
        )
    }
}

struct VRamEntry {
    state: VRamStorageEntryState,
    data: crate::vulkan::Allocation,
}

struct StorageEntry {
    ram: Option<RamEntry>,
    vram: Vec<Option<VRamEntry>>,
}

impl StorageEntry {
    fn is_present(&self) -> bool {
        self.ram.is_some() //TODO: Add VRAM presence checks here
    }
}

pub struct ReadHandle<'a, T: ?Sized> {
    storage: &'a Storage<'a>,
    id: DataId,
    data: &'a T,
}
impl<'a, T: ?Sized> ReadHandle<'a, T> {
    fn new(storage: &'a Storage, id: DataId, data: &'a T) -> Self {
        let mut index = storage.state.index.borrow_mut();
        let RamStorageEntryState::Initialized(ref mut access_state) = &mut index.get_mut(&id).unwrap().ram.as_mut().unwrap().state else {
            panic!("Trying to read uninitialized value");
        };
        *access_state = match *access_state {
            AccessState::ReadBy(n) => AccessState::ReadBy(n + 1),
            AccessState::UnRead(id) => {
                storage.state.lru_manager.borrow_mut().remove(id);
                AccessState::ReadBy(1)
            }
        };

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

    pub fn into_thread_handle(self) -> ThreadReadHandle<'a, T>
    where
        T: Send,
    {
        let ret = ThreadReadHandle {
            id: self.id,
            data: self.data,
            panic_handle: Default::default(),
        };
        //Avoid running destructor
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
        let mut index = self.storage.state.index.borrow_mut();

        let RamStorageEntryState::Initialized(ref mut access_state) = &mut index.get_mut(&self.id).unwrap().ram.as_mut().unwrap().state else {
            panic!("Trying to read uninitialized value");
        };

        *access_state = match *access_state {
            AccessState::ReadBy(1) => {
                let lru_id = self.storage.state.lru_manager.borrow_mut().add(self.id);
                AccessState::UnRead(lru_id)
            }
            AccessState::ReadBy(n) => AccessState::ReadBy(n - 1),
            AccessState::UnRead(_id) => {
                panic!("Invalid state");
            }
        };
    }
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
            storage,
            id: self.id,
            data: self.data,
        }
    }
}

pub struct DropError<'a> {
    storage: &'a Storage<'a>,
    id: DataId,
}
impl Drop for DropError<'_> {
    fn drop(&mut self) {
        panic!("The WriteHandle was not marked initialized!");
    }
}
pub struct DropMarkInitialized<'a> {
    storage: &'a Storage<'a>,
    id: DataId,
}
impl Drop for DropMarkInitialized<'_> {
    fn drop(&mut self) {
        self.storage.new_data.borrow_mut().insert(self.id);
        let mut binding = self.storage.state.index.borrow_mut();
        let state = &mut binding
            .get_mut(&self.id)
            .unwrap()
            .ram
            .as_mut()
            .unwrap()
            .state;
        assert!(!state.initialized());
        let lru_id = self.storage.state.lru_manager.borrow_mut().add(self.id);
        *state = RamStorageEntryState::Initialized(AccessState::UnRead(lru_id));
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
                storage,
                id: self.id,
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
                storage,
                id: self.id,
            },
            data: self.data,
        }
    }
}

pub type WriteHandleUninit<'a, T> = WriteHandle<'a, T, DropError<'a>>;
pub type ThreadWriteHandleUninit<'a, T> = ThreadWriteHandle<'a, T, ThreadMarkerUninitialized>;
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

        let id = drop_handler.id;
        let storage = drop_handler.storage;
        // Avoid running destructor which would panic
        std::mem::forget(drop_handler);

        WriteHandle {
            drop_handler: DropMarkInitialized { id, storage },
            data,
        }
    }
    pub fn into_thread_handle(self) -> ThreadWriteHandleUninit<'a, T>
    where
        T: Send,
    {
        let id = self.drop_handler.id;
        std::mem::forget(self.drop_handler);
        ThreadWriteHandle {
            id,
            data: self.data,
            _marker: ThreadMarkerUninitialized,
            _panic_handle: Default::default(),
        }
    }
}
pub type WriteHandleInit<'a, T> = WriteHandle<'a, T, DropMarkInitialized<'a>>;
pub type ThreadWriteHandleInit<'a, T> = ThreadWriteHandle<'a, T, ThreadMarkerInitialized>;
impl<'a, T: ?Sized> WriteHandleInit<'a, T> {
    pub fn new(storage: &'a Storage, id: DataId, data: &'a mut T) -> Self {
        WriteHandle {
            drop_handler: DropMarkInitialized { id, storage },
            data,
        }
    }
    pub fn into_thread_handle(self) -> ThreadWriteHandleInit<'a, T>
    where
        T: Send,
    {
        let id = self.drop_handler.id;
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

type LRUIndex = u64;

#[derive(Default)]
struct LRUManager {
    list: BTreeMap<LRUIndex, DataId>,
    current: LRUIndex,
}

impl LRUManager {
    fn new() -> Self {
        LRUManager {
            list: BTreeMap::new(),
            current: 0,
        }
    }

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
    new_data: RefCell<BTreeSet<DataId>>,
    ram: &'a crate::ram_allocator::Allocator,
    vram: Vec<&'a crate::vulkan::Allocator>,
}

impl<'a> Storage<'a> {
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

    pub fn try_garbage_collect(&self, mut goal_in_bytes: usize) {
        let mut lru = self.state.lru_manager.borrow_mut();
        let mut index = self.state.index.borrow_mut();
        let mut collected = 0;
        for key in lru.drain_lru().into_iter() {
            let entry = index.get_mut(&key).unwrap();
            let ram_entry = entry.ram.take().unwrap();
            let size = ram_entry.size;
            if !entry.is_present() {
                index.remove(&key).unwrap();
            }
            // Safety: all data ptrs in the index have been allocated with the allocator.
            // Deallocation only happens exactly here where the entry is also removed from the
            // index.
            unsafe { self.ram.dealloc(ram_entry.data) };
            self.new_data.borrow_mut().remove(&key);

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
            if !entry.is_present() {
                index.remove(&key).unwrap();
            }
            // Safety: all data ptrs in the index have been allocated with the allocator.
            // Deallocation only happens exactly here where the entry is also removed from the
            // index.
            unsafe { self.ram.dealloc(ram_entry.data) };
            self.new_data.borrow_mut().remove(&key);
            Ok(())
        } else {
            Err(())
        }
    }
    pub(crate) fn newest_data(&self) -> impl Iterator<Item = DataId> {
        let mut place_holder = BTreeSet::new();
        let mut d = self.new_data.borrow_mut();
        std::mem::swap(&mut *d, &mut place_holder);
        place_holder.into_iter()
    }

    // Allocates a GpuOnly storage buffer
    fn alloc_vram(
        &self,
        device: DeviceId,
        key: DataId,
        layout: Layout,
    ) -> Result<ash::vk::Buffer, Error> {
        let vram = &self.vram[device];
        let allocation = vram.allocate(
            layout,
            ash::vk::BufferUsageFlags::STORAGE_BUFFER,
            crate::vulkan::MemoryLocation::GpuOnly,
        );
        // TODO garbage collection

        let mut index = self.state.index.borrow_mut();
        let entry = index
            .entry(key)
            .or_insert_with(|| self.gen_empty_storage_entry());

        let buffer = allocation.buffer;
        let prev = entry.vram[device].replace(VRamEntry {
            state: VRamStorageEntryState::Initializing,
            data: allocation,
        });
        assert!(prev.is_none());

        Ok(buffer)
    }

    fn gen_empty_storage_entry(&self) -> StorageEntry {
        StorageEntry {
            ram: None,
            vram: self.vram.iter().map(|_| None).collect(),
        }
    }

    fn alloc_ram(&self, key: DataId, layout: Layout) -> Result<*mut u8, Error> {
        let data = match self.ram.alloc(layout) {
            Ok(d) => d,
            Err(_e) => {
                let garbage_collect_goal = 1 << 30; //One gigabyte for now maybe???
                self.try_garbage_collect(garbage_collect_goal);
                self.ram.alloc(layout)?
            }
        };

        let mut index = self.state.index.borrow_mut();
        let entry = index
            .entry(key)
            .or_insert_with(|| self.gen_empty_storage_entry());

        let prev = entry.ram.replace(RamEntry {
            state: RamStorageEntryState::Initializing,
            data,
            size: layout.size(),
        });
        assert!(prev.is_none());

        Ok(data)
    }

    pub fn alloc_ram_slot<T: Copy>(
        &self,
        key: DataId,
        size: usize,
    ) -> Result<WriteHandleUninit<[MaybeUninit<T>]>, Error> {
        let layout = Layout::array::<T>(size).unwrap();
        let ptr = self.alloc_ram(key, layout)?;

        let t_ptr = ptr.cast::<MaybeUninit<T>>();

        // Safety: We constructed the pointer with the required layout
        let t_ref = unsafe { std::slice::from_raw_parts_mut(t_ptr, size) };
        Ok(WriteHandleUninit::new(self, key, t_ref))
    }

    /// Safety: The initial allocation for the TaskId must have happened with the same type
    pub unsafe fn read_ram<'b, T: Copy>(&'b self, key: DataId) -> Option<ReadHandle<'b, [T]>> {
        let t_ref = {
            let index = self.state.index.borrow();
            let entry = index.get(&key)?;
            let ram_entry = entry.ram.as_ref()?;

            if !ram_entry.state.initialized() {
                return None;
            }

            let ptr = ram_entry.data;
            let t_ptr = ptr.cast::<T>();

            let num_elements = num_elms_in_array::<T>(ram_entry.size);

            // Safety: Type matches as per contract upheld by caller. There are also no mutable
            // references to the slot since it has already been initialized.
            unsafe { std::slice::from_raw_parts(t_ptr, num_elements) }
        };
        Some(ReadHandle::new(self, key, t_ref))
    }

    /// Safety: The initial allocation for the TaskId must have happened with the same type and the
    /// size must match the initial allocation
    pub unsafe fn try_update_inplace<'b, T: Copy>(
        &'b self,
        old_key: DataId,
        new_key: DataId,
    ) -> Option<Result<InplaceResult<'b, T>, Error>> {
        let mut index = self.state.index.borrow_mut();
        let entry = index.get(&old_key)?;
        let ram_entry = entry.ram.as_ref()?;
        assert!(ram_entry.state.initialized());

        let num_elements = num_elms_in_array::<T>(ram_entry.size);

        let ptr = ram_entry.data;
        let t_ptr = ptr.cast::<T>();

        let in_place_possible = ram_entry.safe_to_delete();
        Some(Ok(if in_place_possible {
            let mut entry = index.remove(&old_key).unwrap();
            let ram_entry = entry.ram.as_mut().unwrap();
            if let Some(lru_index) = ram_entry.state.lru_index() {
                self.state.lru_manager.borrow_mut().remove(lru_index);
            }
            ram_entry.state = RamStorageEntryState::Initializing;

            let prev = index.insert(new_key, entry);
            assert!(prev.is_none());

            // Safety: Type matches as per contract upheld by caller. There are also no other
            // references to the slot since it has (1.) been already initialized and (2.) there are
            // no readers. In other words: safe_to_delete also implies no other references.
            let t_ref = unsafe { std::slice::from_raw_parts_mut(t_ptr, num_elements) };

            InplaceResult::Inplace(WriteHandleInit::new(self, new_key, t_ref))
        } else {
            std::mem::drop(index); // Release borrow for alloc

            // Safety: Type matches as per contract upheld by caller. There are also no mutable
            // references to the slot since it has already been initialized.
            let t_ref = unsafe { std::slice::from_raw_parts(t_ptr, num_elements) };

            let w = match self.alloc_ram_slot(new_key, num_elements) {
                Ok(w) => w,
                Err(e) => return Some(Err(e)),
            };
            let r = ReadHandle::new(self, old_key, t_ref);
            InplaceResult::New(r, w)
        }))
    }
}
