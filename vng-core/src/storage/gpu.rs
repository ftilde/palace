use ahash::HashMapExt;
use std::{
    alloc::Layout,
    cell::{Cell, RefCell},
    collections::VecDeque,
    mem::MaybeUninit,
};

use super::{
    DataLocation, DataLongevity, DataVersion, DataVersionType, Element, LRUIndex, LRUIndexInner,
    LRUManager, LRUManagerInner,
};

use ash::vk;
use gpu_allocator::vulkan::AllocationScheme;

use crate::{
    operator::{DataDescriptor, DataId, OperatorDescriptor, OperatorId},
    runtime::FrameNumber,
    task::OpaqueTaskContext,
    util::Map,
    vulkan::{
        state::VulkanState, CmdBufferEpoch, CommandBuffer, DeviceContext, DeviceId, DstBarrierInfo,
        SrcBarrierInfo,
    },
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BarrierEpoch(usize);

pub(crate) struct BarrierManager {
    current_epoch: Cell<usize>,
    last_issued_ending: RefCell<Map<(SrcBarrierInfo, DstBarrierInfo), BarrierEpoch>>,
}

impl BarrierManager {
    fn new() -> Self {
        Self {
            current_epoch: Cell::new(0),
            last_issued_ending: RefCell::new(Map::new()),
        }
    }

    pub(crate) fn issue(
        &self,
        cmd: &mut CommandBuffer,
        src: SrcBarrierInfo,
        dst: DstBarrierInfo,
    ) -> BarrierEpoch {
        let ending = self.current_epoch.get();
        let starting = ending + 1;
        self.current_epoch.set(starting);
        let ending = BarrierEpoch(ending);

        let memory_barriers = &[vk::MemoryBarrier2::builder()
            .src_stage_mask(src.stage)
            .src_access_mask(src.access)
            .dst_stage_mask(dst.stage)
            .dst_access_mask(dst.access)
            .build()];
        let barrier_info = vk::DependencyInfo::builder().memory_barriers(memory_barriers);

        unsafe {
            let cmd_raw = cmd.raw();
            cmd.functions()
                .cmd_pipeline_barrier2(cmd_raw, &barrier_info);
        }

        let mut last_issued_ending = self.last_issued_ending.borrow_mut();
        last_issued_ending.insert((src, dst), ending);

        ending
    }

    pub(crate) fn current_epoch(&self) -> BarrierEpoch {
        BarrierEpoch(self.current_epoch.get())
    }

    pub(crate) fn is_visible(
        &self,
        src: SrcBarrierInfo,
        dst: DstBarrierInfo,
        created: BarrierEpoch,
    ) -> bool {
        self.last_issued_ending
            .borrow()
            .get(&(src, dst))
            .map(|e| e >= &created)
            .unwrap_or(false)
    }
}

#[derive(Debug, Eq, PartialEq)]
enum AccessState {
    Some(usize),
    None(Option<LRUIndex>, CmdBufferEpoch),
}

#[derive(Debug)]
struct Visibility {
    src: SrcBarrierInfo,
    created: BarrierEpoch,
}

#[derive(Debug)]
enum StorageEntryState {
    Registered,
    Initializing(StorageInfo),
    Initialized(StorageInfo, Visibility, DataVersion),
}

impl StorageEntryState {
    fn storage_info(&self) -> Option<&StorageInfo> {
        if let StorageEntryState::Initializing(info) | StorageEntryState::Initialized(info, _, _) =
            &self
        {
            Some(&info)
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct StorageInfo {
    pub allocation: Allocation,
    pub layout: Layout,
    data_longevity: DataLongevity,
}

struct Entry {
    state: StorageEntryState,
    access: AccessState,
}

struct StateCacheEntry {
    storage: StorageInfo,
    access: AccessState,
}

pub struct StateCacheAccessToken<'a> {
    storage: &'a Storage,
    device: &'a DeviceContext,
    pub id: DataId,
}

//TODO: See if we can deduplicate the code wrt the "normal" AccessToken
impl<'a> StateCacheAccessToken<'a> {
    fn new(storage: &'a Storage, device: &'a DeviceContext, id: DataId) -> Self {
        let mut index = storage.state_cache_index.borrow_mut();
        let entry = index.get_mut(&id).unwrap();

        // We expect there to ever only be exactly one simulatenous access to a state cache item.
        // If this is not the case, something has seriously gone wrong.
        assert!(matches!(
            entry.access,
            AccessState::None(..) | AccessState::Some(0)
        ));

        inc_access(&mut entry.access, storage);

        Self {
            storage,
            id,
            device,
        }
    }
}
impl Drop for StateCacheAccessToken<'_> {
    fn drop(&mut self) {
        let mut index = self.storage.state_cache_index.borrow_mut();
        let vram_entry = index.get_mut(&self.id).unwrap();
        let longevity = vram_entry.storage.data_longevity;

        // We expect there to ever only be exactly one simulatenous access to a state cache item.
        // If this is not the case, something has seriously gone wrong.
        assert!(matches!(vram_entry.access, AccessState::Some(1)));

        let mut lru_manager = self.device.storage.lru_manager.borrow_mut();
        dec_access(
            &mut vram_entry.access,
            &mut lru_manager,
            self.device,
            LRUItem::State(self.id),
            Some(longevity),
        );
    }
}

pub struct AccessToken<'a> {
    storage: &'a Storage,
    device: &'a DeviceContext,
    pub id: DataId,
}

impl std::fmt::Debug for AccessToken<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AccessIntentToken {{ {:?} }}", self.id)
    }
}
impl<'a> AccessToken<'a> {
    fn new(storage: &'a Storage, device: &'a DeviceContext, id: DataId) -> Self {
        let mut index = storage.data_index.borrow_mut();
        let vram_entry = index.get_mut(&id).unwrap();

        inc_access(&mut vram_entry.access, storage);

        Self {
            storage,
            id,
            device,
        }
    }
}

fn inc_access(access: &mut AccessState, storage: &Storage) {
    *access = match *access {
        AccessState::Some(n) => AccessState::Some(n + 1),
        AccessState::None(id, _) => {
            if let Some(id) = id {
                storage.lru_manager.borrow_mut().remove(id);
            }
            AccessState::Some(1)
        }
    };
}
fn dec_access<T: Clone>(
    access: &mut AccessState,
    lru_manager: &mut LRUManager<T>,
    device: &DeviceContext,
    id: T,
    longevity: Option<DataLongevity>,
) {
    *access = match *access {
        AccessState::Some(1) => {
            let lru_id = if let Some(longevity) = longevity {
                Some(lru_manager.add(id, longevity))
            } else {
                None
            };
            AccessState::None(lru_id, device.current_epoch())
        }
        AccessState::Some(n) => AccessState::Some(n - 1),
        AccessState::None(..) => {
            panic!("Invalid state");
        }
    };
}
impl Drop for AccessToken<'_> {
    fn drop(&mut self) {
        let mut index = self.storage.data_index.borrow_mut();
        let vram_entry = index.get_mut(&self.id).unwrap();

        let longevity = vram_entry.state.storage_info().map(|i| i.data_longevity);

        let mut lru_manager = self.storage.lru_manager.borrow_mut();
        dec_access(
            &mut vram_entry.access,
            &mut lru_manager,
            self.device,
            LRUItem::Data(self.id),
            longevity,
        );
    }
}

pub struct WriteHandle<'a> {
    pub buffer: ash::vk::Buffer,
    pub size: u64,
    drop_handler: DropError,
    access: AccessToken<'a>,
}

pub struct DropError;
impl Drop for DropError {
    fn drop(&mut self) {
        panic!("The WriteHandle was not marked initialized!");
    }
}

impl<'a> WriteHandle<'a> {
    /// Safety: The corresponding slot has to have been completely written to.
    pub unsafe fn initialized<'c, 'b>(self, ctx: OpaqueTaskContext<'c, 'b>, src: SrcBarrierInfo) {
        let version = if ctx
            .predicted_preview_tasks
            .borrow()
            .contains(&ctx.current_task)
        {
            DataVersion::Preview(ctx.current_frame)
        } else {
            DataVersion::Final
        };
        self.initialized_with_version(src, version)
    }
    pub unsafe fn initialized_version<'c, 'b>(
        self,
        ctx: OpaqueTaskContext<'c, 'b>,
        src: SrcBarrierInfo,
        version: DataVersionType,
    ) {
        let version = match version {
            DataVersionType::Final => DataVersion::Final,
            DataVersionType::Preview => DataVersion::Preview(ctx.current_frame),
        };
        self.initialized_with_version(src, version)
    }
    unsafe fn initialized_with_version(self, src: SrcBarrierInfo, version: DataVersion) {
        let WriteHandle {
            access,
            drop_handler,
            ..
        } = self;

        // Avoid running destructor which would panic
        std::mem::forget(drop_handler);

        // Mark as initialized
        let mut binding = access.storage.data_index.borrow_mut();

        {
            let entry = &mut binding.get_mut(&access.id).unwrap();

            entry.state = match std::mem::replace(&mut entry.state, StorageEntryState::Registered) {
                StorageEntryState::Registered => {
                    panic!("Entry should be in state Initializing, but is in Registered");
                }
                StorageEntryState::Initialized(..) => {
                    panic!("Entry should be in state Initializing, but is in Initialized");
                }
                StorageEntryState::Initializing(info) => {
                    access.storage.new_data.add(access.id, version.type_());
                    StorageEntryState::Initialized(
                        info,
                        Visibility {
                            src,
                            created: access.storage.barrier_manager.current_epoch(),
                        },
                        version,
                    )
                }
            };
        }
    }
}

#[derive(Debug)]
pub struct ReadHandle<'a> {
    pub buffer: ash::vk::Buffer,
    pub layout: Layout,
    #[allow(unused)]
    access: AccessToken<'a>,
    pub version: DataVersionType,
    pub data_longevity: DataLongevity,
}

pub struct StateCacheHandle<'a> {
    pub buffer: ash::vk::Buffer,
    pub size: u64,
    #[allow(unused)]
    access: StateCacheAccessToken<'a>,
}

pub enum StateCacheResult<'a> {
    New(StateCacheHandle<'a>),
    Existing(StateCacheHandle<'a>),
}

impl<'a> StateCacheResult<'a> {
    pub fn init(self, f: impl FnOnce(&mut StateCacheHandle)) -> StateCacheHandle<'a> {
        match self {
            StateCacheResult::New(mut n) => {
                f(&mut n);
                n
            }
            StateCacheResult::Existing(n) => n,
        }
    }
    pub fn unpack(self) -> StateCacheHandle<'a> {
        match self {
            StateCacheResult::New(n) => n,
            StateCacheResult::Existing(n) => n,
        }
    }
}

pub enum InplaceResult<'a> {
    Inplace(WriteHandle<'a>, DataVersionType),
    New(ReadHandle<'a>, WriteHandle<'a>),
}
impl<'a> InplaceResult<'a> {
    /// Safety: The corresponding slot (either rw inplace or separate w) has to have been
    /// completely written to.
    pub unsafe fn initialized<'b, 'c>(self, ctx: OpaqueTaskContext<'b, 'c>, info: SrcBarrierInfo) {
        match self {
            InplaceResult::Inplace(rw, _v) => unsafe { rw.initialized(ctx, info) },
            InplaceResult::New(_r, w) => unsafe { w.initialized(ctx, info) },
        };
    }
}

struct IndexBrickRef {
    id: DataId,
    acquire_lru_index: LRUIndexInner,
}

pub struct IndexEntry {
    storage: StorageInfo,
    visibility: Visibility,
    present: Map<u64, IndexBrickRef>,
    access: AccessState,
}

// Safety: The caller is responsible for removing the reference from the index in gpu memory
// (i.e. self.storage.allocation.buffer)
unsafe fn unref_brick_in_index(
    device: &DeviceContext,
    brick_index: &mut Map<DataId, Entry>,
    index_lru: &mut LRUManagerInner<(OperatorId, u64, DataId)>,
    brick_lru: &mut LRUManager<LRUItem>,
    brick_ref: IndexBrickRef,
) {
    index_lru.remove(brick_ref.acquire_lru_index);

    let d_entry = brick_index.get_mut(&brick_ref.id).unwrap();
    let longevity = d_entry.state.storage_info().map(|i| i.data_longevity);

    dec_access(
        &mut d_entry.access,
        brick_lru,
        device,
        LRUItem::Data(brick_ref.id),
        longevity,
    );
}

impl IndexEntry {
    fn new(storage: StorageInfo, visibility: Visibility) -> Self {
        Self {
            storage,
            visibility,
            present: Default::default(),
            access: AccessState::Some(0),
        }
    }

    fn release(
        &mut self,
        device: &DeviceContext,
        brick_index: &mut Map<DataId, Entry>,
        index_lru: &mut LRUManagerInner<(OperatorId, u64, DataId)>,
        brick_lru: &mut LRUManager<LRUItem>,
        pos: u64,
    ) {
        device.with_cmd_buffer(|cmd| unsafe {
            let addr: ash::vk::DeviceAddress = 0u64;
            let offset = pos * std::mem::size_of::<ash::vk::DeviceAddress>() as u64;

            device.functions().cmd_update_buffer(
                cmd.raw(),
                self.storage.allocation.buffer,
                offset,
                bytemuck::bytes_of(&addr),
            );
        });

        let brick_ref = self.present.remove(&pos).unwrap();

        // Safety: We have also just removed the reference from the index
        unsafe { unref_brick_in_index(device, brick_index, index_lru, brick_lru, brick_ref) };
    }
}

pub struct IndexHandle<'a> {
    pub(crate) buffer: ash::vk::Buffer,
    pub(crate) num_chunks: usize,
    op: OperatorId,
    device: &'a DeviceContext,
}

impl<'a> IndexHandle<'a> {
    pub fn insert<'h>(&self, pos: u64, brick: ReadHandle<'h>) {
        let mut index = self.device.storage.index_index.borrow_mut();
        let entry = index.get_mut(&self.op).unwrap();

        if entry.present.contains_key(&pos) {
            return;
        }

        let brick_buffer = brick.buffer;

        // We are somewhat fine with racing here, but not sure how to convince the validation
        // layers of this...
        self.device.with_cmd_buffer(|cmd| unsafe {
            let info = ash::vk::BufferDeviceAddressInfo::builder().buffer(brick_buffer);
            let addr = self.device.functions().get_buffer_device_address(&info);
            let offset = pos * std::mem::size_of::<ash::vk::DeviceAddress>() as u64;

            assert!(offset < entry.storage.layout.size() as u64);
            self.device.functions().cmd_update_buffer(
                cmd.raw(),
                self.buffer,
                offset,
                bytemuck::bytes_of(&addr),
            );
        });
        let data_id = brick.access.id;

        let lru_index = self
            .device
            .storage
            .index_lru
            .borrow_mut()
            .add((self.op, pos, data_id));

        // Leak ReadHandle: For now we don't ever remove chunks once they are in the index.
        std::mem::forget(brick);

        entry.present.insert(
            pos,
            IndexBrickRef {
                id: data_id,
                acquire_lru_index: lru_index,
            },
        );
    }
}
impl<'a> Drop for IndexHandle<'a> {
    fn drop(&mut self) {
        let mut index = self.device.storage.index_index.borrow_mut();
        let vram_entry = index.get_mut(&self.op).unwrap();

        let longevity = vram_entry.storage.data_longevity;

        let mut lru_manager = self.device.storage.lru_manager.borrow_mut();
        dec_access(
            &mut vram_entry.access,
            &mut lru_manager,
            self.device,
            LRUItem::Index(self.op),
            Some(longevity),
        );
    }
}

#[derive(Copy, Clone)]
enum LRUItem {
    Data(DataId),
    State(DataId),
    Index(OperatorId),
}

pub struct Storage {
    data_index: RefCell<Map<DataId, Entry>>,
    state_cache_index: RefCell<Map<DataId, StateCacheEntry>>,
    index_index: RefCell<Map<OperatorId, IndexEntry>>,
    old_unused: RefCell<VecDeque<(StorageInfo, CmdBufferEpoch)>>,
    lru_manager: RefCell<super::LRUManager<LRUItem>>,
    index_lru: RefCell<super::LRUManagerInner<(OperatorId, u64, DataId)>>,
    pub(crate) barrier_manager: BarrierManager,
    allocator: Allocator,
    new_data: super::NewDataManager,
    id: DeviceId,
}

impl Storage {
    pub fn new(device: DeviceId, allocator: Allocator) -> Self {
        Self {
            data_index: Default::default(),
            state_cache_index: Default::default(),
            index_index: Default::default(),
            old_unused: Default::default(),
            lru_manager: Default::default(),
            index_lru: Default::default(),
            barrier_manager: BarrierManager::new(),
            allocator,
            new_data: Default::default(),
            id: device,
        }
    }

    /// Safety: Danger zone: The entries cannot be in use anymore! No checking for dangling
    /// references is done!
    pub unsafe fn free_vram(&self) {
        for (info, _) in std::mem::take(&mut *self.old_unused.borrow_mut()) {
            self.allocator.deallocate(info.allocation);
        }

        for (_, entry) in std::mem::take(&mut *self.state_cache_index.borrow_mut()) {
            self.allocator.deallocate(entry.storage.allocation);
        }

        for (_, entry) in std::mem::take(&mut *self.index_index.borrow_mut()) {
            self.allocator.deallocate(entry.storage.allocation);
        }

        for (_, entry) in std::mem::take(&mut *self.data_index.borrow_mut()) {
            match entry.state {
                StorageEntryState::Registered => {}
                StorageEntryState::Initializing(info)
                | StorageEntryState::Initialized(info, _, _) => {
                    self.allocator.deallocate(info.allocation);
                }
            }
        }
    }

    pub fn try_garbage_collect(&self, device: &DeviceContext, mut goal_in_bytes: usize) {
        let mut collected = 0;

        let mut unused = self.old_unused.borrow_mut();
        while unused
            .front()
            .map(|(_, d)| device.cmd_buffer_completed(*d))
            .unwrap_or(false)
        {
            let (info, _) = unused.pop_front().unwrap();
            unsafe { self.allocator.deallocate(info.allocation) };

            collected += info.layout.size();
        }

        let mut lru = self.lru_manager.borrow_mut();
        let mut index_lru = self.index_lru.borrow_mut();
        let mut index = self.data_index.borrow_mut();
        let mut state_cache_index = self.state_cache_index.borrow_mut();
        let mut index_index = self.index_index.borrow_mut();

        let mut indices_to_unref = Vec::new();
        for (longevity, inner_lru) in lru.inner_mut() {
            let mut collected_local = 0;

            if goal_in_bytes == 0 {
                break;
            };

            while let Some(key) = inner_lru.get_next() {
                let info = match key {
                    LRUItem::Data(key) => {
                        let entry = index.get_mut(&key).unwrap();
                        let AccessState::None(_, f) = entry.access else {
                            panic!("Should not be in LRU list");
                        };
                        if !device.cmd_buffer_completed(f) {
                            // All following LRU items will have the same or a later epoch so cannot be deleted
                            // either
                            break;
                        }

                        let entry = index.remove(&key).unwrap();

                        self.new_data.remove(key);
                        match entry.state {
                            StorageEntryState::Registered => panic!("Should not be in LRU list"),
                            StorageEntryState::Initializing(info)
                            | StorageEntryState::Initialized(info, _, _) => info,
                        }
                    }
                    LRUItem::State(key) => {
                        let entry = state_cache_index.get_mut(&key).unwrap();
                        let AccessState::None(_, f) = entry.access else {
                            panic!("Should not be in LRU list");
                        };
                        if !device.cmd_buffer_completed(f) {
                            // All following LRU items will have the same or a later epoch so cannot be deleted
                            // either
                            break;
                        }

                        let entry = state_cache_index.remove(&key).unwrap();

                        self.new_data.remove(key);
                        entry.storage
                    }
                    LRUItem::Index(key) => {
                        let entry = index_index.get_mut(&key).unwrap();
                        let AccessState::None(_, f) = entry.access else {
                            panic!("Should not be in LRU list");
                        };

                        if !device.cmd_buffer_completed(f) {
                            // All following LRU items will have the same or a later epoch so cannot be deleted
                            // either
                            break;
                        }

                        let entry = index_index.remove(&key).unwrap();

                        indices_to_unref.push(entry.present);

                        entry.storage
                    }
                };

                // Safety: All allocations in the index have been allocated with the allocator.
                // Deallocation only happens exactly here where the entry is also removed from the
                // index. The allocation is also not used on the gpu anymore since the last access
                // epoch has already passed.
                unsafe { self.allocator.deallocate(info.allocation) };

                inner_lru.pop_next();

                let size = info.layout.size();
                collected += size;
                collected_local += size;
                goal_in_bytes = goal_in_bytes.saturating_sub(size);
                if goal_in_bytes == 0 && longevity != DataLongevity::Ephemeral {
                    break;
                };
            }
            println!("Garbage collect GPU ({:?}): {}", longevity, collected_local);
        }
        for brick_map in indices_to_unref {
            for (_pos, brick_ref) in brick_map {
                unsafe {
                    unref_brick_in_index(device, &mut index, &mut index_lru, &mut lru, brick_ref)
                };
            }
        }

        let mut unindexed = 0;
        if goal_in_bytes > 0 {
            while let Some((op, pos, data_id)) = index_lru.get_next() {
                let brick_index = index_index.get_mut(&op).unwrap();

                // Releasing the brick also removes it from the index_lru queue
                brick_index.release(device, &mut index, &mut *index_lru, &mut lru, pos);

                let brick_entry = index.get(&data_id).unwrap();
                let brick_info = match &brick_entry.state {
                    StorageEntryState::Registered | StorageEntryState::Initializing(_) => {
                        panic!("Indexed brick should be initialized")
                    }
                    StorageEntryState::Initialized(info, _, _) => info,
                };
                let size = brick_info.layout.size();

                unindexed += size;
                let Some(rest) = goal_in_bytes.checked_sub(size) else {
                    break;
                };
                goal_in_bytes = rest;
            }
        }

        println!(
            "Garbage collect GPU: {}B | Unindexed: {}B",
            collected, unindexed
        );
    }

    pub fn is_readable(&self, id: DataId) -> bool {
        self.data_index
            .borrow()
            .get(&id)
            .map(|e| matches!(e.state, StorageEntryState::Initialized(_, _, _)))
            .unwrap_or(false)
    }

    pub(crate) fn newest_data(
        &self,
    ) -> impl Iterator<Item = (DataId, DataLocation, DataVersionType)> {
        let id = self.id;
        self.new_data
            .drain()
            .map(move |(d, v)| (d, DataLocation::VRam(id), v))
    }

    fn ensure_presence<'a>(
        &self,
        current_frame: FrameNumber,
        entry: crate::util::MapEntry<'a, DataId, Entry>,
    ) -> &'a mut Entry {
        match entry {
            crate::util::MapEntry::Occupied(mut e) => {
                if let StorageEntryState::Initialized(_, _, version) = e.get().state {
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
                            AccessState::None(lru_index, epoch) => {
                                let StorageEntryState::Initialized(info, _, _) = old.state else {
                                    panic!("we just checked that");
                                };
                                if let Some(lru_index) = lru_index {
                                    self.lru_manager.borrow_mut().remove(lru_index);
                                }
                                let mut old_unused = self.old_unused.borrow_mut();
                                old_unused.push_back((info, epoch));
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

    pub fn register_access<'b>(
        &'b self,
        device: &'b DeviceContext,
        current_frame: FrameNumber,
        id: DataId,
    ) -> AccessToken<'b> {
        {
            let mut index = self.data_index.borrow_mut();
            let entry = index.entry(id);
            self.ensure_presence(current_frame, entry);
        }
        AccessToken::new(self, device, id)
    }

    pub fn access_initializing<'a>(
        &self,
        access: AccessToken<'a>,
    ) -> Result<WriteHandle<'a>, AccessToken<'a>> {
        let index = self.data_index.borrow_mut();
        let entry = index.get(&access.id).unwrap();

        if let StorageEntryState::Initializing(info) = &entry.state {
            Ok(WriteHandle {
                buffer: info.allocation.buffer,
                size: info.allocation.size,
                drop_handler: DropError,
                access,
            })
        } else {
            Err(access)
        }
    }

    pub fn allocate(
        &self,
        device: &DeviceContext,
        layout: Layout,
        use_flags: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Allocation {
        match self.allocator.allocate(layout, use_flags, location) {
            Ok(a) => a,
            Err(_e) => {
                let garbage_collect_goal =
                    self.allocator.allocated() / super::GARBAGE_COLLECT_GOAL_FRACTION;
                self.try_garbage_collect(device, garbage_collect_goal as _);
                self.allocator
                    .allocate(layout, use_flags, location)
                    .expect("Out of memory and nothing we can do about it.")
            }
        }
    }

    /// Safety: Allocation must come from this storage
    pub unsafe fn deallocate(&self, allocation: Allocation) {
        self.allocator.deallocate(allocation);
    }

    pub fn allocate_image(
        &self,
        device: &DeviceContext,
        create_desc: vk::ImageCreateInfo,
    ) -> ImageAllocation {
        match self.allocator.allocate_image(create_desc) {
            Ok(a) => a,
            Err(_e) => {
                let garbage_collect_goal =
                    self.allocator.allocated() / super::GARBAGE_COLLECT_GOAL_FRACTION;
                self.try_garbage_collect(device, garbage_collect_goal as _);
                self.allocator
                    .allocate_image(create_desc)
                    .expect("Out of memory and nothing we can do about it.")
            }
        }
    }

    /// Safety: Allocation must come from this storage
    pub unsafe fn deallocate_image(&self, allocation: ImageAllocation) {
        self.allocator.deallocate_image(allocation);
    }

    fn alloc_ssbo<'b>(&'b self, device: &'b DeviceContext, layout: Layout) -> Allocation {
        let flags = ash::vk::BufferUsageFlags::STORAGE_BUFFER
            | ash::vk::BufferUsageFlags::TRANSFER_DST
            | ash::vk::BufferUsageFlags::TRANSFER_SRC;
        let location = MemoryLocation::GpuOnly;
        self.allocate(device, layout, flags, location)
    }

    pub(crate) fn alloc_and_register_ssbo<'b>(
        &'b self,
        device: &'b DeviceContext,
        current_frame: FrameNumber,
        desc: DataDescriptor,
        layout: Layout,
    ) -> (ash::vk::Buffer, AccessToken<'b>) {
        let allocation = self.alloc_ssbo(device, layout);
        let key = desc.id;

        let buffer = allocation.buffer;

        {
            let mut index = self.data_index.borrow_mut();
            let entry = self.ensure_presence(current_frame, index.entry(key));

            let info = StorageInfo {
                allocation,
                layout,
                data_longevity: desc.longevity,
            };

            assert!(
                matches!(entry.state, StorageEntryState::Registered),
                "State should be registered, but is {:?}",
                entry.state
            );

            entry.state = StorageEntryState::Initializing(info);
        }

        (buffer, AccessToken::new(self, device, key))
    }

    pub fn alloc_slot_raw<'b>(
        &'b self,
        device: &'b DeviceContext,
        current_frame: FrameNumber,
        desc: DataDescriptor,
        layout: Layout,
    ) -> WriteHandle<'b> {
        let size = layout.size();
        let (buffer, access) = self.alloc_and_register_ssbo(device, current_frame, desc, layout);

        WriteHandle {
            buffer,
            size: size as u64,
            access,
            drop_handler: DropError,
        }
    }

    pub fn alloc_slot<'b, T: Element>(
        &'b self,
        device: &'b DeviceContext,
        current_frame: FrameNumber,
        desc: DataDescriptor,
        num: usize,
    ) -> WriteHandle<'b> {
        let layout = Layout::array::<T>(num).unwrap();
        self.alloc_slot_raw(device, current_frame, desc, layout)
    }

    pub fn is_visible(&self, id: DataId, dst_info: DstBarrierInfo) -> Result<(), SrcBarrierInfo> {
        let index = self.data_index.borrow();
        let Some(entry) = index.get(&id) else {
            panic!("Should only be called on present, initialized data");
        };
        let StorageEntryState::Initialized(_info, visibility, _) = &entry.state else {
            panic!("Should only be called on present, initialized data");
        };
        if self
            .barrier_manager
            .is_visible(visibility.src, dst_info, visibility.created)
        {
            Ok(())
        } else {
            Err(visibility.src)
        }
    }

    pub async fn get_index<'b, 'inv>(
        &'b self,
        ctx: OpaqueTaskContext<'b, 'inv>,
        device: &'b DeviceContext,
        op: OperatorDescriptor,
        size: usize,
        dst: DstBarrierInfo,
    ) -> IndexHandle<'b> {
        let (src, created, buffer) = {
            let mut index = self.index_index.borrow_mut();

            let entry = index.entry(op.id).or_insert_with(|| {
                let layout = Layout::array::<ash::vk::DeviceAddress>(size).unwrap();
                let allocation = self.alloc_ssbo(device, layout);
                let info = StorageInfo {
                    allocation,
                    layout,
                    data_longevity: op.data_longevity,
                };
                let buffer = info.allocation.buffer;

                device.with_cmd_buffer(|cmd| unsafe {
                    device
                        .functions()
                        .cmd_fill_buffer(cmd.raw(), buffer, 0, vk::WHOLE_SIZE, 0);
                });

                let src = SrcBarrierInfo {
                    stage: vk::PipelineStageFlags2::TRANSFER,
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                };

                let visibility = Visibility {
                    src,
                    created: self.barrier_manager.current_epoch(),
                };
                IndexEntry::new(info, visibility)
            });

            inc_access(&mut entry.access, self);
            (
                entry.visibility.src,
                entry.visibility.created,
                entry.storage.allocation.buffer,
            )
        };

        if !self.barrier_manager.is_visible(src, dst, created) {
            ctx.submit(device.barrier(src, dst)).await;
        }
        assert!(self.barrier_manager.is_visible(src, dst, created));

        IndexHandle {
            buffer,
            op: op.id,
            device,
            num_chunks: size,
        }
    }

    pub fn read<'b, 't: 'b>(
        &'b self,
        access: AccessToken<'t>,
        dst_info: DstBarrierInfo,
    ) -> Result<ReadHandle<'b>, AccessToken<'t>> {
        let index = self.data_index.borrow();
        let Some(entry) = index.get(&access.id) else {
            return Err(access);
        };
        let StorageEntryState::Initialized(info, visibility, version) = &entry.state else {
            return Err(access);
        };
        if !self
            .barrier_manager
            .is_visible(visibility.src, dst_info, visibility.created)
        {
            return Err(access);
        }

        Ok(ReadHandle {
            buffer: info.allocation.buffer,
            layout: info.layout,
            access,
            version: version.type_(),
            data_longevity: info.data_longevity,
        })
    }

    pub fn try_update_inplace<'b, 't: 'b>(
        &'b self,
        device: &'b DeviceContext,
        current_frame: FrameNumber,
        old_access: AccessToken<'t>,
        new_desc: DataDescriptor,
        dst_info: DstBarrierInfo,
    ) -> Result<InplaceResult<'b>, AccessToken<'t>> {
        let old_key = old_access.id;
        let new_key = new_desc.id;

        let mut index = self.data_index.borrow_mut();
        let Some(entry) = index.get(&old_access.id) else {
            return Err(old_access);
        };
        let StorageEntryState::Initialized(info, visibility, version) = &entry.state else {
            return Err(old_access);
        };
        let old_version = version.type_();
        if !self
            .barrier_manager
            .is_visible(visibility.src, dst_info, visibility.created)
        {
            return Err(old_access);
        }

        // Only allow inplace if we are EXACTLY the one reader
        let in_place_possible = matches!(entry.access, AccessState::Some(1));

        Ok(if in_place_possible {
            let layout = info.layout;
            let buffer = info.allocation.buffer;

            let old_entry = index.remove(&old_key).unwrap();

            // Repurpose access key for the read/write handle
            let mut new_access = old_access;
            new_access.id = new_key;

            let new_entry = index.entry(new_key).or_insert_with(|| Entry {
                state: StorageEntryState::Registered,
                access: AccessState::Some(0),
            });

            let StorageEntryState::Initialized(info, _, _) = old_entry.state else {
                panic!("We already checked that it is initialized and are just moving out now");
            };
            new_entry.state = StorageEntryState::Initializing(info);

            new_entry.access = match new_entry.access {
                AccessState::Some(n) => AccessState::Some(n + 1),
                AccessState::None(..) => panic!("If present, entry should have accessors"),
            };

            InplaceResult::Inplace(
                WriteHandle {
                    buffer,
                    size: layout.size() as _,
                    drop_handler: DropError,
                    access: new_access,
                },
                old_version,
            )
        } else {
            let layout = info.layout;
            let buffer = info.allocation.buffer;
            let data_longevity = info.data_longevity;

            std::mem::drop(index); // Release borrow for alloc

            let w = self.alloc_slot_raw(device, current_frame, new_desc, layout);

            let r = ReadHandle {
                buffer,
                layout,
                access: old_access,
                version: old_version,
                data_longevity,
            };
            InplaceResult::New(r, w)
        })
    }

    pub fn access_state_cache<'b>(
        &'b self,
        device: &'b DeviceContext,
        key: DataId,
        layout: Layout,
    ) -> StateCacheResult<'b> {
        let mut new = false;
        let buffer = {
            let index = self.state_cache_index.borrow();
            if let Some(entry) = index.get(&key) {
                entry.storage.allocation.buffer
            } else {
                new = true;
                std::mem::drop(index);
                // Drop index for allocation which may need to free and then access the index
                let storage = StorageInfo {
                    allocation: self.alloc_ssbo(device, layout),
                    layout,
                    data_longevity: DataLongevity::Ephemeral,
                };

                let mut index = self.state_cache_index.borrow_mut();
                let buffer = storage.allocation.buffer;
                index.insert(
                    key,
                    StateCacheEntry {
                        access: AccessState::Some(0),
                        storage,
                    },
                );
                buffer
            }
        };

        let access = StateCacheAccessToken::new(self, device, key);
        let handle = StateCacheHandle {
            buffer,
            size: layout.size() as _,
            access,
        };

        if new {
            StateCacheResult::New(handle)
        } else {
            StateCacheResult::Existing(handle)
        }
    }

    pub fn deinitialize(&mut self) {
        self.allocator.deinitialize();
    }
}

#[derive(Debug)]
pub struct Allocation {
    // Always init except after VulkanState::deinitialize is called
    allocation: MaybeUninit<gpu_allocator::vulkan::Allocation>,
    pub size: u64,
    pub buffer: vk::Buffer,
}

impl Allocation {
    pub fn mapped_ptr(&self) -> Option<std::ptr::NonNull<std::ffi::c_void>> {
        unsafe { self.allocation.assume_init_ref() }.mapped_ptr()
    }
}

impl VulkanState for Allocation {
    unsafe fn deinitialize(&mut self, context: &DeviceContext) {
        let alloc = Allocation {
            allocation: std::mem::replace(&mut self.allocation, MaybeUninit::uninit()),
            size: self.size,
            buffer: self.buffer,
        };
        context.storage.deallocate(alloc)
    }
}

pub struct ImageAllocation {
    allocation: MaybeUninit<gpu_allocator::vulkan::Allocation>,
    pub image: vk::Image,
}

impl VulkanState for ImageAllocation {
    unsafe fn deinitialize(&mut self, context: &DeviceContext) {
        let alloc = ImageAllocation {
            allocation: std::mem::replace(&mut self.allocation, MaybeUninit::uninit()),
            image: self.image,
        };
        context.storage.deallocate_image(alloc)
    }
}

pub struct Allocator {
    allocator: RefCell<Option<gpu_allocator::vulkan::Allocator>>,
    device: ash::Device,
    num_alloced: Cell<u64>,
    capacity: Option<u64>,
}

pub type MemoryLocation = gpu_allocator::MemoryLocation;

impl Allocator {
    pub fn new(
        instance: ash::Instance,
        device: ash::Device,
        physical_device: vk::PhysicalDevice,
        capacity: Option<u64>,
    ) -> Self {
        let allocator = RefCell::new(Some(
            gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
                instance,
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: true,
            })
            .unwrap(),
        ));
        let num_alloced = Cell::new(0);
        Self {
            allocator,
            device,
            num_alloced,
            capacity,
        }
    }
    pub fn allocate(
        &self,
        layout: Layout,
        use_flags: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> gpu_allocator::Result<Allocation> {
        if let Some(capacity) = self.capacity {
            if self.num_alloced.get() + layout.size() as u64 > capacity {
                return Err(gpu_allocator::AllocationError::OutOfMemory);
            }
        }

        let size = layout.size() as u64;

        // Setup vulkan info
        let vk_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(use_flags | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);

        let buffer = unsafe { self.device.create_buffer(&vk_info, None) }.unwrap();
        let mut requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        requirements.alignment = requirements.alignment.max(layout.align() as u64);

        let mut allocator = self.allocator.borrow_mut();
        let allocator = allocator.as_mut().unwrap();
        let allocation = allocator.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "buffer allocation",
            requirements,
            location,
            linear: true, // Buffers are always linear
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        });
        let allocation = match allocation {
            Ok(a) => a,
            Err(e) => {
                unsafe { self.device.destroy_buffer(buffer, None) };
                return Err(e);
            }
        };

        // Bind memory to the buffer
        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap()
        };

        self.num_alloced
            .set(self.num_alloced.get() + allocation.size());

        Ok(Allocation {
            allocation: MaybeUninit::new(allocation),
            size,
            buffer,
        })
    }

    /// Safety: Allocation must come from this allocator
    pub unsafe fn deallocate(&self, allocation: Allocation) {
        let mut allocator = self.allocator.borrow_mut();
        let allocator = allocator.as_mut().unwrap();
        let allocation_inner = allocation.allocation.assume_init();
        let size = allocation_inner.size();
        allocator.free(allocation_inner).unwrap();
        unsafe { self.device.destroy_buffer(allocation.buffer, None) };

        self.num_alloced
            .set(self.num_alloced.get().checked_sub(size).unwrap());
    }

    pub fn allocate_image(
        &self,
        image_info: vk::ImageCreateInfo,
    ) -> gpu_allocator::Result<ImageAllocation> {
        let image = unsafe { self.device.create_image(&image_info, None) }.unwrap();
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };

        let size = requirements.size;

        if let Some(capacity) = self.capacity {
            if self.num_alloced.get() + size > capacity {
                unsafe { self.device.destroy_image(image, None) };
                return Err(gpu_allocator::AllocationError::OutOfMemory);
            }
        }
        self.num_alloced.set(self.num_alloced.get() + size);

        let mut allocator = self.allocator.borrow_mut();
        let allocator = allocator.as_mut().unwrap();
        let allocation = allocator.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "texture allocation",
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: true, // TODO: maybe not linear?
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        });
        let allocation = match allocation {
            Ok(a) => a,
            Err(e) => {
                unsafe { self.device.destroy_image(image, None) };
                return Err(e);
            }
        };

        // Bind memory to the image
        unsafe {
            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .unwrap()
        };

        Ok(ImageAllocation {
            allocation: MaybeUninit::new(allocation),
            image,
        })
    }

    /// Safety: Allocation must come from this allocator
    pub unsafe fn deallocate_image(&self, allocation: ImageAllocation) {
        let mut allocator = self.allocator.borrow_mut();
        let allocator = allocator.as_mut().unwrap();
        let size = allocation.allocation.assume_init_ref().size();
        allocator.free(allocation.allocation.assume_init()).unwrap();
        unsafe { self.device.destroy_image(allocation.image, None) };

        self.num_alloced
            .set(self.num_alloced.get().checked_sub(size).unwrap());
    }

    fn allocated(&self) -> u64 {
        self.num_alloced.get()
    }

    pub fn deinitialize(&mut self) {
        let mut a = self.allocator.borrow_mut();
        let mut tmp = None;
        std::mem::swap(&mut *a, &mut tmp);
    }
}
