use std::{
    alloc::Layout,
    cell::{Cell, RefCell},
    collections::BTreeMap,
};

use ash::vk;
use gpu_allocator::vulkan::AllocationScheme;

use crate::{
    operator::DataId,
    task_graph::LocatedDataId,
    vulkan::{
        CmdBufferEpoch, CommandBuffer, DeviceContext, DeviceId, DstBarrierInfo, SrcBarrierInfo,
    },
    Error,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BarrierEpoch(usize);

pub(crate) struct BarrierManager {
    current_epoch: Cell<usize>,
    last_issued_ending: RefCell<BTreeMap<(SrcBarrierInfo, DstBarrierInfo), BarrierEpoch>>,
}

impl BarrierManager {
    fn new() -> Self {
        Self {
            current_epoch: Cell::new(0),
            last_issued_ending: RefCell::new(BTreeMap::new()),
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

use super::LRUIndex;

#[derive(Debug, Eq, PartialEq)]
enum AccessState {
    Some(usize),
    None(LRUIndex, CmdBufferEpoch),
}

struct Visibility {
    src: SrcBarrierInfo,
    created: BarrierEpoch,
}

enum StorageEntryState {
    Registered,
    Initializing(StorageInfo),
    Initialized(StorageInfo, Visibility),
}

struct StorageInfo {
    pub allocation: Allocation,
    pub layout: Layout,
}

struct Entry {
    state: StorageEntryState,
    access: AccessState,
}

pub struct AccessToken<'a> {
    storage: &'a Storage,
    device: &'a DeviceContext,
    pub id: DataId,
}

impl std::fmt::Debug for AccessToken<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AccessToken {{ {:?} }}", self.id)
    }
}
impl<'a> AccessToken<'a> {
    fn new(storage: &'a Storage, device: &'a DeviceContext, id: DataId) -> Self {
        let mut index = storage.index.borrow_mut();
        let vram_entry = index.get_mut(&id).unwrap();

        vram_entry.access = match vram_entry.access {
            AccessState::Some(n) => AccessState::Some(n + 1),
            AccessState::None(id, _) => {
                storage.lru_manager.borrow_mut().remove(id);
                AccessState::Some(1)
            }
        };

        Self {
            storage,
            id,
            device,
        }
    }
}
impl Drop for AccessToken<'_> {
    fn drop(&mut self) {
        let mut index = self.storage.index.borrow_mut();
        let vram_entry = index.get_mut(&self.id).unwrap();

        vram_entry.access = match vram_entry.access {
            AccessState::Some(1) => {
                let lru_id = self.storage.lru_manager.borrow_mut().add(self.id);
                AccessState::None(lru_id, self.device.current_epoch())
            }
            AccessState::Some(n) => AccessState::Some(n - 1),
            AccessState::None(..) => {
                panic!("Invalid state");
            }
        };
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
    pub unsafe fn initialized(self, src: SrcBarrierInfo) {
        let WriteHandle {
            access,
            drop_handler,
            ..
        } = self;

        // Avoid running destructor which would panic
        std::mem::forget(drop_handler);

        // Mark as initialized
        access.storage.new_data.add(access.id);
        let mut binding = access.storage.index.borrow_mut();

        {
            let entry = &mut binding.get_mut(&access.id).unwrap();

            entry.state = match std::mem::replace(&mut entry.state, StorageEntryState::Registered) {
                StorageEntryState::Registered => {
                    panic!("Entry should be in state Initializing, but is in Registered");
                }
                StorageEntryState::Initialized(..) => {
                    panic!("Entry should be in state Initializing, but is in Initialized");
                }
                StorageEntryState::Initializing(info) => StorageEntryState::Initialized(
                    info,
                    Visibility {
                        src,
                        created: access.storage.barrier_manager.current_epoch(),
                    },
                ),
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
}

pub enum InplaceResult<'a> {
    Inplace(WriteHandle<'a>),
    New(ReadHandle<'a>, WriteHandle<'a>),
}
impl<'a> InplaceResult<'a> {
    /// Safety: The corresponding slot (either rw inplace or separate w) has to have been
    /// completely written to.
    pub unsafe fn initialized(self, info: SrcBarrierInfo) {
        match self {
            InplaceResult::Inplace(rw) => unsafe { rw.initialized(info) },
            InplaceResult::New(_r, w) => unsafe { w.initialized(info) },
        };
    }
}

pub struct Storage {
    index: RefCell<BTreeMap<DataId, Entry>>,
    lru_manager: RefCell<super::LRUManager>,
    pub(crate) barrier_manager: BarrierManager,
    allocator: Allocator,
    new_data: super::NewDataManager,
    id: DeviceId,
}

impl Storage {
    pub fn new(device: DeviceId, allocator: Allocator) -> Self {
        Self {
            index: Default::default(),
            lru_manager: Default::default(),
            barrier_manager: BarrierManager::new(),
            allocator,
            new_data: Default::default(),
            id: device,
        }
    }

    /// Safety: Danger zone: The entries cannot be in use anymore! No checking for dangling
    /// references is done!
    pub unsafe fn free_vram(&self) {
        let mut index = self.index.borrow_mut();
        for entry in index.values_mut() {
            match std::mem::replace(&mut entry.state, StorageEntryState::Registered) {
                StorageEntryState::Registered => {}
                StorageEntryState::Initializing(info) | StorageEntryState::Initialized(info, _) => {
                    self.allocator.deallocate(info.allocation);
                }
            }
        }
    }

    pub fn try_garbage_collect(&self, device: &DeviceContext, mut goal_in_bytes: usize) {
        let mut lru = self.lru_manager.borrow_mut();
        let mut index = self.index.borrow_mut();
        let mut collected = 0;
        while let Some(key) = lru.get_next() {
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

            let info = match entry.state {
                StorageEntryState::Registered => panic!("Should not be in LRU list"),
                StorageEntryState::Initializing(info) | StorageEntryState::Initialized(info, _) => {
                    info
                }
            };

            // Safety: All allocations in the index have been allocated with the allocator.
            // Deallocation only happens exactly here where the entry is also removed from the
            // index. The allocation is also not used on the gpu anymore since the last access
            // epoch has already passed.
            unsafe { self.allocator.deallocate(info.allocation) };

            self.new_data.remove(key);

            lru.pop_next();

            let size = info.layout.size();
            collected += size;
            let Some(rest) = goal_in_bytes.checked_sub(size) else {
                break;
            };
            goal_in_bytes = rest;
        }
        println!("Garbage collect GPU: {}B", collected);
    }

    pub fn is_readable(&self, id: DataId) -> bool {
        self.index
            .borrow()
            .get(&id)
            .map(|e| matches!(e.state, StorageEntryState::Initialized(_, _)))
            .unwrap_or(false)
    }

    pub(crate) fn newest_data(&self) -> impl Iterator<Item = LocatedDataId> {
        let id = self.id;
        self.new_data
            .drain()
            .map(move |d| d.in_location(super::DataLocation::VRam(id)))
    }

    pub fn register_access<'b>(&'b self, device: &'b DeviceContext, id: DataId) -> AccessToken<'b> {
        {
            let mut index = self.index.borrow_mut();
            index.entry(id).or_insert_with(|| {
                Entry {
                    state: StorageEntryState::Registered,
                    access: AccessState::Some(0), // Will be overwritten immediately when generating token
                }
            });
        }
        AccessToken::new(self, device, id)
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
                let garbage_collect_goal = self.allocator.allocated() / 2;
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

    // Allocates a GpuOnly storage buffer
    fn alloc_ssbo<'b>(
        &'b self,
        device: &'b DeviceContext,
        key: DataId,
        layout: Layout,
    ) -> Result<(ash::vk::Buffer, AccessToken<'b>), Error> {
        let flags = ash::vk::BufferUsageFlags::STORAGE_BUFFER
            | ash::vk::BufferUsageFlags::TRANSFER_DST
            | ash::vk::BufferUsageFlags::TRANSFER_SRC;
        let location = MemoryLocation::GpuOnly;
        let allocation = self.allocate(device, layout, flags, location);

        let buffer = allocation.buffer;

        {
            let mut index = self.index.borrow_mut();
            let entry = index.entry(key).or_insert_with(|| Entry {
                state: StorageEntryState::Registered,
                access: AccessState::Some(0), // Will be overwritten immediately when generating token
            });

            let info = StorageInfo { allocation, layout };

            assert!(matches!(entry.state, StorageEntryState::Registered));

            entry.state = StorageEntryState::Initializing(info);
        }

        Ok((buffer, AccessToken::new(self, device, key)))
    }

    pub fn alloc_slot_raw<'b>(
        &'b self,
        device: &'b DeviceContext,
        key: DataId,
        layout: Layout,
    ) -> Result<WriteHandle<'b>, Error> {
        let size = layout.size();
        let (buffer, access) = self.alloc_ssbo(device, key, layout)?;

        Ok(WriteHandle {
            buffer,
            size: size as u64,
            access,
            drop_handler: DropError,
        })
    }

    pub fn alloc_slot<'b, T: Copy + crevice::std430::Std430>(
        &'b self,
        device: &'b DeviceContext,
        key: DataId,
        num: usize,
    ) -> Result<WriteHandle<'b>, Error> {
        //TODO: Not sure if this actually works with std430
        let layout = Layout::array::<T>(num).unwrap();
        self.alloc_slot_raw(device, key, layout)
    }

    pub fn is_visible(&self, id: DataId, dst_info: DstBarrierInfo) -> Result<(), SrcBarrierInfo> {
        let index = self.index.borrow();
        let Some(entry) = index.get(&id) else {
            panic!("Should only be called on present, initialized data");
        };
        let StorageEntryState::Initialized(_info, visibility) = &entry.state else {
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

    pub fn read<'b, 't: 'b>(
        &'b self,
        access: AccessToken<'t>,
        dst_info: DstBarrierInfo,
    ) -> Result<ReadHandle<'b>, AccessToken<'t>> {
        let index = self.index.borrow();
        let Some(entry) = index.get(&access.id) else {
            return Err(access);
        };
        let StorageEntryState::Initialized(info, visibility) = &entry.state else {
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
        })
    }

    pub fn try_update_inplace<'b, 't: 'b>(
        &'b self,
        device: &'b DeviceContext,
        old_access: AccessToken<'t>,
        new_key: DataId,
        dst_info: DstBarrierInfo,
    ) -> Result<Result<InplaceResult<'b>, Error>, AccessToken<'t>> {
        let old_key = old_access.id;

        let mut index = self.index.borrow_mut();
        let Some(entry) = index.get(&old_access.id) else {
            return Err(old_access);
        };
        let StorageEntryState::Initialized(info, visibility) = &entry.state else {
            return Err(old_access)
        };
        if !self
            .barrier_manager
            .is_visible(visibility.src, dst_info, visibility.created)
        {
            return Err(old_access);
        }

        // Only allow inplace if we are EXACTLY the one reader
        let in_place_possible = matches!(entry.access, AccessState::Some(1));

        Ok(Ok(if in_place_possible {
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

            let StorageEntryState::Initialized(info, _) = old_entry.state else {
                panic!("We already checked that it is initialized and are just moving out now");
            };
            new_entry.state = StorageEntryState::Initializing(info);

            new_entry.access = match new_entry.access {
                AccessState::Some(n) => AccessState::Some(n + 1),
                AccessState::None(..) => panic!("If present, entry should have accessors"),
            };

            InplaceResult::Inplace(WriteHandle {
                buffer,
                size: layout.size() as _,
                drop_handler: DropError,
                access: new_access,
            })
        } else {
            let layout = info.layout;
            let buffer = info.allocation.buffer;

            std::mem::drop(index); // Release borrow for alloc

            let w = match self.alloc_slot_raw(device, new_key, layout) {
                Ok(w) => w,
                Err(e) => return Ok(Err(e)),
            };
            let r = ReadHandle {
                buffer,
                layout,
                access: old_access,
            };
            InplaceResult::New(r, w)
        }))
    }

    pub fn deinitialize(&mut self) {
        self.allocator.deinitialize();
    }
}

pub struct Allocation {
    allocation: gpu_allocator::vulkan::Allocation,
    pub buffer: vk::Buffer,
}

impl Allocation {
    pub fn mapped_ptr(&self) -> Option<std::ptr::NonNull<std::ffi::c_void>> {
        self.allocation.mapped_ptr()
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

        // Setup vulkan info
        let vk_info = vk::BufferCreateInfo::builder()
            .size(layout.size() as u64)
            .usage(use_flags | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);

        let buffer = unsafe { self.device.create_buffer(&vk_info, None) }.unwrap();
        let mut requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        requirements.alignment = requirements.alignment.max(layout.align() as u64);

        let mut allocator = self.allocator.borrow_mut();
        let allocator = allocator.as_mut().unwrap();
        let allocation = allocator.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "some allocation",
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

        Ok(Allocation { allocation, buffer })
    }

    /// Safety: Allocation must come from this allocator
    pub unsafe fn deallocate(&self, allocation: Allocation) {
        let mut allocator = self.allocator.borrow_mut();
        let allocator = allocator.as_mut().unwrap();
        let size = allocation.allocation.size();
        allocator.free(allocation.allocation).unwrap();
        unsafe { self.device.destroy_buffer(allocation.buffer, None) };

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
