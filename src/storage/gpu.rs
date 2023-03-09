use std::{alloc::Layout, cell::RefCell, collections::BTreeMap};

use ash::vk;

use crate::{
    operator::DataId,
    task_graph::LocatedDataId,
    vulkan::{CmdBufferEpoch, DeviceContext, DeviceId},
    Error,
};

#[derive(Debug, Eq, PartialEq)]
enum VRamAccessState {
    Some(usize),
    None(/*LRUIndex, */ CmdBufferEpoch),
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

struct VRamEntry {
    state: VRamStorageEntryState,
    access: VRamAccessState,
}

pub struct VRamAccessToken<'a> {
    storage: &'a Storage,
    device: &'a DeviceContext,
    pub id: DataId,
}
impl<'a> VRamAccessToken<'a> {
    fn new(storage: &'a Storage, device: &'a DeviceContext, id: DataId) -> Self {
        let mut index = storage.index.borrow_mut();
        let vram_entry = index.get_mut(&id).unwrap();

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
        let mut index = self.storage.index.borrow_mut();
        let vram_entry = index.get_mut(&self.id).unwrap();

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
        access.storage.new_data.add(access.id);
        let mut binding = access.storage.index.borrow_mut();

        {
            let entry = &mut binding.get_mut(&access.id).unwrap();

            entry.state =
                match std::mem::replace(&mut entry.state, VRamStorageEntryState::Registered) {
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

pub struct Storage {
    index: RefCell<BTreeMap<DataId, VRamEntry>>,
    _lru_manager: RefCell<super::LRUManager>,
    allocator: crate::vulkan::Allocator,
    new_data: super::NewDataManager,
    id: DeviceId,
}

impl Storage {
    pub fn new(device: DeviceId, allocator: crate::vulkan::Allocator) -> Self {
        Self {
            index: Default::default(),
            _lru_manager: Default::default(),
            allocator,
            new_data: Default::default(),
            id: device,
        }
    }

    pub(crate) fn allocator(&self) -> &crate::vulkan::Allocator {
        &self.allocator
    }

    /// Safety: Danger zone: The entries cannot be in use anymore! No checking for dangling
    /// references is done!
    pub unsafe fn free_vram(&self) {
        let mut index = self.index.borrow_mut();
        for entry in index.values_mut() {
            match std::mem::replace(&mut entry.state, VRamStorageEntryState::Registered) {
                VRamStorageEntryState::Registered => {}
                VRamStorageEntryState::Initializing(info)
                | VRamStorageEntryState::Initialized(info) => {
                    self.allocator.deallocate(info.allocation);
                }
            }
        }
    }

    pub fn is_readable(&self, id: DataId) -> bool {
        self.index
            .borrow()
            .get(&id)
            .map(|e| matches!(e.state, VRamStorageEntryState::Initialized(_)))
            .unwrap_or(false)
    }

    pub(crate) fn newest_data(&self) -> impl Iterator<Item = LocatedDataId> {
        let id = self.id;
        self.new_data
            .drain()
            .map(move |d| d.in_location(super::DataLocation::VRam(id)))
    }

    pub fn register_vram_access<'b>(
        &'b self,
        device: &'b DeviceContext,
        id: DataId,
    ) -> VRamAccessToken<'b> {
        {
            let mut index = self.index.borrow_mut();
            index.entry(id).or_insert_with(|| {
                VRamEntry {
                    state: VRamStorageEntryState::Registered,
                    access: VRamAccessState::Some(0), // Will be overwritten immediately when generating token
                }
            });
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
        let allocation = self.allocator.allocate(
            layout,
            ash::vk::BufferUsageFlags::STORAGE_BUFFER
                | ash::vk::BufferUsageFlags::TRANSFER_DST
                | ash::vk::BufferUsageFlags::TRANSFER_SRC,
            crate::vulkan::MemoryLocation::GpuOnly,
        );
        let buffer = allocation.buffer;

        {
            let mut index = self.index.borrow_mut();
            let entry = index.entry(key).or_insert_with(|| VRamEntry {
                state: VRamStorageEntryState::Registered,
                access: VRamAccessState::Some(0), // Will be overwritten immediately when generating token
            });

            let info = VRamStorageInfo { allocation, layout };

            assert!(matches!(entry.state, VRamStorageEntryState::Registered));

            entry.state = VRamStorageEntryState::Initializing(info);
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
        access: VRamAccessToken<'t>,
    ) -> Result<VRamReadHandle<'b>, VRamAccessToken<'t>> {
        let index = self.index.borrow();
        let Some(entry) = index.get(&access.id) else {
            return Err(access);
        };
        let VRamStorageEntryState::Initialized(info) = &entry.state else {
            return Err(access);
        };

        Ok(VRamReadHandle {
            buffer: info.allocation.buffer,
            layout: info.layout,
            access,
        })
    }

    pub fn deinitialize(&mut self) {
        self.allocator.deinitialize();
    }
}
