use ahash::HashMapExt;
use bytemuck::{Pod, Zeroable};
use id::Identify;
use std::{
    alloc::Layout,
    cell::{Cell, RefCell},
    collections::BTreeMap,
    mem::MaybeUninit,
};

use super::{
    DataLocation, DataLongevity, DataVersion, DataVersionType, GarbageCollectId, LRUIndex,
    LRUIndexInner, LRUManager,
};

use ash::vk;
use gpu_allocator::{vulkan::AllocationScheme, AllocatorReport};

use crate::{
    array::ChunkIndex,
    dtypes::ElementType,
    operator::{DataDescriptor, DataId, OperatorDescriptor, OperatorId},
    runtime::FrameNumber,
    task::{AllocationId, AllocationRequest, OpaqueTaskContext, Request, RequestType},
    util::{IdGenerator, Map},
    vulkan::{
        memory::TempRessource, pipeline::ComputePipelineBuilder, shader::Shader,
        state::VulkanState, CmdBufferEpoch, CommandBuffer, DeviceContext, DeviceId, DstBarrierInfo,
        SrcBarrierInfo,
    },
};

mod page_table {
    use std::alloc::Layout;

    pub const LEVELS: u64 = 3;

    pub const BITS_PER_LEVEL: u64 = 15;
    pub const LEVEL_TABLE_SIZE: u64 = (2 << BITS_PER_LEVEL) + 1; //One extra entry for PageTableIndex

    pub const PAGE_LAYOUT: Layout = match Layout::array::<u64>(LEVEL_TABLE_SIZE as usize) {
        Ok(l) => l,
        Err(_) => panic!(),
    };

    pub const LEVEL_IDENTIFIER_BITS: u64 = 2;
    pub const LEVEL_IDENTIFIER_SIZE: u64 = 2 << LEVEL_IDENTIFIER_BITS;
    pub const LEVEL_IDENTIFIER_MASK: u64 = LEVEL_IDENTIFIER_SIZE - 1;
}

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

        let memory_barriers = &[vk::MemoryBarrier2::default()
            .src_stage_mask(src.stage)
            .src_access_mask(src.access)
            .dst_stage_mask(dst.stage)
            .dst_access_mask(dst.access)];
        let barrier_info = vk::DependencyInfo::default().memory_barriers(memory_barriers);

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
    longevity: Option<DataLongevity>, // May be None if entry is only in Registered state
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
        let version = if let StorageEntryState::Initialized(_, _, v) = vram_entry.state {
            v
        } else {
            DataVersion::Final
        };

        let mut lru_manager = self.storage.lru_manager.borrow_mut();
        dec_access(
            &mut vram_entry.access,
            &mut lru_manager,
            self.device,
            LRUItem::Data(self.id, version),
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
        if !std::thread::panicking() {
            // Avoid additional panics (-> aborts) while already panicking (unwinding)
            panic!("The WriteHandle was not marked initialized!");
        }
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
                    access.storage.new_data.add(access.id);

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

    pub fn into_thread_handle(self) -> ThreadWriteHandle {
        let id = self.access.id;
        std::mem::forget(self.drop_handler);
        std::mem::forget(self.access);
        ThreadWriteHandle {
            id,
            buffer: self.buffer,
            size: self.size,
            _panic_handle: Default::default(),
        }
    }
}

pub struct ThreadWriteHandle {
    id: DataId,
    pub buffer: ash::vk::Buffer,
    pub size: u64,
    _panic_handle: super::ThreadHandleDropPanic,
}

impl ThreadWriteHandle {
    pub fn into_main_handle<'a>(self, ctx: &'a DeviceContext) -> WriteHandle<'a> {
        self._panic_handle.dismiss();
        WriteHandle {
            buffer: self.buffer,
            size: self.size,
            drop_handler: DropError,
            access: AccessToken {
                storage: &ctx.storage,
                device: ctx,
                id: self.id,
            },
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
    storage: &'a Storage,
    device: &'a DeviceContext,
    id: DataId,
    frame: FrameNumber,
}

impl<'a> StateCacheHandle<'a> {
    pub fn buffer_address(&self) -> BufferAddress {
        buffer_address(self.device, self.buffer)
    }
}

pub enum StateCacheResult<'a> {
    New(StateCacheHandle<'a>),
    Existing(StateCacheHandle<'a>),
}

impl Drop for StateCacheHandle<'_> {
    fn drop(&mut self) {
        let mut index = self.storage.data_index.borrow_mut();
        let vram_entry = index.get_mut(&self.id).unwrap();

        let longevity = vram_entry.state.storage_info().map(|i| i.data_longevity);

        let mut lru_manager = self.storage.lru_manager.borrow_mut();
        dec_access(
            &mut vram_entry.access,
            &mut lru_manager,
            self.device,
            LRUItem::Data(self.id, DataVersion::Preview(self.frame)),
            longevity,
        );
    }
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

pub enum InplaceResult<'a, 'inv> {
    Inplace(WriteHandle<'a>, DataVersionType),
    New(ReadHandle<'a>, Request<'a, 'inv, WriteHandle<'a>>),
}

impl<'a, 'inv> InplaceResult<'a, 'inv> {
    pub fn alloc(self) -> Request<'a, 'inv, InplaceHandle<'a>> {
        match self {
            InplaceResult::Inplace(a, b) => Request::ready(InplaceHandle::Inplace(a, b)),
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

pub enum InplaceHandle<'a> {
    Inplace(WriteHandle<'a>, DataVersionType), //TODO: Why do we need the data version type here?
    New(ReadHandle<'a>, WriteHandle<'a>),
}
impl<'a> InplaceHandle<'a> {
    /// Safety: The corresponding slot (either rw inplace or separate w) has to have been
    /// completely written to.
    pub unsafe fn initialized<'b, 'c>(self, ctx: OpaqueTaskContext<'b, 'c>, info: SrcBarrierInfo) {
        match self {
            InplaceHandle::Inplace(rw, _v) => unsafe { rw.initialized(ctx, info) },
            InplaceHandle::New(_r, w) => unsafe { w.initialized(ctx, info) },
        };
    }
}

enum PageTableChildType {
    Inner(Allocation),
    Leaf(DataId, ChunkIndex),
}
struct PageTableChild {
    page_table_lru_index: Option<LRUIndexInner>,
    type_: PageTableChildType,
}

pub struct PageTableState {
    root_storage: StorageInfo,
    visibility: Visibility,
    children: Map<BufferAddress, PageTableChild>,
    access: AccessState,
}

// Safety: The caller is responsible for removing the reference from the index in gpu memory
// (i.e. self.storage.allocation.buffer)
unsafe fn unref_in_page_table(
    device: &DeviceContext,
    data_index: &mut Map<DataId, Entry>,
    page_table_lru: &mut PageTableLRU,
    chunk_lru: &mut LRUManager<LRUItem>,
    page_table_entry: &PageTableChild,
) {
    // unwrap: page_table_entry should always be some if page is managed in the LRU.
    page_table_lru.remove(page_table_entry.page_table_lru_index.unwrap());

    match page_table_entry.type_ {
        PageTableChildType::Inner(_) => {}
        PageTableChildType::Leaf(data_id, _) => {
            let d_entry = data_index.get_mut(&data_id).unwrap();
            let longevity = d_entry.state.storage_info().map(|i| i.data_longevity);

            let version = if let StorageEntryState::Initialized(_, _, v) = d_entry.state {
                v
            } else {
                DataVersion::Final
            };

            dec_access(
                &mut d_entry.access,
                chunk_lru,
                device,
                LRUItem::Data(data_id, version),
                longevity,
            );
        }
    }
}

#[derive(Copy, Clone, Identify)]
enum DispatchDeleteTarget {
    Inner(BufferAddress),
    Leaf(ChunkIndex),
}

fn dispatch_page_table_release(
    device: &DeviceContext,
    root: vk::Buffer,
    to_delete: DispatchDeleteTarget,
) {
    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct PushConstants {
        to_delete: u64,
        page_table_root: BufferAddress,
    }

    let delete_mode = match to_delete {
        DispatchDeleteTarget::Inner(_) => "MODE_INNER",
        DispatchDeleteTarget::Leaf(_) => "MODE_LEAF",
    };
    let pipeline = device.request_state(delete_mode, |device, delete_mode| {
        ComputePipelineBuilder::new(
            Shader::new(include_str!("page_table_delete.glsl")).define(delete_mode, 1),
        )
        .use_push_descriptor(true)
        .build(device)
    });
    let pipeline = match pipeline {
        Ok(o) => o,
        Err(e) => {
            println!("{}", e);
            panic!("Shader compilation failed");
        }
    };

    device.with_cmd_buffer(|cmd| {
        let consts = PushConstants {
            to_delete: match to_delete {
                DispatchDeleteTarget::Inner(buffer_address) => buffer_address.0,
                DispatchDeleteTarget::Leaf(chunk_index) => chunk_index.0,
            },
            page_table_root: buffer_address(device, root),
        };
        unsafe {
            let mut pipeline = pipeline.bind(cmd);

            pipeline.push_constant_pod(consts);
            pipeline.dispatch3d(crate::vec::Vector::fill(1));
        }
    });
}

#[repr(transparent)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, Hash, PartialEq, Eq, PartialOrd, Ord, Identify)]
pub struct BufferAddress(pub u64);

pub fn buffer_address(device: &DeviceContext, buffer: vk::Buffer) -> BufferAddress {
    let info = ash::vk::BufferDeviceAddressInfo::default().buffer(buffer);
    BufferAddress(unsafe { device.functions().get_buffer_device_address(&info) })
}

fn dispatch_page_table_insert(
    device: &DeviceContext,
    root: BufferAddress,
    table_buf1: BufferAddress,
    table_buf2: BufferAddress,
    chunk_id: ChunkIndex,
    chunk_buffer_addr: BufferAddress,
) {
    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct PushConstants {
        root: BufferAddress,
        chunk_id: u64,
        chunk_buffer_addr: BufferAddress,
        table_buf1: BufferAddress,
        table_buf2: BufferAddress,
    }
    let pipeline = device.request_state((), |device, ()| {
        ComputePipelineBuilder::new(Shader::new(include_str!("page_table_insert.glsl")))
            .build(device)
    });
    let pipeline = match pipeline {
        Ok(o) => o,
        Err(e) => {
            println!("{}", e);
            panic!("");
        }
    };
    device.with_cmd_buffer(|cmd| {
        let consts = PushConstants {
            root,
            chunk_id: chunk_id.0,
            chunk_buffer_addr,
            table_buf1,
            table_buf2,
        };
        unsafe {
            let mut pipeline = pipeline.bind(cmd);

            pipeline.push_constant_pod(consts);
            pipeline.dispatch3d(crate::vec::Vector::fill(1));
        }
    });
}

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageTableIndex(pub(crate) u64);

impl PageTableIndex {
    fn level(&self) -> usize {
        (self.0 >> (page_table::LEVELS * page_table::BITS_PER_LEVEL)
            & page_table::LEVEL_IDENTIFIER_MASK) as usize
    }
}

impl From<ChunkIndex> for PageTableIndex {
    fn from(value: ChunkIndex) -> Self {
        let ret = Self(value.0);
        assert_eq!(ret.level(), 0);
        ret
    }
}

impl TryInto<ChunkIndex> for PageTableIndex {
    type Error = ();

    fn try_into(self) -> Result<ChunkIndex, Self::Error> {
        if self.level() == 0 {
            Ok(ChunkIndex(self.0))
        } else {
            Err(())
        }
    }
}

impl PageTableState {
    fn new(storage: StorageInfo, visibility: Visibility) -> Self {
        Self {
            root_storage: storage,
            visibility,
            children: Default::default(),
            access: AccessState::Some(0),
        }
    }

    fn note_use(&mut self, buffer_addr: BufferAddress, page_table_lru: &mut PageTableLRU) {
        // Chunk may not be present anymore due to intermittent garbage collection
        if let Some(pt_entry) = self.children.get_mut(&buffer_addr) {
            pt_entry.page_table_lru_index =
                Some(page_table_lru.note_use(pt_entry.page_table_lru_index.unwrap()));
        }
    }

    fn release(
        &mut self,
        device: &DeviceContext,
        data_index: &mut Map<DataId, Entry>,
        page_table_lru: &mut PageTableLRU,
        chunk_lru: &mut LRUManager<LRUItem>,
        addr: BufferAddress,
    ) -> PageTableChildType {
        //println!("Remove {:?}", addr);
        let pt_entry = self.children.remove(&addr).unwrap();

        let to_delete = match &pt_entry.type_ {
            PageTableChildType::Inner(_) => DispatchDeleteTarget::Inner(addr),
            PageTableChildType::Leaf(_, chunk_index) => DispatchDeleteTarget::Leaf(*chunk_index),
        };
        dispatch_page_table_release(device, self.root_storage.allocation.buffer, to_delete);

        // Safety: We have also just removed the reference from the index
        unsafe { unref_in_page_table(device, data_index, page_table_lru, chunk_lru, &pt_entry) };
        pt_entry.type_
    }
}

#[derive(Default)]
struct PageTablePageCache {
    available: RefCell<Vec<Allocation>>,
    returned: RefCell<BTreeMap<CmdBufferEpoch, Vec<Allocation>>>,
}

impl PageTablePageCache {
    pub fn insert(&self, device: &DeviceContext, alloc: Allocation) {
        {
            let mut returned = self.returned.borrow_mut();
            returned
                .entry(device.current_epoch())
                .or_default()
                .push(alloc);
        }

        self.try_reclaim_returned(device);
    }

    pub fn try_reclaim_returned(&self, device: &DeviceContext) {
        let current_epoch = device.current_epoch();
        let mut returned = self.returned.borrow_mut();
        let mut available = self.available.borrow_mut();
        while returned
            .first_key_value()
            .map(|(k, _)| k < &current_epoch)
            .unwrap_or(false)
        {
            available.extend(returned.pop_first().unwrap().1);
            println!("Available: {}", available.len());
        }
    }

    pub fn get<'a, 'inv>(&self, device: &'a DeviceContext) -> Request<'a, 'inv, Allocation> {
        let mut available = self.available.borrow_mut();
        if let Some(alloc) = available.pop() {
            Request::ready(alloc)
        } else {
            device.storage.request_allocate_page_table_page(&device)
        }
    }
}

pub struct PageTableHandle<'a> {
    pub(crate) buffer: ash::vk::Buffer,
    op: OperatorId,
    device: &'a DeviceContext,
}

impl<'a> PageTableHandle<'a> {
    pub fn root(&self) -> BufferAddress {
        buffer_address(&self.device, self.buffer)
    }

    pub async fn insert<'h, 'inv>(
        &self,
        ctx: OpaqueTaskContext<'h, 'inv>,
        pos: ChunkIndex,
        chunk: ReadHandle<'h>,
    ) {
        let buf1 = ctx
            .submit(
                self.device
                    .storage
                    .request_allocate_page_table_page(&self.device),
            )
            .await;
        let buf2 = ctx
            .submit(
                self.device
                    .storage
                    .request_allocate_page_table_page(&self.device),
            )
            .await;

        self.device.with_cmd_buffer(|cmd| unsafe {
            //TODO: Why do we always have to zero them, even if we never use them?
            for buf in [buf1.buffer, buf2.buffer] {
                self.device
                    .functions()
                    .cmd_fill_buffer(cmd.raw(), buf, 0, vk::WHOLE_SIZE, 0);
            }

            self.device.storage.barrier_manager.issue(
                cmd,
                SrcBarrierInfo {
                    stage: vk::PipelineStageFlags2::TRANSFER,
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                },
                DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                },
            );
        });

        let buf1_addr = buffer_address(&self.device, buf1.buffer);
        let buf2_addr = buffer_address(&self.device, buf2.buffer);
        let root_buffer_addr = buffer_address(&self.device, self.buffer);

        let mut index = self.device.storage.page_table_index.borrow_mut();
        let entry = index.get_mut(&self.op).unwrap();

        let chunk_buffer_addr = buffer_address(self.device, chunk.buffer);

        dispatch_page_table_insert(
            self.device,
            root_buffer_addr,
            buf1_addr,
            buf2_addr,
            pos,
            chunk_buffer_addr,
        );

        if !entry.children.contains_key(&chunk_buffer_addr) {
            let data_id = chunk.access.id;
            assert_eq!(data_id, DataId::new(self.op, pos));

            // After allocations (potential preemptions): Give management to LRU
            let lru_index = self
                .device
                .storage
                .page_table_lru
                .borrow_mut()
                .add((self.op, chunk_buffer_addr));

            // Insert page in index so that following calls return in the `contains_key` guard above.
            entry.children.insert(
                chunk_buffer_addr,
                PageTableChild {
                    page_table_lru_index: Some(lru_index),
                    type_: PageTableChildType::Leaf(data_id, pos),
                },
            );

            // Leak ReadHandle: This handle is now "owned" by the index
            std::mem::forget(chunk);
        } else {
            // Nothing. We already have a child entry for the chunk, but either way we have added
            // it to the page table.

            //println!("{:?} should already be present", pos);
        }

        for (buf, buf_addr) in [(buf1, buf1_addr), (buf2, buf2_addr)] {
            let lru_index = self
                .device
                .storage
                .page_table_lru
                .borrow_mut()
                .add((self.op, buf_addr));

            entry.children.insert(
                buf_addr,
                PageTableChild {
                    page_table_lru_index: Some(lru_index),
                    type_: PageTableChildType::Inner(buf),
                },
            );
            //println!("Inner insert {:?}", buf_addr);
        }
    }

    pub fn note_use<'h>(&self, buffer_addr: BufferAddress) {
        let mut page_table_index = self.device.storage.page_table_index.borrow_mut();
        let index_entry = page_table_index.get_mut(&self.op).unwrap();

        let mut page_table_lru = self.device.storage.page_table_lru.borrow_mut();

        index_entry.note_use(buffer_addr, &mut page_table_lru);
    }
}
impl<'a> Drop for PageTableHandle<'a> {
    fn drop(&mut self) {
        let mut index = self.device.storage.page_table_index.borrow_mut();
        let vram_entry = index.get_mut(&self.op).unwrap();

        let longevity = vram_entry.root_storage.data_longevity;

        let mut lru_manager = self.device.storage.lru_manager.borrow_mut();
        dec_access(
            &mut vram_entry.access,
            &mut lru_manager,
            self.device,
            LRUItem::PageTableRoot(self.op),
            Some(longevity),
        );
    }
}

#[derive(Copy, Clone, Debug)]
enum LRUItem {
    Data(DataId, DataVersion),
    PageTableRoot(OperatorId),
}

type PageTableLRU = super::LRUManagerInner<(OperatorId, BufferAddress)>;

pub struct Storage {
    data_index: RefCell<Map<DataId, Entry>>,
    old_preview_data_index: RefCell<
        Map<DataId, BTreeMap<FrameNumber, (StorageEntryState, Option<LRUIndex>, CmdBufferEpoch)>>,
    >,
    page_table_index: RefCell<Map<OperatorId, PageTableState>>,
    // Manage (unreferenced) items (chunks as well as page tables) and free them once we are able
    lru_manager: RefCell<super::LRUManager<LRUItem>>,
    // Purpose: Keep track of when chunks in page table were used and possibly remove them from the
    // corresponding table (thus (possibly) adding them to "normal" lru_manager)
    page_table_lru: RefCell<PageTableLRU>,
    pub(crate) barrier_manager: BarrierManager,
    allocator: Allocator,
    new_data: super::NewDataManager,
    id: DeviceId,
    garbage_collect_id_gen: IdGenerator<GarbageCollectId>,
    manual_garbage_returns: Cell<u64>,
    page_table_page_cache: PageTablePageCache,
}

impl Storage {
    pub fn new(device: DeviceId, allocator: Allocator) -> Self {
        Self {
            data_index: Default::default(),
            old_preview_data_index: Default::default(),
            page_table_index: Default::default(),
            lru_manager: Default::default(),
            page_table_lru: Default::default(),
            barrier_manager: BarrierManager::new(),
            allocator,
            new_data: Default::default(),
            id: device,
            garbage_collect_id_gen: Default::default(),
            manual_garbage_returns: 0.into(),
            page_table_page_cache: Default::default(),
        }
    }

    pub fn print_usage(&self) {
        #[derive(Debug)]
        #[allow(unused)] //For some reason the Debug does not silence unused warnings
        enum AccessStatePrint {
            Some(usize),
            None(Option<LRUIndex>, CmdBufferEpoch),
        }

        let report = self.allocator.generate_report();
        let a = bytesize::to_string(report.total_allocated_bytes, true);
        let b = bytesize::to_string(report.total_capacity_bytes, true);
        println!("allocated {}, capacity {}", a, b);
        for block in report.blocks {
            dbg!(bytesize::to_string(block.size, true));
            dbg!(block.allocations.len());
        }

        let lru = self.lru_manager.borrow();
        for inner in &lru.inner {
            println!("{:?}", inner);
        }

        let index = self.data_index.borrow();
        let mut entries: Vec<_> = index
            .iter()
            .map(|(k, v)| {
                let (state, size) = match &v.state {
                    StorageEntryState::Registered => ("registered", 0),
                    StorageEntryState::Initializing(storage_info) => {
                        ("initializing", storage_info.layout.size())
                    }
                    StorageEntryState::Initialized(storage_info, _visibility, _data_version) => {
                        ("initialized", storage_info.layout.size())
                    }
                };
                let access = match v.access {
                    AccessState::Some(n) => AccessStatePrint::Some(n),
                    AccessState::None(lruindex, cmd_buffer_epoch) => {
                        AccessStatePrint::None(lruindex, cmd_buffer_epoch)
                    }
                };
                (k, state, size, access)
            })
            .collect();

        entries.sort_by_key(|v| v.2);

        for entry in entries {
            println!(
                "{:?} {} {:?} {}",
                entry.0,
                entry.1,
                entry.3,
                bytesize::to_string(entry.2 as _, true)
            );
        }

        let old_preview_index = self.old_preview_data_index.borrow();
        println!("opv len {}", old_preview_index.len());
    }

    /// Safety: Danger zone: The entries cannot be in use anymore! No checking for dangling
    /// references is done!
    pub unsafe fn free_vram(&self) {
        {
            let mut pti = self.page_table_index.borrow_mut();
            for (operator, buf_addr) in
                std::mem::take(&mut *self.page_table_lru.borrow_mut()).drain_lru()
            {
                let child = pti
                    .get_mut(&operator)
                    .unwrap()
                    .children
                    .remove(&buf_addr)
                    .unwrap();
                match child.type_ {
                    PageTableChildType::Inner(allocation) => {
                        self.allocator.deallocate(allocation);
                    }
                    PageTableChildType::Leaf(_, _) => {
                        // Will be deallocated below from data_index
                    }
                }
            }
        }

        for (_, entry) in std::mem::take(&mut *self.page_table_index.borrow_mut()) {
            self.allocator.deallocate(entry.root_storage.allocation);
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

        for (_, entries) in std::mem::take(&mut *self.old_preview_data_index.borrow_mut()) {
            for (_, entry) in entries {
                match entry.0 {
                    StorageEntryState::Registered => {}
                    StorageEntryState::Initializing(info)
                    | StorageEntryState::Initialized(info, _, _) => {
                        self.allocator.deallocate(info.allocation);
                    }
                }
            }
        }
    }

    pub fn next_garbage_collect(&self) -> GarbageCollectId {
        self.garbage_collect_id_gen.preview_next()
    }

    pub fn wait_garbage_collect<'a, 'inv>(&self) -> Request<'a, 'inv, ()> {
        Request::garbage_collect(DataLocation::GPU(self.id), self.next_garbage_collect())
    }

    pub fn try_promote_previous_preview<'b>(
        &'b self,
        device: &'b DeviceContext,
        d: DataId,
        current_frame: FrameNumber,
    ) -> Result<WriteHandle<'b>, ()> {
        // Ensure there is already an entry for the current frame in the index. This may
        // (otherwise) not be the case if the datum was requested from another device/datalocation.
        {
            let mut index = self.data_index.borrow_mut();
            let entry = index.entry(d);
            self.ensure_presence(current_frame, entry);
        }

        let mut old_preview_data_index = self.old_preview_data_index.borrow_mut();
        if let Some(entries) = old_preview_data_index.get_mut(&d) {
            {
                let entry = entries.pop_last().unwrap();
                if entries.is_empty() {
                    old_preview_data_index.remove(&d);
                }

                let (old_state, lru_index, epoch) = entry.1;

                let mut index = self.data_index.borrow_mut();

                let existing_access = index.remove(&d).map(|v| {
                    assert!(matches!(v.state, StorageEntryState::Registered));
                    v.access
                });
                let access = match existing_access {
                    Some(AccessState::Some(n)) => {
                        if let Some(lru_index) = lru_index {
                            self.lru_manager.borrow_mut().remove(lru_index);
                        }
                        AccessState::Some(n)
                    }
                    Some(AccessState::None(_, _)) => {
                        panic!("New entry should already have accesses")
                    }
                    None => AccessState::None(lru_index, epoch),
                };

                let StorageEntryState::Initialized(info, _visibility, _version) = old_state else {
                    panic!("Invalid state");
                };

                let state = StorageEntryState::Initializing(info);

                index.insert(d, Entry { state, access });
            }

            let access = self.register_access(device, current_frame, d);

            Ok(self.access_initializing(access).unwrap())
        } else {
            Err(())
        }
    }

    pub fn try_garbage_collect(
        &self,
        device: &DeviceContext,
        goal_in_bytes: usize,
        current_frame: FrameNumber,
    ) -> usize {
        let mut collected = self.manual_garbage_returns.get() as usize;
        self.manual_garbage_returns.set(0);

        let mut lru = self.lru_manager.borrow_mut();
        let mut page_table_lru = self.page_table_lru.borrow_mut();
        let mut index = self.data_index.borrow_mut();
        let mut old_preview_index = self.old_preview_data_index.borrow_mut();
        let mut page_table_index = self.page_table_index.borrow_mut();

        let mut page_tables_to_unref = Vec::new();
        for (longevity, inner_lru) in lru.inner_mut() {
            let mut collected_local = 0;

            if goal_in_bytes <= collected {
                break;
            };

            while let Some(key) = inner_lru.get_next() {
                let info = match key {
                    LRUItem::Data(key, data_version) => {
                        let (old_preview_frame, epoch) =
                            if let DataVersion::Preview(v) = data_version {
                                if let Some((_, _, epoch)) =
                                    old_preview_index.get_mut(&key).and_then(|m| m.get_mut(&v))
                                {
                                    (Some(v), *epoch)
                                } else {
                                    let entry = index.get_mut(&key).unwrap();
                                    let AccessState::None(_, epoch) = entry.access else {
                                        panic!("Should not be in LRU list");
                                    };
                                    (None, epoch)
                                }
                            } else {
                                let entry = index.get_mut(&key).unwrap();
                                let AccessState::None(_, epoch) = entry.access else {
                                    panic!("Should not be in LRU list");
                                };
                                (None, epoch)
                            };

                        if !device.cmd_buffer_completed(epoch) {
                            // All following LRU items will have the same or a later epoch so cannot be deleted
                            // either
                            break;
                        }
                        if matches!(longevity, DataLongevity::Cache) {
                            // This is a "young" cache item (i.e. possibly from the last frame) so
                            // we do not (yet) want to delete it.
                            if data_version
                                .of_frame()
                                .map(|v| current_frame.diff(v) < 2)
                                .unwrap_or(false)
                            {
                                break;
                            }
                        }

                        let state = if let Some(old_preview_frame) = old_preview_frame {
                            let old_versions = old_preview_index.get_mut(&key).unwrap();
                            let ret = old_versions.remove(&old_preview_frame).unwrap().0;
                            if old_versions.is_empty() {
                                old_preview_index.remove(&key);
                            }
                            ret
                        } else {
                            index.remove(&key).unwrap().state
                        };

                        self.new_data.remove(key);
                        match state {
                            StorageEntryState::Registered => panic!("Should not be in LRU list"),
                            StorageEntryState::Initializing(info)
                            | StorageEntryState::Initialized(info, _, _) => info,
                        }
                    }
                    LRUItem::PageTableRoot(key) => {
                        let entry = page_table_index.get_mut(&key).unwrap();
                        let AccessState::None(_, f) = entry.access else {
                            panic!("Should not be in LRU list");
                        };

                        if !device.cmd_buffer_completed(f) {
                            // All following LRU items will have the same or a later epoch so cannot be deleted
                            // either
                            break;
                        }

                        let entry = page_table_index.remove(&key).unwrap();

                        page_tables_to_unref.push(entry.children);

                        entry.root_storage
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
                if goal_in_bytes <= collected && longevity != DataLongevity::Ephemeral {
                    break;
                };
            }
            if collected_local > 0 {
                println!(
                    "Garbage collect GPU{}: ({:?}) {}",
                    self.id.inner(),
                    longevity,
                    bytesize::to_string(collected_local as _, true)
                );
            }
        }
        // Unref all chunks that were used in a page table deleted above
        for pt_entries in page_tables_to_unref {
            for (_pos, pt_entry) in pt_entries {
                unsafe {
                    unref_in_page_table(
                        device,
                        &mut index,
                        &mut page_table_lru,
                        &mut lru,
                        &pt_entry,
                    );

                    if let PageTableChildType::Inner(allocation) = pt_entry.type_ {
                        //self.page_table_page_cache.insert(device, allocation);
                        std::mem::drop(TempRessource::new(device, allocation));
                    }
                };
            }
        }

        // Potentially unref chunks from indices that are still active
        let mut unindexed = 0;
        if goal_in_bytes > collected {
            // Pop items one by one from the respective indices. Note that since chunks are
            // inserted in the lru only when they are first inserted into the index, we effectively
            // a "least recently added" strategy here. This is probably not what we actually
            // want... However, we (so far) have no feedback about chunks being used via the index.
            while let Some((op, buf_addr)) = page_table_lru.get_next() {
                let page_table = page_table_index.get_mut(&op).unwrap();

                // Releasing the chunk also removes it from the page table queue
                let child = page_table.release(
                    device,
                    &mut index,
                    &mut *page_table_lru,
                    &mut lru,
                    buf_addr,
                );

                let size = match child {
                    PageTableChildType::Leaf(data_id, _) => {
                        let chunk_entry = index.get(&data_id).unwrap();
                        let chunk_info = match &chunk_entry.state {
                            StorageEntryState::Registered | StorageEntryState::Initializing(_) => {
                                panic!("Indexed chunk should be initialized")
                            }
                            StorageEntryState::Initialized(info, _, _) => info,
                        };
                        chunk_info.layout.size()
                    }
                    PageTableChildType::Inner(allocation) => {
                        let size = allocation.size;

                        // Free page once epoch is over and no one can be referencing it anymore
                        std::mem::drop(TempRessource::new(device, allocation));
                        //self.page_table_page_cache.insert(device, allocation);

                        size as usize
                    }
                };

                unindexed += size;
                if goal_in_bytes <= collected + unindexed {
                    break;
                };
            }
        }

        println!(
            "Garbage collect GPU{}: {} | Unindexed: {}",
            self.id.inner(),
            bytesize::to_string(collected as _, true),
            bytesize::to_string(unindexed as _, true),
        );

        if collected > 0 {
            let _ = self.garbage_collect_id_gen.next();
        }
        collected
    }

    pub fn is_initializing(&self, id: DataId) -> bool {
        self.data_index
            .borrow()
            .get(&id)
            .map(|e| matches!(e.state, StorageEntryState::Initializing(_)))
            .unwrap_or(false)
    }

    pub fn is_readable(&self, id: DataId, requested_version: DataVersion) -> bool {
        self.data_index
            .borrow()
            .get(&id)
            .map(|e| {
                if let StorageEntryState::Initialized(_, _, version) = e.state {
                    version >= requested_version
                } else {
                    false
                }
            })
            .unwrap_or(false)
    }

    pub(crate) fn newest_data(&self) -> impl Iterator<Item = (DataId, DataLocation)> {
        let id = self.id;
        self.new_data
            .drain()
            .map(move |d| (d, DataLocation::GPU(id)))
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
                                let DataVersion::Preview(frame_number) = version else {
                                    panic!("invalid state since version is smaller than current preview");
                                };
                                self.old_preview_data_index
                                    .borrow_mut()
                                    .entry(*e.key())
                                    .or_default()
                                    .insert(frame_number, (old.state, lru_index, epoch));
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

        match &entry.state {
            StorageEntryState::Registered => Err(access),
            StorageEntryState::Initializing(info) => Ok(WriteHandle {
                buffer: info.allocation.buffer,
                size: info.allocation.size,
                drop_handler: DropError,
                access,
            }),
            StorageEntryState::Initialized(_, _, _) => {
                panic!("Trying to access an initialized value");
            }
        }
    }
    pub(crate) fn access_initializing_state_cache<'a>(
        &self,
        access: AccessToken<'a>,
        current_frame: FrameNumber,
    ) -> Result<StateCacheHandle<'a>, AccessToken<'a>> {
        self.access_initializing(access).map(|r| {
            let WriteHandle {
                buffer,
                size,
                drop_handler,
                access,
            } = r;

            std::mem::forget(drop_handler);

            let res = StateCacheHandle {
                buffer,
                size,
                storage: access.storage,
                device: access.device,
                id: access.id,
                frame: current_frame,
            };
            std::mem::forget(access);
            res
        })
    }

    pub(crate) fn allocate_raw(
        &self,
        layout: Layout,
        use_flags: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<Allocation, ()> {
        self.allocator
            .allocate(layout, use_flags, location)
            .map_err(|_| ())
    }

    /// Safety: Allocation must come from this storage
    pub unsafe fn deallocate(&self, allocation: Allocation) {
        self.allocator.deallocate(allocation);
    }

    pub fn request_allocate_raw<'req, 'inv>(
        &'req self,
        device: &'req DeviceContext,
        layout: Layout,
        use_flags: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Request<'req, 'inv, Allocation> {
        let (result_sender, result_receiver) = oneshot::channel();

        Request {
            type_: RequestType::Allocation(
                AllocationId::next(),
                AllocationRequest::VRamBufRaw(
                    device.id,
                    layout,
                    use_flags,
                    location,
                    result_sender,
                ),
            ),
            gen_poll: Box::new(move |_ctx| {
                Box::new(move || match result_receiver.try_recv() {
                    Ok(res) => Some(res),
                    Err(oneshot::TryRecvError::Empty) => None,
                    Err(oneshot::TryRecvError::Disconnected) => {
                        panic!("Either polled twice or the compute thread was interrupted")
                    }
                })
            }),
            _marker: Default::default(),
        }
    }

    fn request_allocate_page_table_page<'req, 'inv>(
        &'req self,
        device: &'req DeviceContext,
    ) -> Request<'req, 'inv, Allocation> {
        let layout = page_table::PAGE_LAYOUT;
        let use_flags = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let location = MemoryLocation::GpuOnly;

        self.request_allocate_raw(device, layout, use_flags, location)
    }

    pub fn allocated(&self) -> u64 {
        self.allocator.allocated()
    }

    pub fn capacity(&self) -> u64 {
        self.allocator.max_capacity
    }

    pub(crate) fn allocate_image(
        &self,
        create_desc: vk::ImageCreateInfo,
    ) -> Result<ImageAllocation, ()> {
        self.allocator.allocate_image(create_desc).map_err(|_| ())
    }

    /// Safety: Allocation must come from this storage
    pub unsafe fn deallocate_image(&self, allocation: ImageAllocation) {
        self.allocator.deallocate_image(allocation);
    }

    pub fn request_allocate_image<'a, 'inv>(
        &self,
        device: &DeviceContext,
        create_desc: vk::ImageCreateInfo<'static>,
    ) -> Request<'a, 'inv, ImageAllocation> {
        let (result_sender, result_receiver) = oneshot::channel();

        Request {
            type_: RequestType::Allocation(
                AllocationId::next(),
                AllocationRequest::VRamImageRaw(device.id, create_desc, result_sender),
            ),
            gen_poll: Box::new(move |_ctx| {
                Box::new(move || match result_receiver.try_recv() {
                    Ok(res) => Some(res),
                    Err(oneshot::TryRecvError::Empty) => None,
                    Err(oneshot::TryRecvError::Disconnected) => {
                        panic!("Either polled twice or the compute thread was interrupted")
                    }
                })
            }),
            _marker: Default::default(),
        }
    }

    fn alloc_ssbo<'b>(&'b self, layout: Layout) -> Result<Allocation, ()> {
        let flags = ash::vk::BufferUsageFlags::STORAGE_BUFFER
            | ash::vk::BufferUsageFlags::TRANSFER_DST
            | ash::vk::BufferUsageFlags::TRANSFER_SRC;
        let location = MemoryLocation::GpuOnly;
        self.allocate_raw(layout, flags, location)
    }

    pub(crate) fn alloc_and_register_ssbo<'b>(
        &'b self,
        device: &'b DeviceContext,
        current_frame: FrameNumber,
        desc: DataDescriptor,
        layout: Layout,
    ) -> Result<(ash::vk::Buffer, AccessToken<'b>), ()> {
        let allocation = self.alloc_ssbo(layout)?;
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

        #[cfg(feature = "alloc_fill_nan")]
        device.with_cmd_buffer(|cmd| {
            let init_value = f32::NAN.to_bits();
            unsafe {
                device
                    .functions()
                    .cmd_fill_buffer(cmd.raw(), buffer, 0, vk::WHOLE_SIZE, init_value)
            };
            self.barrier_manager.issue(
                cmd,
                SrcBarrierInfo {
                    stage: vk::PipelineStageFlags2::TRANSFER,
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                },
                DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                    access: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                },
            )
        });

        Ok((buffer, AccessToken::new(self, device, key)))
    }

    pub fn request_alloc_slot_raw<'req, 'inv>(
        &'req self,
        device: &'req DeviceContext,
        current_frame: FrameNumber,
        data_descriptor: DataDescriptor,
        layout: Layout,
    ) -> Request<'req, 'inv, WriteHandle<'req>> {
        let mut access = Some(device.storage.register_access(
            device,
            current_frame,
            data_descriptor.id,
        ));

        Request {
            type_: RequestType::Allocation(
                AllocationId::next(),
                AllocationRequest::VRam(device.id, layout, data_descriptor),
            ),
            gen_poll: Box::new(move |_ctx| {
                Box::new(move || {
                    access = match device.storage.access_initializing(access.take().unwrap()) {
                        Ok(r) => return Some(r),
                        Err(acc) => Some(acc),
                    };
                    None
                })
            }),
            _marker: Default::default(),
        }
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

    // TODO taking the ctx here and this being a async fn instead of returning a request is not
    // particularly pretty. We should try to disentangle this.
    pub async fn get_page_table<'b, 'inv>(
        &'b self,
        ctx: OpaqueTaskContext<'b, 'inv>,
        device: &'b DeviceContext,
        op: OperatorDescriptor,
        dst: DstBarrierInfo,
    ) -> PageTableHandle<'b> {
        let (src, created, buffer) = {
            let mut index = self.page_table_index.borrow_mut();

            if let Some(entry) = index.get_mut(&op.id) {
                inc_access(&mut entry.access, self);
                (
                    entry.visibility.src,
                    entry.visibility.created,
                    entry.root_storage.allocation.buffer,
                )
            } else {
                std::mem::drop(index);

                let allocation = ctx
                    .submit(self.request_allocate_page_table_page(device))
                    .await;

                let mut index = self.page_table_index.borrow_mut();

                let entry = match index.entry(op.id) {
                    std::collections::hash_map::Entry::Occupied(o) => {
                        // Someone else was faster than us in allocating a slot for the index. just
                        // throw the allocation away.
                        // TODO: Not particularly pretty. Maybe we should adopt a reservation
                        // approach like for data slots here, too.
                        unsafe { self.deallocate(allocation) };
                        o.into_mut()
                    }
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        let info = StorageInfo {
                            allocation,
                            layout: page_table::PAGE_LAYOUT,
                            data_longevity: op.data_longevity,
                        };
                        let buffer = info.allocation.buffer;

                        device.with_cmd_buffer(|cmd| unsafe {
                            device.functions().cmd_fill_buffer(
                                cmd.raw(),
                                buffer,
                                0,
                                vk::WHOLE_SIZE,
                                0,
                            );
                        });

                        let src = SrcBarrierInfo {
                            stage: vk::PipelineStageFlags2::TRANSFER,
                            access: vk::AccessFlags2::TRANSFER_WRITE,
                        };

                        let visibility = Visibility {
                            src,
                            created: self.barrier_manager.current_epoch(),
                        };
                        slot.insert(PageTableState::new(info, visibility))
                    }
                };

                inc_access(&mut entry.access, self);
                (
                    entry.visibility.src,
                    entry.visibility.created,
                    entry.root_storage.allocation.buffer,
                )
            }
        };

        if !self.barrier_manager.is_visible(src, dst, created) {
            ctx.submit(device.barrier(src, dst)).await;
        }
        assert!(self.barrier_manager.is_visible(src, dst, created));

        PageTableHandle {
            buffer,
            op: op.id,
            device,
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

    pub fn try_update_inplace<'b, 't: 'b, 'inv, W: ElementType>(
        &'b self,
        device: &'b DeviceContext,
        current_frame: FrameNumber,
        old_access: AccessToken<'t>,
        new_desc: DataDescriptor,
        num_elements: usize,
        dst_info: DstBarrierInfo,
        write_dtype: W,
    ) -> Result<InplaceResult<'b, 'inv>, AccessToken<'t>> {
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

        let out_layout = write_dtype.array_layout(num_elements);
        let same_layout = info.layout == out_layout;
        // Only allow inplace if we are EXACTLY the one reader
        let in_place_possible = matches!(entry.access, AccessState::Some(1)) && same_layout;

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
            let in_layout = info.layout;
            let buffer = info.allocation.buffer;
            let data_longevity = info.data_longevity;

            std::mem::drop(index); // Release borrow for alloc

            let w = self.request_alloc_slot_raw(device, current_frame, new_desc, out_layout);

            let r = ReadHandle {
                buffer,
                layout: in_layout,
                access: old_access,
                version: old_version,
                data_longevity,
            };
            InplaceResult::New(r, w)
        })
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
        context
            .storage
            .manual_garbage_returns
            .set(context.storage.manual_garbage_returns.get() + self.size);
        context.storage.deallocate(alloc)
    }
}

pub struct ImageAllocation {
    allocation: MaybeUninit<gpu_allocator::vulkan::Allocation>,
    pub size: u64,
    pub image: vk::Image,
}

impl VulkanState for ImageAllocation {
    unsafe fn deinitialize(&mut self, context: &DeviceContext) {
        let alloc = ImageAllocation {
            allocation: std::mem::replace(&mut self.allocation, MaybeUninit::uninit()),
            size: self.size,
            image: self.image,
        };
        context
            .storage
            .manual_garbage_returns
            .set(context.storage.manual_garbage_returns.get() + self.size);
        context.storage.deallocate_image(alloc)
    }
}

pub struct Allocator {
    allocator: RefCell<Option<gpu_allocator::vulkan::Allocator>>,
    device: ash::Device,
    allocator_capacity: Cell<u64>,
    max_capacity: u64,
}

pub type MemoryLocation = gpu_allocator::MemoryLocation;

impl Allocator {
    pub fn new(
        instance: ash::Instance,
        device: ash::Device,
        physical_device: vk::PhysicalDevice,
        capacity: u64,
    ) -> Self {
        let allocator = RefCell::new(Some(
            gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
                instance,
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: true,
                allocation_sizes: Default::default(), //TODO: See if we can read good values from
                                                      //device
            })
            .unwrap(),
        ));
        let allocator_capacity = Cell::new(0);
        Self {
            allocator,
            device,
            allocator_capacity,
            max_capacity: capacity,
        }
    }
    pub fn allocate(
        &self,
        layout: Layout,
        use_flags: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> gpu_allocator::Result<Allocation> {
        assert_ne!(layout.size(), 0);

        let size = layout.size() as u64;

        // Setup vulkan info
        let vk_info = vk::BufferCreateInfo::default()
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
            allow_capacity_increase: self.allocator_capacity.get() < self.max_capacity,
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

        self.allocator_capacity.set(allocator_capacity(&allocator));

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
        allocator.free(allocation_inner).unwrap();
        unsafe { self.device.destroy_buffer(allocation.buffer, None) };

        self.allocator_capacity.set(allocator_capacity(&allocator));
    }

    pub fn allocate_image(
        &self,
        image_info: vk::ImageCreateInfo,
    ) -> gpu_allocator::Result<ImageAllocation> {
        let image = unsafe { self.device.create_image(&image_info, None) }.unwrap();
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };

        let size = requirements.size;

        let mut allocator = self.allocator.borrow_mut();
        let allocator = allocator.as_mut().unwrap();
        let allocation = allocator.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "texture allocation",
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: true, // TODO: maybe not linear?
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            allow_capacity_increase: self.allocator_capacity.get() < self.max_capacity,
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

        self.allocator_capacity.set(allocator_capacity(&allocator));

        Ok(ImageAllocation {
            allocation: MaybeUninit::new(allocation),
            size,
            image,
        })
    }

    /// Safety: Allocation must come from this allocator
    pub unsafe fn deallocate_image(&self, allocation: ImageAllocation) {
        let mut allocator = self.allocator.borrow_mut();
        let allocator = allocator.as_mut().unwrap();
        allocator.free(allocation.allocation.assume_init()).unwrap();
        unsafe { self.device.destroy_image(allocation.image, None) };

        self.allocator_capacity.set(allocator_capacity(&allocator));
    }

    fn allocated(&self) -> u64 {
        self.allocator_capacity.get()
    }

    fn generate_report(&self) -> AllocatorReport {
        let borrow = self.allocator.borrow();
        let a = borrow.as_ref().unwrap();
        a.generate_report()
    }

    pub fn deinitialize(&mut self) {
        let mut a = self.allocator.borrow_mut();
        let mut tmp = None;
        std::mem::swap(&mut *a, &mut tmp);
    }
}

fn allocator_capacity(allocator: &gpu_allocator::vulkan::Allocator) -> u64 {
    // Note: This is not completely free, since we iterate over all storage blocks, but also
    // not too expensive, since the size of memory blocks (~256MB) is quite large compared to
    // the expected device memory (a couple of GB). If device memory is too large and this
    // becomes a problem, we probably want larger memory blocks anyways.
    allocator.capacity()
}
