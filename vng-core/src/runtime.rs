use ahash::{HashMapExt, HashSetExt};
use std::{
    cell::RefCell,
    collections::VecDeque,
    num::NonZeroU64,
    sync::mpsc,
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
    time::{Duration, Instant},
};

use crate::{
    operator::{DataId, OpaqueOperator, OperatorDescriptor, OperatorId, TypeErased},
    storage::{ram::Storage, DataLocation, DataVersionType, VisibleDataLocation},
    task::{DataRequest, OpaqueTaskContext, Request, RequestInfo, RequestType, Task},
    task_graph::{Priority, RequestId, TaskClass, TaskGraph, TaskId, VisibleDataId},
    task_manager::{TaskManager, ThreadSpawner},
    threadpool::{ComputeThreadPool, IoThreadPool, JobInfo},
    util::{Map, Set},
    vulkan::{BarrierInfo, DeviceContext, VulkanContext},
    Error,
};

const CMD_BUF_EAGER_CYCLE_TIME: Duration = Duration::from_millis(10);
const WAIT_TIMEOUT_GPU: Duration = Duration::from_micros(100);
const WAIT_TIMEOUT_CPU: Duration = Duration::from_micros(100);
const STUCK_TIMEOUT: Duration = Duration::from_secs(5);

struct DataRequestItem {
    id: DataId,
    item: TypeErased,
}

impl PartialEq for DataRequestItem {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id)
    }
}
impl Eq for DataRequestItem {}
impl PartialOrd for DataRequestItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.id.partial_cmp(&other.id)
    }
}
impl Ord for DataRequestItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}
impl std::hash::Hash for DataRequestItem {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

#[derive(Default)]
struct TaskIdManager {
    counts: Map<OperatorId, usize>,
}

impl TaskIdManager {
    fn gen_id(&mut self, op: OperatorId) -> TaskId {
        let v = self.counts.entry(op).or_insert(0);
        let ret = *v;
        *v = *v + 1;
        TaskId::new(op, ret)
    }
}

struct RequestBatch<'inv> {
    items: Set<DataRequestItem>,
    op: &'inv dyn OpaqueOperator,
    batch_id: TaskId,
}

#[derive(Default)]
struct RequestBatcher<'inv> {
    pending_batches: Map<OperatorId, RequestBatch<'inv>>,
    task_id_manager: TaskIdManager,
}

enum BatchAddResult {
    New(TaskId),
    Existing(TaskId),
}

impl<'inv> RequestBatcher<'inv> {
    fn add(&mut self, request: DataRequest<'inv>) -> BatchAddResult {
        let source = &*request.source;
        let op_id = source.id();
        let req_item = DataRequestItem {
            id: request.id,
            item: request.item,
        };
        match self.pending_batches.entry(op_id) {
            crate::util::MapEntry::Vacant(o) => {
                let mut items = Set::new();
                items.insert(req_item);
                let batch_id = self.task_id_manager.gen_id(op_id);
                o.insert(RequestBatch {
                    items,
                    op: source,
                    batch_id,
                });
                BatchAddResult::New(batch_id)
            }
            crate::util::MapEntry::Occupied(mut o) => {
                let entry = o.get_mut();
                entry.items.insert(req_item);
                BatchAddResult::Existing(entry.batch_id)
            }
        }
    }

    fn gen_single_task_id(&mut self, op_id: OperatorId) -> TaskId {
        self.task_id_manager.gen_id(op_id)
    }

    fn get(&mut self, op: OperatorId) -> (&'inv dyn OpaqueOperator, Vec<TypeErased>) {
        let batch = self.pending_batches.remove(&op).unwrap();
        let items = batch.items.into_iter().map(|i| i.item).collect::<Vec<_>>();
        (batch.op, items)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum BarrierItem {
    Data(VisibleDataId),
    Barrier(BarrierInfo),
}

impl Into<RequestId> for BarrierItem {
    fn into(self) -> RequestId {
        match self {
            BarrierItem::Data(d) => RequestId::Data(d),
            BarrierItem::Barrier(info) => RequestId::Barrier(info),
        }
    }
}

struct BarrierBatcher {
    pending: Map<BarrierInfo, (TaskId, Set<BarrierItem>)>,
    pending_inv: Map<TaskId, BarrierInfo>,
    transfer_count: usize,
    op_id: OperatorId,
}

impl BarrierBatcher {
    fn new() -> Self {
        Self {
            pending: Map::new(),
            pending_inv: Map::new(),
            transfer_count: 0,
            op_id: OperatorId::new("BarrierBatcher"),
        }
    }
    fn add(&mut self, info: BarrierInfo, item: BarrierItem) -> BatchAddResult {
        match self.pending.entry(info) {
            crate::util::MapEntry::Vacant(o) => {
                let c = self.transfer_count;
                self.transfer_count += 1;
                let task_id = TaskId::new(self.op_id, c);
                o.insert((task_id, {
                    let mut s = Set::new();
                    s.insert(item);
                    s
                }));
                self.pending_inv.insert(task_id, info);
                BatchAddResult::New(task_id)
            }
            crate::util::MapEntry::Occupied(mut o) => {
                o.get_mut().1.insert(item);
                BatchAddResult::Existing(o.get().0)
            }
        }
    }

    fn get<'cref, 'inv>(
        &mut self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        task_id: TaskId,
    ) -> Option<Task<'cref>> {
        if let Some(t) = self.pending_inv.remove(&task_id) {
            let (_, items) = self.pending.remove(&t).unwrap();
            Some(
                async move {
                    let device = &ctx.device_contexts[t.device];
                    device.with_cmd_buffer(|cmd| {
                        let _ = device.storage.barrier_manager.issue(cmd, t.src, t.dst);
                        //println!("barrier: {:?}, {:?}", t.src, t.dst);
                    });
                    for item in items {
                        ctx.completed_requests.add(item.into());
                    }
                    Ok(())
                }
                .into(),
            )
        } else {
            None
        }
    }
}

pub struct RunTime {
    pub ram: crate::storage::ram::Storage,
    pub vulkan: VulkanContext,
    pub compute_thread_pool: ComputeThreadPool,
    pub io_thread_pool: IoThreadPool,
    pub async_result_receiver: mpsc::Receiver<JobInfo>,
    frame: FrameNumber,
}

impl RunTime {
    pub fn new(
        storage_size: usize,
        gpu_storage_size: Option<u64>,
        num_compute_threads: Option<usize>,
    ) -> Result<Self, Error> {
        let num_compute_threads = num_compute_threads.unwrap_or(num_cpus::get());
        let (async_result_sender, async_result_receiver) = mpsc::channel();
        let vulkan = VulkanContext::new(gpu_storage_size)?;
        let ram = crate::storage::ram::Storage::new(storage_size)?;
        let frame = FrameNumber(1.try_into().unwrap());
        Ok(RunTime {
            ram,
            compute_thread_pool: ComputeThreadPool::new(
                async_result_sender.clone(),
                num_compute_threads,
            ),
            io_thread_pool: IoThreadPool::new(async_result_sender),
            async_result_receiver,
            vulkan,
            frame,
        })
    }

    pub fn resolve<
        'call,
        R: 'static,
        F: for<'cref, 'inv> FnOnce(
            OpaqueTaskContext<'cref, 'inv>,
            &'inv &'call (), //Note: This is just a marker to indicate that 'call: 'inv
        ) -> Task<'cref, R>,
    >(
        &'call mut self,
        deadline: Option<Instant>,
        task: F,
    ) -> Result<R, Error> {
        let request_queue = RequestQueue::new();
        let hints = TaskHints::default();
        let thread_spawner = ThreadSpawner::new();
        let completed_requests = Default::default();
        let predicted_preview_tasks = Default::default();
        let frame = self.frame;
        self.frame = FrameNumber(frame.0.checked_add(1).unwrap());

        let data = ContextData {
            request_queue,
            hints,
            completed_requests,
            thread_spawner,
            storage: &self.ram,
            device_contexts: self.vulkan.device_contexts(),
            frame: self.frame,
            predicted_preview_tasks,
        };
        let mut executor = {
            Executor {
                data: &data,
                task_manager: TaskManager::new(
                    &mut self.compute_thread_pool,
                    &mut self.io_thread_pool,
                    &mut self.async_result_receiver,
                ),
                waker: dummy_waker(),
                task_graph: TaskGraph::new(),
                statistics: Statistics::new(),
                operator_info: Default::default(),
                transfer_manager: Default::default(),
                request_batcher: Default::default(),
                barrier_batcher: BarrierBatcher::new(),
                deadline: deadline.unwrap_or_else(|| {
                    Instant::now() + std::time::Duration::from_secs(1 << 32)
                } /* basically: never */),
            }
        };

        executor.resolve(|ctx| task(ctx, &&()))
    }
}

impl Drop for RunTime {
    fn drop(&mut self) {
        // Safety: The runtime (including all references to storage) is dropped now, so no dangling
        // references will be left behind
        for device in self.vulkan.device_contexts() {
            unsafe { device.storage.free_vram() };
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[repr(transparent)]
pub struct FrameNumber(NonZeroU64);

/// An object that contains all data that will be later lent out to `TaskContexts` via `Executor`.
///
/// A note on lifetime names:
/// `'cref` refers to lifetimes that live at least as long as the context data (i.e., storage,
/// device contexts). This is also the lifetime of all `Tasks`, for example.
///
/// `'inv`, an invariant (!) lifetime (due to the use in a `RefCell` in `RequestQueue`), specifies
/// the lifetime of `OpaqueOperator` references as handled during the evaluation of the operator
/// network (used in `RequestQueue` and `RequestBatcher`).
struct ContextData<'cref, 'inv> {
    request_queue: RequestQueue<'inv>,
    hints: TaskHints,
    completed_requests: CompletedRequests,
    thread_spawner: ThreadSpawner,
    pub storage: &'cref Storage,
    device_contexts: &'cref [DeviceContext],
    frame: FrameNumber,
    predicted_preview_tasks: RefCell<Set<TaskId>>,
}

struct Executor<'cref, 'inv> {
    pub data: &'cref ContextData<'cref, 'inv>,
    task_manager: TaskManager<'cref>,
    request_batcher: RequestBatcher<'inv>,
    barrier_batcher: BarrierBatcher,
    task_graph: TaskGraph,
    transfer_manager: crate::vulkan::memory::TransferManager,
    statistics: Statistics,
    operator_info: Map<OperatorId, OperatorDescriptor>,
    waker: Waker,
    deadline: Instant,
}

impl<'cref, 'inv> Executor<'cref, 'inv> {
    //pub fn statistics(&self) -> &Statistics {
    //    &self.statistics
    //}
    fn context(&self, current_task: TaskId) -> OpaqueTaskContext<'cref, 'inv> {
        OpaqueTaskContext {
            requests: &self.data.request_queue,
            storage: &self.data.storage,
            hints: &self.data.hints,
            completed_requests: &self.data.completed_requests,
            thread_pool: &self.data.thread_spawner,
            device_contexts: self.data.device_contexts,
            current_task,
            current_op: self.operator_info.get(&current_task.operator()).cloned(),
            current_frame: self.data.frame,
            predicted_preview_tasks: &self.data.predicted_preview_tasks,
            deadline: self.deadline,
        }
    }

    fn construct_task(&mut self, id: TaskId) -> Task<'cref> {
        if let Some(t) = self.barrier_batcher.get(self.context(id), id) {
            t
        } else {
            let (op, batch) = self.request_batcher.get(id.operator());
            let context = self.context(id);
            //Safety: The argument batch is precisely for the returned operator, and thus of the right
            //type.
            unsafe { op.compute(context, batch) }
        }
    }

    fn try_resolve_implied(&mut self) -> Result<(), Error> {
        self.wait_for_async_results();

        enum StuckState {
            Not,
            Reported,
            WaitingSince(std::time::Instant),
        }
        let mut stuck_state = StuckState::Not;
        loop {
            self.cycle_cmd_buffers(CMD_BUF_EAGER_CYCLE_TIME);

            let ready = self.task_graph.next_implied_ready();
            if let Some(task_id) = ready {
                stuck_state = StuckState::Not;

                let resolved_deps =
                    if let Some(resolved_deps) = self.task_graph.resolved_deps(task_id) {
                        let mut tmp = Set::new();
                        std::mem::swap(&mut tmp, resolved_deps);
                        tmp
                    } else {
                        Set::new()
                    };
                let old_hints = self.data.hints.completed.replace(resolved_deps);
                // Hints should only contain during the loop body and returned back to the task
                // graph once the task was polled.
                assert!(old_hints.is_empty());

                // TODO: Try to clean up API here
                if !self.task_manager.has_task(task_id) {
                    let task = self.construct_task(task_id);
                    self.task_manager.add_task(task_id, task);
                }
                let task = self.task_manager.get_task(task_id).unwrap();

                let mut ctx = Context::from_waker(&self.waker);

                // The queue should always just contain the requests that are enqueued by polling
                // the following task! This is important so that we know the calling tasks id for
                // the generated requests.
                assert!(self.data.request_queue.is_empty());

                match task.as_mut().poll(&mut ctx) {
                    Poll::Ready(Ok(_)) => {
                        assert!(self.data.request_queue.is_empty());
                        // Drain hints
                        self.data.hints.completed.replace(Set::new());
                        self.register_produced_data();
                        self.task_graph.task_done(task_id);
                        self.task_manager.remove_task(task_id).unwrap();
                        self.statistics.tasks_executed += 1;
                    }
                    Poll::Ready(e) => {
                        self.register_produced_data();
                        println!("Execution errored, exporting task graph.");
                        crate::task_graph::export(&self.task_graph);
                        return e;
                    }
                    Poll::Pending => {
                        // Return hints back to the task graph
                        let old_hints = self.data.hints.completed.replace(Set::new());
                        if let Some(resolved_deps) = self.task_graph.resolved_deps(task_id) {
                            *resolved_deps = old_hints;
                        }
                        self.register_produced_data();
                        self.enqueue_requested(task_id);
                    }
                };
            } else {
                if self.task_graph.has_open_tasks() {
                    let stuck_time = match stuck_state {
                        StuckState::Not => Some(std::time::Instant::now()),
                        StuckState::WaitingSince(t) => Some(t),
                        StuckState::Reported => None,
                    };
                    if let Some(stuck_time) = stuck_time {
                        if stuck_time.elapsed() > STUCK_TIMEOUT {
                            eprintln!("Execution appears to be stuck. Generating dependency file");
                            crate::task_graph::export(&self.task_graph);
                            //eprintln!("Device states:");
                            //for device in self.data.device_contexts {
                            //    let buf = device.current_command_buffer.borrow();
                            //    eprintln!("{}: {:?}", device.id, buf.id());
                            //    let waiting = device.waiting_command_buffers.borrow();
                            //    eprintln!("Waiting: {}", waiting.len());
                            //    for w in waiting.iter() {
                            //        eprintln!("{}: {:?}", device.id, w.0);
                            //    }
                            //}
                            stuck_state = StuckState::Reported;
                        } else {
                            stuck_state = StuckState::WaitingSince(stuck_time);
                        }
                    }
                    self.wait_for_async_results();
                } else {
                    return Ok(());
                }
            }
        }
    }

    fn register_produced_data_from(
        &mut self,
        items: impl Iterator<Item = (DataId, DataLocation, DataVersionType)>,
    ) {
        for (id, produced_loc, produced_ver) in items {
            for requested in self.task_graph.requested_locations(id) {
                let r_id = VisibleDataId {
                    id,
                    location: requested.0,
                }
                .into();
                if let DataVersionType::Preview = produced_ver {
                    let mut m = self.data.predicted_preview_tasks.borrow_mut();
                    for dependent in self.task_graph.dependents(r_id) {
                        m.insert(*dependent);
                    }
                }
                let req_prio = requested
                    .1
                    .iter()
                    .map(|tid| self.task_graph.get_priority(*tid))
                    .max()
                    .unwrap();
                if let Some(task_id) =
                    self.try_make_available(id, produced_loc, requested.0, req_prio)
                {
                    self.task_graph.will_provide_data(task_id, id);
                } else {
                    self.task_graph.resolved_implied(r_id);
                }
            }
        }
    }
    fn register_produced_data(&mut self) {
        self.register_produced_data_from(self.data.storage.newest_data());
        for device in self.data.device_contexts {
            self.register_produced_data_from(device.storage.newest_data());
        }
        for item in self.data.completed_requests.take() {
            self.task_graph.resolved_implied(item.into());
        }
    }

    fn try_make_available(
        &mut self,
        id: DataId,
        from: DataLocation,
        to: VisibleDataLocation,
        req_prio: Priority,
    ) -> Option<TaskId> {
        match (from, to) {
            (DataLocation::Ram, VisibleDataLocation::Ram) => None,
            (DataLocation::VRam(source), VisibleDataLocation::VRam(target, dst_info))
                if target == source =>
            {
                let device = &self.data.device_contexts[target];
                match device.storage.is_visible(id, dst_info) {
                    Ok(()) => None,
                    Err(src_info) => {
                        let b_info = BarrierInfo {
                            device: device.id,
                            src: src_info,
                            dst: dst_info,
                        };
                        let task_id = match self
                            .barrier_batcher
                            .add(b_info, BarrierItem::Data(id.with_visibility(to)))
                        {
                            BatchAddResult::New(t) => {
                                self.task_graph
                                    .add_implied(t, req_prio.downstream(TaskClass::Barrier));
                                t
                            }
                            BatchAddResult::Existing(t) => t,
                        };
                        Some(task_id)
                    }
                }
            }
            (DataLocation::VRam(_source), VisibleDataLocation::VRam(_target, _)) => {
                panic!("VRam to VRam transfer not implemented, yet")
            }
            (DataLocation::Ram, VisibleDataLocation::VRam(target_id, _)) => {
                let task_id = self.transfer_manager.next_id();
                let access = self.data.storage.register_access(id);
                let transfer_task = self.transfer_manager.transfer_to_gpu(
                    self.context(task_id),
                    &self.data.device_contexts[target_id],
                    access,
                );
                self.task_manager.add_task(task_id, transfer_task);
                self.task_graph
                    .add_implied(task_id, req_prio.downstream(TaskClass::Transfer));
                Some(task_id)
            }
            (DataLocation::VRam(source_id), VisibleDataLocation::Ram) => {
                let task_id = self.transfer_manager.next_id();
                let device = &self.data.device_contexts[source_id];
                let access = device.storage.register_access(device, self.data.frame, id);
                let transfer_task = self.transfer_manager.transfer_to_cpu(
                    self.context(task_id),
                    &self.data.device_contexts[source_id],
                    access,
                );
                self.task_manager.add_task(task_id, transfer_task);
                self.task_graph
                    .add_implied(task_id, req_prio.downstream(TaskClass::Transfer));
                Some(task_id)
            }
        }
    }

    fn find_available_location(&self, datum: DataId) -> Option<DataLocation> {
        if self.data.storage.is_readable(datum) {
            Some(DataLocation::Ram)
        } else {
            for device in self.data.device_contexts {
                if device.storage.is_readable(datum) {
                    return Some(DataLocation::VRam(device.id));
                }
            }
            None
        }
    }

    fn enqueue(&mut self, from: TaskId, req: RequestInfo<'inv>) {
        let req_id = req.id();
        //TODO: We also want to increase the priority of a task if it was already requested...
        let already_requested = self.task_graph.already_requested(req_id);
        let req_prio = self.task_graph.get_priority(from);
        self.task_graph
            .add_dependency(from, req_id, req.progress_indicator);
        match req.task {
            RequestType::YieldOnce => {
                self.task_graph.resolved_implied(req_id);
            }
            RequestType::ExternalProgress => {}
            RequestType::Ready => {
                panic!("Ready request should never reach the executor");
            }
            RequestType::Data(data_request) => {
                let op_id = data_request.source.id();
                self.operator_info
                    .entry(op_id)
                    .or_insert(OperatorDescriptor {
                        id: op_id,
                        data_longevity: data_request.source.longevity(),
                    });

                let data_id = data_request.id;
                let data_req_loc = data_request.location;
                if !already_requested {
                    let fulfiller_task_id = if let Some(available) =
                        self.find_available_location(data_id)
                    {
                        // Data should not already be present => unwrap
                        self.try_make_available(data_id, available, data_req_loc, req_prio)
                            .unwrap()
                    } else {
                        match data_request.source.granularity() {
                            crate::operator::ItemGranularity::Single => {
                                // Spawn task immediately since we have all information we need
                                let task_id = self
                                    .request_batcher
                                    .gen_single_task_id(data_request.source.id());
                                let context = self.context(task_id);
                                let items = vec![data_request.item];
                                //Safety: The item is precisely for the returned operator, and thus of the right type.
                                let task = unsafe { data_request.source.compute(context, items) };
                                self.task_graph
                                    .add_implied(task_id, req_prio.downstream(TaskClass::Data));
                                self.task_manager.add_task(task_id, task);
                                task_id
                            }
                            crate::operator::ItemGranularity::Batched => {
                                // Add item to batcher to spawn later
                                match self.request_batcher.add(data_request) {
                                    BatchAddResult::New(id) => {
                                        self.task_graph
                                            .add_implied(id, req_prio.downstream(TaskClass::Data));
                                        id
                                    }
                                    BatchAddResult::Existing(id) => {
                                        self.task_graph.try_increase_priority(
                                            id,
                                            req_prio.downstream(TaskClass::Data),
                                        );
                                        id
                                    }
                                }
                            }
                        }
                    };
                    self.task_graph
                        .will_provide_data(fulfiller_task_id, data_id);
                } else {
                    let s = self.task_graph.who_will_provide_data(data_id).clone();
                    for fulfiller_task_id in s {
                        self.task_graph
                            .will_provide_data(fulfiller_task_id, data_id);
                    }
                };
            }
            RequestType::Barrier(b_info) => {
                let id = match self
                    .barrier_batcher
                    .add(b_info, BarrierItem::Barrier(b_info))
                {
                    BatchAddResult::New(id) => {
                        self.task_graph
                            .add_implied(id, req_prio.downstream(TaskClass::Barrier));
                        id
                    }
                    BatchAddResult::Existing(id) => {
                        self.task_graph
                            .try_increase_priority(id, req_prio.downstream(TaskClass::Data));
                        id
                    }
                };
                self.task_graph.will_fullfil_req(id, req_id);
            }
            RequestType::CmdBufferCompletion(_id) => {}
            RequestType::CmdBufferSubmission(_id) => {}
            RequestType::Allocation(id, alloc) => {
                let op_name = match alloc {
                    crate::task::AllocationRequest::Ram(_, _) => "allocator_ram",
                    crate::task::AllocationRequest::VRam(_, _, _) => "allocator_vram",
                    crate::task::AllocationRequest::VRamBufRaw(_, _, _, _, _) => {
                        "allocator_vram_raw"
                    }
                    crate::task::AllocationRequest::VRamImageRaw(_, _, _) => "allocator_vram_image",
                };
                let task_id = TaskId::new(OperatorId::new(op_name), id.inner() as _);
                let ctx = self.context(task_id);
                let task = match alloc {
                    crate::task::AllocationRequest::Ram(layout, data_descriptor) => async move {
                        loop {
                            if let Ok(_) = ctx.storage.alloc(data_descriptor, layout) {
                                break;
                            } else {
                                ctx.submit(ctx.storage.wait_garbage_collect()).await;
                            }
                        }
                        ctx.completed_requests.add(id.into());
                        Ok(())
                    }
                    .into(),
                    crate::task::AllocationRequest::VRam(device_id, layout, data_descriptor) => {
                        async move {
                            let device = &ctx.device_contexts[device_id];
                            loop {
                                if let Ok(_) = device.storage.alloc_and_register_ssbo(
                                    device,
                                    ctx.current_frame,
                                    data_descriptor,
                                    layout,
                                ) {
                                    break;
                                } else {
                                    ctx.submit(device.storage.wait_garbage_collect()).await;
                                }
                            }
                            ctx.completed_requests.add(id.into());
                            Ok(())
                        }
                        .into()
                    }
                    crate::task::AllocationRequest::VRamBufRaw(
                        device_id,
                        layout,
                        use_flags,
                        location,
                        result_sender,
                    ) => async move {
                        let device = &ctx.device_contexts[device_id];
                        let res = loop {
                            if let Ok(res) =
                                device.storage.allocate_raw(layout, use_flags, location)
                            {
                                break res;
                            } else {
                                ctx.submit(device.storage.wait_garbage_collect()).await;
                            }
                        };
                        result_sender.send(res).unwrap();
                        ctx.completed_requests.add(id.into());
                        Ok(())
                    }
                    .into(),
                    crate::task::AllocationRequest::VRamImageRaw(
                        device_id,
                        create_desc,
                        result_sender,
                    ) => async move {
                        let device = &ctx.device_contexts[device_id];
                        let res = device.storage.allocate_image(device, create_desc);
                        result_sender.send(res).unwrap();
                        ctx.completed_requests.add(id.into());
                        Ok(())
                    }
                    .into(),
                };
                self.task_manager.add_task(task_id, task);
                self.task_graph
                    .add_implied(task_id, req_prio.downstream(TaskClass::Alloc));
                self.task_graph.will_fullfil_req(task_id, req_id);
            }
            RequestType::ThreadPoolJob(job, type_) => {
                self.task_manager.spawn_job(job, type_).unwrap();
            }
            RequestType::Group(group) => {
                for v in group.all {
                    self.task_graph.in_group(v.id(), group.id);
                    self.enqueue(
                        from,
                        RequestInfo {
                            task: v,
                            progress_indicator:
                                crate::task_graph::ProgressIndicator::PartialPossible,
                        },
                    );
                }
            }
            RequestType::GarbageCollect(l) => {
                let task_id = match l {
                    DataLocation::Ram => TaskId::new(OperatorId::new("garbage_collect_ram"), 0),
                    DataLocation::VRam(d) => {
                        TaskId::new(OperatorId::new("garbage_collect_vram"), d)
                    }
                };
                if !already_requested {
                    let ctx = self.context(task_id);

                    let task = match l {
                        DataLocation::Ram => async move {
                            let garbage_collect_goal = ctx.storage.size()
                                / crate::storage::GARBAGE_COLLECT_GOAL_FRACTION as usize;

                            loop {
                                if ctx.storage.try_garbage_collect(garbage_collect_goal) > 0 {
                                    break;
                                } else {
                                    ctx.submit(Request::external_progress()).await;
                                }
                            }

                            ctx.completed_requests.add(req_id);
                            Ok(())
                        }
                        .into(),
                        DataLocation::VRam(did) => async move {
                            let device = &ctx.device_contexts[did];
                            let garbage_collect_goal = device.storage.bytes_allocated()
                                / crate::storage::GARBAGE_COLLECT_GOAL_FRACTION as usize;

                            loop {
                                if device
                                    .storage
                                    .try_garbage_collect(device, garbage_collect_goal)
                                    > 0
                                {
                                    break;
                                } else {
                                    ctx.submit(Request::external_progress()).await;
                                }
                            }
                            ctx.completed_requests.add(req_id);
                            Ok(())
                        }
                        .into(),
                    };
                    self.task_manager.add_task(task_id, task);
                    self.task_graph
                        .add_implied(task_id, req_prio.downstream(TaskClass::GarbageCollect));
                }
                self.task_graph.will_fullfil_req(task_id, req_id);
            }
        }
    }
    fn enqueue_requested(&mut self, from: TaskId) {
        for req in self.data.request_queue.drain() {
            self.enqueue(from, req);
        }
    }

    fn cycle_cmd_buffers(&mut self, min_age: Duration) {
        for device in self.data.device_contexts {
            if device.cmd_buffer_age() >= min_age {
                if let Some(submitted) = device.try_submit_and_cycle_command_buffer() {
                    //println!("Cycling command buffer {:?}", submitted);
                    self.task_graph
                        .resolved_implied(RequestId::CmdBufferSubmission(submitted));
                }
            }
        }
    }

    fn wait_for_async_results(&mut self) {
        self.cycle_cmd_buffers(Duration::from_secs(0));

        let mut external_progress = false;

        let mut gpu_work_done = false;
        for device in self.data.device_contexts {
            let done_cmd_buffers = device.wait_for_cmd_buffers(WAIT_TIMEOUT_GPU);
            external_progress |= !done_cmd_buffers.is_empty();
            for done in done_cmd_buffers {
                self.task_graph
                    .resolved_implied(RequestId::CmdBufferCompletion(done.into()));
                gpu_work_done = true;
            }
        }

        let cpu_wait_timeout = if gpu_work_done {
            Duration::from_secs(0)
        } else {
            WAIT_TIMEOUT_CPU
        };
        let jobs = self.task_manager.wait_for_jobs(cpu_wait_timeout);
        external_progress |= !jobs.is_empty();
        for job_id in jobs {
            self.task_graph.resolved_implied(job_id.into());
        }

        if external_progress {
            self.task_graph
                .resolved_implied(RequestId::ExternalProgress);
        }
    }

    fn resolve<'call, R, F: FnOnce(OpaqueTaskContext<'cref, 'inv>) -> Task<'call, R>>(
        &'call mut self,
        task: F,
    ) -> Result<R, Error> {
        // It is not important that we have a unique id here since no persistent results are
        // generated by the associated task. This is ensured by specifying an output type of
        // `Never`.
        let op_id = OperatorId::new("RunTime::resolve");
        let task_id = self.request_batcher.task_id_manager.gen_id(op_id);
        let mut task = task(self.context(task_id));

        loop {
            let mut ctx = Context::from_waker(&self.waker);
            match task.as_mut().poll(&mut ctx) {
                Poll::Ready(res) => {
                    return res;
                }
                Poll::Pending => {
                    self.enqueue_requested(task_id);
                }
            };
            self.try_resolve_implied()?;
            assert!(self.data.request_queue.is_empty());
        }
    }
}

fn dummy_raw_waker() -> RawWaker {
    fn no_op(_: *const ()) {}
    fn clone(_: *const ()) -> RawWaker {
        dummy_raw_waker()
    }

    let vtable = &RawWakerVTable::new(clone, no_op, no_op, no_op);
    RawWaker::new(0 as *const (), vtable)
}

fn dummy_waker() -> Waker {
    let raw = dummy_raw_waker();
    // Safety: The dummy waker literally does nothing and thus upholds all cantracts of
    // `RawWaker`/`RawWakerVTable`.
    unsafe { Waker::from_raw(raw) }
}

pub struct Statistics {
    pub tasks_executed: usize,
}

impl Statistics {
    fn new() -> Self {
        Self { tasks_executed: 0 }
    }
}

//TODO: refactor these into some "completed requests" thingy. Should be easy enough for these two
//below, but should (if possible) also include completed data requests
#[derive(Default)]
pub(crate) struct CompletedRequests {
    completed: RefCell<Set<RequestId>>,
}
impl CompletedRequests {
    //fn set(&self, items: Set<RequestId>) {
    //    let mut completed = self.completed.borrow_mut();
    //    std::mem::replace(&mut *completed, items);
    //}
    fn add(&self, item: RequestId) {
        let mut completed = self.completed.borrow_mut();
        completed.insert(item);
    }
    fn take(&self) -> Set<RequestId> {
        let mut completed = self.completed.borrow_mut();
        std::mem::replace(&mut *completed, Set::new())
    }
}

#[derive(Default)]
pub struct TaskHints {
    pub completed: RefCell<Set<RequestId>>,
}

impl TaskHints {
    pub fn noticed_completion(&self, id: RequestId) {
        let mut completed = self.completed.borrow_mut();
        completed.remove(&id);
    }
}

pub struct RequestQueue<'inv> {
    buffer: RefCell<VecDeque<RequestInfo<'inv>>>,
}

impl<'inv> RequestQueue<'inv> {
    pub fn new() -> Self {
        Self {
            buffer: RefCell::new(VecDeque::new()),
        }
    }
    pub fn push(&self, req: RequestInfo<'inv>) {
        self.buffer.borrow_mut().push_back(req)
    }
    pub fn drain<'b>(&'b self) -> impl Iterator<Item = RequestInfo<'inv>> + 'b {
        self.buffer
            .borrow_mut()
            .drain(..)
            .collect::<Vec<_>>()
            .into_iter()
    }
    pub fn is_empty(&self) -> bool {
        self.buffer.borrow().is_empty()
    }
}
