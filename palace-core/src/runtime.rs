use ahash::{HashMapExt, HashSetExt};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    num::NonZeroU64,
    path::Path,
    sync::mpsc,
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
    time::{Duration, Instant},
};

use crate::{
    array::ChunkIndex,
    operator::{DataId, OpaqueOperator, OperatorDescriptor, OperatorId},
    storage::{disk, gpu::BarrierEpoch, ram, CpuDataLocation, DataLocation, VisibleDataLocation},
    task::{DataRequest, OpaqueTaskContext, Request, RequestInfo, RequestType, Task},
    task_graph::{Priority, RequestId, TaskClass, TaskGraph, TaskId, VisibleDataId},
    task_manager::{TaskManager, ThreadSpawner},
    threadpool::{ComputeThreadPool, IoThreadPool, JobInfo},
    util::{Map, Set},
    vulkan::{memory::TransferTaskResult, BarrierInfo, DeviceContext, DeviceId, VulkanContext},
    Error,
};

const CMD_BUF_EAGER_CYCLE_TIME: Duration = Duration::from_micros(300);
const WAIT_TIMEOUT_GPU: Duration = Duration::from_micros(100);
const WAIT_TIMEOUT_CPU: Duration = Duration::from_micros(100);
const STUCK_TIMEOUT: Duration = Duration::from_secs(5);

struct DataRequestItem {
    id: DataId,
    item: ChunkIndex,
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

struct UnfinishedBatch {
    items: Set<DataRequestItem>,
    id: TaskId,
}

struct OperatorBatches<'inv> {
    unfinished: Map<DataLocation, UnfinishedBatch>,
    finished: Map<TaskId, (DataLocation, Set<DataRequestItem>)>,
    requestor: Option<TaskId>,
    op: &'inv dyn OpaqueOperator,
    task_counter: crate::util::IdGenerator<u64>,
}

impl<'inv> OperatorBatches<'inv> {
    fn new(source: &'inv dyn OpaqueOperator) -> Self {
        let task_counter: crate::util::IdGenerator<u64> = Default::default();
        OperatorBatches {
            unfinished: Map::new(),
            finished: Map::new(),
            requestor: None,
            op: source,
            task_counter,
        }
    }
}

#[derive(Default)]
struct RequestBatcher<'inv> {
    pending_batches: Map<OperatorId, OperatorBatches<'inv>>,
}

enum BatchAddResult {
    New(TaskId),
    Existing(TaskId),
}

impl<'inv> RequestBatcher<'inv> {
    fn add(
        &mut self,
        request: DataRequest<'inv>,
        max_batch_size: usize,
        from: TaskId,
    ) -> BatchAddResult {
        let source = &*request.source;
        let op_id = source.op_id();
        let location = request.location.into();
        let req_item = DataRequestItem {
            id: request.id,
            item: request.item,
        };

        let batches = self
            .pending_batches
            .entry(op_id)
            .or_insert_with(|| OperatorBatches::new(source));

        let mut new_batch = false;
        let unfinished = batches.unfinished.entry(location).or_insert_with(|| {
            new_batch = true;
            UnfinishedBatch {
                id: TaskId::new(batches.op.op_id(), batches.task_counter.next() as _),
                items: Default::default(),
            }
        });

        unfinished.items.insert(req_item);
        let current_id = unfinished.id;

        let overly_full = unfinished.items.len() >= max_batch_size
            || batches.requestor.map(|t| t != from).unwrap_or(false);

        if overly_full {
            let batch = batches.unfinished.remove(&location).unwrap();
            batches.finished.insert(batch.id, (location, batch.items));
        }

        if new_batch {
            batches.requestor = Some(from); //TODO: why the hell would we need this?
            BatchAddResult::New(current_id)
        } else {
            BatchAddResult::Existing(current_id)
        }
    }

    fn get(&mut self, tid: TaskId) -> (&'inv dyn OpaqueOperator, DataLocation, Vec<ChunkIndex>) {
        let batches = self.pending_batches.get_mut(&tid.operator()).unwrap();
        let (loc, items) = if let Some((loc, _)) = batches.unfinished.iter().find(|v| v.1.id == tid)
        {
            let loc = *loc;
            let batch = batches.unfinished.remove(&loc).unwrap();
            assert_eq!(tid, batch.id);
            (loc, batch.items)
        } else {
            batches.finished.remove(&tid).unwrap()
        };

        let items = items.into_iter().map(|i| i.item).collect::<Vec<_>>();

        (batches.op, loc, items)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum BarrierItem {
    Data(VisibleDataId),
    Barrier(BarrierInfo, BarrierEpoch),
}

impl Into<RequestId> for BarrierItem {
    fn into(self) -> RequestId {
        match self {
            BarrierItem::Data(d) => RequestId::Data(d),
            BarrierItem::Barrier(info, e) => RequestId::Barrier(info, e),
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
            op_id: OperatorId::new("builtin::barrier_bat"),
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
                    let device = &ctx.device_contexts[&t.device];
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
pub struct RunTimeBuilder<'a> {
    num_compute_threads: usize,
    disk_cache_size: Option<usize>,
    disk_cache_path: &'a Path,
    devices: Vec<usize>, //Empty: any
    max_parallel_tasks: usize,
    max_requests_per_task: usize,
}

impl<'a> RunTimeBuilder<'a> {
    pub fn new() -> Self {
        Self {
            num_compute_threads: num_cpus::get(),
            disk_cache_size: None,
            disk_cache_path: &Path::new("./disk.cache"),
            devices: Vec::new(),
            max_parallel_tasks: crate::task_graph::DEFAULT_MAX_PARALLEL_TASKS_PER_OPERATOR,
            max_requests_per_task: 64,
        }
    }
    pub fn num_compute_threads(mut self, n: usize) -> Self {
        self.num_compute_threads = n;
        self
    }
    pub fn num_compute_threads_opt(self, n: Option<usize>) -> Self {
        if let Some(n) = n {
            self.num_compute_threads(n)
        } else {
            self
        }
    }
    pub fn disk_cache_size(mut self, n: usize) -> Self {
        self.disk_cache_size = Some(n);
        self
    }
    pub fn disk_cache_size_opt(self, n: Option<usize>) -> Self {
        if let Some(n) = n {
            self.disk_cache_size(n)
        } else {
            self
        }
    }
    pub fn disk_cache_path<'b>(self, path: &'b Path) -> RunTimeBuilder<'b>
    where
        'a: 'b,
    {
        let mut ret: RunTimeBuilder<'b> = self;
        ret.disk_cache_path = path;
        ret
    }
    pub fn devices(mut self, devices: Vec<usize>) -> Self {
        self.devices = devices;
        self
    }
    pub fn max_requests_per_task(mut self, n: usize) -> Self {
        self.max_requests_per_task = n;
        self
    }
    pub fn max_requests_per_task_opt(self, n: Option<usize>) -> Self {
        if let Some(n) = n {
            self.max_requests_per_task(n)
        } else {
            self
        }
    }
    pub fn max_parallel_tasks(mut self, n: usize) -> Self {
        self.max_parallel_tasks = n;
        self
    }
    pub fn max_parallel_tasks_opt(self, n: Option<usize>) -> Self {
        if let Some(n) = n {
            self.max_parallel_tasks(n)
        } else {
            self
        }
    }
    pub fn finish(self, storage_size: usize, gpu_storage_size: u64) -> Result<RunTime, Error> {
        let (async_result_sender, async_result_receiver) = mpsc::channel();
        let vulkan = VulkanContext::new(gpu_storage_size, self.devices)?;
        let ram = crate::storage::ram::RamAllocator::new(storage_size)?;
        let ram = crate::storage::cpu::Storage::new(ram);
        let disk = if let Some(size) = self.disk_cache_size {
            let disk = crate::storage::disk::MmapAllocator::new(self.disk_cache_path, size)?;
            Some(crate::storage::cpu::Storage::new(disk))
        } else {
            None
        };
        let frame = FrameNumber::first();
        Ok(RunTime {
            ram,
            disk,
            compute_thread_pool: ComputeThreadPool::new(
                async_result_sender.clone(),
                self.num_compute_threads,
            ),
            io_thread_pool: IoThreadPool::new(async_result_sender),
            async_result_receiver,
            vulkan,
            frame,
            max_parallel_tasks: self.max_parallel_tasks,
            max_requests_per_task: self.max_requests_per_task,
        })
    }
}

pub struct RunTime {
    pub ram: crate::storage::ram::Storage,
    pub disk: Option<crate::storage::disk::Storage>,
    pub vulkan: VulkanContext,
    pub compute_thread_pool: ComputeThreadPool,
    pub io_thread_pool: IoThreadPool,
    pub async_result_receiver: mpsc::Receiver<JobInfo>,
    frame: FrameNumber,
    max_parallel_tasks: usize,
    max_requests_per_task: usize,
}

#[derive(Copy, Clone, Debug)]
pub struct Deadline {
    pub interactive: Instant,
    pub refinement: Instant,
}

impl Deadline {
    pub fn for_frame_duration(last_frame: Instant, duration: Duration) -> Self {
        Self {
            interactive: last_frame + duration,
            refinement: last_frame + 3 * duration,
        }
    }

    pub fn never() -> Self {
        Self {
            interactive: Instant::now() + std::time::Duration::from_secs(1 << 32),
            refinement: Instant::now() + std::time::Duration::from_secs(1 << 32),
        }
    }
}

impl RunTime {
    pub fn build() -> RunTimeBuilder<'static> {
        RunTimeBuilder::new()
    }
    pub fn checked_device_id(&self, raw_id: usize) -> Option<DeviceId> {
        self.vulkan.checked_device_id(raw_id)
    }

    pub fn all_devices(&self) -> Vec<DeviceId> {
        self.vulkan
            .device_contexts()
            .values()
            .map(|v| v.id)
            .collect()
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
        deadline: Option<Deadline>,
        save_task_stream: bool,
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
            disk_cache: self.disk.as_ref(),
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
                task_graph: TaskGraph::new(self.max_parallel_tasks, save_task_stream),
                statistics: Statistics::new(),
                operator_info: Default::default(),
                transfer_manager: Default::default(),
                request_batcher: Default::default(),
                barrier_batcher: BarrierBatcher::new(),
                deadline: deadline.unwrap_or(Deadline::never()),
                start: Instant::now(),
                max_requests_per_task: self.max_requests_per_task,
            }
        };

        let res = executor.resolve(|ctx| task(ctx, &&()));

        if save_task_stream {
            crate::task_graph::save_task_stream(
                &executor.task_graph,
                Path::new("hltaskeventstream.json"),
            );
        }

        res
    }
}

impl Drop for RunTime {
    fn drop(&mut self) {
        // Safety: The runtime (including all references to storage) is dropped now, so no dangling
        // references will be left behind
        for (_, device) in self.vulkan.device_contexts() {
            unsafe { device.storage.free_vram() };
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[repr(transparent)]
pub struct FrameNumber(NonZeroU64);

impl FrameNumber {
    pub fn diff(self, other: Self) -> u64 {
        self.0.get() - other.0.get()
    }
    pub fn first() -> Self {
        FrameNumber(1.try_into().unwrap())
    }
}

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
    pub storage: &'cref ram::Storage,
    pub disk_cache: Option<&'cref disk::Storage>,
    device_contexts: &'cref BTreeMap<DeviceId, DeviceContext>,
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
    deadline: Deadline,
    start: Instant,
    max_requests_per_task: usize,
}

impl<'cref, 'inv> Executor<'cref, 'inv> {
    //pub fn statistics(&self) -> &Statistics {
    //    &self.statistics
    //}
    fn context(&self, current_task: TaskId) -> OpaqueTaskContext<'cref, 'inv> {
        OpaqueTaskContext {
            requests: &self.data.request_queue,
            storage: &self.data.storage,
            disk_cache: self.data.disk_cache,
            hints: &self.data.hints,
            completed_requests: &self.data.completed_requests,
            thread_pool: &self.data.thread_spawner,
            device_contexts: self.data.device_contexts,
            current_task,
            current_op: self.operator_info.get(&current_task.operator()).cloned(),
            current_frame: self.data.frame,
            predicted_preview_tasks: &self.data.predicted_preview_tasks,
            deadline: self.deadline,
            start: self.start,
        }
    }

    fn construct_task(&mut self, id: TaskId) -> Task<'cref> {
        if let Some(t) = self.barrier_batcher.get(self.context(id), id) {
            t
        } else {
            let (op, loc, batch) = self.request_batcher.get(id);
            let context = self.context(id);
            //Safety: The argument batch is precisely for the returned operator, and thus of the right
            //type.
            unsafe { op.compute(context, batch, loc) }
        }
    }

    fn try_resolve_implied(&mut self, root: TaskId) -> Result<(), Error> {
        self.wait_for_async_results();

        enum StuckState {
            Not,
            Reported,
            WaitingSince(std::time::Instant),
        }
        let mut stuck_state = StuckState::Not;
        loop {
            self.cycle_cmd_buffers(CMD_BUF_EAGER_CYCLE_TIME);
            self.wait_for_async_results_with_timeout(
                Duration::from_millis(0),
                Duration::from_millis(0),
            );

            let ready = self.task_graph.next_ready();
            if let Some(task_id) = ready {
                if task_id == root {
                    return Ok(());
                }

                stuck_state = StuckState::Not;

                let mut resolved_deps = Set::new();
                std::mem::swap(&mut resolved_deps, self.task_graph.resolved_deps(task_id));
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

                let cache_results = self.data.disk_cache.is_some()
                    && self
                        .operator_info
                        .get(&task_id.operator())
                        .map(|o| o.cache_results)
                        .unwrap_or(false);

                match task.as_mut().poll(&mut ctx) {
                    Poll::Ready(Ok(())) => {
                        assert!(self.data.request_queue.is_empty());
                        // Drain hints
                        self.data.hints.completed.replace(Set::new());
                        self.register_produced_data(task_id, cache_results);
                        self.task_graph.task_done(task_id);
                        self.task_manager.remove_task(task_id).unwrap();
                        self.statistics.tasks_executed += 1;
                    }
                    Poll::Ready(Err(e)) => {
                        self.register_produced_data(task_id, cache_results);
                        println!("Execution errored {}", e);
                        crate::task_graph::save(&self.task_graph, Path::new("task_graph.svg"));
                        crate::task_graph::save_task_stream(
                            &self.task_graph,
                            Path::new("hltaskeventstream.json"),
                        );
                        return Err(e);
                    }
                    Poll::Pending => {
                        // Return hints back to the task graph
                        *self.task_graph.resolved_deps(task_id) =
                            self.data.hints.completed.replace(Set::new());
                        self.register_produced_data(task_id, cache_results);
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

                            let s = bytesize::to_string(self.data.storage.size() as _, true);
                            eprintln!("Ram utilization: ({} capacity):", s,);
                            self.data.storage.print_usage();

                            for d in self.data.device_contexts.values() {
                                let c = d.storage.capacity();
                                let c = bytesize::to_string(c, true);
                                let a = bytesize::to_string(d.storage.allocated(), true);
                                eprintln!(
                                    "VRam utilization {:?}: {}/{}, epoch {:?}",
                                    d.id,
                                    a,
                                    c,
                                    d.current_epoch()
                                );
                                d.storage.print_usage();
                            }
                            crate::task_graph::save(&self.task_graph, Path::new("task_graph.svg"));
                            crate::task_graph::save_full_detail(
                                &self.task_graph,
                                Path::new("task_graph_detailed.svg"),
                            );
                            crate::task_graph::save_task_stream(
                                &self.task_graph,
                                Path::new("hltaskeventstream.json"),
                            );

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
                            self.task_graph.run_sanity_check();
                        } else {
                            stuck_state = StuckState::WaitingSince(stuck_time);
                        }
                    }
                    // Wait for commandbuffers, if anything at all has been written to it
                    // (-> Smallest min_age above 0)
                    self.cycle_cmd_buffers(Duration::from_nanos(1));
                    self.wait_for_async_results();
                } else {
                    return Ok(());
                }
            }
        }
    }

    fn register_produced_data_from(
        &mut self,
        task_id: TaskId,
        items: impl Iterator<Item = (DataId, DataLocation)>,
        and_cache: bool,
    ) {
        for (id, produced_loc) in items {
            self.task_graph.has_produced_data(task_id, id);

            let mut requested_locations = self.task_graph.data_requests(id);
            if and_cache {
                let e = requested_locations
                    .entry(VisibleDataLocation::CPU(CpuDataLocation::Disk))
                    .or_default();
                e.insert(TaskId::new(OperatorId::new("builtin::cacher"), 0));
            }
            for requested in requested_locations {
                // Prefer the "best" available location for transfers
                if self
                    .find_available_location(id, requested.0.into())
                    .unwrap()
                    == produced_loc
                {
                    let r_id = VisibleDataId {
                        id,
                        location: requested.0,
                    }
                    .into();
                    let req_prio = requested
                        .1
                        .iter()
                        .map(|tid| {
                            self.task_graph
                                .get_priority(*tid)
                                .unwrap_or(crate::task_graph::ROOT_PRIO)
                        })
                        .max()
                        .unwrap();

                    if let Some(task_id) =
                        self.try_make_available(id, produced_loc, requested.0, req_prio)
                    {
                        for requestor in requested.1.iter() {
                            if !self.task_graph.is_currently_being_fulfilled(*requestor, id) {
                                self.task_graph
                                    .will_provide_data_for(task_id, id, *requestor);
                            }
                        }
                    } else {
                        self.task_graph.resolved_implied(r_id);
                    }
                }
            }
        }
    }
    fn register_produced_data(&mut self, from: TaskId, and_cache: bool) {
        self.register_produced_data_from(from, self.data.storage.newest_data(), and_cache);
        if let Some(disk) = self.data.disk_cache {
            self.register_produced_data_from(from, disk.newest_data(), false);
        }
        for device in self.data.device_contexts.values() {
            self.register_produced_data_from(from, device.storage.newest_data(), and_cache);
        }
        for item in self.data.completed_requests.take() {
            if let RequestId::Data(id) = item {
                self.task_graph.has_produced_data(from, id.id);
            }
            self.task_graph.resolved_implied(item.into());
        }
    }

    fn gpu_to_cpu(
        &mut self,
        id: DataId,
        source_id: DeviceId,
        target: CpuDataLocation,
        req_prio: Priority,
    ) -> Option<TaskId> {
        let task_id = self.transfer_manager.next_id();
        let device = &self.data.device_contexts[&source_id];
        let access = device.storage.register_access(device, self.data.frame, id);
        let transfer_task = self.transfer_manager.transfer_gpu_to_cpu(
            self.context(task_id),
            &self.data.device_contexts[&source_id],
            access,
            target,
        );
        Some(match transfer_task {
            TransferTaskResult::New(t) => {
                self.task_manager.add_task(task_id, t);
                self.task_graph
                    .add_task(task_id, req_prio.downstream(TaskClass::Transfer));
                task_id
            }
            TransferTaskResult::Existing(task_id) => task_id,
        })
    }

    fn try_make_available(
        &mut self,
        id: DataId,
        from: DataLocation,
        to: VisibleDataLocation,
        req_prio: Priority,
    ) -> Option<TaskId> {
        match (from, to) {
            (
                DataLocation::CPU(CpuDataLocation::Ram),
                VisibleDataLocation::CPU(CpuDataLocation::Ram),
            ) => None,
            (
                DataLocation::CPU(CpuDataLocation::Disk),
                VisibleDataLocation::CPU(CpuDataLocation::Disk),
            ) => None,
            (DataLocation::GPU(source), VisibleDataLocation::GPU(target, dst_info))
                if target == source =>
            {
                let device = &self.data.device_contexts[&target];
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
                                    .add_task(t, req_prio.downstream(TaskClass::Barrier));
                                t
                            }
                            BatchAddResult::Existing(t) => t,
                        };
                        Some(task_id)
                    }
                }
            }
            (DataLocation::GPU(source), VisibleDataLocation::GPU(_, _)) => {
                // There is no direct way to transfer memory from one gpu to another. Hence we
                // first transfer it to the cpu. Once it arrives there, the transfer machinery will
                // be activated again to transfer it to the actual target gpu.
                // This is similar to how transfers happen first and the results are made visible
                // afterwards (see: other branch with source == target).
                self.gpu_to_cpu(id, source, CpuDataLocation::Ram, req_prio)
            }
            (DataLocation::CPU(source), VisibleDataLocation::GPU(target_id, _)) => {
                let task_id = self.transfer_manager.next_id();

                let ctx = self.context(task_id);
                let transfer_task = match source {
                    CpuDataLocation::Ram => {
                        let s = self.data.storage;
                        let access = s.register_access(self.data.frame, id);
                        let Ok(source) = s.read_raw(access) else {
                            panic!("Data should already be in ram");
                        };
                        self.transfer_manager.transfer_cpu_to_gpu(
                            ctx,
                            &self.data.device_contexts[&target_id],
                            source,
                        )
                    }
                    CpuDataLocation::Disk => {
                        let s = self.data.disk_cache.unwrap();
                        let access = s.register_access(self.data.frame, id);
                        let Ok(source) = s.read_raw(access) else {
                            panic!("Data should already be in disk cache");
                        };
                        self.transfer_manager.transfer_cpu_to_gpu(
                            ctx,
                            &self.data.device_contexts[&target_id],
                            source,
                        )
                    }
                };
                Some(match transfer_task {
                    TransferTaskResult::New(t) => {
                        self.task_manager.add_task(task_id, t);
                        self.task_graph
                            .add_task(task_id, req_prio.downstream(TaskClass::Transfer));
                        task_id
                    }
                    TransferTaskResult::Existing(task_id) => task_id,
                })
            }
            (DataLocation::GPU(source_id), VisibleDataLocation::CPU(target)) => {
                self.gpu_to_cpu(id, source_id, target, req_prio)
            }
            (DataLocation::CPU(source), VisibleDataLocation::CPU(target)) => {
                let task_id = self.transfer_manager.next_id();

                let ctx = self.context(task_id);
                let transfer_task = match source {
                    CpuDataLocation::Ram => {
                        let s = self.data.storage;
                        let access = s.register_access(self.data.frame, id);
                        let Ok(source) = s.read_raw(access) else {
                            panic!("Data should already be in ram");
                        };
                        self.transfer_manager
                            .transfer_cpu_to_cpu(ctx, source, target)
                    }
                    CpuDataLocation::Disk => {
                        let s = self.data.disk_cache.unwrap();
                        let access = s.register_access(self.data.frame, id);
                        let Ok(source) = s.read_raw(access) else {
                            panic!("Data should already be in disk cache");
                        };
                        self.transfer_manager
                            .transfer_cpu_to_cpu(ctx, source, target)
                    }
                };
                Some(match transfer_task {
                    TransferTaskResult::New(t) => {
                        self.task_manager.add_task(task_id, t);
                        self.task_graph
                            .add_task(task_id, req_prio.downstream(TaskClass::Transfer));
                        task_id
                    }
                    TransferTaskResult::Existing(task_id) => task_id,
                })
            }
        }
    }

    fn is_available_in(&self, datum: DataId, loc: DataLocation) -> bool {
        let version = crate::storage::DataVersion::Preview(self.data.frame);
        match loc {
            DataLocation::CPU(CpuDataLocation::Ram) => {
                self.data.storage.is_readable(datum, version)
            }
            DataLocation::CPU(CpuDataLocation::Disk) => {
                if let Some(disk) = self.data.disk_cache {
                    disk.is_readable(datum, version)
                } else {
                    false
                }
            }
            DataLocation::GPU(i) => self.data.device_contexts[&i]
                .storage
                .is_readable(datum, version),
        }
    }

    fn find_available_location(
        &self,
        datum: DataId,
        preferred: DataLocation,
    ) -> Option<DataLocation> {
        let mut locs = vec![preferred, DataLocation::CPU(CpuDataLocation::Ram)];
        locs.extend(
            self.data
                .device_contexts
                .keys()
                .cloned()
                .map(DataLocation::GPU),
        );
        locs.push(DataLocation::CPU(CpuDataLocation::Disk));

        for loc in locs {
            if self.is_available_in(datum, loc) {
                return Some(loc);
            }
        }
        None
    }

    fn enqueue(&mut self, from: TaskId, req: RequestInfo<'inv>) {
        let req_id = req.id();
        //TODO: We also want to increase the priority of a task if it was already requested...

        let req_prio = self.task_graph.get_priority(from).unwrap();
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
                // There is possibly a task running that fulfills this request. In that case we
                // don't need to do anything
                if !self
                    .task_graph
                    .is_currently_being_fulfilled(from, data_request.id)
                {
                    let op_id = data_request.source.op_id();
                    self.operator_info
                        .entry(op_id)
                        .or_insert(data_request.source.operator_descriptor());

                    let data_id = data_request.id;
                    let data_req_loc = data_request.location;

                    if let Some(available) =
                        self.find_available_location(data_id, data_req_loc.into())
                    {
                        // Data should not already be present => unwrap
                        let fulfiller_task_id = self
                            .try_make_available(data_id, available, data_req_loc, req_prio)
                            .unwrap();
                        self.task_graph
                            .will_provide_data_for(fulfiller_task_id, data_id, from);
                    } else {
                        let fullfillers = self.task_graph.who_will_provide_data(data_id);
                        if fullfillers.is_empty() {
                            let batch_size = match data_request.source.granularity() {
                                crate::operator::ItemGranularity::Single => 1,
                                crate::operator::ItemGranularity::Batched => {
                                    self.max_requests_per_task
                                }
                            };
                            // Add item to batcher to spawn later
                            let fulfiller_task_id =
                                match self.request_batcher.add(data_request, batch_size, from) {
                                    BatchAddResult::New(id) => {
                                        self.task_graph
                                            .add_task(id, req_prio.downstream(TaskClass::Data));
                                        id
                                    }
                                    BatchAddResult::Existing(id) => {
                                        assert_ne!(batch_size, 1);
                                        self.task_graph.try_increase_priority(
                                            id,
                                            req_prio.downstream(TaskClass::Data),
                                        );
                                        id
                                    }
                                };
                            self.task_graph
                                .will_provide_data_for(fulfiller_task_id, data_id, from);
                        } else {
                            assert_eq!(fullfillers.len(), 1);
                            if !self
                                .task_graph
                                .is_currently_being_fulfilled(from, data_request.id)
                            {
                                for fulfiller_task_id in fullfillers {
                                    self.task_graph.will_provide_data_for(
                                        fulfiller_task_id,
                                        data_id,
                                        from,
                                    );
                                }
                            }
                        }
                    }
                }
            }
            RequestType::Barrier(b_info, e) => {
                let id = match self
                    .barrier_batcher
                    .add(b_info, BarrierItem::Barrier(b_info, e))
                {
                    BatchAddResult::New(id) => {
                        self.task_graph
                            .add_task(id, req_prio.downstream(TaskClass::Barrier));
                        id
                    }
                    BatchAddResult::Existing(id) => {
                        self.task_graph
                            .try_increase_priority(id, req_prio.downstream(TaskClass::Barrier));
                        id
                    }
                };
                self.task_graph.will_fullfil_req(id, req_id);
            }
            RequestType::CmdBufferCompletion(_id) => {}
            RequestType::CmdBufferSubmission(_id) => {}
            RequestType::Allocation(id, alloc) => {
                let op_name = match alloc {
                    crate::task::AllocationRequest::Ram(_, _, CpuDataLocation::Ram) => {
                        "builtin::alloc_ram"
                    }
                    crate::task::AllocationRequest::Ram(_, _, CpuDataLocation::Disk) => {
                        "builtin::alloc_disk"
                    }
                    crate::task::AllocationRequest::VRam(_, _, _) => "builtin::alloc_vram",
                    crate::task::AllocationRequest::VRamBufRaw(_, _, _, _, _) => {
                        "builtin::alloc_vram_raw"
                    }
                    crate::task::AllocationRequest::VRamImageRaw(_, _, _) => {
                        "builtin::alloc_vram_img"
                    }
                };
                let task_id = TaskId::new(OperatorId::new(op_name), id.inner() as _);
                let ctx = self.context(task_id);
                let task = match alloc {
                    crate::task::AllocationRequest::Ram(
                        layout,
                        data_descriptor,
                        CpuDataLocation::Ram,
                    ) => async move {
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
                    crate::task::AllocationRequest::Ram(
                        layout,
                        data_descriptor,
                        CpuDataLocation::Disk,
                    ) => async move {
                        let disk = ctx.disk_cache.unwrap();
                        loop {
                            if let Ok(_) = disk.alloc(data_descriptor, layout) {
                                break;
                            } else {
                                ctx.submit(disk.wait_garbage_collect()).await;
                            }
                        }
                        ctx.completed_requests.add(id.into());
                        Ok(())
                    }
                    .into(),
                    crate::task::AllocationRequest::VRam(device_id, layout, data_descriptor) => {
                        async move {
                            let device = &ctx.device_contexts[&device_id];
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
                        let device = &ctx.device_contexts[&device_id];
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
                        let device = &ctx.device_contexts[&device_id];
                        let res = loop {
                            if let Ok(res) = device.storage.allocate_image(create_desc) {
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
                };
                self.task_manager.add_task(task_id, task);
                self.task_graph
                    .add_task(task_id, req_prio.downstream(TaskClass::Alloc));
                self.task_graph.will_fullfil_req(task_id, req_id);
            }
            RequestType::ThreadPoolJob(job, type_) => {
                self.task_manager.spawn_job(job, type_);
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
                    DataLocation::CPU(CpuDataLocation::Ram) => {
                        TaskId::new(OperatorId::new("builtin::gc_ram"), 0)
                    }
                    DataLocation::CPU(CpuDataLocation::Disk) => {
                        TaskId::new(OperatorId::new("builtin::gc_disk"), 0)
                    }
                    DataLocation::GPU(d) => {
                        TaskId::new(OperatorId::new("builtin::gc_vram"), d.inner())
                    }
                };
                let already_queued = self.task_graph.has_task(task_id);
                if !already_queued {
                    let ctx = self.context(task_id);

                    let task = match l {
                        DataLocation::CPU(CpuDataLocation::Ram) => async move {
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
                        DataLocation::CPU(CpuDataLocation::Disk) => async move {
                            let disk = ctx.disk_cache.unwrap();
                            let garbage_collect_goal = disk.size()
                                / crate::storage::GARBAGE_COLLECT_GOAL_FRACTION as usize;

                            loop {
                                if disk.try_garbage_collect(garbage_collect_goal) > 0 {
                                    break;
                                } else {
                                    panic!("Progress on disk cache should always be possible");
                                }
                            }

                            ctx.completed_requests.add(req_id);
                            Ok(())
                        }
                        .into(),
                        DataLocation::GPU(did) => {
                            let frame = self.data.frame;
                            async move {
                                let device = &ctx.device_contexts[&did];
                                let c = device.storage.capacity();
                                let garbage_collect_goal = c as usize
                                    / crate::storage::GARBAGE_COLLECT_GOAL_FRACTION as usize;

                                loop {
                                    if device.storage.try_garbage_collect(
                                        device,
                                        garbage_collect_goal,
                                        frame,
                                    ) > 0
                                    {
                                        break;
                                    } else {
                                        ctx.submit(Request::external_progress()).await;
                                    }
                                }
                                ctx.completed_requests.add(req_id);
                                Ok(())
                            }
                        }
                        .into(),
                    };
                    self.task_manager.add_task(task_id, task);
                    self.task_graph
                        .add_task(task_id, req_prio.downstream(TaskClass::GarbageCollect));
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
        for device in self.data.device_contexts.values() {
            match device.try_submit_and_cycle_command_buffer(min_age) {
                crate::vulkan::CmdBufferCycleResult::Submitted(id) => {
                    self.task_graph
                        .resolved_implied(RequestId::CmdBufferSubmission(id));
                }
                crate::vulkan::CmdBufferCycleResult::TooYoung => {}
                crate::vulkan::CmdBufferCycleResult::EmptyFinished(id) => {
                    self.task_graph
                        .resolved_implied(RequestId::CmdBufferSubmission(id));
                    self.task_graph
                        .resolved_implied(RequestId::CmdBufferCompletion(id));
                }
            }
        }
    }

    fn wait_for_async_results(&mut self) {
        self.wait_for_async_results_with_timeout(WAIT_TIMEOUT_CPU, WAIT_TIMEOUT_GPU);
    }
    fn wait_for_async_results_with_timeout(
        &mut self,
        timeout_cpu: Duration,
        timeout_gpu: Duration,
    ) {
        let mut external_progress = false;

        let mut gpu_work_done = false;
        for device in self.data.device_contexts.values() {
            let done_cmd_buffers = device.wait_for_cmd_buffers(timeout_gpu);
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
            timeout_cpu
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
        let task_id = TaskId::new(op_id, 0);
        let mut task = task(self.context(task_id));
        self.task_graph
            .add_task(task_id, crate::task_graph::ROOT_ORIGIN);

        let res = loop {
            let mut ctx = Context::from_waker(&self.waker);

            let mut resolved_deps = Set::new();
            std::mem::swap(&mut resolved_deps, self.task_graph.resolved_deps(task_id));
            let old_hints = self.data.hints.completed.replace(resolved_deps);
            assert!(old_hints.is_empty());

            match task.as_mut().poll(&mut ctx) {
                Poll::Ready(res) => {
                    self.task_graph.task_done(task_id);
                    break res;
                }
                Poll::Pending => {
                    self.enqueue_requested(task_id);
                }
            };

            *self.task_graph.resolved_deps(task_id) = self.data.hints.completed.replace(Set::new());

            self.try_resolve_implied(task_id)?;
            assert!(self.data.request_queue.is_empty());
        };

        // Resolve open caching tasks
        while self.task_graph.has_open_tasks() {
            self.try_resolve_implied(task_id)?;
        }

        res
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

    pub fn swap(&self, s: &mut Set<RequestId>) {
        std::mem::swap(&mut *self.completed.borrow_mut(), s)
    }
}

pub struct RequestQueue<'inv> {
    buffer: RefCell<Vec<RequestInfo<'inv>>>,
}

impl<'inv> RequestQueue<'inv> {
    pub fn new() -> Self {
        Self {
            buffer: RefCell::new(Vec::new()),
        }
    }
    pub fn push(&self, req: RequestInfo<'inv>) {
        self.buffer.borrow_mut().push(req)
    }
    pub fn replace(&self, v: Vec<RequestInfo<'inv>>) -> Vec<RequestInfo<'inv>> {
        std::mem::replace(&mut *self.buffer.borrow_mut(), v)
    }
    pub fn drain(&self) -> Vec<RequestInfo<'inv>> {
        std::mem::take(&mut *self.buffer.borrow_mut())
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.borrow().is_empty()
    }
}
