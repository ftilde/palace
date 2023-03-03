use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::mpsc,
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
    time::Duration,
};

use crate::{
    operator::{DataId, OpaqueOperator, OperatorId, TypeErased},
    storage::{DataLocation, Storage, StorageState},
    task::{DataRequest, OpaqueTaskContext, RequestInfo, RequestType, Task, TaskContext},
    task_graph::{RequestId, TaskGraph, TaskId},
    task_manager::{TaskManager, ThreadSpawner},
    threadpool::{ComputeThreadPool, IoThreadPool, JobInfo},
    vulkan::{DeviceContext, VulkanManager},
    Error,
};

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

#[derive(Default)]
struct TaskIdManager {
    counts: BTreeMap<OperatorId, usize>,
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
    items: BTreeSet<DataRequestItem>,
    op: &'inv dyn OpaqueOperator,
    //batch_id: TaskId,
}

#[derive(Default)]
struct RequestBatcher<'inv> {
    pending_batches: BTreeMap<OperatorId, RequestBatch<'inv>>,
    task_id_manager: TaskIdManager,
}

impl<'inv> RequestBatcher<'inv> {
    fn add(&mut self, request: DataRequest<'inv>) -> Option<TaskId> {
        let source = &*request.source;
        let op_id = source.id();
        let req_item = DataRequestItem {
            id: request.id.id,
            item: request.item,
        };
        match self.pending_batches.entry(op_id) {
            std::collections::btree_map::Entry::Vacant(o) => {
                let mut items = BTreeSet::new();
                items.insert(req_item);
                let batch_id = self.task_id_manager.gen_id(op_id);
                o.insert(RequestBatch {
                    items,
                    op: source,
                    //batch_id,
                });
                Some(batch_id)
            }
            std::collections::btree_map::Entry::Occupied(mut o) => {
                o.get_mut().items.insert(req_item);
                None
            }
        }
    }

    fn get(&mut self, op: OperatorId) -> (&'inv dyn OpaqueOperator, Vec<TypeErased>) {
        let batch = self.pending_batches.remove(&op).unwrap();
        let items = batch.items.into_iter().map(|i| i.item).collect::<Vec<_>>();
        (batch.op, items)
    }
}

pub struct RunTime {
    pub storage: StorageState,
    pub ram: crate::ram_allocator::Allocator,
    pub vulkan: VulkanManager,
    pub compute_thread_pool: ComputeThreadPool,
    pub io_thread_pool: IoThreadPool,
    pub async_result_receiver: mpsc::Receiver<JobInfo>,
}

impl RunTime {
    pub fn new(storage_size: usize, num_compute_threads: Option<usize>) -> Result<Self, Error> {
        let num_compute_threads = num_compute_threads.unwrap_or(num_cpus::get());
        let (async_result_sender, async_result_receiver) = mpsc::channel();
        let vulkan = VulkanManager::new()?;
        let ram = crate::ram_allocator::Allocator::new(storage_size)?;
        let storage = Default::default();
        Ok(RunTime {
            storage,
            ram,
            compute_thread_pool: ComputeThreadPool::new(
                async_result_sender.clone(),
                num_compute_threads,
            ),
            io_thread_pool: IoThreadPool::new(async_result_sender),
            async_result_receiver,
            vulkan,
        })
    }

    pub fn context_anchor(&mut self) -> ContextAnchor {
        ContextAnchor {
            data: ContextData::new(&self.storage, &self.ram, self.vulkan.device_contexts()),
            compute_thread_pool: &mut self.compute_thread_pool,
            io_thread_pool: &mut self.io_thread_pool,
            async_result_receiver: &mut self.async_result_receiver,
        }
    }
}

/// An object that contains all data that will be later lent out to `TaskContexts` via `Executor`.
///
/// A note on lifetime names:
/// `'cref` refers to lifetimes that live at least as long as the context data (i.e., this anchor).
/// This is also the lifetime of all `Tasks`, for example.
///
/// `'inv`, an invariant (!) lifetime (due to the use in a `RefCell` in `RequestQueue`), specifies
/// the lifetime of `OpaqueOperator` references as handled during the evaluation of the operator
/// network (used in `RequestQueue` and `RequestBatcher`).
pub struct ContextAnchor<'cref, 'inv> {
    data: ContextData<'cref, 'inv>,
    compute_thread_pool: &'cref mut ComputeThreadPool,
    io_thread_pool: &'cref mut IoThreadPool,
    pub async_result_receiver: &'cref mut mpsc::Receiver<JobInfo>,
}

impl<'cref, 'inv> ContextAnchor<'cref, 'inv> {
    pub fn executor(&'cref mut self) -> Executor<'cref, 'inv> {
        Executor {
            data: &self.data,
            task_manager: TaskManager::new(
                self.compute_thread_pool,
                self.io_thread_pool,
                self.async_result_receiver,
            ),
            waker: dummy_waker(),
            task_graph: TaskGraph::new(),
            statistics: Statistics::new(),
            transfer_manager: Default::default(),
            request_batcher: Default::default(),
        }
    }
}

pub struct ContextData<'cref, 'inv> {
    request_queue: RequestQueue<'inv>,
    hints: TaskHints,
    thread_spawner: ThreadSpawner,
    pub storage: Storage<'cref>,
    device_contexts: &'cref [DeviceContext],
}

impl<'cref> ContextData<'cref, '_> {
    pub fn new(
        storage: &'cref StorageState,
        ram: &'cref crate::ram_allocator::Allocator,
        device_contexts: &'cref [DeviceContext],
    ) -> Self {
        let request_queue = RequestQueue::new();
        let hints = TaskHints::new();
        let thread_spawner = ThreadSpawner::new();
        let vram = device_contexts.iter().map(|c| c.allocator()).collect();
        //TODO: Argue safety
        let storage = unsafe { Storage::new(storage, ram, vram) };
        ContextData {
            request_queue,
            hints,
            thread_spawner,
            storage,
            device_contexts,
        }
    }
}

pub struct Executor<'cref, 'inv> {
    pub data: &'cref ContextData<'cref, 'inv>,
    task_manager: TaskManager<'cref>,
    request_batcher: RequestBatcher<'inv>,
    task_graph: TaskGraph,
    transfer_manager: crate::vulkan::TransferManager,
    statistics: Statistics,
    waker: Waker,
}

impl<'cref, 'inv> Executor<'cref, 'inv> {
    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }
    fn context(&self, current_task: TaskId) -> OpaqueTaskContext<'cref, 'inv> {
        OpaqueTaskContext {
            requests: &self.data.request_queue,
            storage: &self.data.storage,
            hints: &self.data.hints,
            thread_pool: &self.data.thread_spawner,
            device_contexts: self.data.device_contexts,
            current_task,
        }
    }

    fn construct_task(&mut self, id: TaskId) -> Task<'cref> {
        let (op, batch) = self.request_batcher.get(id.operator());
        let context = self.context(id);
        //Safety: The argument batch is precisely for the returned operator, and thus of the right
        //type.
        unsafe { op.compute(context, batch) }
    }

    fn try_resolve_implied(&mut self) -> Result<(), Error> {
        loop {
            let ready = self.task_graph.next_implied_ready();
            if ready.is_empty() {
                if self.task_graph.has_open_tasks() {
                    self.wait_for_async_results();
                    continue;
                }

                return Ok(());
            }

            for task_id in ready {
                let resolved_deps =
                    if let Some(resolved_deps) = self.task_graph.resolved_deps(task_id) {
                        let mut tmp = BTreeSet::new();
                        std::mem::swap(&mut tmp, resolved_deps);
                        tmp
                    } else {
                        BTreeSet::new()
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
                        self.data.hints.completed.replace(BTreeSet::new());
                        self.task_graph.task_done(task_id);
                        self.task_manager.remove_task(task_id).unwrap();
                        self.statistics.tasks_executed += 1;
                    }
                    Poll::Ready(e) => {
                        return e;
                    }
                    Poll::Pending => {
                        // Return hints back to the task graph
                        let old_hints = self.data.hints.completed.replace(BTreeSet::new());
                        if let Some(resolved_deps) = self.task_graph.resolved_deps(task_id) {
                            *resolved_deps = old_hints;
                        }
                        self.enqueue_requested(task_id);
                    }
                };
                for d in self.data.storage.newest_data() {
                    for requested in self.task_graph.requested_locations(d.id) {
                        let data = d.id;
                        match (d.location, requested) {
                            (DataLocation::Ram, DataLocation::Ram) => {
                                self.task_graph.resolved_implied(d.into());
                            }
                            (DataLocation::VRam(source), DataLocation::VRam(target))
                                if target == source =>
                            {
                                self.task_graph.resolved_implied(d.into());
                            }
                            (DataLocation::VRam(_source), DataLocation::VRam(_target)) => {
                                panic!("VRam to VRam transfer not implemented, yet")
                            }
                            (DataLocation::Ram, target @ DataLocation::VRam(target_id)) => {
                                if !self.data.storage.present(data.in_location(target)) {
                                    let task_id = self.transfer_manager.next_id();
                                    let transfer_task = self.transfer_manager.transfer_to_gpu(
                                        self.context(task_id),
                                        &self.data.device_contexts[target_id],
                                        data,
                                    );
                                    self.task_manager.add_task(task_id, transfer_task);
                                    self.task_graph.add_implied(task_id);
                                }
                            }
                            (DataLocation::VRam(source_id), target @ DataLocation::Ram) => {
                                if !self.data.storage.present(data.in_location(target)) {
                                    let task_id = self.transfer_manager.next_id();
                                    let transfer_task = self.transfer_manager.transfer_to_cpu(
                                        self.context(task_id),
                                        &self.data.device_contexts[source_id],
                                        data,
                                    );
                                    self.task_manager.add_task(task_id, transfer_task);
                                    self.task_graph.add_implied(task_id);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn enqueue(&mut self, from: TaskId, req: RequestInfo<'inv>) {
        let req_id = req.id();
        let already_requested = self.task_graph.already_requested(req_id);
        self.task_graph
            .add_dependency(from, req_id, req.progress_indicator);
        match req.task {
            RequestType::Data(data_request) => {
                if !already_requested {
                    if let Some(new_batch_id) = self.request_batcher.add(data_request) {
                        self.task_graph.add_implied(new_batch_id);
                    }
                }
            }
            RequestType::CmdBufferCompletion(_id) => {}
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
        }
    }
    fn enqueue_requested(&mut self, from: TaskId) {
        for req in self.data.request_queue.drain() {
            self.enqueue(from, req);
        }
    }

    fn wait_for_async_results(&mut self) {
        for device in self.data.device_contexts {
            device.try_submit_and_cycle_command_buffer();
        }

        let timeout = Duration::from_micros(100);
        for device in self.data.device_contexts {
            for done in device.wait_for_cmd_buffers(timeout) {
                self.task_graph.resolved_implied(done.into());
            }
        }

        let jobs = self.task_manager.wait_for_jobs(timeout);
        for job_id in jobs {
            self.task_graph.resolved_implied(job_id.into());
        }
    }

    pub fn resolve<'call, R, F: FnOnce(TaskContext<'cref, 'inv, (), Never>) -> Task<'call, R>>(
        &'call mut self,
        task: F,
    ) -> Result<R, Error> {
        // It is not important that we have a unique id here since no persistent results are
        // generated by the associated task. This is ensured by specifying an output type of
        // `Never`.
        let op_id = OperatorId::new("RunTime::resolve");
        let task_id = self.request_batcher.task_id_manager.gen_id(op_id);
        let mut task = task(TaskContext::new(self.context(task_id)));

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

// TODO: Use ! when stable
pub enum Never {}

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

pub struct TaskHints {
    pub completed: RefCell<BTreeSet<RequestId>>,
}

impl TaskHints {
    pub fn new() -> Self {
        TaskHints {
            completed: RefCell::new(BTreeSet::new()),
        }
    }

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
