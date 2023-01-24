use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

use crate::{
    operator::{DataId, OpaqueOperator, OperatorId, TypeErased},
    storage::Storage,
    task::{
        DataRequest, OpaqueTaskContext, RequestInfo, RequestType, Task, TaskContext, ThreadPoolJob,
    },
    task_graph::{RequestId, TaskGraph, TaskId},
    task_manager::{TaskManager, ThreadSpawner},
    threadpool::ThreadPool,
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
            id: request.id,
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
    pub storage: Storage,
    pub thread_pool: ThreadPool,
}

impl RunTime {
    pub fn new(storage_size: usize, num_threads: usize) -> Self {
        RunTime {
            storage: Storage::new(storage_size),
            thread_pool: ThreadPool::new(num_threads),
        }
    }

    pub fn context_anchor(&mut self) -> ContextAnchor {
        ContextAnchor {
            data: ContextData::new(&self.storage),
            thread_pool: &mut self.thread_pool,
        }
    }
}

/// An object that contains all data that will be later lent out to `TastContexts` via `Executor`.
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
    thread_pool: &'cref mut ThreadPool,
}

impl<'cref, 'inv> ContextAnchor<'cref, 'inv> {
    pub fn executor(&'cref mut self) -> Executor<'cref, 'inv> {
        Executor {
            data: &self.data,
            task_manager: TaskManager::new(self.thread_pool),
            waker: dummy_waker(),
            task_graph: TaskGraph::new(),
            thread_pool_waiting: VecDeque::new(),
            statistics: Statistics::new(),
            request_batcher: Default::default(),
        }
    }
}

pub struct ContextData<'cref, 'inv> {
    request_queue: RequestQueue<'inv>,
    hints: TaskHints,
    thread_spawner: ThreadSpawner,
    pub storage: &'cref Storage,
}

impl<'cref> ContextData<'cref, '_> {
    pub fn new(storage: &'cref Storage) -> Self {
        let request_queue = RequestQueue::new();
        let hints = TaskHints::new();
        let thread_spawner = ThreadSpawner::new();
        ContextData {
            request_queue,
            hints,
            thread_spawner,
            storage,
        }
    }
}

pub struct Executor<'cref, 'inv> {
    pub data: &'cref ContextData<'cref, 'inv>,
    task_manager: TaskManager<'cref>,
    request_batcher: RequestBatcher<'inv>,
    task_graph: TaskGraph,
    thread_pool_waiting: VecDeque<ThreadPoolJob>,
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
            self.handle_thread_pool();

            let ready = self.task_graph.next_ready();
            if ready.is_empty() {
                return Ok(());
            }
            for ready in ready {
                let task_id = ready.id;
                let resolved_deps = ready.resolved_deps;
                let old_hints = self.data.hints.completed.replace(resolved_deps);
                // We require polled tasks to empty the resolved_deps before yielding
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
                        self.task_graph.task_done(task_id);
                        self.task_manager.remove_task(task_id).unwrap();
                        self.statistics.tasks_executed += 1;
                    }
                    Poll::Ready(e) => {
                        return e;
                    }
                    Poll::Pending => {
                        self.enqueue_requested(task_id);
                    }
                };
                for d in self.data.storage.newest_data() {
                    self.task_graph.resolved_implied(d.into());
                }
            }
        }
    }

    fn enqueue_requested(&mut self, from: TaskId) {
        for req in self.data.request_queue.drain() {
            let req_id = req.id();
            match req.task {
                RequestType::Data(data_request) => {
                    if let Some(new_batch_id) = self.request_batcher.add(data_request) {
                        self.task_graph.add_implied(new_batch_id);
                    }
                    self.task_graph
                        .add_dependency(from, req_id, req.progress_indicator);
                }
                RequestType::ThreadPoolJob(job) => {
                    self.thread_pool_waiting.push_back(job);
                    self.task_graph
                        .add_dependency(from, req_id, req.progress_indicator);
                }
            }
        }
    }

    fn handle_thread_pool(&mut self) {
        while let Some(task) = self.task_manager.job_finished() {
            self.task_graph.resolved_implied(task.into());
        }

        // Kind of ugly with the checks and unwraps, but I think this is the easiest way to not
        // pull something from either collection if only either one is ready.
        while self.task_manager.pool_worker_available() && !self.thread_pool_waiting.is_empty() {
            let job = self.thread_pool_waiting.pop_front().unwrap();
            self.task_manager.spawn_job(job).unwrap();
        }
    }

    /// Safety: The specified type must be the result of the operation
    pub fn resolve<'call, R, F: FnOnce(TaskContext<'cref, 'inv, (), ()>) -> Task<'call, R>>(
        &'call mut self,
        task: F,
    ) -> Result<R, Error> {
        //It is not important that we have a unique id here since no persistent results are
        //generated by the associated task.
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
    completed: std::cell::Cell<Vec<RequestId>>,
}

impl TaskHints {
    pub fn new() -> Self {
        TaskHints {
            completed: std::cell::Cell::new(Vec::new()),
        }
    }
    pub fn drain_completed(&self) -> Vec<RequestId> {
        self.completed.take()
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
