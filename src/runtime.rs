use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

use crate::{
    operator::{DataId, OpaqueOperator, OperatorId, TypeErased},
    storage::Storage,
    task::{DataRequest, RequestInfo, RequestType, Task, TaskContext, ThreadPoolJob},
    task_graph::{RequestId, TaskGraph, TaskId},
    task_manager::{TaskManager, ThreadSpawner},
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

struct RequestBatch<'op> {
    items: BTreeSet<DataRequestItem>,
    op: &'op dyn OpaqueOperator,
    //batch_id: TaskId,
}

#[derive(Default)]
struct RequestBatcher<'op> {
    pending_batches: BTreeMap<OperatorId, RequestBatch<'op>>,
    task_id_manager: TaskIdManager,
}

impl<'op> RequestBatcher<'op> {
    unsafe fn add(&mut self, request: DataRequest) -> Option<TaskId> {
        let source = &*request.source;
        let op_id = source.id();
        let req_item = DataRequestItem {
            id: request.id,
            item: request.item,
        };
        match self.pending_batches.entry(*op_id) {
            std::collections::btree_map::Entry::Vacant(o) => {
                let mut items = BTreeSet::new();
                items.insert(req_item);
                let batch_id = self.task_id_manager.gen_id(*op_id);
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

    fn get(&mut self, op: OperatorId) -> (&'op dyn OpaqueOperator, Vec<TypeErased>) {
        let batch = self.pending_batches.remove(&op).unwrap();
        let items = batch.items.into_iter().map(|i| i.item).collect::<Vec<_>>();
        (batch.op, items)
    }
}

pub struct RunTime<'tasks, 'queue, 'op> {
    waker: Waker,
    task_graph: TaskGraph,
    thread_pool_waiting: VecDeque<ThreadPoolJob>,
    request_batcher: RequestBatcher<'tasks>,
    storage: &'tasks Storage,
    task_manager: TaskManager<'tasks>,
    thread_spawner: &'tasks ThreadSpawner,
    request_queue: &'queue RequestQueue<'op>,
    hints: &'queue TaskHints,
    statistics: Statistics,
}

impl<'tasks, 'queue: 'tasks, 'op: 'queue> RunTime<'tasks, 'queue, 'op> {
    pub fn new(
        storage: &'tasks Storage,
        task_manager: TaskManager<'tasks>,
        thread_spawner: &'tasks ThreadSpawner,
        request_queue: &'queue RequestQueue<'op>,
        hints: &'queue TaskHints,
    ) -> Self {
        RunTime {
            waker: dummy_waker(),
            task_graph: TaskGraph::new(),
            thread_pool_waiting: VecDeque::new(),
            request_batcher: Default::default(),
            storage,
            task_manager,
            thread_spawner,
            request_queue,
            hints,
            statistics: Statistics::new(),
        }
    }
    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }
    fn context(&self, current_task: TaskId, current_op: OperatorId) -> TaskContext<'tasks> {
        TaskContext {
            requests: self.request_queue,
            storage: self.storage,
            hints: self.hints,
            thread_pool: self.thread_spawner,
            current_task,
            current_op,
        }
    }

    fn construct_task(&mut self, id: TaskId) -> Task<'tasks> {
        let (op, batch) = self.request_batcher.get(id.operator());
        let context = self.context(id, id.operator());
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
                let old_hints = self.hints.completed.replace(resolved_deps);
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
                assert!(self.request_queue.is_empty());

                match task.as_mut().poll(&mut ctx) {
                    Poll::Ready(Ok(_)) => {
                        assert!(self.request_queue.is_empty());
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
                for d in self.storage.newest_data() {
                    self.task_graph.resolved_implied(d.into());
                }
            }
        }
    }

    fn enqueue_requested(&mut self, from: TaskId) {
        for req in self.request_queue.drain() {
            let req_id = req.id();
            match req.task {
                RequestType::Data(data_request) => {
                    if let Some(new_batch_id) = unsafe { self.request_batcher.add(data_request) } {
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
    pub fn resolve<'call, R, F: FnOnce(TaskContext<'tasks>) -> Task<'call, R>>(
        &'call mut self,
        task: F,
    ) -> Result<R, Error> {
        let op_id = OperatorId::new("bleh"); //TODO unique ID
        let task_id = self.request_batcher.task_id_manager.gen_id(op_id);
        let mut task = task(self.context(task_id, op_id));

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
            assert!(self.request_queue.is_empty());
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

pub struct RequestQueue<'op> {
    buffer: RefCell<VecDeque<RequestInfo>>,
    _marker: std::marker::PhantomData<&'op ()>,
}
// CHANGE_ME add unsafety info here somewhere
impl<'op> RequestQueue<'op> {
    pub fn new() -> Self {
        Self {
            buffer: RefCell::new(VecDeque::new()),
            _marker: Default::default(),
        }
    }
    pub fn push(&self, req: RequestInfo) {
        self.buffer.borrow_mut().push_back(req)
    }
    pub fn drain<'b>(&'b self) -> impl Iterator<Item = RequestInfo> + 'b {
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
