use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

use crate::{
    operator::OperatorId,
    storage::Storage,
    task::{DatumRequest, RequestType, Task, TaskContext, TaskId, TaskInfo, ThreadPoolJob},
    task_manager::{TaskManager, ThreadSpawner},
    Error,
};

pub struct RunTime<'tasks, 'queue, 'op> {
    waker: Waker,
    task_graph: TaskGraph,
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
    fn context(&self, current: TaskId) -> TaskContext<'tasks, 'op> {
        TaskContext {
            requests: self.request_queue,
            storage: self.storage,
            hints: self.hints,
            thread_pool: self.thread_spawner,
            current_task: current,
        }
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

                let mut ctx = Context::from_waker(&self.waker);
                let task = self.task_manager.get_task(task_id).unwrap();

                // The queue should always just contain the tasks that are enqueued by polling the
                // following task! This is important so that we know the calling tasks id for the
                // generated requests.
                assert!(self.request_queue.is_empty());

                match task.as_mut().poll(&mut ctx) {
                    Poll::Ready(Ok(_)) => {
                        assert!(self.request_queue.is_empty());
                        self.task_graph.resolved_implied(task_id);
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
            }
        }
    }

    fn enqueue_requested(&mut self, from: TaskId) {
        for req in self.request_queue.drain() {
            let task_id = req.id();
            match req.task {
                RequestType::Data(task_constr) => {
                    let context = self.context(task_id);
                    if !self.task_graph.exists(task_id) {
                        let task = (task_constr)(context);
                        self.task_manager.add_task(task_id, task);
                        self.task_graph.add_implied(task_id);
                    }
                    self.task_graph
                        .add_dependency(from, task_id, req.progress_indicator);
                }
                RequestType::ThreadPoolJob(job) => {
                    self.task_graph
                        .thread_pool_waiting
                        .push_back((task_id, job));
                    self.task_graph
                        .add_dependency(from, task_id, req.progress_indicator);
                }
            }
        }
    }

    fn handle_thread_pool(&mut self) {
        while let Some(task) = self.task_manager.job_finished() {
            self.task_graph.resolved_implied(task);
        }

        // Kind of ugly with the checks and unwraps, but I think this is the easiest way to not
        // pull something from either collection if only either one is ready.
        while self.task_manager.pool_worker_available()
            && !self.task_graph.thread_pool_waiting.is_empty()
        {
            let (id, job) = self.task_graph.thread_pool_waiting.pop_front().unwrap();
            self.task_manager.spawn_job(id, job).unwrap();
        }
    }

    /// Safety: The specified type must be the result of the operation
    pub fn resolve<'call, R, F: FnOnce(TaskContext<'tasks, 'op>) -> Task<'call, R>>(
        &'call mut self,
        task: F,
    ) -> Result<R, Error> {
        // TODO: Not sure if passing F here is a valid way to generate a unique id? E.g. when
        // running in a loop with changed parameters. We should try this out
        let op_id = OperatorId::new::<F>(&[]);
        let task_id = TaskId::new(op_id, &DatumRequest::Value);
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
            assert!(self.request_queue.is_empty());
        }
    }
}

pub enum ProgressIndicator {
    PartialPossible,
    WaitForComplete,
}

struct TaskGraph {
    implied_tasks: BTreeSet<TaskId>,
    thread_pool_waiting: VecDeque<(TaskId, ThreadPoolJob)>,
    deps: BTreeMap<TaskId, BTreeMap<TaskId, ProgressIndicator>>, // key requires values
    rev_deps: BTreeMap<TaskId, BTreeSet<TaskId>>,                // values require key
    ready: BTreeSet<TaskId>,
    resolved_deps: BTreeMap<TaskId, Vec<TaskId>>,
}

impl TaskGraph {
    fn new() -> Self {
        Self {
            implied_tasks: BTreeSet::new(),
            thread_pool_waiting: VecDeque::new(),
            deps: BTreeMap::new(),
            rev_deps: BTreeMap::new(),
            ready: BTreeSet::new(),
            resolved_deps: BTreeMap::new(),
        }
    }

    fn exists(&self, id: TaskId) -> bool {
        self.implied_tasks.contains(&id)
    }

    fn add_implied(&mut self, id: TaskId) {
        let inserted = self.implied_tasks.insert(id);
        assert!(inserted, "Tried to insert task twice");
        self.ready.insert(id);
    }

    fn add_dependency(
        &mut self,
        wants: TaskId,
        wanted: TaskId,
        progress_indicator: ProgressIndicator,
    ) {
        assert_ne!(wants, wanted, "Tasks cannot wait on themselves");
        self.deps
            .entry(wants)
            .or_default()
            .insert(wanted, progress_indicator);
        self.rev_deps.entry(wanted).or_default().insert(wants);
        self.ready.remove(&wants);
    }

    fn resolved_implied(&mut self, id: TaskId) {
        self.implied_tasks.remove(&id);
        self.ready.remove(&id);

        for rev_dep in self.rev_deps.remove(&id).iter().flatten() {
            let deps_of_rev_dep = self.deps.get_mut(&rev_dep).unwrap();
            let progress_indicator = deps_of_rev_dep.remove(&id).unwrap();
            self.resolved_deps.entry(*rev_dep).or_default().push(id);
            if deps_of_rev_dep.is_empty()
                || matches!(progress_indicator, ProgressIndicator::PartialPossible)
            {
                self.ready.insert(*rev_dep);
            }
        }
    }

    fn next_ready(&mut self) -> Vec<ReadyTask> {
        self.ready
            .iter()
            .filter(|t| self.implied_tasks.contains(&t))
            .map(|t| ReadyTask {
                id: *t,
                resolved_deps: self.resolved_deps.remove(t).unwrap_or_default(),
            })
            .collect()
    }
}

struct ReadyTask {
    id: TaskId,
    resolved_deps: Vec<TaskId>,
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
    completed: std::cell::Cell<Vec<TaskId>>,
}

impl TaskHints {
    pub fn new() -> Self {
        TaskHints {
            completed: std::cell::Cell::new(Vec::new()),
        }
    }
    pub fn drain_completed(&self) -> Vec<TaskId> {
        self.completed.take()
    }
}

pub struct RequestQueue<'op> {
    buffer: RefCell<VecDeque<TaskInfo<'op>>>,
}
impl<'op> RequestQueue<'op> {
    pub fn new() -> Self {
        Self {
            buffer: RefCell::new(VecDeque::new()),
        }
    }
    pub fn push(&self, req: TaskInfo<'op>) {
        self.buffer.borrow_mut().push_back(req)
    }
    pub fn drain<'b>(&'b self) -> impl Iterator<Item = TaskInfo<'op>> + 'b {
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
