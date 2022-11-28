use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

use crate::{
    operator::OperatorId,
    storage::Storage,
    task::{DatumRequest, Task, TaskContext, TaskId, TaskInfo},
    Error,
};

pub struct RunTime<'op, 'queue, 'tasks> {
    waker: Waker,
    tasks: TaskGraph<'tasks>,
    storage: &'tasks Storage,
    request_queue: &'queue RequestQueue<'op>,
    hints: &'queue TaskHints,
    statistics: Statistics,
}

impl<'op, 'queue, 'tasks> RunTime<'op, 'queue, 'tasks>
where
    'op: 'queue,
    'queue: 'tasks,
{
    pub fn new(
        storage: &'tasks Storage,
        request_queue: &'queue RequestQueue<'op>,
        hints: &'queue TaskHints,
    ) -> Self {
        RunTime {
            waker: dummy_waker(),
            tasks: TaskGraph::new(),
            storage,
            request_queue,
            hints,
            statistics: Statistics::new(),
        }
    }
    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }
    fn context(&self) -> TaskContext<'op, 'tasks> {
        TaskContext {
            requests: self.request_queue,
            storage: self.storage,
            hints: self.hints,
        }
    }

    fn try_resolve_implied(&mut self) -> Result<(), Error> {
        loop {
            let ready = self.tasks.next_ready();
            if ready.is_empty() {
                return Ok(());
            }
            for ready in ready {
                let task_id = ready.id;
                let resolved_deps = ready.resolved_deps;
                self.hints.completed.set(resolved_deps);

                let mut ctx = Context::from_waker(&self.waker);
                let task = self.tasks.get_implied_mut(task_id);

                // The queue should always just contain the tasks that are enqueued by polling the
                // following task! This is important so that we know the calling tasks id for the
                // generated requests.
                assert!(self.request_queue.is_empty());

                match task.as_mut().poll(&mut ctx) {
                    Poll::Ready(Ok(_)) => {
                        self.tasks.resolved_implied(task_id);
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
            let context = self.context();
            if !self.tasks.exists(task_id) {
                let task = (req.task)(context);
                self.tasks.add_implied(task_id, task);
            }
            self.tasks
                .add_dependency(from, task_id, req.progress_indicator);
        }
    }

    /// Safety: The specified type must be the result of the operation
    pub fn resolve<'call, R, F: FnOnce(TaskContext<'op, 'tasks>) -> Task<'call, R>>(
        &'call mut self,
        task: F,
    ) -> Result<R, Error> {
        let op_id = OperatorId::new::<F>(&[]);
        let task_id = TaskId::new(op_id, &DatumRequest::Value);
        let mut task = task(self.context());

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
        }
    }
}

pub enum ProgressIndicator {
    PartialPossible,
    WaitForComplete,
}

struct TaskGraph<'a> {
    implied_tasks: BTreeMap<TaskId, Task<'a>>,
    deps: BTreeMap<TaskId, BTreeMap<TaskId, ProgressIndicator>>, // key requires values
    rev_deps: BTreeMap<TaskId, BTreeSet<TaskId>>,                // values require key
    ready: BTreeSet<TaskId>,
    resolved_deps: BTreeMap<TaskId, Vec<TaskId>>,
}

impl<'a> TaskGraph<'a> {
    fn new() -> Self {
        Self {
            implied_tasks: BTreeMap::new(),
            deps: BTreeMap::new(),
            rev_deps: BTreeMap::new(),
            ready: BTreeSet::new(),
            resolved_deps: BTreeMap::new(),
        }
    }

    fn exists(&self, id: TaskId) -> bool {
        self.implied_tasks.contains_key(&id)
    }

    fn add_implied(&mut self, id: TaskId, task: Task<'a>) {
        let prev = self.implied_tasks.insert(id, task);
        assert!(prev.is_none(), "Tried to insert task twice");
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
        let removed = self.implied_tasks.remove(&id);
        assert!(removed.is_some(), "Task was not present");
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

    fn get_implied_mut(&mut self, id: TaskId) -> &mut Task<'a> {
        self.implied_tasks.get_mut(&id).unwrap() //TODO: make api here nicer to avoid unwraps etc.
    }

    fn next_ready(&mut self) -> Vec<ReadyTask> {
        self.ready
            .iter()
            .filter(|t| self.implied_tasks.contains_key(&t))
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
