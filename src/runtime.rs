use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

use crate::{
    operator::{Network, OperatorId},
    storage::Storage,
    task::{DatumRequest, Task, TaskContext, TaskId, TaskInfo},
    Error,
};

pub struct RunTime<'a> {
    network: &'a Network,
    waker: Waker,
    tasks: TaskGraph<'a>,
    storage: &'a Storage,
    request_queue: &'a RequestQueue,
    statistics: Statistics,
}

impl<'a> RunTime<'a> {
    pub fn new(
        network: &'a Network,
        storage: &'a Storage,
        request_queue: &'a RequestQueue,
    ) -> Self {
        RunTime {
            network,
            waker: dummy_waker(),
            tasks: TaskGraph::new(),
            storage,
            request_queue,
            statistics: Statistics::new(),
        }
    }
    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }
    fn context(&self, op: OperatorId) -> TaskContext<'a> {
        TaskContext {
            requests: self.request_queue,
            storage: self.storage,
            active: op,
        }
    }

    pub fn run(&mut self) -> Result<(), Error> {
        loop {
            let ready = self.tasks.ready();
            if ready.is_empty() {
                return Ok(());
            }
            for task_id in ready {
                let mut ctx = Context::from_waker(&self.waker);
                let task = self.tasks.get_mut(task_id);

                // The queue should always just contain the tasks that are enqueued by polling the
                // following task! This is important so that we know the calling tasks id for the
                // generated requests.
                assert!(self.request_queue.is_empty());

                match task.as_mut().poll(&mut ctx) {
                    Poll::Ready(Ok(_)) => {
                        self.tasks.resolved(task_id);
                        self.statistics.tasks_executed += 1;
                    }
                    Poll::Ready(Err(e)) => {
                        panic!("Task executed with error: {}", e)
                    }
                    Poll::Pending => {
                        let caller_id = task_id;
                        for req in self.request_queue.drain() {
                            let Some(op) = self.network.get(req.operator) else {
                                return Err("Operator with specified id not found".into());
                            };

                            let task_id = req.id();
                            if !self.tasks.exists(task_id) {
                                let task = op.compute(self.context(req.operator), req.data);

                                self.tasks.add(task_id, task);
                            }
                            self.tasks.add_dependency(caller_id, task_id);
                        }
                    }
                };
            }
        }
    }

    /// Safety: The specified type must be the result of the operation
    pub unsafe fn request_blocking<T>(
        &mut self,
        op_id: OperatorId,
        req: DatumRequest,
    ) -> Result<&'a T, Error> {
        let info = TaskInfo::new(op_id, req);
        let Some(op) = self.network.get(op_id) else {
            return Err("Operator with specified id not found".into());
        };
        let task_id = info.id();
        let task = op.compute(self.context(op_id), info.data);

        self.tasks.add(task_id, task);

        // TODO this can probably be optimized to only compute the values necessary for the
        // requested task. (Although the question is if such a situation ever occurs with the
        // current API...)
        self.run()?;

        Ok(self.storage.read_ram(task_id).unwrap())
    }
}

struct TaskGraph<'a> {
    tasks: BTreeMap<TaskId, Task<'a>>,
    deps: BTreeMap<TaskId, BTreeSet<TaskId>>, // key requires values
    rev_deps: BTreeMap<TaskId, BTreeSet<TaskId>>, // values require key
    ready: BTreeSet<TaskId>,
}

impl<'a> TaskGraph<'a> {
    fn new() -> Self {
        Self {
            tasks: BTreeMap::new(),
            deps: BTreeMap::new(),
            rev_deps: BTreeMap::new(),
            ready: BTreeSet::new(),
        }
    }

    fn exists(&self, id: TaskId) -> bool {
        self.tasks.contains_key(&id)
    }

    fn add(&mut self, id: TaskId, task: Task<'a>) {
        let prev = self.tasks.insert(id, task);
        assert!(prev.is_none(), "Tried to insert task twice");
        self.ready.insert(id);
    }

    fn add_dependency(&mut self, wants: TaskId, wanted: TaskId) {
        self.deps.entry(wants).or_default().insert(wanted);
        self.rev_deps.entry(wanted).or_default().insert(wants);
        self.ready.remove(&wants);
    }

    fn resolved(&mut self, id: TaskId) {
        let removed = self.tasks.remove(&id);
        assert!(removed.is_some(), "Task was not present");
        self.ready.remove(&id);

        for rev_dep in self.rev_deps.remove(&id).iter().flatten() {
            let deps_of_rev_dep = self.deps.get_mut(&rev_dep).unwrap();
            let removed = deps_of_rev_dep.remove(&id);
            assert!(removed);
            if deps_of_rev_dep.is_empty() {
                let inserted = self.ready.insert(*rev_dep);
                assert!(inserted);
            }
        }
    }

    fn get_mut(&mut self, id: TaskId) -> &mut Task<'a> {
        self.tasks.get_mut(&id).unwrap() //TODO: make api here nicer to avoid unwraps etc.
    }

    fn ready(&self) -> Vec<TaskId> {
        self.ready.iter().cloned().collect()
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

pub struct RequestQueue {
    buffer: RefCell<VecDeque<TaskInfo>>,
}
impl RequestQueue {
    pub fn new() -> Self {
        Self {
            buffer: RefCell::new(VecDeque::new()),
        }
    }
    pub fn push(&self, req: TaskInfo) {
        self.buffer.borrow_mut().push_back(req)
    }
    pub fn drain<'a>(&'a self) -> impl Iterator<Item = TaskInfo> + 'a {
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
