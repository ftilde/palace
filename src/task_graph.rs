use std::collections::{BTreeMap, BTreeSet};

use derive_more::From;

use crate::{
    operator::{DataId, OperatorId},
    threadpool::JobId,
};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, From)]
pub enum RequestId {
    Data(DataId),
    Job(JobId),
}

impl RequestId {
    pub fn unwrap_data(&self) -> DataId {
        match self {
            RequestId::Data(d) => *d,
            RequestId::Job(_) => panic!("Tried to unwrap DataId from RequestId::Job"),
        }
    }
}

pub enum ProgressIndicator {
    PartialPossible,
    WaitForComplete,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct TaskId {
    op: OperatorId,
    num: usize,
}

impl TaskId {
    pub fn new(op: OperatorId, num: usize) -> Self {
        Self { op, num }
    }
    pub fn operator(&self) -> OperatorId {
        self.op
    }
}

#[derive(Default)]
pub struct TaskGraph {
    implied_tasks: BTreeSet<TaskId>,
    waits_on: BTreeMap<TaskId, BTreeMap<RequestId, ProgressIndicator>>,
    required_by: BTreeMap<RequestId, BTreeSet<TaskId>>,
    ready: BTreeSet<TaskId>,
    resolved_deps: BTreeMap<TaskId, Vec<RequestId>>,
}

impl TaskGraph {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn add_dependency(
        &mut self,
        wants: TaskId,
        wanted: RequestId,
        progress_indicator: ProgressIndicator,
    ) {
        self.required_by.entry(wanted).or_default().insert(wants);

        self.waits_on
            .entry(wants)
            .or_default()
            .insert(wanted, progress_indicator);
        self.ready.remove(&wants);
    }

    pub fn add_implied(&mut self, id: TaskId) {
        let inserted = self.implied_tasks.insert(id);
        self.waits_on.insert(id, BTreeMap::new());
        assert!(inserted, "Tried to insert task twice");
        self.ready.insert(id);
    }

    pub fn resolved_implied(&mut self, id: RequestId) {
        for rev_dep in self.required_by.remove(&id).iter().flatten() {
            let deps_of_rev_dep = self.waits_on.get_mut(&rev_dep).unwrap();
            let progress_indicator = deps_of_rev_dep.remove(&id).unwrap();
            self.resolved_deps.entry(*rev_dep).or_default().push(id);
            if deps_of_rev_dep.is_empty()
                || matches!(progress_indicator, ProgressIndicator::PartialPossible)
            {
                self.ready.insert(*rev_dep);
            }
        }
    }

    pub fn task_done(&mut self, id: TaskId) {
        self.implied_tasks.remove(&id);
        self.ready.remove(&id);
        let deps = self.waits_on.remove(&id).unwrap();
        assert!(deps.is_empty());
    }

    pub fn next_ready(&mut self) -> Vec<ReadyTask> {
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

pub struct ReadyTask {
    pub id: TaskId,
    pub resolved_deps: Vec<RequestId>,
}
