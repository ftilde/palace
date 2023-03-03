use std::collections::{BTreeMap, BTreeSet};

use derive_more::From;

use crate::{
    id::Id,
    operator::{DataId, OperatorId},
    storage::DataLocation,
    threadpool::JobId,
    vulkan::CmdBufferSubmissionId,
};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct LocatedDataId {
    pub id: DataId,
    pub location: DataLocation,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, From)]
pub enum RequestId {
    CmdBufferCompletion(CmdBufferSubmissionId),
    Data(LocatedDataId),
    Job(JobId),
    Group(GroupId),
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct GroupId(pub Id);

impl RequestId {
    pub fn unwrap_data(&self) -> LocatedDataId {
        match self {
            RequestId::Data(d) => *d,
            RequestId::Job(_) => panic!("Tried to unwrap DataId from RequestId::Job"),
            RequestId::Group(_) => panic!("Tried to unwrap DataId from RequestId::Group"),
            RequestId::CmdBufferCompletion(_) => {
                panic!("Tried to unwrap DataId from RequestId::CmdBufferCompletion")
            }
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
    implied_ready: BTreeSet<TaskId>,
    resolved_deps: BTreeMap<TaskId, BTreeSet<RequestId>>,
    in_groups: BTreeMap<RequestId, BTreeSet<GroupId>>,
    requested_locations: BTreeMap<DataId, BTreeSet<DataLocation>>,
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
        self.implied_ready.remove(&wants);

        if let RequestId::Data(d) = wanted {
            let entry = self.requested_locations.entry(d.id).or_default();
            entry.insert(d.location);
        }
    }

    pub fn requested_locations(&self, id: DataId) -> BTreeSet<DataLocation> {
        self.requested_locations.get(&id).unwrap().clone()
    }

    pub fn in_group(&mut self, in_: RequestId, group: GroupId) {
        let entry = self.in_groups.entry(in_).or_default();
        entry.insert(group);
    }

    pub fn add_implied(&mut self, id: TaskId) {
        let inserted = self.implied_tasks.insert(id);
        self.waits_on.insert(id, BTreeMap::new());
        assert!(inserted, "Tried to insert task twice");
        self.implied_ready.insert(id);
    }

    pub fn already_requested(&self, rid: RequestId) -> bool {
        self.required_by.contains_key(&rid)
    }

    pub fn resolved_implied(&mut self, id: RequestId) {
        if let RequestId::Data(d) = id {
            let entry = self.requested_locations.get_mut(&d.id).unwrap();
            entry.remove(&d.location);
            if entry.is_empty() {
                self.requested_locations.remove(&d.id);
            }
        }
        for rev_dep in self.required_by.remove(&id).iter().flatten() {
            let deps_of_rev_dep = self.waits_on.get_mut(&rev_dep).unwrap();
            let progress_indicator = deps_of_rev_dep.remove(&id).unwrap();
            let resolved_deps = self.resolved_deps.entry(*rev_dep).or_default();
            resolved_deps.insert(id);
            if let Some(groups) = self.in_groups.remove(&id) {
                for group in groups {
                    resolved_deps.insert(group.into());
                }
            }
            if deps_of_rev_dep.is_empty()
                || matches!(progress_indicator, ProgressIndicator::PartialPossible)
            {
                if self.implied_tasks.contains(rev_dep) {
                    self.implied_ready.insert(*rev_dep);
                }
            }
        }
    }

    pub fn task_done(&mut self, id: TaskId) {
        self.implied_tasks.remove(&id);
        self.implied_ready.remove(&id);
        self.resolved_deps.remove(&id);
        let deps = self.waits_on.remove(&id).unwrap();
        assert!(deps.iter().all(|(v, _)| matches!(v, RequestId::Group(_))));
    }

    pub fn next_implied_ready(&mut self) -> BTreeSet<TaskId> {
        let mut ready = BTreeSet::new();
        std::mem::swap(&mut self.implied_ready, &mut ready);
        ready
    }

    pub fn has_open_tasks(&self) -> bool {
        !self.implied_tasks.is_empty()
    }

    pub fn resolved_deps(&mut self, task: TaskId) -> Option<&mut BTreeSet<RequestId>> {
        self.resolved_deps.get_mut(&task)
    }
}
