use std::{collections::VecDeque, hash::Hash};

use crate::{
    operator::{DataId, OperatorId},
    storage::{gpu::BarrierEpoch, DataLocation, VisibleDataLocation},
    task::AllocationId,
    threadpool::JobId,
    util::{Map, Set},
    vulkan::{BarrierInfo, CmdBufferSubmissionId},
};
use ahash::HashMapExt;
use gs_core::EventStreamBuilder;
use id::Id;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct LocatedDataId {
    pub id: DataId,
    pub location: DataLocation,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct VisibleDataId {
    pub id: DataId,
    pub location: VisibleDataLocation,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum RequestId {
    CmdBufferCompletion(CmdBufferSubmissionId),
    CmdBufferSubmission(CmdBufferSubmissionId),
    Barrier(BarrierInfo, BarrierEpoch),
    Allocation(AllocationId),
    Data(VisibleDataId),
    Job(JobId),
    Group(GroupId),
    GarbageCollect(DataLocation),
    Ready,
    YieldOnce,
    ExternalProgress,
}

impl From<VisibleDataId> for RequestId {
    fn from(d: VisibleDataId) -> Self {
        RequestId::Data(d)
    }
}
impl From<JobId> for RequestId {
    fn from(value: JobId) -> Self {
        RequestId::Job(value)
    }
}
impl From<GroupId> for RequestId {
    fn from(value: GroupId) -> Self {
        RequestId::Group(value)
    }
}
impl From<AllocationId> for RequestId {
    fn from(value: AllocationId) -> Self {
        RequestId::Allocation(value)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct GroupId(pub Id);

impl RequestId {
    pub fn unwrap_data(&self) -> VisibleDataId {
        match self {
            RequestId::Data(d) => *d,
            RequestId::Allocation(..) => {
                panic!("Tried to unwrap DataId from RequestId::Allocation")
            }
            RequestId::Job(_) => panic!("Tried to unwrap DataId from RequestId::Job"),
            RequestId::Group(_) => panic!("Tried to unwrap DataId from RequestId::Group"),
            RequestId::CmdBufferCompletion(_) => {
                panic!("Tried to unwrap DataId from RequestId::CmdBufferCompletion")
            }
            RequestId::CmdBufferSubmission(_) => {
                panic!("Tried to unwrap DataId from RequestId::CmdBufferSubmission")
            }
            RequestId::Barrier(..) => {
                panic!("Tried to unwrap DataId from RequestId::Barrier")
            }
            RequestId::Ready => {
                panic!("Tried to unwrap DataId from RequestId::Ready")
            }
            RequestId::YieldOnce => {
                panic!("Tried to unwrap DataId from RequestId::YieldOnce")
            }
            RequestId::ExternalProgress => {
                panic!("Tried to unwrap DataId from RequestId::ExternalProgress")
            }
            RequestId::GarbageCollect(_) => {
                panic!("Tried to unwrap DataId from RequestId::GarbageCollect")
            }
        }
    }
}

#[derive(Copy, Clone)]
pub enum ProgressIndicator {
    PartialPossible,
    WaitForComplete,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
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

struct TaskMetadata {
    priority: Priority,
}

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum TaskClass {
    GarbageCollect = 1,
    Barrier = 2,
    Data = 3,
    Alloc = 4,
    Transfer = 5,
}

pub const ROOT_ORIGIN: TaskOrigin = TaskOrigin {
    level: 0,
    class: TaskClass::Data,
    progress: 0,
};
pub const ROOT_PRIO: Priority = Priority {
    origin: ROOT_ORIGIN,
    progress: 0,
    ts: 0,
};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct TaskOrigin {
    level: u32,
    class: TaskClass,
    progress: u32,
}

impl TaskOrigin {
    pub fn merge(&self, other: TaskOrigin) -> Self {
        //TODO: Maybe revisit this
        let progress = self.progress.max(other.progress);
        Self {
            level: self.level.max(other.level),
            class: self.class.max(other.class),
            progress,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Priority {
    pub origin: TaskOrigin,
    progress: u32,
    ts: u32,
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.prio().cmp(&other.prio())
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Priority {
    fn prio(&self) -> u64 {
        let progress = self.total_progress();
        // NOTE: The following does currently not work anymore since progress is not < (1<<32>
        //(((1 << 8) - (self.origin.level as u64)) << 56) + ((progress as u64) << 32) + (1 << 32)
        //    - (self.ts as u64)
        ((self.origin.class as u8 as u64) << 56)
            + (((1 << 8) - (self.origin.level as u64)) << 48)
            + (progress as u64)
    }

    fn total_progress(&self) -> u32 {
        self.progress + self.origin.progress
    }

    pub fn downstream(&self, class: TaskClass) -> TaskOrigin {
        TaskOrigin {
            level: self.origin.level + 1,
            class: class.max(self.origin.class),
            progress: self.total_progress(),
        }
    }
}

struct GraphEventStream {
    inner: EventStreamBuilder,
    enabled: bool,
}

impl GraphEventStream {
    pub fn new(enable_stream_recording: bool) -> Self {
        Self {
            inner: Default::default(),
            enabled: enable_stream_recording,
        }
    }
    fn emit_node(&mut self, action: StreamAction, item: impl EventStreamNode) {
        if self.enabled {
            let node = gs_core::Node {
                id: item.to_eventstream_id(),
                label: item.label(),
            };
            let e = match action {
                StreamAction::Add => gs_core::Event::AddNode(node),
                StreamAction::Remove => gs_core::Event::RemoveNode(node),
            };
            self.inner.add(e);
        }
    }

    fn node_add(&mut self, item: impl EventStreamNode) {
        self.emit_node(StreamAction::Add, item);
    }
    fn node_remove(&mut self, item: impl EventStreamNode) {
        self.emit_node(StreamAction::Remove, item);
    }

    fn edge(
        &mut self,
        action: StreamAction,
        from: impl EventStreamNode,
        to: impl EventStreamNode,
        count: usize,
    ) {
        if self.enabled {
            let edge = gs_core::Edge {
                from: from.to_eventstream_id(),
                to: to.to_eventstream_id(),
                label: format!("{}", count),
            };
            let e = match action {
                StreamAction::Add => gs_core::Event::AddEdge(edge),
                StreamAction::Remove => gs_core::Event::RemoveEdge(edge),
            };
            self.inner.add(e);
        }
    }
    fn edge_update(
        &mut self,
        from: impl EventStreamNode,
        to: impl EventStreamNode,
        count: usize,
        new_count: usize,
    ) {
        if self.enabled {
            let edge = gs_core::Edge {
                from: from.to_eventstream_id(),
                to: to.to_eventstream_id(),
                label: format!("{}", count),
            };
            let e = gs_core::Event::UpdateEdgeLabel(edge, format!("{}", new_count));
            self.inner.add(e);
        }
    }
    fn edge_add(&mut self, from: impl EventStreamNode, to: impl EventStreamNode, count: usize) {
        self.edge(StreamAction::Add, from, to, count)
    }
    fn edge_remove(&mut self, from: impl EventStreamNode, to: impl EventStreamNode, count: usize) {
        self.edge(StreamAction::Remove, from, to, count)
    }
}

struct HighLevelGraph {
    depends_on: Map<TaskId, Map<TaskId, Set<RequestId>>>,
    //provides_for: Map<TaskId, Map<TaskId, usize>>,
    event_stream: GraphEventStream,
}

impl Default for HighLevelGraph {
    fn default() -> Self {
        Self::new(false)
    }
}

fn pseudo_tid(t: TaskId) -> (bool, TaskId) {
    match t.op.1 {
        "builtin::alloc_ram"
        | "builtin::alloc_disk"
        | "builtin::alloc_vram"
        | "builtin::alloc_vram_raw"
        | "builtin::alloc_vram_img"
        | "builtin::gc_disk"
        | "builtin::gc_ram"
        | "builtin::transfer_mgr" => (true, TaskId::new(t.op, 0)),
        "builtin::gc_vram" => (true, t),
        _ => (false, t),
    }
}

impl HighLevelGraph {
    pub fn new(enable_stream_recording: bool) -> Self {
        Self {
            depends_on: Default::default(),
            event_stream: GraphEventStream::new(enable_stream_recording),
        }
    }
    fn add_task(&mut self, t: TaskId) {
        let (pseudo, t) = pseudo_tid(t);

        let e = self.depends_on.entry(t);
        if matches!(e, crate::util::MapEntry::Vacant(_)) {
            self.event_stream.node_add(t);
        } else {
            assert!(
                pseudo,
                "{:?} is already present in the map, and is not a pseudo task",
                t
            );
        }
        e.or_default();
        //assert!(self.provides_for.insert(t, Default::default()).is_none());
    }
    fn add_dependency(&mut self, from: TaskId, to: TaskId, req: RequestId) {
        let (_pseudo, from) = pseudo_tid(from);
        let (_pseudo, to) = pseudo_tid(to);

        let edge_entry = self
            .depends_on
            .entry(from)
            .or_default()
            .entry(to)
            .or_default();
        if edge_entry.insert(req) {
            let len = edge_entry.len();
            if len == 1 {
                self.event_stream.edge_add(from, to, 1);
            } else {
                self.event_stream.edge_update(from, to, len - 1, len);
            }
        }

        //let edge_entry = self
        //    .provides_for
        //    .entry(to)
        //    .or_default()
        //    .entry(from)
        //    .or_default();
        //*edge_entry += 1;
    }
    fn remove_dependency(&mut self, from: TaskId, to: TaskId, req: RequestId) {
        let (pseudo_to, from) = pseudo_tid(from);
        let (pseudo_from, to) = pseudo_tid(to);

        let from_entry = self.depends_on.get_mut(&from).unwrap();
        let edge_entry = from_entry.get_mut(&to);

        if let Some(edge_entry) = edge_entry {
            if edge_entry.remove(&req) {
                let len = edge_entry.len();
                if len == 0 {
                    self.event_stream.edge_remove(from, to, 1);
                    self.depends_on.get_mut(&from).unwrap().remove(&to).unwrap();
                } else {
                    self.event_stream.edge_update(from, to, len + 1, len);
                }
            }
        } else {
            if !pseudo_to && !pseudo_from {
                // If any of the tasks are pseudotasks, there might be duplicate removals (which we
                // just ignore)
                panic!("{:?} should dep on {:?} for {:?}", from, to, req);
            }
        }
        //let edge_entry = self
        //    .provides_for
        //    .entry(to)
        //    .or_default()
        //    .get_mut(&from)
        //    .unwrap();
        //*edge_entry -= 1;
    }
    fn remove_task(&mut self, t: TaskId) {
        let (pseudo, t) = pseudo_tid(t);

        if !pseudo {
            let depends_on = self.depends_on.get(&t).unwrap().clone();
            //let depends_on = self.depends_on.remove(&t).unwrap();
            //assert!(depends_on.is_empty(), "{:?} dep on {:?}", t, depends_on);
            for (dep, n) in depends_on {
                for req in n {
                    // This can happen when another task fulfills a requested unsolicitedly
                    self.remove_dependency(t, dep, req);
                }
                //assert!(n.is_empty(), "{:?} deps on {:?} for {:?}", t, dep, n);
            }
            //let _provided = self.provides_for.remove(&t).unwrap();
            //for provided in provided.into_iter() {
            //    let e = self
            //        .depends_on
            //        .get_mut(&provided.0)
            //        .unwrap()
            //        .remove(&t)
            //        .unwrap();
            //    assert_eq!(e, 0, "for {:?}", t);
            //    //self.event_stream.edge_remove(provided.0, t, provided.1);
            //}
            self.event_stream.node_remove(t);
        }
    }
}

const NUM_PARALLEL_TASKS_PER_OPERATOR: usize = 2;
#[derive(Default)]
struct PostponedOperatorTask {
    queue: VecDeque<(TaskId, Priority)>,
    num_active: usize,
}

#[derive(Default)]
pub struct TaskGraph {
    tasks: Map<TaskId, TaskMetadata>,
    waits_on: Map<TaskId, Map<RequestId, ProgressIndicator>>,
    required_by: Map<RequestId, Set<TaskId>>,
    will_provide_data: Map<TaskId, Set<DataId>>,
    data_provided_by: Map<DataId, Set<TaskId>>,
    will_fullfil_req: Map<TaskId, Set<RequestId>>,
    req_fullfil_by: Map<RequestId, TaskId>,
    ready: priority_queue::PriorityQueue<TaskId, Priority>,
    resolved_deps: Map<TaskId, Set<RequestId>>,
    in_groups: Map<RequestId, Set<GroupId>>,
    groups: Map<GroupId, Set<RequestId>>,
    data_requests: Map<DataId, Map<VisibleDataLocation, Set<TaskId>>>,
    request_to_active_fulfillers: Map<DataId, Map<TaskId /*req*/, TaskId /*fulfiller*/>>,
    high_level: HighLevelGraph,
    ts_counter: u32,
    postponed_data_operator_tasks: Map<OperatorId, PostponedOperatorTask>,
}

trait EventStreamNode {
    fn to_eventstream_id(&self) -> u64;
    fn label(&self) -> String;
}

impl EventStreamNode for TaskId {
    fn to_eventstream_id(&self) -> u64 {
        let mut hasher = xxhash_rust::xxh3::Xxh3Builder::new().with_seed(0).build();
        self.hash(&mut hasher);
        hasher.digest()
    }
    fn label(&self) -> String {
        //format!("{}{}{:?}", self.op.1, self.num, self.op.inner())
        format!("{}{}", self.op.1, self.num)
    }
}

impl EventStreamNode for RequestId {
    fn to_eventstream_id(&self) -> u64 {
        let mut hasher = xxhash_rust::xxh3::Xxh3Builder::new().with_seed(1).build();
        self.hash(&mut hasher);
        hasher.digest()
    }
    fn label(&self) -> String {
        format!("{:?}", self)
    }
}

impl EventStreamNode for DataId {
    fn to_eventstream_id(&self) -> u64 {
        let mut hasher = xxhash_rust::xxh3::Xxh3Builder::new().with_seed(2).build();
        self.hash(&mut hasher);
        hasher.digest()
    }
    fn label(&self) -> String {
        format!("{:?}", self)
    }
}

impl Drop for TaskGraph {
    fn drop(&mut self) {
        if std::thread::panicking() {
            println!("Panic detected, exporting task graph state...");
            export(&self);
        }
    }
}

impl TaskGraph {
    pub fn new(enable_stream_recording: bool) -> Self {
        let mut s = Self::default();
        s.high_level = HighLevelGraph::new(enable_stream_recording);
        s
    }

    pub fn add_dependency(
        &mut self,
        wants: TaskId,
        wanted: RequestId,
        progress_indicator: ProgressIndicator,
    ) {
        self.waits_on
            .entry(wants)
            .or_default()
            .insert(wanted, progress_indicator);

        self.required_by.entry(wanted).or_default().insert(wants);
        self.ready.remove(&wants);

        if let RequestId::Data(d) = wanted {
            let entry = self.data_requests.entry(d.id).or_default();
            let e = entry.entry(d.location.into());

            let loc = e.or_default();
            loc.insert(wants);
        }
    }

    //TODO: Try to avoid clone
    pub fn data_requests(&self, id: DataId) -> Map<VisibleDataLocation, Set<TaskId>> {
        // Note: May be None due to builtin::cacher
        self.data_requests.get(&id).cloned().unwrap_or_default()
    }

    pub fn is_currently_being_fulfilled(&self, requestor: TaskId, id: DataId) -> bool {
        self.request_to_active_fulfillers
            .get(&id)
            .map(|m| m.contains_key(&requestor))
            .unwrap_or(false)
    }

    pub fn in_group(&mut self, in_: RequestId, group: GroupId) {
        let entry = self.in_groups.entry(in_).or_default();
        entry.insert(group);
        let entry = self.groups.entry(group).or_default();
        entry.insert(in_);
    }

    pub fn who_will_provide_data(&self, data: DataId) -> Set<TaskId> {
        self.data_provided_by
            .get(&data)
            .cloned()
            .unwrap_or_default()
    }

    pub fn who_will_fullfil_req(&self, req: RequestId) -> Option<TaskId> {
        self.req_fullfil_by.get(&req).copied()
    }

    pub fn will_provide_data_for(&mut self, provider: TaskId, data: DataId, requestor: TaskId) {
        let entries = self.will_provide_data.entry(provider).or_default();
        entries.insert(data);
        let entry = self.data_provided_by.entry(data).or_default();
        entry.insert(provider);

        let prev = self
            .request_to_active_fulfillers
            .entry(data)
            .or_default()
            .insert(requestor, provider);
        if let Some(prev) = prev {
            panic!(
                "{:?} for {:?} will now be provided by {:?}, but is already by {:?}",
                data, requestor, provider, prev
            );
        }

        if let Some(location) = self.data_requests.get(&data).and_then(|r| {
            r.iter()
                .filter_map(|v| {
                    if v.1.contains(&requestor) {
                        Some(v.0)
                    } else {
                        None
                    }
                })
                .next()
        }) {
            self.high_level.add_dependency(
                requestor,
                provider,
                RequestId::Data(VisibleDataId {
                    id: data,
                    location: *location,
                }),
            );
        } else {
            if requestor.operator().1 != "builtin::cacher" {
                panic!(
                    "Trying to create dependency from unknown task {:?}",
                    requestor
                );
            }
        }
    }
    pub fn has_produced_data(&mut self, task: TaskId, data: DataId) {
        // Tasks may produce data that has not been requested, so the None-case here (when no task
        // was asked to provide this) ...
        if let Some(dpb_entry) = self.data_provided_by.get_mut(&data) {
            // and false-case here (when a task other than the requested provided it) are also possible
            if dpb_entry.remove(&task) {
                if dpb_entry.is_empty() {
                    self.data_provided_by.remove(&data);
                }

                let entry = self.will_provide_data.get_mut(&task).unwrap();
                assert!(entry.remove(&data));
                if entry.is_empty() {
                    self.will_provide_data.remove(&task);
                }
            }
        }

        // Again, may be none if not requested
        if let Some(fulfiller_entries) = self.request_to_active_fulfillers.get_mut(&data) {
            if let Some(data_request_entry) = self.data_requests.get(&data) {
                // Note: May be none if builtin::cacher produced it
                for loc in data_request_entry.iter() {
                    for requestor in loc.1 {
                        if fulfiller_entries.get(requestor) == Some(&task) {
                            self.high_level.remove_dependency(
                                *requestor,
                                task,
                                RequestId::Data(VisibleDataId {
                                    id: data,
                                    location: *loc.0,
                                }),
                            );
                        }
                    }
                }
            }

            fulfiller_entries.retain(|_k, v| *v != task);
            if fulfiller_entries.is_empty() {
                self.request_to_active_fulfillers.remove(&data);
            }
        }
    }

    pub fn will_fullfil_req(&mut self, task: TaskId, req: RequestId) {
        let entries = self.will_fullfil_req.entry(task).or_default();
        let newly_inserted = entries.insert(req);
        if newly_inserted {
            assert!(self.req_fullfil_by.insert(req, task).is_none());
        }

        for requestor in &self.required_by[&req] {
            self.high_level.add_dependency(*requestor, task, req);
        }
    }

    pub fn add_task(&mut self, id: TaskId, origin: TaskOrigin) {
        let next_ts = self.ts_counter + 1;
        self.ts_counter = next_ts;
        let priority = Priority {
            origin,
            progress: 0,
            ts: next_ts,
        };
        let inserted = self.tasks.insert(id, TaskMetadata { priority });
        assert!(inserted.is_none(), "Tried to insert task twice");

        self.resolved_deps.insert(id, Default::default());

        self.waits_on.insert(id, Map::new());

        self.high_level.add_task(id);

        let run_now = if origin.class == TaskClass::Data {
            let entry = self
                .postponed_data_operator_tasks
                .entry(id.operator())
                .or_default();
            if entry.num_active < NUM_PARALLEL_TASKS_PER_OPERATOR {
                entry.num_active += 1;
                true
            } else {
                entry.queue.push_back((id, priority));
                false
            }
        } else {
            true
        };

        if run_now {
            self.ready.push(id, priority);
        }
    }
    pub fn has_task(&self, id: TaskId) -> bool {
        self.tasks.contains_key(&id)
    }
    pub fn try_increase_priority(&mut self, id: TaskId, origin: TaskOrigin) {
        // Note that this does not change the priority of downstream tasks recursively! This is
        // fine, however, and in practice only meant for batched tasks (barriers and operator
        // tasks) whose priority is only updated as long as they are not started (and thus have no
        // downstream tasks).

        let task_md = self.tasks.get_mut(&id).unwrap();
        task_md.priority.origin = task_md.priority.origin.merge(origin);

        self.ready.change_priority(&id, task_md.priority);
    }
    pub fn get_priority(&mut self, id: TaskId) -> Option<Priority> {
        self.tasks.get(&id).map(|m| m.priority)
    }

    pub fn dependents(&self, id: RequestId) -> &Set<TaskId> {
        self.required_by.get(&id).unwrap()
    }

    pub fn resolved_implied(&mut self, id: RequestId) {
        if let RequestId::Data(d) = id {
            let entry = self.data_requests.get_mut(&d.id).unwrap();
            let _fulfilled_locs = entry.remove(&d.location).unwrap();

            if entry.is_empty() {
                self.data_requests.remove(&d.id);
            }
        }

        let required_by = self.required_by.remove(&id);

        if let Some(by) = self.req_fullfil_by.remove(&id) {
            for rev_dep in required_by.iter().flatten() {
                self.high_level.remove_dependency(*rev_dep, by, id);
            }
        }

        for rev_dep in required_by.iter().flatten() {
            let deps_of_rev_dep = self.waits_on.get_mut(&rev_dep).unwrap();
            let progress_indicator = deps_of_rev_dep.remove(&id).unwrap();
            let resolved_deps = self.resolved_deps.get_mut(rev_dep).unwrap();
            resolved_deps.insert(id);

            if let Some(md) = self.tasks.get_mut(&rev_dep) {
                if let RequestId::Data(_) = id {
                    md.priority.progress += 1;
                }

                if deps_of_rev_dep.is_empty()
                    || matches!(progress_indicator, ProgressIndicator::PartialPossible)
                {
                    self.ready.push(*rev_dep, md.priority);
                }
            }
        }

        let mut resolved_groups = Vec::new();
        if let Some(groups) = self.in_groups.remove(&id) {
            for group in groups {
                let group_members = self.groups.get_mut(&group).unwrap();
                group_members.remove(&id);
                if group_members.is_empty() {
                    resolved_groups.push(group);
                    self.groups.remove(&group);
                }
            }
        }
        for group in resolved_groups {
            self.resolved_implied(group.into());
        }
    }

    pub fn run_sanity_check(&mut self) {
        assert!(self.ready.is_empty());
        for t in self.tasks.keys() {
            if let Some(waiting_for) = self.waits_on.get(t) {
                if waiting_for.is_empty() {
                    println!("Task {:?} waiting requests is empty", t);
                    if let Some(post) = self.postponed_data_operator_tasks.get(&t.operator()) {
                        println!(
                            "Operator {:?} has postponed tasks, {} active",
                            t.operator(),
                            post.num_active
                        );
                        if post.queue.iter().find(|i| i.0 == *t).is_some() {
                            println!("And its in the waiting queue");
                        }
                    }
                } else {
                    println!("Task {:?} waits on", t);
                    for w in waiting_for {
                        println!("\t{:?}", w.0);
                        if let RequestId::Data(id) = w.0 {
                            println!(
                                "Will be provided by: {:?}",
                                self.data_provided_by.get(&id.id)
                            );
                            println!("Requested at: {:?}", self.data_requests.get(&id.id));
                        }
                    }
                }
            } else {
                println!("Task {:?} is not in 'waiting_for'", t);
            }
        }
    }

    pub fn task_done(&mut self, id: TaskId) {
        self.tasks.remove(&id);
        self.ready.remove(&id);
        self.resolved_deps.remove(&id);

        let _wpd = self.will_provide_data.remove(&id);
        //assert!(wpd.is_none());
        let _wfr = self.will_fullfil_req.remove(&id);

        let deps = self.waits_on.remove(&id).unwrap();
        assert!(deps.is_empty());
        //assert!(deps.iter().all(|(v, _)| matches!(v, RequestId::Group(_))));

        self.high_level.remove_task(id);

        if let Some(entry) = self.postponed_data_operator_tasks.get_mut(&id.operator()) {
            if let Some((next_id, priority)) = entry.queue.pop_front() {
                self.ready.push(next_id, priority);
            } else {
                entry.num_active -= 1;
            }
            if entry.num_active == 0 {
                self.postponed_data_operator_tasks.remove(&id.operator());
            }
        }
    }

    pub fn next_ready(&mut self) -> Option<TaskId> {
        self.ready.pop().map(|(k, _v)| {
            //println!("Scheduling {:?} with prio {:?}", k, _v);
            k
        })
    }

    pub fn has_open_tasks(&self) -> bool {
        !self.tasks.is_empty()
    }

    pub fn resolved_deps(&mut self, task: TaskId) -> &mut Set<RequestId> {
        self.resolved_deps.get_mut(&task).unwrap()
    }
}

enum StreamAction {
    Add,
    Remove,
}

pub fn export_full_detail(task_graph: &TaskGraph) {
    use graphviz_rust::attributes::EdgeAttributes;
    use graphviz_rust::cmd::*;
    use graphviz_rust::dot_structures::{Edge, Graph, Id, Node, NodeId, Stmt};
    use graphviz_rust::dot_structures::{EdgeTy, Vertex};
    use graphviz_rust::exec;
    use graphviz_rust::printer::*;
    use graphviz_rust::{
        attributes::{self, color, NodeAttributes},
        into_attr::IntoAttribute,
    };

    let mut stmts = Vec::new();
    let mut id_counter = 0;
    let task_nodes = task_graph
        .waits_on
        .keys()
        .map(|k| {
            let label = format!("\"{}{}\"", k.op.1, k.num);
            id_counter += 1;
            let id = id_counter.to_string();
            let node_id = NodeId(Id::Plain(id), None);
            let mut attributes = Vec::new();
            attributes.push(NodeAttributes::label(label));
            attributes.push(NodeAttributes::shape(attributes::shape::rectangle));
            let node = Node {
                id: node_id.clone(),
                attributes,
            };
            stmts.push(Stmt::Node(node));
            (*k, node_id)
        })
        .collect::<Map<_, _>>();

    let request_nodes = task_graph
        .required_by
        .keys()
        .map(|k| {
            let label = format!("\"{:?}\"", k);
            id_counter += 1;
            let id = id_counter.to_string();
            let node_id = NodeId(Id::Plain(id), None);
            let mut attributes = Vec::new();
            attributes.push(color::default().into_attr());
            attributes.push(NodeAttributes::label(label));
            attributes.push(NodeAttributes::shape(attributes::shape::ellipse));
            let node = Node {
                id: node_id.clone(),
                attributes,
            };
            stmts.push(Stmt::Node(node));
            (*k, node_id)
        })
        .collect::<Map<_, _>>();

    let data_nodes = task_graph
        .data_requests
        .keys()
        .map(|k| {
            let label = format!("\"{:?}\"", k);
            id_counter += 1;
            let id = id_counter.to_string();
            let node_id = NodeId(Id::Plain(id), None);
            let mut attributes = Vec::new();
            attributes.push(color::default().into_attr());
            attributes.push(NodeAttributes::label(label));
            attributes.push(NodeAttributes::shape(attributes::shape::ellipse));
            let node = Node {
                id: node_id.clone(),
                attributes,
            };
            stmts.push(Stmt::Node(node));
            (*k, node_id)
        })
        .collect::<Map<_, _>>();

    for (t, r) in &task_graph.waits_on {
        for (r, _dep_type) in r {
            let mut attributes = Vec::new();
            attributes.push(color::default().into_attr());
            attributes.push(EdgeAttributes::arrowhead(attributes::arrowhead::vee));
            let edge = Edge {
                ty: EdgeTy::Pair(
                    Vertex::N(task_nodes.get(t).unwrap().clone()),
                    Vertex::N(request_nodes.get(r).unwrap().clone()),
                ),
                attributes,
            };
            stmts.push(Stmt::Edge(edge))
        }
    }

    for (t, r) in &task_graph.will_provide_data {
        for r in r {
            if let Some(r) = data_nodes.get(r).cloned() {
                let mut attributes = Vec::new();
                attributes.push(color::default().into_attr());
                attributes.push(EdgeAttributes::arrowhead(attributes::arrowhead::vee));
                let edge = Edge {
                    ty: EdgeTy::Pair(Vertex::N(r), Vertex::N(task_nodes.get(t).unwrap().clone())),
                    attributes,
                };
                stmts.push(Stmt::Edge(edge))
            }
        }
    }

    for (t, r) in &task_graph.will_fullfil_req {
        for r in r {
            let r = request_nodes.get(r).cloned().unwrap();
            let mut attributes = Vec::new();
            attributes.push(color::default().into_attr());
            attributes.push(EdgeAttributes::arrowhead(attributes::arrowhead::vee));
            let edge = Edge {
                ty: EdgeTy::Pair(Vertex::N(r), Vertex::N(task_nodes.get(t).unwrap().clone())),
                attributes,
            };
            stmts.push(Stmt::Edge(edge))
        }
    }

    for (d, l) in &task_graph.data_requests {
        for l in l.keys() {
            let mut attributes = Vec::new();
            attributes.push(color::default().into_attr());
            let r_id = d.with_visibility(*l).into();
            let edge = Edge {
                ty: EdgeTy::Pair(
                    Vertex::N(request_nodes.get(&r_id).unwrap().clone()),
                    Vertex::N(data_nodes.get(d).unwrap().clone()),
                ),
                attributes,
            };
            stmts.push(Stmt::Edge(edge))
        }
    }

    let graph = Graph::DiGraph {
        id: Id::Plain(format!("TaskGraph")),
        strict: true,
        stmts,
    };

    let filename = "taskgraph.svg";

    let mut ctx = PrinterContext::default();
    ctx.always_inline();
    let _empty = exec(
        graph,
        &mut ctx,
        vec![
            CommandArg::Format(Format::Svg),
            CommandArg::Layout(Layout::Dot),
            CommandArg::Output(filename.to_string()),
        ],
    )
    .unwrap();
    println!("Finished writing dependency graph to file: {}", filename);

    let filename = "hltaskeventstream.json";

    if task_graph.high_level.event_stream.enabled {
        task_graph
            .high_level
            .event_stream
            .inner
            .stream
            .save(std::path::Path::new(filename));
        println!(
            "Finished writing high level event stream to file: {}",
            filename
        );
    } else {
        println!("Task event stream recording was not enabled",);
    }
}

pub fn export(task_graph: &TaskGraph) {
    use graphviz_rust::attributes::EdgeAttributes;
    use graphviz_rust::cmd::*;
    use graphviz_rust::dot_structures::{Edge, Graph, Id, Node, NodeId, Stmt};
    use graphviz_rust::dot_structures::{EdgeTy, Vertex};
    use graphviz_rust::exec;
    use graphviz_rust::printer::*;
    use graphviz_rust::{
        attributes::{self, color, NodeAttributes},
        into_attr::IntoAttribute,
    };

    let mut stmts = Vec::new();
    let mut id_counter = 0;
    let task_nodes = task_graph
        .waits_on
        .keys()
        .map(|k| {
            let label = format!("\"{}{}\"", k.op.1, k.num);
            id_counter += 1;
            let id = id_counter.to_string();
            let node_id = NodeId(Id::Plain(id), None);
            let mut attributes = Vec::new();
            attributes.push(NodeAttributes::label(label));
            attributes.push(NodeAttributes::shape(attributes::shape::rectangle));
            let node = Node {
                id: node_id.clone(),
                attributes,
            };
            stmts.push(Stmt::Node(node));
            (*k, node_id)
        })
        .collect::<Map<_, _>>();

    let request_nodes = task_graph
        .required_by
        .keys()
        .filter_map(|k| {
            if !matches!(k, RequestId::Data(_) | RequestId::Group(_)) {
                let label = format!("\"{:?}\"", k);
                id_counter += 1;
                let id = id_counter.to_string();
                let node_id = NodeId(Id::Plain(id), None);
                let mut attributes = Vec::new();
                attributes.push(color::default().into_attr());
                attributes.push(NodeAttributes::label(label));
                attributes.push(NodeAttributes::shape(attributes::shape::ellipse));
                let node = Node {
                    id: node_id.clone(),
                    attributes,
                };
                stmts.push(Stmt::Node(node));
                Some((*k, node_id))
            } else {
                None
            }
        })
        .collect::<Map<_, _>>();

    for (source, data_ids) in &task_graph.will_provide_data {
        for (target, request_ids) in &task_graph.waits_on {
            let mut count = 0;
            for req in request_ids {
                if let RequestId::Data(d) = req.0 {
                    if data_ids.contains(&d.id) {
                        count += 1;
                    }
                }
            }
            if count > 0 {
                let mut attributes = Vec::new();
                attributes.push(color::default().into_attr());
                attributes.push(EdgeAttributes::arrowhead(attributes::arrowhead::vee));
                let edge = Edge {
                    ty: EdgeTy::Chain(vec![
                        Vertex::N(task_nodes.get(&target).unwrap().clone()),
                        Vertex::N({
                            id_counter += 1;
                            let id = id_counter.to_string();
                            let node_id = NodeId(Id::Plain(id), None);
                            let mut attributes = Vec::new();
                            let label = format!("\"{:?}\"", count);
                            attributes.push(color::default().into_attr());
                            attributes.push(NodeAttributes::label(label));
                            attributes.push(NodeAttributes::shape(attributes::shape::ellipse));

                            stmts.push(Stmt::Node(Node {
                                id: node_id.clone(),
                                attributes,
                            }));
                            node_id
                        }),
                        Vertex::N(task_nodes.get(&source).unwrap().clone()),
                    ]),
                    attributes,
                };
                stmts.push(Stmt::Edge(edge))
            }
        }
    }

    for (t, r) in &task_graph.waits_on {
        for (r, _dep_type) in r {
            if !matches!(r, RequestId::Data(_) | RequestId::Group(_)) {
                let mut attributes = Vec::new();
                attributes.push(color::default().into_attr());
                attributes.push(EdgeAttributes::arrowhead(attributes::arrowhead::vee));
                let edge = Edge {
                    ty: EdgeTy::Pair(
                        Vertex::N(task_nodes.get(t).unwrap().clone()),
                        Vertex::N(request_nodes.get(r).unwrap().clone()),
                    ),
                    attributes,
                };
                stmts.push(Stmt::Edge(edge))
            }
        }
    }

    for (t, r) in &task_graph.will_fullfil_req {
        for r in r {
            let r = request_nodes.get(r).cloned().unwrap();
            let mut attributes = Vec::new();
            attributes.push(color::default().into_attr());
            attributes.push(EdgeAttributes::arrowhead(attributes::arrowhead::vee));
            let edge = Edge {
                ty: EdgeTy::Pair(Vertex::N(r), Vertex::N(task_nodes.get(t).unwrap().clone())),
                attributes,
            };
            stmts.push(Stmt::Edge(edge))
        }
    }

    let graph = Graph::DiGraph {
        id: Id::Plain(format!("TaskGraph")),
        strict: true,
        stmts,
    };

    let filename = "taskgraph.svg";

    let mut ctx = PrinterContext::default();
    ctx.always_inline();
    let _empty = exec(
        graph,
        &mut ctx,
        vec![
            CommandArg::Format(Format::Svg),
            CommandArg::Layout(Layout::Dot),
            CommandArg::Output(filename.to_string()),
        ],
    )
    .unwrap();
    println!("Finished writing dependency graph to file: {}", filename);

    let filename = "hltaskeventstream.json";
    if task_graph.high_level.event_stream.enabled {
        task_graph
            .high_level
            .event_stream
            .inner
            .stream
            .save(std::path::Path::new(filename));
        println!(
            "Finished writing high level event stream to file: {}",
            filename
        );
    } else {
        println!("Task event stream recording was not enabled",);
    }
}
