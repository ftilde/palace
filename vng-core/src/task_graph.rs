use std::hash::Hash;

use crate::{
    id::Id,
    operator::{DataId, OperatorId},
    storage::{DataLocation, VisibleDataLocation},
    task::AllocationId,
    threadpool::JobId,
    util::{Map, Set},
    vulkan::{BarrierInfo, CmdBufferSubmissionId},
};
use ahash::HashMapExt;
use gs_core::EventStream;

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
    Barrier(BarrierInfo),
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
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum TaskClass {
    GarbageCollect = 1,
    Barrier = 2,
    Data = 3,
    Alloc = 4,
    Transfer = 5,
}

const ROOT_PRIO: Priority = Priority {
    level: 0,
    progress: 0,
    class: TaskClass::Data,
};

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Priority {
    level: u32,
    progress: u32,
    class: TaskClass,
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
    pub fn downstream(&self, class: TaskClass) -> Priority {
        Priority {
            level: self.level + 1,
            progress: 0,
            class,
        }
    }
    fn prio(&self) -> u32 {
        self.progress << 16 + self.level
    }
}

#[derive(Default)]
struct GraphEventStream(EventStream);
impl GraphEventStream {
    // Eventstream stuff:
    fn emit_node(&mut self, action: StreamAction, item: impl EventStreamNode) {
        let node = gs_core::Node {
            id: item.to_eventstream_id(),
            label: item.label(),
        };
        let e = match action {
            StreamAction::Add => gs_core::Event::AddNode(node),
            StreamAction::Remove => gs_core::Event::RemoveNode(node),
        };
        self.0.add(e);
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
        let edge = gs_core::Edge {
            from: from.to_eventstream_id(),
            to: to.to_eventstream_id(),
            label: format!("{}", count),
        };
        let e = match action {
            StreamAction::Add => gs_core::Event::AddEdge(edge),
            StreamAction::Remove => gs_core::Event::RemoveEdge(edge),
        };
        self.0.add(e);
    }
    fn edge_update(
        &mut self,
        from: impl EventStreamNode,
        to: impl EventStreamNode,
        count: usize,
        new_count: usize,
    ) {
        let edge = gs_core::Edge {
            from: from.to_eventstream_id(),
            to: to.to_eventstream_id(),
            label: format!("{}", count),
        };
        let e = gs_core::Event::UpdateEdgeLabel(edge, format!("{}", new_count));
        self.0.add(e);
    }
    fn edge_add(&mut self, from: impl EventStreamNode, to: impl EventStreamNode, count: usize) {
        self.edge(StreamAction::Add, from, to, count)
    }
    fn edge_remove(&mut self, from: impl EventStreamNode, to: impl EventStreamNode, count: usize) {
        self.edge(StreamAction::Remove, from, to, count)
    }
}

#[derive(Default)]
struct HighLevelGraph {
    depends_on: Map<TaskId, Map<TaskId, usize>>,
    //provides_for: Map<TaskId, Map<TaskId, usize>>,
    event_stream: GraphEventStream,
}

impl HighLevelGraph {
    fn add_task(&mut self, t: TaskId) {
        assert!(self.depends_on.insert(t, Default::default()).is_none());
        //assert!(self.provides_for.insert(t, Default::default()).is_none());
        self.event_stream.node_add(t);
    }
    fn add_dependency(&mut self, from: TaskId, to: TaskId) {
        let edge_entry = self
            .depends_on
            .entry(from)
            .or_default()
            .entry(to)
            .or_default();
        if *edge_entry == 0 {
            self.event_stream.edge_add(from, to, 1);
        } else {
            self.event_stream
                .edge_update(from, to, *edge_entry, *edge_entry + 1);
        }
        *edge_entry += 1;

        //let edge_entry = self
        //    .provides_for
        //    .entry(to)
        //    .or_default()
        //    .entry(from)
        //    .or_default();
        //*edge_entry += 1;
    }
    fn remove_dependency(&mut self, from: TaskId, to: TaskId) {
        let edge_entry = self.depends_on.get_mut(&from).unwrap().get_mut(&to);
        assert!(edge_entry.is_some(), "F {:?} T {:?}", from, to);
        let edge_entry = edge_entry.unwrap();
        *edge_entry -= 1;
        if *edge_entry == 0 {
            self.event_stream.edge_remove(from, to, 1);
            //TODO: Do we need to remove from set?
        } else {
            self.event_stream
                .edge_update(from, to, *edge_entry + 1, *edge_entry);
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
        let depends_on = self.depends_on.remove(&t).unwrap();
        //assert!(depends_on.is_empty(), "{:?} dep on {:?}", t, depends_on);
        for (dep, n) in depends_on {
            assert_eq!(n, 0, "{:?} deps on {:?}", t, dep);
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

#[derive(Default)]
pub struct TaskGraph {
    implied_tasks: Map<TaskId, TaskMetadata>,
    waits_on: Map<TaskId, Map<RequestId, ProgressIndicator>>,
    required_by: Map<RequestId, Set<TaskId>>,
    will_provide_data: Map<TaskId, Set<DataId>>,
    data_provided_by: Map<DataId, Set<TaskId>>,
    will_fullfil_req: Map<TaskId, Set<RequestId>>,
    req_fullfil_by: Map<RequestId, TaskId>,
    implied_ready: priority_queue::PriorityQueue<TaskId, Priority>,
    resolved_deps: Map<TaskId, Set<RequestId>>,
    in_groups: Map<RequestId, Set<GroupId>>,
    groups: Map<GroupId, Set<RequestId>>,
    requested_locations: Map<DataId, Map<VisibleDataLocation, Set<TaskId>>>,
    event_stream: GraphEventStream,
    high_level: HighLevelGraph,
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
        // Hm, only relevant for the root task, i think. Could be moved somewhere else possibly
        if !self.waits_on.contains_key(&wants) {
            self.event_stream.node_add(wants);
            self.high_level.add_task(wants);
        }
        if !self.required_by.contains_key(&wanted) {
            self.event_stream.node_add(wanted);
        }

        if self
            .waits_on
            .entry(wants)
            .or_default()
            .insert(wanted, progress_indicator)
            .is_none()
        {
            self.event_stream.edge_add(wants, wanted, 1);
        } else {
            //println!("Already waiting {:?}", wants);
        }
        self.required_by.entry(wanted).or_default().insert(wants);
        self.implied_ready.remove(&wants);

        if let RequestId::Data(d) = wanted {
            let entry = self.requested_locations.entry(d.id).or_default();
            let e = entry.entry(d.location);

            if matches!(e, crate::util::MapEntry::Vacant(_)) {
                self.event_stream.node_add(d.id);
                self.event_stream.edge_add(wanted, d.id, 1);
            }

            let loc = e.or_default();
            loc.insert(wants);
        }
    }

    //TODO: Try to avoid clone
    pub fn requested_locations(&self, id: DataId) -> Map<VisibleDataLocation, Set<TaskId>> {
        self.requested_locations.get(&id).unwrap().clone()
    }

    pub fn in_group(&mut self, in_: RequestId, group: GroupId) {
        let entry = self.in_groups.entry(in_).or_default();
        entry.insert(group);
        let entry = self.groups.entry(group).or_default();
        entry.insert(in_);
    }

    pub fn will_provide_data(&mut self, task: TaskId, data: DataId) {
        let entries = self.will_provide_data.entry(task).or_default();
        entries.insert(data);
        self.event_stream.edge_add(data, task, 1);
        let entry = self.data_provided_by.entry(data).or_default();
        assert!(entry.insert(task));

        for locations in self.requested_locations[&data].values() {
            for requestor in locations {
                self.high_level.add_dependency(*requestor, task);
            }
        }
    }

    pub fn will_fullfil_req(&mut self, task: TaskId, req: RequestId) {
        let entries = self.will_fullfil_req.entry(task).or_default();
        let newly_inserted = entries.insert(req);
        if !newly_inserted {
            assert!(self.req_fullfil_by.insert(req, task).is_none());
            self.event_stream.edge_add(req, task, 1);

            for requestor in &self.required_by[&req] {
                self.high_level.add_dependency(*requestor, task);
            }
        }
    }

    pub fn add_implied(&mut self, id: TaskId, priority: Priority) {
        let inserted = self.implied_tasks.insert(id, TaskMetadata { priority });
        self.waits_on.insert(id, Map::new());

        self.high_level.add_task(id);
        self.event_stream.node_add(id);

        assert!(inserted.is_none(), "Tried to insert task twice");
        self.implied_ready.push(id, priority);
    }
    pub fn try_increase_priority(&mut self, id: TaskId, priority: Priority) {
        // Note that this does not change the priority of downstream tasks recursively! This is
        // fine, however, and in practice only meant for batched tasks (barriers and operator
        // tasks) whose priority is only updated as long as they are not started (and thus have no
        // downstream tasks).

        let task_md = self.implied_tasks.get_mut(&id).unwrap();
        task_md.priority = task_md.priority.max(priority);

        self.implied_ready.change_priority(&id, task_md.priority);
    }
    pub fn get_priority(&mut self, id: TaskId) -> Priority {
        self.implied_tasks
            .get(&id)
            .map(|m| m.priority)
            .unwrap_or(ROOT_PRIO)
    }

    pub fn already_requested(&self, rid: RequestId) -> bool {
        self.required_by.contains_key(&rid)
    }

    pub fn dependents(&self, id: RequestId) -> &Set<TaskId> {
        self.required_by.get(&id).unwrap()
    }

    pub fn resolved_implied(&mut self, id: RequestId) {
        if let RequestId::Data(d) = id {
            let entry = self.requested_locations.get_mut(&d.id).unwrap();
            let fulfilled_locs = entry.remove(&d.location).unwrap();
            self.event_stream.edge_remove(id, d.id, 1);

            let by = self.data_provided_by.remove(&d.id).unwrap();
            for from in fulfilled_locs {
                for by in &by {
                    self.high_level.remove_dependency(from, *by);
                }
            }
            if entry.is_empty() {
                self.requested_locations.remove(&d.id);
                self.event_stream.node_remove(d.id);
            }
        }

        let required_by = self.required_by.remove(&id);
        let was_requested = required_by.is_some();

        if let Some(by) = self.req_fullfil_by.remove(&id) {
            for rev_dep in required_by.iter().flatten() {
                self.high_level.remove_dependency(*rev_dep, by);
            }
        }

        for rev_dep in required_by.iter().flatten() {
            self.event_stream.edge_remove(*rev_dep, id, 1);

            let deps_of_rev_dep = self.waits_on.get_mut(&rev_dep).unwrap();
            let progress_indicator = deps_of_rev_dep.remove(&id).unwrap();
            let resolved_deps = self.resolved_deps.entry(*rev_dep).or_default();
            resolved_deps.insert(id);

            if let Some(md) = self.implied_tasks.get_mut(&rev_dep) {
                if let RequestId::Data(_) = id {
                    md.priority.progress += 1;
                }

                if deps_of_rev_dep.is_empty()
                    || matches!(progress_indicator, ProgressIndicator::PartialPossible)
                {
                    self.implied_ready.push(*rev_dep, md.priority);
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

        // E.g. cmdbuffer completions are marked as resolved even if no one requested them
        if was_requested {
            self.event_stream.node_remove(id);
        }
    }

    pub fn task_done(&mut self, id: TaskId) {
        self.implied_tasks.remove(&id);
        self.implied_ready.remove(&id);
        self.resolved_deps.remove(&id);

        let wpd = self.will_provide_data.remove(&id);
        for did in wpd.into_iter().flatten() {
            self.event_stream.edge_remove(did, id, 1);

            self.data_provided_by
                .get_mut(&did)
                .and_then(|v| Some(v.remove(&id)));

            if let Some(locations) = self.requested_locations.get_mut(&did) {
                for (_loc, tids) in locations {
                    for tid in tids.iter() {
                        self.high_level.remove_dependency(*tid, id);
                    }
                }
            }
        }
        let wfr = self.will_fullfil_req.remove(&id);
        for rid in wfr.into_iter().flatten() {
            self.event_stream.edge_remove(rid, id, 1);
        }

        let deps = self.waits_on.remove(&id).unwrap();
        assert!(deps.is_empty());
        //assert!(deps.iter().all(|(v, _)| matches!(v, RequestId::Group(_))));

        self.event_stream.node_remove(id);
        self.high_level.remove_task(id);
    }

    pub fn next_implied_ready(&mut self) -> Option<TaskId> {
        self.implied_ready.pop().map(|(k, _v)| k)
    }

    pub fn has_open_tasks(&self) -> bool {
        !self.implied_tasks.is_empty()
    }

    pub fn resolved_deps(&mut self, task: TaskId) -> Option<&mut Set<RequestId>> {
        self.resolved_deps.get_mut(&task)
    }
}

enum StreamAction {
    Add,
    Remove,
}

pub fn export_full_detail(graph: &TaskGraph) {
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
    let task_nodes = graph
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

    let request_nodes = graph
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

    let data_nodes = graph
        .requested_locations
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

    for (t, r) in &graph.waits_on {
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

    for (t, r) in &graph.will_provide_data {
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

    for (t, r) in &graph.will_fullfil_req {
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

    for (d, l) in &graph.requested_locations {
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
    task_graph
        .high_level
        .event_stream
        .0
        .save(std::path::Path::new(filename));
    println!(
        "Finished writing high level event stream to file: {}",
        filename
    );

    let filename = "taskeventstream.json";
    task_graph
        .event_stream
        .0
        .save(std::path::Path::new(filename));
    println!("Finished writing event stream to file: {}", filename);
}
