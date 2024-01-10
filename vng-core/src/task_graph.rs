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
pub enum TaskClass {
    Alloc = 0,
    GarbageCollect = 1,
    Barrier = 2,
    Data = 3,
    Transfer = 4,
}

const ROOT_PRIO: Priority = Priority { level: 0, prio: 0 };

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Priority {
    level: u32,
    prio: u32,
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.prio.cmp(&other.prio)
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
            prio: {
                match class {
                    TaskClass::Alloc => self.level,
                    TaskClass::GarbageCollect => 1 << 16,
                    _ => self.prio.max(((class as u8 as u32) << 16) + self.level),
                }
            },
        }
    }
}

#[derive(Default)]
pub struct TaskGraph {
    implied_tasks: Map<TaskId, TaskMetadata>,
    waits_on: Map<TaskId, Map<RequestId, ProgressIndicator>>,
    required_by: Map<RequestId, Set<TaskId>>,
    will_provide_data: Map<TaskId, Set<DataId>>,
    will_fullfil_req: Map<TaskId, Set<RequestId>>,
    implied_ready: priority_queue::PriorityQueue<TaskId, Priority>,
    resolved_deps: Map<TaskId, Set<RequestId>>,
    in_groups: Map<RequestId, Set<GroupId>>,
    groups: Map<GroupId, Set<RequestId>>,
    requested_locations: Map<DataId, Map<VisibleDataLocation, Set<TaskId>>>,
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
            let loc = entry.entry(d.location).or_default();
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
    }

    pub fn will_fullfil_req(&mut self, task: TaskId, req: RequestId) {
        let entries = self.will_fullfil_req.entry(task).or_default();
        entries.insert(req);
    }

    pub fn add_implied(&mut self, id: TaskId, priority: Priority) {
        let inserted = self.implied_tasks.insert(id, TaskMetadata { priority });
        self.waits_on.insert(id, Map::new());
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
            .get_mut(&id)
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
            if deps_of_rev_dep.is_empty()
                || matches!(progress_indicator, ProgressIndicator::PartialPossible)
            {
                if let Some(m) = self.implied_tasks.get_mut(rev_dep) {
                    self.implied_ready.push(*rev_dep, m.priority);
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

    pub fn task_done(&mut self, id: TaskId) {
        self.implied_tasks.remove(&id);
        self.implied_ready.remove(&id);
        self.resolved_deps.remove(&id);

        self.will_provide_data.remove(&id);
        self.will_fullfil_req.remove(&id);

        let deps = self.waits_on.remove(&id).unwrap();
        assert!(deps.is_empty());
        //assert!(deps.iter().all(|(v, _)| matches!(v, RequestId::Group(_))));
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

pub fn export(graph: &TaskGraph) {
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

    for (source, data_ids) in &graph.will_provide_data {
        for (target, request_ids) in &graph.waits_on {
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

    for (t, r) in &graph.waits_on {
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
