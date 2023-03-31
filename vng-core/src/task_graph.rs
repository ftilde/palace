use std::collections::{BTreeMap, BTreeSet};

use crate::{
    id::Id,
    operator::{DataId, OperatorId},
    storage::{DataLocation, VisibleDataLocation},
    threadpool::JobId,
    vulkan::{BarrierInfo, CmdBufferSubmissionId},
};

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

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum RequestId {
    CmdBufferCompletion(CmdBufferSubmissionId),
    CmdBufferSubmission(CmdBufferSubmissionId),
    Barrier(BarrierInfo),
    Data(VisibleDataId),
    Job(JobId),
    Group(GroupId),
}

impl From<VisibleDataId> for RequestId {
    fn from(value: VisibleDataId) -> Self {
        RequestId::Data(value)
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

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct GroupId(pub Id);

impl RequestId {
    pub fn unwrap_data(&self) -> VisibleDataId {
        match self {
            RequestId::Data(d) => *d,
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
    will_provide: BTreeMap<TaskId, BTreeSet<DataId>>,
    implied_ready: BTreeSet<TaskId>,
    resolved_deps: BTreeMap<TaskId, BTreeSet<RequestId>>,
    in_groups: BTreeMap<RequestId, BTreeSet<GroupId>>,
    groups: BTreeMap<GroupId, BTreeSet<RequestId>>,
    requested_locations: BTreeMap<DataId, BTreeSet<VisibleDataLocation>>,
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

    pub fn requested_locations(&self, id: DataId) -> BTreeSet<VisibleDataLocation> {
        self.requested_locations.get(&id).unwrap().clone()
    }

    pub fn in_group(&mut self, in_: RequestId, group: GroupId) {
        let entry = self.in_groups.entry(in_).or_default();
        entry.insert(group);
        let entry = self.groups.entry(group).or_default();
        entry.insert(in_);
    }

    pub fn will_provide(&mut self, task: TaskId, data: DataId) {
        let entries = self.will_provide.entry(task).or_default();
        entries.insert(data);
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
            if deps_of_rev_dep.is_empty()
                || matches!(progress_indicator, ProgressIndicator::PartialPossible)
            {
                if self.implied_tasks.contains(rev_dep) {
                    self.implied_ready.insert(*rev_dep);
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

        self.will_provide.remove(&id);

        let deps = self.waits_on.remove(&id).unwrap();
        assert!(deps.is_empty());
        //assert!(deps.iter().all(|(v, _)| matches!(v, RequestId::Group(_))));
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
        .collect::<BTreeMap<_, _>>();

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
        .collect::<BTreeMap<_, _>>();

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
        .collect::<BTreeMap<_, _>>();

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

    for (t, r) in &graph.will_provide {
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

    for (d, l) in &graph.requested_locations {
        for l in l {
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
