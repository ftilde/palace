use std::collections::BTreeMap;

use crate::id::Id;
use crate::task::{DatumRequest, Task, TaskContext};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct OperatorId(Id);
impl OperatorId {
    pub fn new<T>(inputs: &[OperatorId]) -> Self {
        // TODO: Maybe it's more efficient to use the sha.update method directly.
        let mut id = Id::from_data(std::any::type_name::<T>().as_ref());
        for i in inputs {
            id = Id::combine(&[id, i.0]);
        }
        OperatorId(id)
    }
    pub fn inner(&self) -> Id {
        self.0
    }
}

impl From<Id> for OperatorId {
    fn from(inner: Id) -> Self {
        Self(inner)
    }
}

pub trait Operator {
    fn id(&self) -> OperatorId;
    fn compute<'a>(&'a self, rt: TaskContext<'a>, info: DatumRequest) -> Task<'a>;
}

pub struct Network {
    operators: BTreeMap<OperatorId, Box<dyn Operator>>,
}

impl Network {
    pub fn new() -> Self {
        Network {
            operators: BTreeMap::new(),
        }
    }

    pub fn add(&mut self, op: impl Operator + 'static) -> OperatorId {
        let id = op.id();
        self.operators.insert(op.id(), Box::new(op));
        id
    }

    pub fn get(&self, op: OperatorId) -> Option<&dyn Operator> {
        self.operators.get(&op).map(|op| &**op)
    }
}
