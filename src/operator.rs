use crate::id::Id;

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
}
