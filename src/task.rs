use std::future::Future;
use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::task::Poll;

use crate::data::Storage;
use crate::data::{BrickPosition, Datum};
use crate::id::Id;
use crate::operator::OperatorId;
use crate::runtime::RequestQueue;
use crate::Error;

use derive_more::{Constructor, Deref, DerefMut};

#[derive(Deref, DerefMut)]
pub struct Task<'a>(Pin<Box<dyn Future<Output = Result<Datum, Error>> + 'a>>);

impl<'a, F> From<F> for Task<'a>
where
    F: Future<Output = Result<Datum, Error>> + 'a,
{
    fn from(inner: F) -> Self {
        Self(Box::pin(inner))
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct TaskId(Id);
impl TaskId {
    pub fn new(op: OperatorId, d: &DatumRequest) -> Self {
        TaskId(Id::combine(&[op.inner(), d.id()]))
    }
}

#[derive(Constructor)]
pub struct TaskInfo {
    pub operator: OperatorId,
    pub data: DatumRequest,
}
impl TaskInfo {
    pub fn id(&self) -> TaskId {
        TaskId::new(self.operator, &self.data)
    }
}

#[derive(Copy, Clone)]
pub struct TaskContext<'a> {
    pub requests: &'a RequestQueue,
    pub storage: &'a Storage,
}

impl<'a> TaskContext<'a> {
    pub async fn request(&'a self, info: TaskInfo) -> Result<Datum, Error> {
        let task_id = info.id();
        if let Some(data) = self.storage.read_ram(task_id) {
            return Ok(data.clone());
        }
        self.requests.push(info);
        std::future::poll_fn(|_ctx| loop {
            if let Some(data) = self.storage.read_ram(task_id) {
                return Poll::Ready(Ok(data.clone()));
            } else {
                return Poll::Pending;
            }
        })
        .await
    }
}

#[non_exhaustive]
#[derive(Hash)]
pub enum DatumRequest {
    Value,
    Brick(BrickPosition),
}

impl DatumRequest {
    pub fn id(&self) -> Id {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        let v = hasher.finish();
        let hash = bytemuck::bytes_of(&v);
        return Id::from_data(hash);
    }
}
