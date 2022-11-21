use std::future::Future;
use std::hash::{Hash, Hasher};
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::task::Poll;

use crate::data::BrickPosition;
use crate::id::Id;
use crate::operator::OperatorId;
use crate::runtime::RequestQueue;
use crate::storage::Storage;
use crate::Error;

use bytemuck::AnyBitPattern;
use derive_more::{Constructor, Deref, DerefMut};

#[derive(Deref, DerefMut)]
pub struct Task<'a>(Pin<Box<dyn Future<Output = Result<(), Error>> + 'a>>);

impl<'a, F> From<F> for Task<'a>
where
    F: Future<Output = Result<(), Error>> + 'a,
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
    pub active: OperatorId,
}

impl<'a> TaskContext<'a> {
    /// Safety: The requested type must match the task's result
    pub async unsafe fn request<'r, T: AnyBitPattern + 'r>(
        &'r self,
        info: TaskInfo,
    ) -> Result<&'r T, Error> {
        let task_id = info.id();
        if let Some(data) = self.storage.read_ram(task_id) {
            return Ok(data);
        }
        self.requests.push(info);
        std::future::poll_fn(|_ctx| loop {
            if let Some(data) = self.storage.read_ram(task_id) {
                return Poll::Ready(Ok(data));
            } else {
                return Poll::Pending;
            }
        })
        .await
    }

    /// Safety: The requested type must match the task's result
    pub async unsafe fn request_slice<'r, T: AnyBitPattern + 'r>(
        &'r self,
        info: TaskInfo,
        size: usize,
    ) -> Result<&'r [T], Error> {
        let task_id = info.id();
        if let Some(data) = self.storage.read_ram_slice(task_id, size) {
            return Ok(data);
        }
        self.requests.push(info);
        std::future::poll_fn(|_ctx| loop {
            if let Some(data) = self.storage.read_ram_slice(task_id, size) {
                return Poll::Ready(Ok(data));
            } else {
                return Poll::Pending;
            }
        })
        .await
    }

    /// Safety: the MaybeUninit needs to be written to in f (i.e., made valid).
    pub unsafe fn with_ram_slot<
        T: AnyBitPattern,
        F: FnOnce(&mut MaybeUninit<T>) -> Result<(), Error>,
    >(
        &self,
        request: &DatumRequest,
        f: F,
    ) -> Result<(), Error> {
        let id = TaskId::new(self.active, request);
        self.storage.with_ram_slot(id, f)
    }

    /// Safety: the MaybeUninit needs to be written to in f (i.e., made valid).
    pub unsafe fn with_ram_slot_slice<
        T: AnyBitPattern,
        F: FnOnce(&mut [MaybeUninit<T>]) -> Result<(), Error>,
    >(
        &self,
        request: &DatumRequest,
        size: usize,
        f: F,
    ) -> Result<(), Error> {
        let id = TaskId::new(self.active, request);
        self.storage.with_ram_slot_slice(id, size, f)
    }

    pub fn write_to_ram<T: AnyBitPattern>(
        &self,
        request: &DatumRequest,
        value: T,
    ) -> Result<(), Error> {
        unsafe {
            self.with_ram_slot(request, |v| {
                v.write(value);
                Ok(())
            })
        }
    }
}

#[non_exhaustive]
#[derive(Hash, Copy, Clone)]
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
