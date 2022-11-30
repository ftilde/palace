use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::task::Poll;

use crate::data::BrickPosition;
use crate::id::Id;
use crate::operator::OperatorId;
use crate::runtime::{ProgressIndicator, RequestQueue, TaskHints};
use crate::storage::Storage;
use crate::threadpool::ThreadPool;
use crate::Error;
use futures::stream::StreamExt;

use bytemuck::AnyBitPattern;
use derive_more::{Constructor, Deref, DerefMut, From};

#[derive(Deref, DerefMut)]
pub struct Task<'a, R = ()>(Pin<Box<dyn Future<Output = Result<R, Error>> + 'a>>);

impl<'a, F, R> From<F> for Task<'a, R>
where
    F: Future<Output = Result<R, Error>> + 'a,
{
    fn from(inner: F) -> Self {
        Self(Box::pin(inner))
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, From)]
pub struct TaskId(Id);
impl TaskId {
    pub fn new(op: OperatorId, d: &DatumRequest) -> Self {
        TaskId(Id::combine(&[op.inner(), d.id()]))
    }
}

#[derive(Constructor)]
pub struct TaskInfo<'op> {
    pub id: TaskId,
    pub task: Box<dyn 'op + for<'tasks> FnOnce(TaskContext<'op, 'tasks>) -> Task<'tasks>>,
    pub progress_indicator: ProgressIndicator,
}
impl TaskInfo<'_> {
    pub fn id(&self) -> TaskId {
        self.id
    }
}

type ResultPoll<'op, V> = Box<dyn for<'tasks> FnMut(TaskContext<'op, 'tasks>) -> Option<&'tasks V>>;

pub struct Request<'op, V: ?Sized> {
    pub id: TaskId,
    pub compute: Box<dyn 'op + for<'tasks> FnOnce(TaskContext<'op, 'tasks>) -> Task<'tasks>>,
    pub poll: ResultPoll<'op, V>,
}

#[derive(Copy, Clone)]
pub struct TaskContext<'op, 'tasks>
where
    'op: 'tasks,
{
    pub requests: &'tasks RequestQueue<'op>,
    pub storage: &'tasks Storage,
    pub hints: &'tasks TaskHints,
    pub thread_pool: &'tasks ThreadPool<'tasks>,
}

// Workaround for a compiler bug(?)
// See https://github.com/rust-lang/rust/issues/34511#issuecomment-373423999
pub trait WeUseThisLifetime<'a> {}
impl<'a, T: ?Sized> WeUseThisLifetime<'a> for T {}

impl<'op, 'tasks> TaskContext<'op, 'tasks>
where
    'op: 'tasks,
{
    pub fn submit<V: ?Sized + 'tasks>(
        self,
        mut request: Request<'op, V>,
    ) -> impl Future<Output = &'tasks V> + 'tasks + WeUseThisLifetime<'op> {
        let initial_ready = (request.poll)(self);
        if initial_ready.is_none() {
            self.requests.push(TaskInfo {
                id: request.id,
                task: request.compute,
                progress_indicator: ProgressIndicator::WaitForComplete,
            });
        }
        std::future::poll_fn(move |_ctx| loop {
            if let Some(data) = initial_ready {
                return Poll::Ready(data);
            } else if let Some(data) = (request.poll)(self) {
                return Poll::Ready(data);
            } else {
                return Poll::Pending;
            }
        })
    }

    pub fn submit_unordered<V: 'tasks + ?Sized>(
        self,
        requests: impl Iterator<Item = Request<'op, V>> + 'tasks,
    ) -> impl StreamExt<Item = &'tasks V> + 'tasks + WeUseThisLifetime<'op> {
        self.submit_unordered_with_data(requests.map(|r| (r, ())))
            .map(|(r, ())| r)
    }

    pub fn submit_unordered_with_data<V: 'tasks + ?Sized, D: 'tasks>(
        self,
        requests: impl Iterator<Item = (Request<'op, V>, D)>,
    ) -> impl StreamExt<Item = (&'tasks V, D)> + 'tasks + WeUseThisLifetime<'op> {
        let mut initial_ready = Vec::new();
        let mut task_map = requests
            .into_iter()
            .filter_map(|(mut req, data)| {
                if let Some(r) = (req.poll)(self) {
                    initial_ready.push((r, data));
                    return None;
                }
                self.requests.push(TaskInfo {
                    id: req.id,
                    task: req.compute,
                    progress_indicator: ProgressIndicator::PartialPossible,
                });
                Some((req.id, (req.poll, data)))
            })
            .collect::<BTreeMap<_, _>>();
        let mut completed = VecDeque::new();
        futures::stream::poll_fn(move |_f_ctx| -> Poll<Option<(&'tasks V, D)>> {
            completed.extend(self.hints.drain_completed());
            if let Some(r) = initial_ready.pop() {
                return Poll::Ready(Some(r));
            }
            if task_map.is_empty() {
                return Poll::Ready(None);
            }
            assert!(!completed.is_empty());

            loop {
                let Some(completed_id) = completed.pop_front() else {
                        return Poll::Pending;
                    };
                let Some((mut result_poll, data)) = task_map.remove(&completed_id) else { continue };

                match result_poll(self) {
                    Some(v) => return Poll::Ready(Some((v, data))),
                    None => panic!("Task should have been ready!"),
                }
            }
        })
    }

    /// Safety: the MaybeUninit needs to be written to in f (i.e., made valid).
    pub unsafe fn with_ram_slot<
        T: AnyBitPattern,
        F: FnOnce(&mut MaybeUninit<T>) -> Result<(), Error>,
    >(
        &self,
        id: TaskId,
        f: F,
    ) -> Result<(), Error> {
        self.storage.with_ram_slot(id, f)
    }

    /// Safety: the MaybeUninit needs to be written to in f (i.e., made valid).
    pub unsafe fn with_ram_slot_slice<
        T: AnyBitPattern,
        F: FnOnce(&mut [MaybeUninit<T>]) -> Result<(), Error>,
    >(
        &self,
        id: TaskId,
        size: usize,
        f: F,
    ) -> Result<(), Error> {
        self.storage.with_ram_slot_slice(id, size, f)
    }

    pub fn write_to_ram<T: AnyBitPattern>(&self, id: TaskId, value: T) -> Result<(), Error> {
        unsafe {
            self.with_ram_slot(id, |v| {
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
