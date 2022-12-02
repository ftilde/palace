use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::task::Poll;

use crate::data::BrickPosition;
use crate::id::Id;
use crate::operator::OperatorId;
use crate::runtime::{ProgressIndicator, RequestQueue, TaskHints};
use crate::storage::Storage;
use crate::threadpool::{self, ThreadSpawner};
use crate::Error;
use futures::stream::StreamExt;

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
    pub task: RequestType<'op>,
    pub progress_indicator: ProgressIndicator,
}
impl TaskInfo<'_> {
    pub fn id(&self) -> TaskId {
        self.id
    }
}

type ResultPoll<'op, V> =
    Box<dyn for<'tasks> FnMut(&TaskContext<'op, 'tasks>) -> Option<&'tasks V>>;

pub type TaskConstructor<'op> =
    Box<dyn 'op + for<'tasks> FnOnce(TaskContext<'op, 'tasks>) -> Task<'tasks>>;

#[derive(From)]
pub enum RequestType<'op> {
    Data(TaskConstructor<'op>),
    ThreadPoolJob(threadpool::Job),
}

pub struct Request<'req, 'op, V: ?Sized> {
    pub id: TaskId,
    pub type_: RequestType<'op>,
    pub poll: ResultPoll<'op, V>,
    pub _marker: std::marker::PhantomData<&'req ()>,
}

#[derive(Copy, Clone)]
pub struct TaskContext<'op, 'tasks>
where
    'op: 'tasks,
{
    pub requests: &'tasks RequestQueue<'op>,
    pub storage: &'tasks Storage,
    pub hints: &'tasks TaskHints,
    pub thread_pool: &'tasks ThreadSpawner,
}

// Workaround for a compiler bug(?)
// See https://github.com/rust-lang/rust/issues/34511#issuecomment-373423999
pub trait WeUseThisLifetime<'a> {}
impl<'a, T: ?Sized> WeUseThisLifetime<'a> for T {}

impl<'op, 'tasks> TaskContext<'op, 'tasks>
where
    'op: 'tasks,
{
    pub fn submit<'req, V: ?Sized + 'req>(
        &'req self,
        mut request: Request<'req, 'op, V>,
    ) -> impl Future<Output = &'req V> + 'req + WeUseThisLifetime<'op> {
        async move {
            let _ = self.hints.drain_completed();
            match request.type_ {
                RequestType::Data(_) => {
                    if let Some(res) = (request.poll)(self) {
                        return std::future::ready(res).await;
                    }
                }
                RequestType::ThreadPoolJob(_) => {}
            };

            self.requests.push(TaskInfo {
                id: request.id,
                task: request.type_,
                progress_indicator: ProgressIndicator::WaitForComplete,
            });

            futures::pending!();

            loop {
                let _ = self.hints.drain_completed();
                if let Some(data) = (request.poll)(self) {
                    return std::future::ready(data).await;
                } else {
                    futures::pending!();
                }
            }
        }
    }

    #[allow(unused)] //We will probably use this at some point
    pub fn submit_unordered<'req, V: 'req + ?Sized>(
        &'req self,
        requests: impl Iterator<Item = Request<'req, 'op, V>> + 'req,
    ) -> impl StreamExt<Item = &'req V> + 'req + WeUseThisLifetime<'op>
    where
        'tasks: 'req,
    {
        self.submit_unordered_with_data(requests.map(|r| (r, ())))
            .map(|(r, ())| r)
    }

    pub fn submit_unordered_with_data<'req, V: 'req + ?Sized, D: 'tasks>(
        &'req self,
        requests: impl Iterator<Item = (Request<'req, 'op, V>, D)> + 'req,
    ) -> impl StreamExt<Item = (&'req V, D)> + 'req + WeUseThisLifetime<'op>
    where
        'tasks: 'req,
    {
        let mut initial_ready = Vec::new();
        let mut task_map = requests
            .into_iter()
            .filter_map(|(mut req, data)| {
                match req.type_ {
                    RequestType::Data(_) => {
                        if let Some(r) = (req.poll)(self) {
                            initial_ready.push((r, data));
                            return None;
                        }
                    }
                    RequestType::ThreadPoolJob(_) => {}
                };
                self.requests.push(TaskInfo {
                    id: req.id,
                    task: req.type_,
                    progress_indicator: ProgressIndicator::PartialPossible,
                });
                Some((req.id, (req.poll, data)))
            })
            .collect::<BTreeMap<_, _>>();
        let mut completed = VecDeque::new();
        futures::stream::poll_fn(move |_f_ctx| -> Poll<Option<(&'req V, D)>> {
            completed.extend(self.hints.drain_completed());
            if let Some(r) = initial_ready.pop() {
                return Poll::Ready(Some(r));
            }
            if task_map.is_empty() {
                return Poll::Ready(None);
            }

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
