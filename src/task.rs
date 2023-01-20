use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::task::Poll;

use crate::operator::{DataId, OpaqueOperator, OperatorId, TypeErased};
use crate::runtime::{RequestQueue, TaskHints};
use crate::storage::{Storage, WriteHandleUninit};
use crate::task_graph::{ProgressIndicator, RequestId, TaskId};
use crate::task_manager::ThreadSpawner;
use crate::threadpool::JobId;
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

#[derive(Constructor)]
pub struct RequestInfo {
    pub task: RequestType,
    pub progress_indicator: ProgressIndicator,
}
impl RequestInfo {
    pub fn id(&self) -> RequestId {
        self.task.id()
    }
}

type ResultPoll<'rt, V> = Box<dyn FnMut(PollContext<'rt>) -> Option<V>>;

pub struct DataRequest {
    pub id: DataId,
    pub source: *const dyn OpaqueOperator,
    pub item: TypeErased,
}

#[derive(From)]
pub enum RequestType {
    Data(DataRequest),
    ThreadPoolJob(ThreadPoolJob),
}

impl RequestType {
    fn id(&self) -> RequestId {
        match self {
            RequestType::Data(d) => RequestId::Data(d.id),
            RequestType::ThreadPoolJob(j) => RequestId::Job(j.id),
        }
    }
}

pub struct ThreadPoolJob {
    pub id: JobId,
    pub waiting_id: TaskId,
    pub job: crate::threadpool::Job,
}

pub struct Request<'req, V> {
    pub type_: RequestType,
    pub poll: ResultPoll<'req, V>,
    pub _marker: std::marker::PhantomData<&'req ()>,
}

impl<V> Request<'_, V> {
    pub fn id(&self) -> RequestId {
        self.type_.id()
    }
}

#[derive(Copy, Clone)]
pub struct PollContext<'rt> {
    pub storage: &'rt Storage,
}

#[derive(Copy, Clone)]
pub struct OpaqueTaskContext<'tasks> {
    pub requests: &'tasks RequestQueue<'tasks>,
    pub storage: &'tasks Storage,
    pub hints: &'tasks TaskHints,
    pub thread_pool: &'tasks ThreadSpawner,
    pub current_task: TaskId,
}

// Workaround for a compiler bug(?)
// See https://github.com/rust-lang/rust/issues/34511#issuecomment-373423999
pub trait WeUseThisLifetime<'a> {}
impl<'a, T: ?Sized> WeUseThisLifetime<'a> for T {}

impl<'tasks> OpaqueTaskContext<'tasks> {
    pub fn spawn_job<'req>(&'req self, f: impl FnOnce() + Send + 'req) -> Request<'req, ()> {
        self.thread_pool.spawn(self.current_task, f)
    }

    pub fn current_op(&self) -> OperatorId {
        self.current_task.operator()
    }

    pub fn submit<'req, V: 'req>(
        &'req self,
        mut request: Request<'req, V>,
    ) -> impl Future<Output = V> + 'req {
        async move {
            let _ = self.hints.drain_completed();
            match request.type_ {
                RequestType::Data(_) => {
                    if let Some(res) = (request.poll)(PollContext {
                        storage: self.storage,
                    }) {
                        return std::future::ready(res).await;
                    }
                }
                RequestType::ThreadPoolJob(_) => {}
            };

            self.requests.push(RequestInfo {
                task: request.type_,
                progress_indicator: ProgressIndicator::WaitForComplete,
            });

            futures::pending!();

            loop {
                let _ = self.hints.drain_completed();
                if let Some(data) = (request.poll)(PollContext {
                    storage: self.storage,
                }) {
                    return std::future::ready(data).await;
                } else {
                    futures::pending!();
                }
            }
        }
    }

    #[allow(unused)] //We will probably use this at some point
    pub fn submit_unordered<'req, V: 'req>(
        &'req self,
        requests: impl Iterator<Item = Request<'req, V>> + 'req,
    ) -> impl StreamExt<Item = V> + 'req
    where
        'tasks: 'req,
    {
        self.submit_unordered_with_data(requests.map(|r| (r, ())))
            .map(|(r, ())| r)
    }

    pub fn submit_unordered_with_data<'req, V: 'req, D: 'tasks>(
        &'req self,
        requests: impl Iterator<Item = (Request<'req, V>, D)> + 'req,
    ) -> impl StreamExt<Item = (V, D)> + 'req
    where
        'tasks: 'req,
    {
        let mut initial_ready = Vec::new();
        let mut task_map = requests
            .into_iter()
            .filter_map(|(mut req, data)| {
                match req.type_ {
                    RequestType::Data(_) => {
                        if let Some(r) = (req.poll)(PollContext {
                            storage: self.storage,
                        }) {
                            initial_ready.push((r, data));
                            return None;
                        }
                    }
                    RequestType::ThreadPoolJob(_) => {}
                };
                let id = req.type_.id();
                self.requests.push(RequestInfo {
                    task: req.type_,
                    progress_indicator: ProgressIndicator::PartialPossible,
                });
                Some((id, (req.poll, data)))
            })
            .collect::<BTreeMap<_, _>>();
        let mut completed = VecDeque::new();
        futures::stream::poll_fn(move |_f_ctx| -> Poll<Option<(V, D)>> {
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

                match result_poll(PollContext {
                    storage: self.storage,
                }) {
                    Some(v) => return Poll::Ready(Some((v, data))),
                    None => panic!("Task should have been ready!"),
                }
            }
        })
    }
}

#[derive(Copy, Clone)]
pub struct TaskContext<'tasks, ItemDescriptor, Output: ?Sized> {
    inner: OpaqueTaskContext<'tasks>,
    _output_marker: std::marker::PhantomData<(ItemDescriptor, Output)>,
}

impl<'tasks, ItemDescriptor: bytemuck::NoUninit, Output: bytemuck::AnyBitPattern + ?Sized>
    std::ops::Deref for TaskContext<'tasks, ItemDescriptor, Output>
{
    type Target = OpaqueTaskContext<'tasks>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'tasks, ItemDescriptor: bytemuck::NoUninit, Output: bytemuck::AnyBitPattern + ?Sized>
    TaskContext<'tasks, ItemDescriptor, Output>
{
    pub(crate) fn new(inner: OpaqueTaskContext<'tasks>) -> Self {
        Self {
            inner,
            _output_marker: Default::default(),
        }
    }

    pub fn alloc_slot(
        &self,
        item: ItemDescriptor,
        size: usize,
    ) -> Result<WriteHandleUninit<[MaybeUninit<Output>]>, Error> {
        let id = DataId::new(self.inner.current_op(), &item);
        self.inner.storage.alloc_ram_slot_slice(id, size)
    }
}

impl<'tasks, Output: bytemuck::AnyBitPattern> TaskContext<'tasks, (), Output> {
    pub fn write(&self, value: Output) -> Result<(), Error> {
        let data_id = DataId::new(self.current_op(), &());
        self.inner.storage.write_to_ram(data_id, value)
    }
}
