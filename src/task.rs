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
use crate::threadpool::{JobId, JobType};
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
pub struct RequestInfo<'inv> {
    pub task: RequestType<'inv>,
    pub progress_indicator: ProgressIndicator,
}
impl RequestInfo<'_> {
    pub fn id(&self) -> RequestId {
        self.task.id()
    }
}

type ResultPoll<'a, V> = Box<dyn FnMut(PollContext<'a>) -> Option<V>>;

pub struct DataRequest<'inv> {
    pub id: DataId,
    pub source: &'inv dyn OpaqueOperator,
    pub item: TypeErased,
}

#[derive(From)]
pub enum RequestType<'inv> {
    Data(DataRequest<'inv>),
    ThreadPoolJob(ThreadPoolJob, JobType),
}

impl RequestType<'_> {
    fn id(&self) -> RequestId {
        match self {
            RequestType::Data(d) => RequestId::Data(d.id),
            RequestType::ThreadPoolJob(j, _) => RequestId::Job(j.id),
        }
    }
}

pub struct ThreadPoolJob {
    pub id: JobId,
    pub waiting_id: TaskId,
    pub job: crate::threadpool::Job,
}

pub struct Request<'req, 'inv, V> {
    pub type_: RequestType<'inv>,
    pub poll: ResultPoll<'req, V>,
    pub _marker: std::marker::PhantomData<&'req ()>,
}

impl<V> Request<'_, '_, V> {
    pub fn id(&self) -> RequestId {
        self.type_.id()
    }
}

#[derive(Copy, Clone)]
pub struct PollContext<'cref> {
    pub storage: &'cref Storage,
}

#[derive(Copy, Clone)]
pub struct OpaqueTaskContext<'cref, 'inv> {
    pub requests: &'cref RequestQueue<'inv>,
    pub storage: &'cref Storage,
    pub hints: &'cref TaskHints,
    pub thread_pool: &'cref ThreadSpawner,
    pub current_task: TaskId,
}

// Workaround for a compiler bug(?)
// See https://github.com/rust-lang/rust/issues/34511#issuecomment-373423999
pub trait WeUseThisLifetime<'a> {}
impl<'a, T: ?Sized> WeUseThisLifetime<'a> for T {}

#[derive(Copy, Clone)]
pub struct TaskContext<'cref, 'inv, ItemDescriptor, Output: ?Sized> {
    inner: OpaqueTaskContext<'cref, 'inv>,
    _output_marker: std::marker::PhantomData<(ItemDescriptor, Output)>,
}
impl<'cref, 'inv, ItemDescriptor: bytemuck::NoUninit, Output: ?Sized>
    TaskContext<'cref, 'inv, ItemDescriptor, Output>
{
    pub(crate) fn new(inner: OpaqueTaskContext<'cref, 'inv>) -> Self {
        Self {
            inner,
            _output_marker: Default::default(),
        }
    }

    pub fn submit<'req, V: 'req>(
        &'req self,
        mut request: Request<'req, 'inv, V>,
    ) -> impl Future<Output = V> + 'req + WeUseThisLifetime<'inv> {
        async move {
            match request.type_ {
                RequestType::Data(_) => {
                    if let Some(res) = (request.poll)(PollContext {
                        storage: self.inner.storage,
                    }) {
                        return std::future::ready(res).await;
                    }
                }
                RequestType::ThreadPoolJob(_, _) => {}
            };

            let request_id = request.id();

            self.inner.requests.push(RequestInfo {
                task: request.type_,
                progress_indicator: ProgressIndicator::WaitForComplete,
            });

            futures::pending!();

            loop {
                if let Some(data) = (request.poll)(PollContext {
                    storage: self.inner.storage,
                }) {
                    let mut completed = self.inner.hints.completed.borrow_mut();
                    completed.remove(&request_id);
                    return std::future::ready(data).await;
                } else {
                    futures::pending!();
                }
            }
        }
    }
    /// Spawn a job on the io pool. This job is allowed to hold locks/do IO, but should not do
    /// excessive computation.
    pub fn spawn_io<'req, R: Send + 'static>(
        &'req self,
        f: impl FnOnce() -> R + Send + 'req,
    ) -> Request<'req, 'inv, R> {
        self.inner
            .thread_pool
            .spawn(JobType::Io, self.inner.current_task, f)
    }

    /// Spawn a job on the compute pool. This job is assumed to not block (i.e., do IO or hold
    /// locks), but instead to fully utilize the compute capabilities of the core it is running on.
    pub fn spawn_compute<'req, R: Send + 'static>(
        &'req self,
        f: impl FnOnce() -> R + Send + 'req,
    ) -> Request<'req, 'inv, R> {
        self.inner
            .thread_pool
            .spawn(JobType::Compute, self.inner.current_task, f)
    }

    pub fn current_op(&self) -> OperatorId {
        self.inner.current_task.operator()
    }

    #[allow(unused)] //We will probably use this at some point
    pub fn submit_unordered<'req, V: 'req>(
        &'req self,
        requests: impl Iterator<Item = Request<'req, 'inv, V>> + 'req,
    ) -> impl StreamExt<Item = V> + 'req + WeUseThisLifetime<'inv>
    where
        'inv: 'req,
    {
        self.submit_unordered_with_data(requests.map(|r| (r, ())))
            .map(|(r, ())| r)
    }

    pub fn submit_unordered_with_data<'req, V: 'req, D: 'inv>(
        &'req self,
        requests: impl Iterator<Item = (Request<'req, 'inv, V>, D)> + 'req,
    ) -> impl StreamExt<Item = (V, D)> + 'req + WeUseThisLifetime<'inv>
    where
        'inv: 'req,
    {
        let mut initial_ready = Vec::new();
        let mut task_map = requests
            .into_iter()
            .filter_map(|(mut req, data)| {
                match req.type_ {
                    RequestType::Data(_) => {
                        if let Some(r) = (req.poll)(PollContext {
                            storage: self.inner.storage,
                        }) {
                            initial_ready.push((r, data));
                            return None;
                        }
                    }
                    RequestType::ThreadPoolJob(_, _) => {}
                };
                let id = req.type_.id();
                self.inner.requests.push(RequestInfo {
                    task: req.type_,
                    progress_indicator: ProgressIndicator::PartialPossible,
                });
                Some((id, (req.poll, data)))
            })
            .collect::<BTreeMap<_, _>>();
        let mut completed = VecDeque::new();
        futures::stream::poll_fn(move |_f_ctx| -> Poll<Option<(V, D)>> {
            {
                let mut newly_completed = Vec::new();
                for c in self.inner.hints.completed.borrow().iter() {
                    if task_map.contains_key(&c) {
                        newly_completed.push(*c);
                    }
                }
                for c in newly_completed {
                    self.inner.hints.noticed_completion(c);
                    completed.push_back(c);
                }
            }
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
                    storage: self.inner.storage,
                }) {
                    Some(v) => return Poll::Ready(Some((v, data))),
                    None => panic!("Task should have been ready!"),
                }
            }
        })
    }
}

impl<'cref, 'inv, ItemDescriptor: bytemuck::NoUninit, Output: bytemuck::AnyBitPattern + ?Sized>
    TaskContext<'cref, 'inv, ItemDescriptor, Output>
{
    pub fn alloc_slot(
        &self,
        item: ItemDescriptor,
        size: usize,
    ) -> Result<WriteHandleUninit<[MaybeUninit<Output>]>, Error> {
        let id = DataId::new(self.current_op(), &item);
        self.inner.storage.alloc_ram_slot(id, size)
    }
}

impl<'cref, 'inv, Output: bytemuck::AnyBitPattern> TaskContext<'cref, 'inv, (), Output> {
    pub fn write(&self, value: Output) -> Result<(), Error> {
        let mut slot = self.alloc_slot((), 1)?;
        slot[0].write(value);
        unsafe { slot.initialized() };

        Ok(())
    }
}
