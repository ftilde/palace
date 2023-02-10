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
use futures::Stream;

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

type ResultPoll<'a, V> = Box<dyn FnMut(PollContext<'a>) -> Option<V> + 'a>;

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

pub trait RequestStream<'req, 'inv, V1, D1, V2, D2, F: Fn(V1, D1) -> (Request<'req, 'inv, V2>, D2)>
{
    fn then_req(
        self,
        ctx: OpaqueTaskContext<'req, 'inv>,
        f: F,
    ) -> Box<RequestThen<'req, 'inv, V2, D2, F, Self>>
    where
        Self: Sized;
}

impl<
        'req,
        'inv,
        V1,
        D1,
        V2,
        D2,
        F: Fn(V1, D1) -> (Request<'req, 'inv, V2>, D2),
        S: Stream<Item = (V1, D1)>,
    > RequestStream<'req, 'inv, V1, D1, V2, D2, F> for S
{
    fn then_req(
        self,
        ctx: OpaqueTaskContext<'req, 'inv>,
        f: F,
    ) -> Box<RequestThen<'req, 'inv, V2, D2, F, Self>>
    where
        Self: Sized,
    {
        Box::new(RequestThen {
            inner: self,
            then: RequestStreamSource::empty(ctx),
            f,
        })
    }
}

pub struct RequestThen<'req, 'inv, V2, D2, F, I> {
    inner: I,
    then: Box<RequestStreamSource<'req, 'inv, V2, D2>>,
    f: F,
}

impl<
        'req,
        'inv,
        V1,
        D1,
        V2,
        D2,
        I: Stream<Item = (V1, D1)> + std::marker::Unpin,
        F: Fn(V1, D1) -> (Request<'req, 'inv, V2>, D2),
    > futures::Stream for Box<RequestThen<'req, 'inv, V2, D2, F, I>>
{
    type Item = (V2, D2);

    fn poll_next(
        mut self: Pin<&mut Self>,
        ctx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let input_empty = loop {
            match self.inner.poll_next_unpin(ctx) {
                Poll::Ready(Some((v, d))) => {
                    let to_push = (self.f)(v, d);
                    self.then.push_request(to_push);
                }
                Poll::Ready(None) => {
                    break true;
                }
                Poll::Pending => {
                    break false;
                }
            }
        };
        match self.then.poll_next_unpin(ctx) {
            // If there are currently no further items in the `then` stream, but we still have
            // unprocessed items in the `inner` stream, we are not done, yet!
            Poll::Ready(None) if !input_empty => Poll::Pending,
            o => o,
        }
    }
}

struct RequestStreamSource<'req, 'inv, V, D> {
    task_map: BTreeMap<RequestId, (ResultPoll<'req, V>, D)>,
    ready: VecDeque<(V, D)>,
    task_context: OpaqueTaskContext<'req, 'inv>,
}

impl<'req, 'inv, V, D> RequestStreamSource<'req, 'inv, V, D> {
    fn empty(task_context: OpaqueTaskContext<'req, 'inv>) -> Box<Self> {
        Box::new(Self {
            task_map: BTreeMap::new(),
            ready: VecDeque::new(),
            task_context,
        })
    }
    fn unordered(
        task_context: OpaqueTaskContext<'req, 'inv>,
        requests: impl Iterator<Item = (Request<'req, 'inv, V>, D)> + 'req,
    ) -> Box<Self> {
        let mut ready = VecDeque::new();
        let task_map = requests
            .into_iter()
            .filter_map(|(mut req, data)| {
                match req.type_ {
                    RequestType::Data(_) => {
                        if let Some(r) = (req.poll)(PollContext {
                            storage: task_context.storage,
                        }) {
                            ready.push_back((r, data));
                            return None;
                        }
                    }
                    RequestType::ThreadPoolJob(_, _) => {}
                }
                let id = req.type_.id();
                task_context.requests.push(RequestInfo {
                    task: req.type_,
                    progress_indicator: ProgressIndicator::PartialPossible,
                });
                Some((id, (req.poll, data)))
            })
            .collect::<BTreeMap<_, _>>();

        Box::new(Self {
            task_map,
            ready,
            task_context,
        })
    }

    pub fn push_request(&mut self, (mut req, data): (Request<'req, 'inv, V>, D)) {
        match req.type_ {
            RequestType::Data(_) => {
                if let Some(r) = (req.poll)(PollContext {
                    storage: self.task_context.storage,
                }) {
                    self.ready.push_back((r, data));
                    return;
                }
            }
            RequestType::ThreadPoolJob(_, _) => {}
        }
        let id = req.type_.id();
        self.task_context.requests.push(RequestInfo {
            task: req.type_,
            progress_indicator: ProgressIndicator::PartialPossible,
        });
        self.task_map.insert(id, (req.poll, data));
    }

    //pub fn then(self) -> Then
}

impl<'req, 'inv, V, D> futures::Stream for Box<RequestStreamSource<'req, 'inv, V, D>> {
    type Item = (V, D);

    fn poll_next(
        self: Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let s = self.get_mut();
        {
            let mut newly_completed = Vec::new();
            for c in s.task_context.hints.completed.borrow().iter() {
                if s.task_map.contains_key(&c) {
                    newly_completed.push(*c);
                }
            }
            // TODO maybe try to merge these loops. Problem: borrow of completed and usage in
            // noticed_completion
            for c in newly_completed {
                s.task_context.hints.noticed_completion(c);

                let (mut result_poll, data) = s.task_map.remove(&c).unwrap();
                let v = result_poll(PollContext {
                    storage: s.task_context.storage,
                })
                .expect("Task should have been ready!");
                s.ready.push_back((v, data));
            }
        }
        if let Some(r) = s.ready.pop_front() {
            return Poll::Ready(Some(r));
        }
        if s.task_map.is_empty() {
            Poll::Ready(None)
        } else {
            Poll::Pending
        }
    }
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
impl<'cref, 'inv, ItemDescriptor, Output> Into<OpaqueTaskContext<'cref, 'inv>>
    for TaskContext<'cref, 'inv, ItemDescriptor, Output>
{
    fn into(self) -> OpaqueTaskContext<'cref, 'inv> {
        self.inner
    }
}
impl<'cref, 'inv, ItemDescriptor: std::hash::Hash, Output: ?Sized>
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
                    self.inner.hints.noticed_completion(request_id);
                    return std::future::ready(data).await;
                } else {
                    futures::pending!();
                }
            }
        }
    }
    /// Spawn a job on the io pool. This job is allowed to hold locks/do IO, but should not do
    /// excessive computation.
    pub fn spawn_io<'req, R: Send + 'req>(
        &'req self,
        f: impl FnOnce() -> R + Send + 'req,
    ) -> Request<'req, 'inv, R> {
        self.inner
            .thread_pool
            .spawn(JobType::Io, self.inner.current_task, f)
    }

    /// Spawn a job on the compute pool. This job is assumed to not block (i.e., do IO or hold
    /// locks), but instead to fully utilize the compute capabilities of the core it is running on.
    pub fn spawn_compute<'req, R: Send + 'req>(
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
        RequestStreamSource::unordered(self.inner, requests)
    }

    // TODO: We may not want to expose the storage directly in the future. Currently this is used
    // for the into_main_handle methods of Thread*Handle (see storage.rs), but we could change them
    // to take a context argument instead.
    pub fn storage(&self) -> &Storage {
        &self.inner.storage
    }
}

impl<'cref, 'inv, ItemDescriptor: std::hash::Hash, Output: Copy + ?Sized>
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

impl<'cref, 'inv, Output: Copy> TaskContext<'cref, 'inv, (), Output> {
    pub fn write(&self, value: Output) -> Result<(), Error> {
        let mut slot = self.alloc_slot((), 1)?;
        slot[0].write(value);
        unsafe { slot.initialized() };

        Ok(())
    }
}
