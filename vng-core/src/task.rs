use ahash::HashMapExt;
use std::alloc::Layout;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::future::Future;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::task::Poll;
use std::time::Instant;

use crate::id::Id;
use crate::operator::{
    DataDescriptor, DataId, OpaqueOperator, OperatorDescriptor, OperatorId, TypeErased,
};
use crate::runtime::{CompletedBarrierItems, FrameNumber, RequestQueue, TaskHints};
use crate::storage::gpu::{StateCacheResult, WriteHandle};
use crate::storage::ram::{Storage, WriteHandleUninit};
use crate::storage::{Element, VisibleDataLocation};
use crate::task_graph::{GroupId, ProgressIndicator, RequestId, TaskId, VisibleDataId};
use crate::task_manager::ThreadSpawner;
use crate::threadpool::{JobId, JobType};
use crate::util::{Map, Set};
use crate::vulkan::{BarrierInfo, DeviceContext};
use crate::Error;
use futures::stream::StreamExt;
use futures::Stream;
use pin_project::pin_project;

use derive_more::{Constructor, Deref, DerefMut};

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

/// Note: Per contract, the poll function must not be called after it has returned Some once.
type ResultPoll<'a, V> = Box<dyn FnMut() -> Option<V> + 'a>;

pub struct DataRequest<'inv> {
    pub id: DataId,
    pub location: VisibleDataLocation,
    pub source: &'inv dyn OpaqueOperator,
    pub item: TypeErased,
}

pub struct RequestGroup<'inv> {
    pub id: GroupId,
    pub all: Vec<RequestType<'inv>>,
}

pub enum RequestType<'inv> {
    Data(DataRequest<'inv>),
    ThreadPoolJob(ThreadPoolJob, JobType),
    CmdBufferCompletion(crate::vulkan::CmdBufferSubmissionId),
    CmdBufferSubmission(crate::vulkan::CmdBufferSubmissionId),
    Barrier(BarrierInfo),
    Group(RequestGroup<'inv>),
}

impl RequestType<'_> {
    pub fn id(&self) -> RequestId {
        match self {
            RequestType::CmdBufferCompletion(d) => RequestId::CmdBufferCompletion(*d),
            RequestType::CmdBufferSubmission(d) => RequestId::CmdBufferSubmission(*d),
            RequestType::Barrier(i) => RequestId::Barrier(*i),
            RequestType::Data(d) => RequestId::Data(VisibleDataId {
                id: d.id,
                location: d.location,
            }),
            RequestType::ThreadPoolJob(j, _) => RequestId::Job(j.id),
            RequestType::Group(g) => RequestId::Group(g.id),
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
    pub gen_poll: Box<dyn FnOnce(PollContext<'req>) -> ResultPoll<'req, V> + 'req>,
    pub _marker: std::marker::PhantomData<&'req ()>,
}

impl<'req, 'inv, V: 'req> Request<'req, 'inv, V> {
    pub fn id(&self) -> RequestId {
        self.type_.id()
    }
}

#[derive(Copy, Clone)]
pub struct PollContext<'cref> {
    pub storage: &'cref Storage,
    pub device_contexts: &'cref [DeviceContext],
    pub current_frame: FrameNumber,
}

#[derive(Copy, Clone)]
pub struct OpaqueTaskContext<'cref, 'inv> {
    pub(crate) requests: &'cref RequestQueue<'inv>,
    pub(crate) storage: &'cref Storage,
    pub(crate) barrier_completions: &'cref CompletedBarrierItems,
    pub(crate) hints: &'cref TaskHints,
    pub(crate) thread_pool: &'cref ThreadSpawner,
    pub(crate) device_contexts: &'cref [DeviceContext],
    pub(crate) predicted_preview_tasks: &'cref RefCell<Set<TaskId>>,
    pub(crate) current_task: TaskId,
    pub(crate) current_op: Option<OperatorDescriptor>, //Only present if task originated from an operator
    pub(crate) current_frame: FrameNumber,
    pub(crate) deadline: Instant,
}

impl<'cref, 'inv> OpaqueTaskContext<'cref, 'inv> {
    pub fn submit<'req, V: 'req>(
        &'req self,
        request: Request<'req, 'inv, V>,
    ) -> impl Future<Output = V> + 'req + WeUseThisLifetime<'inv> {
        async move {
            let request_id = request.id();
            let mut poll = (request.gen_poll)(PollContext {
                storage: self.storage,
                device_contexts: self.device_contexts,
                current_frame: self.current_frame,
            });
            match request.type_ {
                RequestType::Data(_)
                | RequestType::Group(_)
                | RequestType::Barrier(..)
                | RequestType::CmdBufferCompletion(_)
                | RequestType::CmdBufferSubmission(_) => {
                    if let Some(res) = poll() {
                        return std::future::ready(res).await;
                    }
                }
                RequestType::ThreadPoolJob(_, _) => {}
            };

            let progress_indicator = if let RequestType::Group(_) = request.type_ {
                ProgressIndicator::PartialPossible
            } else {
                ProgressIndicator::WaitForComplete
            };

            self.requests.push(RequestInfo {
                task: request.type_,
                progress_indicator,
            });

            futures::pending!();

            loop {
                if let Some(data) = poll() {
                    self.hints.noticed_completion(request_id);
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
        self.thread_pool.spawn(JobType::Io, self.current_task, f)
    }

    /// Spawn a job on the compute pool. This job is assumed to not block (i.e., do IO or hold
    /// locks), but instead to fully utilize the compute capabilities of the core it is running on.
    pub fn spawn_compute<'req, R: Send + 'req>(
        &'req self,
        f: impl FnOnce() -> R + Send + 'req,
    ) -> Request<'req, 'inv, R> {
        self.thread_pool
            .spawn(JobType::Compute, self.current_task, f)
    }

    pub fn current_op(&self) -> OperatorId {
        self.current_task.operator()
    }

    pub fn current_op_desc(&self) -> Option<OperatorDescriptor> {
        self.current_op
    }

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
        RequestStreamSource::unordered(*self, requests)
    }

    pub fn group<'req, V: 'req>(
        &'req self,
        r: impl IntoIterator<Item = Request<'req, 'inv, V>>,
    ) -> Request<'req, 'inv, Vec<V>> {
        let r = r.into_iter();
        let l = r.size_hint().0;
        let mut types = Vec::with_capacity(l);
        let mut polls = Vec::with_capacity(l);
        let mut ids: Vec<Id> = Vec::with_capacity(l);
        let mut done = Vec::with_capacity(l);

        for (i, r) in r.into_iter().enumerate() {
            //TODO: Try to remove the hack with hash ids and ESPECIALLY the differenciation of
            //CmdBufferSubmission/Completion.
            match r.id() {
                RequestId::Data(d) => ids.push(Id::hash(&d)),
                RequestId::Job(i) => ids.push(Id::hash(&i)),
                RequestId::CmdBufferCompletion(i) => ids.push(Id::hash(&(0, i))),
                RequestId::CmdBufferSubmission(i) => ids.push(Id::hash(&(1, i))),
                RequestId::Barrier(b_info) => ids.push(Id::hash(&b_info)),
                RequestId::Group(g) => ids.push(g.0),
            }
            let mut poll = (r.gen_poll)(PollContext {
                storage: self.storage,
                device_contexts: self.device_contexts,
                current_frame: self.current_frame,
            });
            match r.type_ {
                RequestType::Data(_)
                | RequestType::Group(_)
                | RequestType::Barrier(..)
                | RequestType::CmdBufferCompletion(_)
                | RequestType::CmdBufferSubmission(_) => {
                    if let Some(v) = poll() {
                        done.push(MaybeUninit::new(v));
                        continue;
                    }
                }
                RequestType::ThreadPoolJob(_, _) => {}
            }
            done.push(MaybeUninit::uninit());
            polls.push((i, poll));
            types.push(r.type_);
        }
        let id = Id::combine(&ids[..]);
        Request {
            type_: RequestType::Group(RequestGroup {
                id: GroupId(id),
                all: types,
            }),
            gen_poll: Box::new(move |_ctx| {
                Box::new(move || {
                    // TODO: possibly add some kind of information which ones are ready?
                    polls.retain_mut(|(i, poll)| match poll() {
                        Some(v) => {
                            done[*i].write(v);
                            false
                        }
                        None => true,
                    });
                    if polls.is_empty() {
                        let ret = std::mem::take(&mut done);
                        // TODO: The following "should" basically be a noop. Maybe this can be
                        // optimized, but maybe the compiler already does a good job for this.
                        let ret = ret
                            .into_iter()
                            .map(|v| unsafe { v.assume_init() })
                            .collect();
                        Some(ret)
                    } else {
                        None
                    }
                })
            }),
            _marker: Default::default(),
        }
    }

    // TODO: We may not want to expose the storage directly in the future. Currently this is used
    // for the into_main_handle methods of Thread*Handle (see storage.rs), but we could change them
    // to take a context argument instead.
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn vulkan_device(&self) -> &DeviceContext {
        self.device_contexts.first().unwrap()
    }

    pub fn past_deadline(&self) -> bool {
        self.deadline < Instant::now()
    }
}

pub trait RequestStream<'req, 'inv, V> {
    // TODO: Possibly change once impl trait ist stable
    fn then_req<V2: 'req, F: Fn(V) -> Request<'req, 'inv, V2> + 'req>(
        self,
        ctx: OpaqueTaskContext<'req, 'inv>,
        f: F,
    ) -> Box<dyn Stream<Item = V2> + std::marker::Unpin + 'req>
    where
        Self: Sized;

    fn then_req_with_data<V2, D2, F: Fn(V) -> (Request<'req, 'inv, V2>, D2)>(
        self,
        ctx: OpaqueTaskContext<'req, 'inv>,
        f: F,
    ) -> RequestThenWithData<'req, 'inv, V2, D2, futures::stream::Map<Self, F>>
    where
        Self: Sized;
}

impl<'req, 'inv, V, S: Stream<Item = V> + std::marker::Unpin + 'req> RequestStream<'req, 'inv, V>
    for S
{
    fn then_req<V2: 'req, F: Fn(V) -> Request<'req, 'inv, V2> + 'req>(
        self,
        ctx: OpaqueTaskContext<'req, 'inv>,
        f: F,
    ) -> Box<dyn Stream<Item = V2> + std::marker::Unpin + 'req>
    where
        Self: Sized,
    {
        Box::new(
            RequestThenWithData {
                inner: self.map(move |i| (f(i), ())),
                then: RequestStreamSource::empty(ctx),
            }
            .map(|(v, ())| v),
        )
    }
    fn then_req_with_data<V2, D2, F: Fn(V) -> (Request<'req, 'inv, V2>, D2)>(
        self,
        ctx: OpaqueTaskContext<'req, 'inv>,
        f: F,
    ) -> RequestThenWithData<'req, 'inv, V2, D2, futures::stream::Map<Self, F>>
    where
        Self: Sized,
    {
        RequestThenWithData {
            inner: self.map(f),
            then: RequestStreamSource::empty(ctx),
        }
    }
}

#[pin_project]
pub struct RequestThenWithData<'req, 'inv, V2, D2, I> {
    inner: I,
    then: RequestStreamSource<'req, 'inv, V2, D2>,
}

impl<'req, 'inv, V2, D2, I: Stream<Item = (Request<'req, 'inv, V2>, D2)> + std::marker::Unpin>
    futures::Stream for RequestThenWithData<'req, 'inv, V2, D2, I>
{
    type Item = (V2, D2);

    fn poll_next(
        mut self: Pin<&mut Self>,
        ctx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let input_empty = loop {
            match self.inner.poll_next_unpin(ctx) {
                Poll::Ready(Some(v)) => {
                    self.then.push_request(v);
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

#[pin_project]
struct RequestStreamSource<'req, 'inv, V, D> {
    task_map: Map<RequestId, Vec<(ResultPoll<'req, V>, D)>>,
    ready: VecDeque<(V, D)>,
    task_context: OpaqueTaskContext<'req, 'inv>,
}

impl<'req, 'inv, V, D> RequestStreamSource<'req, 'inv, V, D> {
    fn empty(task_context: OpaqueTaskContext<'req, 'inv>) -> Self {
        Self {
            task_map: Map::new(),
            ready: VecDeque::new(),
            task_context,
        }
    }

    fn unordered(
        task_context: OpaqueTaskContext<'req, 'inv>,
        requests: impl Iterator<Item = (Request<'req, 'inv, V>, D)> + 'req,
    ) -> Self {
        let mut ret = Self {
            task_map: Default::default(),
            ready: Default::default(),
            task_context,
        };
        for r in requests {
            ret.push_request(r);
        }
        ret
    }

    pub fn push_request(&mut self, (req, data): (Request<'req, 'inv, V>, D)) {
        let mut poll = (req.gen_poll)(PollContext {
            storage: self.task_context.storage,
            device_contexts: self.task_context.device_contexts,
            current_frame: self.task_context.current_frame,
        });
        match req.type_ {
            RequestType::Data(_)
            | RequestType::Group(_)
            | RequestType::Barrier(..)
            | RequestType::CmdBufferCompletion(_)
            | RequestType::CmdBufferSubmission(_) => {
                if let Some(r) = poll() {
                    self.ready.push_back((r, data));
                    return;
                }
            }
            RequestType::ThreadPoolJob(_, _) => {}
        }
        let id = req.type_.id();
        let entry = self.task_map.entry(id).or_default();
        self.task_context.requests.push(RequestInfo {
            task: req.type_,
            progress_indicator: ProgressIndicator::PartialPossible,
        });
        entry.push((poll, data));
    }
}

impl<'req, 'inv, V, D> futures::Stream for RequestStreamSource<'req, 'inv, V, D> {
    type Item = (V, D);

    fn poll_next(
        self: Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let s = self.get_mut();
        {
            // TODO Try to remove double loop with BtreeSet::drain_filter once it's stable.
            let mut newly_completed = Vec::new();
            for c in s.task_context.hints.completed.borrow().iter() {
                if s.task_map.contains_key(&c) {
                    newly_completed.push(*c);
                }
            }
            for c in newly_completed {
                let t = s.task_map.get_mut(&c).unwrap();
                *t = std::mem::take(t)
                    .into_iter()
                    .filter_map(|(mut result_poll, data)| {
                        if let Some(v) = result_poll() {
                            s.task_context.hints.noticed_completion(c);
                            s.ready.push_back((v, data));
                            None
                        } else {
                            Some((result_poll, data))
                        }
                    })
                    .collect();
                if t.is_empty() {
                    s.task_map.remove(&c);
                }
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

impl<'cref, 'inv, I, O: ?Sized> std::ops::Deref for TaskContext<'cref, 'inv, I, O> {
    type Target = OpaqueTaskContext<'cref, 'inv>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
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
}

impl<'cref, 'inv, ItemDescriptor: std::hash::Hash, Output: Element + ?Sized>
    TaskContext<'cref, 'inv, ItemDescriptor, Output>
{
    pub fn alloc_slot(
        &'cref self,
        item: ItemDescriptor,
        size: usize,
    ) -> WriteHandleUninit<'cref, [MaybeUninit<Output>]> {
        let id = DataDescriptor::new(self.current_op_desc().unwrap(), &item);
        self.inner.storage.alloc_slot(id, size)
    }
}

impl<'cref, 'inv, ItemDescriptor: std::hash::Hash, Output: Element + ?Sized>
    TaskContext<'cref, 'inv, ItemDescriptor, Output>
{
    pub fn alloc_slot_gpu<'a>(
        &'a self,
        device: &'a DeviceContext,
        item: ItemDescriptor,
        size: usize,
    ) -> WriteHandle<'a> {
        let id = DataDescriptor::new(self.current_op_desc().unwrap(), &item);
        device
            .storage
            .alloc_slot::<Output>(device, self.current_frame, id, size)
    }

    pub fn access_state_cache<'a>(
        &'a self,
        device: &'a DeviceContext,
        item: ItemDescriptor,
        name: &str,
        layout: Layout,
    ) -> Result<StateCacheResult<'a>, Error> {
        let base_id = DataId::new(self.current_op(), &item);
        let id = DataId(Id::combine(&[base_id.0, Id::hash(name)]));

        Ok(device.storage.access_state_cache(device, id, layout))
    }
}

impl<'cref, 'inv, Output: Element> TaskContext<'cref, 'inv, (), Output> {
    pub fn write(&self, value: Output) {
        let mut slot = self.alloc_slot((), 1);
        slot[0].write(value);
        unsafe { slot.initialized(**self) };
    }
}
impl<'cref, 'inv, Output: Element> TaskContext<'cref, 'inv, (), Output> {
    pub fn alloc_scalar_gpu<'a>(&'a self, device: &'a DeviceContext) -> WriteHandle<'a> {
        let id = DataDescriptor::new(self.current_op_desc().unwrap(), &());
        device
            .storage
            .alloc_slot::<Output>(device, self.current_frame, id, 1)
    }
}
