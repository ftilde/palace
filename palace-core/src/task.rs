use ahash::HashMapExt;
use std::alloc::Layout;
use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::sync::atomic::AtomicU64;
use std::task::Poll;
use std::time::{Duration, Instant};

use crate::array::ChunkIndex;
use crate::data::LocalCoordinate;
use crate::dim::DynDimension;
use crate::dtypes::{ConversionError, DType, ElementType, StaticElementType};
use crate::operator::{DataDescriptor, DataId, OpaqueOperator, OperatorDescriptor, OperatorId};
use crate::runtime::{CompletedRequests, Deadline, FrameNumber, RequestQueue, TaskHints};
use crate::storage::cpu::{ConcurrentWriteAccessError, ReadHandle};
use crate::storage::gpu::{BarrierEpoch, MemoryLocation, WriteHandle};
use crate::storage::ram::{self, RamAllocator, RawWriteHandleUninit, WriteHandleUninit};
use crate::storage::{disk, CpuDataLocation, DataVersionType, Element};
use crate::storage::{DataLocation, GarbageCollectId, VisibleDataLocation};
use crate::task_graph::{GroupId, ProgressIndicator, RequestId, TaskId, VisibleDataId};
use crate::task_manager::ThreadSpawner;
use crate::threadpool::{JobId, JobType};
use crate::util::{Map, Set};
use crate::vec::Vector;
use crate::vulkan::{BarrierInfo, DeviceContext, DeviceId};
use crate::Error;
use futures::stream::StreamExt;
use futures::{FutureExt, Stream};
use id::{Id, Identify};
use pin_project::pin_project;

pub use futures::join;

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

#[non_exhaustive]
pub struct DataRequest<'inv> {
    pub id: DataId,
    pub location: VisibleDataLocation,
    pub source: &'inv dyn OpaqueOperator,
    pub item: ChunkIndex,
    _use_the_constructor: (),
}

impl<'inv> DataRequest<'inv> {
    pub(crate) fn new(
        id: DataId,
        location: VisibleDataLocation,
        source: &'inv dyn OpaqueOperator,
        item: ChunkIndex,
    ) -> Self {
        Self {
            id,
            location,
            source,
            item,
            _use_the_constructor: (),
        }
    }
}

pub struct RequestGroup<'inv> {
    pub id: GroupId,
    pub all: Vec<RequestType<'inv>>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct AllocationId(u64);

impl AllocationId {
    pub fn inner(&self) -> u64 {
        self.0
    }
    pub fn next() -> AllocationId {
        static ALLOC_ID_COUNTER: AtomicU64 = AtomicU64::new(0);
        let id_raw = ALLOC_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        AllocationId(id_raw)
    }
}

pub enum AllocationRequest {
    Ram(Layout, DataDescriptor, CpuDataLocation),
    VRam(DeviceId, Layout, DataDescriptor),
    VRamBufRaw(
        DeviceId,
        Layout,
        ash::vk::BufferUsageFlags,
        MemoryLocation,
        oneshot::Sender<crate::storage::gpu::Allocation>,
    ),
    VRamImageRaw(
        DeviceId,
        ash::vk::ImageCreateInfo<'static>,
        oneshot::Sender<crate::storage::gpu::ImageAllocation>,
    ),
}

pub enum RequestType<'inv> {
    Data(DataRequest<'inv>),
    Allocation(AllocationId, AllocationRequest),
    ThreadPoolJob(ThreadPoolJob, JobType),
    CmdBufferCompletion(crate::vulkan::CmdBufferSubmissionId),
    CmdBufferSubmission(crate::vulkan::CmdBufferSubmissionId),
    Barrier(BarrierInfo, BarrierEpoch),
    Group(RequestGroup<'inv>),
    GarbageCollect(DataLocation),
    Ready,
    YieldOnce,
    ExternalProgress,
}

impl RequestType<'_> {
    pub fn id(&self) -> RequestId {
        match self {
            RequestType::Allocation(id, ..) => RequestId::Allocation(*id),
            RequestType::CmdBufferCompletion(d) => RequestId::CmdBufferCompletion(*d),
            RequestType::CmdBufferSubmission(d) => RequestId::CmdBufferSubmission(*d),
            RequestType::Barrier(i, e) => RequestId::Barrier(*i, *e),
            RequestType::Data(d) => RequestId::Data(VisibleDataId {
                id: d.id,
                location: d.location,
            }),
            RequestType::ThreadPoolJob(j, _) => RequestId::Job(j.id),
            RequestType::Group(g) => RequestId::Group(g.id),
            RequestType::Ready => RequestId::Ready,
            RequestType::YieldOnce => RequestId::YieldOnce,
            RequestType::ExternalProgress => RequestId::ExternalProgress,
            RequestType::GarbageCollect(l) => RequestId::GarbageCollect(*l),
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
    pub fn ready(v: V) -> Self {
        Self {
            type_: RequestType::Ready,
            gen_poll: Box::new(move |_ctx| {
                let mut v = Some(v);
                Box::new(move || v.take())
            }),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn id(&self) -> RequestId {
        self.type_.id()
    }

    pub fn map<R: 'req>(self, f: impl FnOnce(V) -> R + 'req) -> Request<'req, 'inv, R> {
        Request {
            type_: self.type_,
            _marker: self._marker,
            gen_poll: Box::new(move |ctx| {
                let mut inner_poll = (self.gen_poll)(ctx);
                let mut f = Some(f);
                Box::new(move || inner_poll().map(|v| (f.take().unwrap())(v)))
            }),
        }
    }
}
impl<'req, 'inv, V: 'req, E: 'req + std::fmt::Debug> Request<'req, 'inv, Result<V, E>> {
    pub fn unwrap_value(self) -> Request<'req, 'inv, V> {
        self.map(|v| v.unwrap())
    }
}

impl<'req, 'inv> Request<'req, 'inv, ()> {
    pub fn yield_once() -> Request<'req, 'inv, ()> {
        Self {
            type_: RequestType::YieldOnce,
            gen_poll: Box::new(move |_ctx| Box::new(move || Some(()))),
            _marker: std::marker::PhantomData,
        }
    }
    pub fn external_progress() -> Request<'req, 'inv, ()> {
        Self {
            type_: RequestType::ExternalProgress,
            gen_poll: Box::new(move |_ctx| Box::new(move || Some(()))),
            _marker: std::marker::PhantomData,
        }
    }
    pub fn garbage_collect(
        location: DataLocation,
        gid: GarbageCollectId,
    ) -> Request<'req, 'inv, ()> {
        Self {
            type_: RequestType::GarbageCollect(location),
            gen_poll: Box::new(move |ctx| {
                Box::new(move || match location {
                    DataLocation::CPU(CpuDataLocation::Ram) => {
                        if ctx.storage.next_garbage_collect() > gid {
                            Some(())
                        } else {
                            None
                        }
                    }
                    DataLocation::CPU(CpuDataLocation::Disk) => {
                        if ctx.disk_cache.unwrap().next_garbage_collect() > gid {
                            Some(())
                        } else {
                            None
                        }
                    }
                    DataLocation::GPU(id) => {
                        if ctx.device_contexts[&id].storage.next_garbage_collect() > gid {
                            Some(())
                        } else {
                            None
                        }
                    }
                })
            }),
            _marker: std::marker::PhantomData,
        }
    }
}

#[derive(Copy, Clone)]
pub struct PollContext<'cref> {
    pub storage: &'cref ram::Storage,
    pub disk_cache: Option<&'cref disk::Storage>,
    pub device_contexts: &'cref BTreeMap<DeviceId, DeviceContext>,
    pub(crate) predicted_preview_tasks: &'cref RefCell<Set<TaskId>>,
    pub(crate) current_task: TaskId,
    pub current_frame: FrameNumber,
}

impl PollContext<'_> {
    pub fn register_dependency_dataversion(&self, version: DataVersionType) {
        if version == DataVersionType::Preview {
            self.predicted_preview_tasks
                .borrow_mut()
                .insert(self.current_task);
        }
    }
}

#[derive(Copy, Clone)]
pub struct OpaqueTaskContext<'cref, 'inv> {
    pub(crate) requests: &'cref RequestQueue<'inv>,
    pub(crate) storage: &'cref ram::Storage,
    pub(crate) disk_cache: Option<&'cref disk::Storage>,
    pub(crate) completed_requests: &'cref CompletedRequests,
    pub(crate) hints: &'cref TaskHints,
    pub(crate) thread_pool: &'cref ThreadSpawner,
    pub(crate) device_contexts: &'cref BTreeMap<DeviceId, DeviceContext>,
    pub(crate) predicted_preview_tasks: &'cref RefCell<Set<TaskId>>,
    pub(crate) current_task: TaskId,
    pub(crate) current_op: Option<OperatorDescriptor>, //Only present if task originated from an operator
    pub(crate) current_frame: FrameNumber,
    pub(crate) deadline: Deadline,
    pub(crate) start: Instant,
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
                disk_cache: self.disk_cache,
                device_contexts: self.device_contexts,
                current_frame: self.current_frame,
                predicted_preview_tasks: self.predicted_preview_tasks,
                current_task: self.current_task,
            });
            match request.type_ {
                RequestType::Data(_)
                | RequestType::Ready
                | RequestType::Group(_)
                | RequestType::Barrier(..)
                | RequestType::CmdBufferCompletion(_)
                | RequestType::CmdBufferSubmission(_)
                | RequestType::Allocation(..) => {
                    if let Some(res) = poll() {
                        return std::future::ready(res).await;
                    }
                }
                RequestType::ThreadPoolJob(_, _)
                | RequestType::GarbageCollect(_)
                | RequestType::ExternalProgress
                | RequestType::YieldOnce => {}
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

    pub fn current_task(&self) -> TaskId {
        self.current_task
    }

    // Only available if task originated from an operator (may not be the case for transfer tasks
    // for example)
    pub fn current_op_desc(&self) -> Option<OperatorDescriptor> {
        self.current_op
    }

    // Only available if task originated from an operator (may not be the case for transfer tasks
    // for example)
    pub fn data_descriptor(&self, chunk: ChunkIndex) -> Option<DataDescriptor> {
        self.current_op_desc()
            .map(|d| DataDescriptor::new(d, chunk))
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

    pub fn submit_unordered_with_data<'req, V: 'req, D: 'req>(
        &'req self,
        requests: impl Iterator<Item = (Request<'req, 'inv, V>, D)> + 'req,
    ) -> impl StreamExt<Item = (V, D)> + 'req + WeUseThisLifetime<'inv>
    where
        'inv: 'req,
    {
        RequestStreamSource::unordered(*self, requests)
    }

    #[must_use]
    pub async fn run_unordered<V: 'cref>(
        self,
        requests: impl Iterator<Item = ChildTask<'cref, V>> + 'cref,
    ) -> Vec<V> {
        TasksUnordered::new(self, requests).run().await
    }

    pub fn alloc_raw<'req>(
        &'req self,
        data_descriptor: DataDescriptor,
        layout: Layout,
    ) -> Request<'req, 'inv, Result<RawWriteHandleUninit<'req>, ConcurrentWriteAccessError>> {
        self.storage
            .request_alloc_raw(self.current_frame, data_descriptor, layout)
    }

    pub fn alloc_raw_disk<'req>(
        &'req self,
        data_descriptor: DataDescriptor,
        layout: Layout,
    ) -> Request<
        'req,
        'inv,
        Result<crate::storage::disk::RawWriteHandleUninit<'req>, ConcurrentWriteAccessError>,
    > {
        self.disk_cache
            .unwrap()
            .request_alloc_raw(self.current_frame, data_descriptor, layout)
    }

    pub fn alloc_raw_gpu<'req>(
        &'req self,
        device: &'req DeviceContext,
        data_descriptor: DataDescriptor,
        layout: Layout,
    ) -> Request<'req, 'inv, WriteHandle<'req>> {
        device
            .storage
            .request_alloc_slot_raw(device, self.current_frame, data_descriptor, layout)
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
                RequestId::Allocation(i) => ids.push(Id::hash(&i)),
                RequestId::CmdBufferCompletion(i) => ids.push(Id::hash(&(0, i))),
                RequestId::CmdBufferSubmission(i) => ids.push(Id::hash(&(1, i))),
                RequestId::Barrier(b_info, e) => ids.push([Id::hash(&b_info), Id::hash(&e)].id()),
                RequestId::Group(g) => ids.push(g.0),
                RequestId::GarbageCollect(g) => ids.push(Id::hash(&g)),
                RequestId::Ready => {}
                RequestId::YieldOnce => {}
                RequestId::ExternalProgress => {}
            }
            let mut poll = (r.gen_poll)(PollContext {
                storage: self.storage,
                disk_cache: self.disk_cache,
                device_contexts: self.device_contexts,
                current_frame: self.current_frame,
                predicted_preview_tasks: self.predicted_preview_tasks,
                current_task: self.current_task,
            });
            match r.type_ {
                RequestType::Data(_)
                | RequestType::Group(_)
                | RequestType::Ready
                | RequestType::Barrier(..)
                | RequestType::CmdBufferCompletion(_)
                | RequestType::CmdBufferSubmission(_)
                | RequestType::Allocation(..) => {
                    if let Some(v) = poll() {
                        done.push(MaybeUninit::new(v));
                        continue;
                    }
                }
                RequestType::ThreadPoolJob(_, _)
                | RequestType::GarbageCollect(_)
                | RequestType::ExternalProgress
                | RequestType::YieldOnce => {}
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
    pub fn storage(&self) -> &ram::Storage {
        &self.storage
    }

    pub fn preferred_device(&self, loc: DataLocation) -> &DeviceContext {
        let i = match loc {
            DataLocation::CPU(_cpu_data_location) => *self.device_contexts.keys().next().unwrap(),
            DataLocation::GPU(i) => i,
        };
        &self.device_contexts[&i]
    }

    pub fn first_device(&self) -> &DeviceContext {
        &self.device_contexts.values().next().unwrap()
    }

    pub fn device_ctx(&self, id: DeviceId) -> &DeviceContext {
        &self.device_contexts[&id]
    }

    pub fn past_deadline(&self, interactive: bool) -> Option<Lateness> {
        let deadline = if interactive {
            self.deadline.interactive
        } else {
            self.deadline.refinement
        };
        lateness_for_deadline(self.start, deadline)
    }
}

fn lateness_for_deadline(start: Instant, deadline: Instant) -> Option<Lateness> {
    let now = Instant::now();
    if deadline < now {
        let d = now - deadline;
        let total_duration = deadline
            .checked_duration_since(start)
            .unwrap_or(Duration::from_secs(0));
        Some(d.as_millis() as f32 / total_duration.as_millis() as f32)
    } else {
        None
    }
}

pub type Lateness = f32;

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

pub struct ChildTask<'a, R>(Pin<Box<dyn Future<Output = R> + 'a>>);

impl<'a, F, R> From<F> for ChildTask<'a, R>
where
    F: Future<Output = R> + 'a,
{
    fn from(inner: F) -> Self {
        Self(Box::pin(inner))
    }
}

#[pin_project]
pub struct TasksUnordered<'req, 'inv, R> {
    children: Vec<Option<ChildTask<'req, R>>>,
    children_waiting_on_requests: Map<usize, Map<RequestId, ProgressIndicator>>,
    requests_waited_on_by: Map<RequestId, Set<usize>>,
    resolved_deps: Map<usize, Set<RequestId>>,
    //task_map: Map<RequestId, Vec<(ResultPoll<'req, V>, D)>>,
    pending_results: Vec<R>,
    task_context: OpaqueTaskContext<'req, 'inv>,
    ready: Set<usize>,
}

impl<'req, 'inv, R> TasksUnordered<'req, 'inv, R> {
    pub fn new(
        task_context: OpaqueTaskContext<'req, 'inv>,
        requests: impl Iterator<Item = ChildTask<'req, R>> + 'req,
    ) -> Self {
        let requests: Vec<_> = requests.map(Some).collect();
        Self {
            ready: (0..requests.len()).collect(),
            children: requests,
            children_waiting_on_requests: Default::default(),
            requests_waited_on_by: Default::default(),
            pending_results: Default::default(),
            resolved_deps: Default::default(),
            task_context,
        }
    }

    pub async fn run(self) -> Vec<R> {
        self.collect().await
    }
}

impl<'req, 'inv, R> futures::Stream for TasksUnordered<'req, 'inv, R> {
    type Item = R;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let s = self.get_mut();
        {
            let mut completed_hints = Vec::new();
            for c in s.task_context.hints.completed.borrow().iter() {
                if let Some(waiting) = s.requests_waited_on_by.remove(c) {
                    completed_hints.push(*c);
                    for child in waiting.iter() {
                        let waited_on = s.children_waiting_on_requests.get_mut(child).unwrap();
                        let progress_indicator = waited_on.remove(c).unwrap();

                        s.resolved_deps.entry(*child).or_default().insert(*c);

                        if waited_on.is_empty()
                            || matches!(progress_indicator, ProgressIndicator::PartialPossible)
                        {
                            s.ready.insert(*child);
                        }
                        if waited_on.is_empty() {
                            s.children_waiting_on_requests.remove(child);
                        }
                    }
                }
            }
            for c in completed_hints {
                s.task_context.hints.noticed_completion(c);
            }

            let mut all_requested = s.task_context.requests.drain();
            let mut still_ready = Vec::new();
            for child_id in s.ready.drain() {
                let child = &mut s.children[child_id];

                let resolved_deps = s.resolved_deps.entry(child_id).or_default();
                s.task_context.hints.swap(resolved_deps);

                match child.as_mut().unwrap().0.poll_unpin(cx) {
                    Poll::Ready(r) => {
                        s.pending_results.push(r);
                        s.children[child_id] = None;

                        assert!(!s.children_waiting_on_requests.contains_key(&child_id));
                    }
                    Poll::Pending => {
                        let mut newly_requested = s.task_context.requests.drain();
                        //assert!(!newly_requested.is_empty());
                        for r in &mut newly_requested {
                            let r_id = r.id();
                            s.children_waiting_on_requests
                                .entry(child_id)
                                .or_default()
                                .insert(r_id, r.progress_indicator);
                            s.requests_waited_on_by
                                .entry(r_id)
                                .or_default()
                                .insert(child_id);
                            r.progress_indicator = ProgressIndicator::PartialPossible;
                        }
                        if newly_requested.is_empty() {
                            still_ready.push(child_id);
                        }
                        all_requested.append(&mut newly_requested);
                    }
                }

                s.task_context.hints.swap(resolved_deps);
            }
            s.ready.extend(still_ready);
            let tmp = s.task_context.requests.replace(all_requested);
            assert!(tmp.is_empty());
        }
        if let Some(r) = s.pending_results.pop() {
            return Poll::Ready(Some(r));
        }
        if s.children_waiting_on_requests.is_empty() && s.ready.is_empty() {
            assert!(s.requests_waited_on_by.is_empty());
            assert!(s.children_waiting_on_requests.is_empty());
            assert!(s.children.iter().all(Option::is_none));
            Poll::Ready(None)
        } else {
            Poll::Pending
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
            disk_cache: self.task_context.disk_cache,
            device_contexts: self.task_context.device_contexts,
            current_frame: self.task_context.current_frame,
            predicted_preview_tasks: self.task_context.predicted_preview_tasks,
            current_task: self.task_context.current_task,
        });
        match req.type_ {
            RequestType::Data(_)
            | RequestType::Group(_)
            | RequestType::Ready
            | RequestType::Barrier(..)
            | RequestType::CmdBufferCompletion(_)
            | RequestType::CmdBufferSubmission(_)
            | RequestType::Allocation(..) => {
                if let Some(r) = poll() {
                    self.ready.push_back((r, data));
                    return;
                }
            }
            RequestType::ThreadPoolJob(_, _)
            | RequestType::GarbageCollect(_)
            | RequestType::ExternalProgress
            | RequestType::YieldOnce => {}
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
pub struct TaskContext<'cref, 'inv, OutputType> {
    inner: OpaqueTaskContext<'cref, 'inv>,
    dtype: OutputType,
}

impl<'cref, 'inv, O> std::ops::Deref for TaskContext<'cref, 'inv, O> {
    type Target = OpaqueTaskContext<'cref, 'inv>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<'cref, 'inv, Output> Into<OpaqueTaskContext<'cref, 'inv>>
    for TaskContext<'cref, 'inv, Output>
{
    fn into(self) -> OpaqueTaskContext<'cref, 'inv> {
        self.inner
    }
}
impl<'cref, 'inv, OutputType> TaskContext<'cref, 'inv, OutputType> {
    pub(crate) unsafe fn new(inner: OpaqueTaskContext<'cref, 'inv>, dtype: OutputType) -> Self {
        Self { inner, dtype }
    }
}

impl<'cref, 'inv, T> TaskContext<'cref, 'inv, T>
where
    T: TryFrom<DType, Error = ConversionError>,
{
    pub fn try_from(value: TaskContext<'cref, 'inv, DType>) -> Result<Self, ConversionError> {
        Ok(Self {
            inner: value.inner,
            dtype: value.dtype.try_into()?,
        })
    }
}

impl<'cref, 'inv, Output: Element> TaskContext<'cref, 'inv, StaticElementType<Output>> {
    pub fn alloc_slot<'req, D: DynDimension>(
        &'req self,
        item: ChunkIndex,
        size: &Vector<D, LocalCoordinate>,
    ) -> Request<
        'req,
        'inv,
        Result<WriteHandleUninit<'req, [MaybeUninit<Output>]>, ConcurrentWriteAccessError>,
    > {
        self.alloc_slot_num_elements(item, size.hmul())
    }

    pub fn alloc_slot_num_elements<'req>(
        &'req self,
        item: ChunkIndex,
        size: usize,
    ) -> Request<
        'req,
        'inv,
        Result<WriteHandleUninit<'req, [MaybeUninit<Output>]>, ConcurrentWriteAccessError>,
    > {
        let id = DataDescriptor::new(self.current_op_desc().unwrap(), item);
        self.storage
            .request_alloc_slot(self.current_frame, id, size)
    }
}

pub struct ReuseResult<'a, 'inv> {
    pub request: Request<'a, 'inv, WriteHandle<'a>>,
    pub new: bool,
}

impl<'cref, 'inv, OutputType: ElementType> TaskContext<'cref, 'inv, OutputType> {
    pub fn alloc_slot_gpu<'a, D: DynDimension>(
        &'a self,
        device: &'a DeviceContext,
        item: ChunkIndex,
        size: &Vector<D, LocalCoordinate>,
    ) -> Request<'a, 'inv, WriteHandle<'a>> {
        self.alloc_slot_num_elements_gpu(device, item, size.hmul())
    }

    pub fn alloc_slot_num_elements_gpu<'a>(
        &'a self,
        device: &'a DeviceContext,
        item: ChunkIndex,
        size: usize,
    ) -> Request<'a, 'inv, WriteHandle<'a>> {
        let layout = self.dtype.array_layout(size);
        let id = DataDescriptor::new(self.current_op_desc().unwrap(), item);
        self.alloc_raw_gpu(device, id, layout)
    }

    pub fn alloc_try_reuse_gpu<'a>(
        &'a self,
        device: &'a DeviceContext,
        item: ChunkIndex,
        size: usize,
    ) -> ReuseResult<'a, 'inv> {
        let layout = self.dtype.array_layout(size);
        let id = DataId::new(self.current_op(), item);
        let descriptor = DataDescriptor {
            id,
            longevity: crate::storage::DataLongevity::Cache,
        };
        if let Ok(handle) =
            device
                .storage
                .try_promote_previous_preview(device, id, self.current_frame)
        {
            ReuseResult {
                request: Request::ready(handle),
                new: false,
            }
        } else {
            ReuseResult {
                request: self.alloc_raw_gpu(device, descriptor, layout),
                new: true,
            }
        }
    }

    pub fn access_state_cache_gpu<'a>(
        &'a self,
        device: &'a DeviceContext,
        item: ChunkIndex,
        name: &str,
        layout: Layout,
    ) -> Request<'a, 'inv, crate::storage::gpu::StateCacheResult<'a>> {
        let base_id = DataId::new(self.current_op(), item);
        let id = DataId(Id::combine(&[base_id.0, Id::hash(name)]));

        let data_descriptor = DataDescriptor {
            id,
            longevity: crate::storage::DataLongevity::Cache,
        };

        let mut access = Some(
            device
                .storage
                .register_access(device, self.current_frame, id),
        );

        let old = device.storage.is_initializing(id);

        Request {
            type_: RequestType::Allocation(
                AllocationId::next(),
                AllocationRequest::VRam(device.id, layout, data_descriptor),
            ),
            gen_poll: Box::new(move |_ctx| {
                Box::new(move || {
                    access = match device
                        .storage
                        .access_initializing_state_cache(access.take().unwrap(), self.current_frame)
                    {
                        Ok(r) => {
                            return Some(if old {
                                crate::storage::gpu::StateCacheResult::Existing(r)
                            } else {
                                crate::storage::gpu::StateCacheResult::New(r)
                            });
                        }
                        Err(acc) => Some(acc),
                    };
                    None
                })
            }),
            _marker: Default::default(),
        }
    }

    pub fn access_state_cache<'a, S: bytemuck::AnyBitPattern + Send>(
        &'a self,
        item: ChunkIndex,
        name: &str,
        size: usize,
    ) -> Request<'a, 'inv, crate::storage::cpu::StateCacheResult<'a, [MaybeUninit<S>], RamAllocator>>
    {
        let base_id = DataId::new(self.current_op(), item);
        let id = DataId(Id::combine(&[base_id.0, Id::hash(name)]));

        // Safety: We ensure no simulatenous access by taking the ChunkIndex into account for the
        // id.
        unsafe {
            self.storage
                .request_access_state_cache(self.current_frame, id, size)
        }
    }

    pub fn access_state_cache_shared<'req, T: Send + Default>(
        &'req self,
        name: &str,
        size: usize,
    ) -> Request<'req, 'inv, ReadHandle<'req, [T], RamAllocator>> {
        let base_id = self.current_op();
        let id = DataId(Id::combine(&[base_id.inner(), Id::hash(name)]));

        let access = self.storage.register_access(self.current_frame, id);
        let layout = Layout::array::<T>(size).unwrap();

        let longevity = crate::storage::DataLongevity::Cache;
        let data_descriptor = DataDescriptor { id, longevity };

        match unsafe { self.storage.read(access) } {
            Ok(r) => Request::ready(r),
            Err(access) => Request {
                type_: RequestType::Allocation(
                    AllocationId::next(),
                    AllocationRequest::Ram(layout, data_descriptor, CpuDataLocation::Ram),
                ),
                gen_poll: Box::new(move |_ctx| {
                    let mut access = Some(access);
                    Box::new(move || {
                        access = match self.storage.access_initializing(access.take().unwrap()) {
                            Ok(v) => {
                                let mut wh = v
                                    .unwrap()
                                    .map_drop_handler(|h| h.into_error())
                                    .transmute(size);
                                crate::data::fill_uninit_default(&mut *wh);
                                return Some(
                                    unsafe { wh.initialized(self.inner) }
                                        .into_read_handle(longevity),
                                );
                            }
                            Err(acc) => match unsafe { self.storage.read(acc) } {
                                Ok(r) => return Some(r),
                                Err(acc) => Some(acc),
                            },
                        };
                        None
                    })
                }),
                _marker: Default::default(),
            },
        }
    }
}

impl<'cref, 'inv, Output: Element> TaskContext<'cref, 'inv, StaticElementType<Output>> {
    #[must_use]
    pub fn write_scalar<'a>(&'a self, value: Output) -> Request<'a, 'inv, ()> {
        let ctx = **self;
        self.alloc_slot_num_elements(ChunkIndex(0), 1)
            .unwrap_value()
            .map(move |mut slot| {
                slot[0].write(value);
                unsafe { slot.initialized(ctx) };
            })
    }
}
impl<'cref, 'inv, Output: Element> TaskContext<'cref, 'inv, StaticElementType<Output>> {
    pub fn alloc_scalar_gpu<'a>(
        &'a self,
        device: &'a DeviceContext,
    ) -> Request<'a, 'inv, WriteHandle<'a>> {
        let layout = Layout::new::<Output>();
        let id = DataDescriptor::new(self.current_op_desc().unwrap(), ChunkIndex(0));
        self.alloc_raw_gpu(device, id, layout)
    }
}
