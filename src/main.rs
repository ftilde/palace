use clap::Parser;
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs::File,
    future::Future,
    hash::{Hash, Hasher},
    path::PathBuf,
    pin::Pin,
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

mod array;

type SVec3 = cgmath::Vector3<u32>;

fn hmul<S>(s: cgmath::Vector3<S>) -> S
where
    S: std::ops::Mul<S, Output = S>,
{
    s.x * s.y * s.z
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Id([u8; 20]);

impl Id {
    fn from_data(data: &[u8]) -> Self {
        let sha = sha1_smol::Sha1::from(data);
        Self(sha.digest().bytes())
    }
    fn combine(ids: &[Id]) -> Self {
        let mut sha = sha1_smol::Sha1::new();
        for id in ids {
            sha.update(&id.0);
        }
        Self(sha.digest().bytes())
    }
}

#[derive(Copy, Clone)]
struct TaskContext<'a> {
    requests: &'a RequestQueue,
    storage: &'a Storage,
}

impl<'a> TaskContext<'a> {
    async fn request(&'a self, caller: TaskId, info: TaskInfo) -> Result<Datum, Error> {
        let task_id = info.id();
        if let Some(data) = self.storage.read_ram(task_id) {
            return Ok(data.clone());
        }
        self.requests.push(Request { caller, info });
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

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct OperatorId(Id);
impl OperatorId {
    fn new<T>(inputs: &[OperatorId]) -> Self {
        // TODO: Maybe it's more efficient to use the sha.update method directly.
        let mut id = Id::from_data(std::any::type_name::<T>().as_ref());
        for i in inputs {
            id = Id::combine(&[id, i.0]);
        }
        OperatorId(id)
    }
}

impl From<Id> for OperatorId {
    fn from(inner: Id) -> Self {
        Self(inner)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TaskId(Id);
impl TaskId {
    fn new(op: OperatorId, d: &DatumRequest) -> Self {
        TaskId(Id::combine(&[op.0, d.id()]))
    }
}

struct TaskInfo {
    op: OperatorId,
    data: DatumRequest,
}
impl TaskInfo {
    fn new(op: OperatorId, data: DatumRequest) -> Self {
        Self { op, data }
    }
    fn id(&self) -> TaskId {
        TaskId::new(self.op, &self.data)
    }
}

#[derive(Copy, Clone, Hash)]
struct VoxelPosition(SVec3);
#[derive(Copy, Clone, Hash)]
struct BrickPosition(SVec3);

// TODO: Maybe we don't want this to be copy if it gets too large.
#[derive(Copy, Clone)]
struct VolumeMetaData {
    dimensions: VoxelPosition,
    brick_size: VoxelPosition,
}

fn div_round_up(v1: u32, v2: u32) -> u32 {
    (v1 + v2 - 1) / v2
}

impl VolumeMetaData {
    fn num_voxels(&self) -> u64 {
        hmul(self.dimensions.0.cast::<u64>().unwrap())
    }
    fn dimension_in_bricks(&self) -> BrickPosition {
        BrickPosition(self.dimensions.0.zip(self.brick_size.0, div_round_up))
    }
    fn brick_pos(&self, pos: VoxelPosition) -> BrickPosition {
        BrickPosition(pos.0.zip(self.brick_size.0, |a, b| a / b))
    }
    fn brick_begin(&self, pos: BrickPosition) -> VoxelPosition {
        VoxelPosition(pos.0.zip(self.brick_size.0, |a, b| a * b))
    }
    fn brick_end(&self, pos: BrickPosition) -> VoxelPosition {
        let next_pos = pos.0 + cgmath::vec3(1, 1, 1);
        let raw_end = next_pos.zip(self.brick_size.0, |a, b| a * b);
        VoxelPosition(raw_end.zip(self.dimensions.0, std::cmp::min))
    }
    fn brick_dim(&self, pos: BrickPosition) -> VoxelPosition {
        VoxelPosition(self.brick_end(pos).0 - self.brick_begin(pos).0)
    }
}

struct Brick<'a> {
    size: VoxelPosition,
    data: &'a [f32],
}

impl<'a> Brick<'a> {
    fn new(data: &'a BrickData, size: VoxelPosition) -> Self {
        Self {
            data: data.as_slice(),
            size,
        }
    }
    fn voxels(&'a self) -> impl Iterator<Item = f32> + 'a {
        itertools::iproduct! { 0..self.size.0.z, 0..self.size.0.y, 0..self.size.0.x }
            .map(|(z, y, x)| to_linear(cgmath::vec3(x, y, z), self.size.0))
            .map(|i| self.data[i])
    }
}

type BrickData = Vec<f32>;

#[derive(Clone)] //TODO remove clone bound
#[non_exhaustive]
enum Datum {
    Float(f32),
    Volume(VolumeMetaData),
    Brick(BrickData),
}

impl Datum {
    fn float(self) -> Result<f32, Error> {
        if let Datum::Float(v) = self {
            Ok(v)
        } else {
            Err("Value is not a float".into())
        }
    }
    fn volume(self) -> Result<VolumeMetaData, Error> {
        if let Datum::Volume(v) = self {
            Ok(v)
        } else {
            Err("Value is not a volume".into())
        }
    }
    fn brick(self) -> Result<BrickData, Error> {
        if let Datum::Brick(v) = self {
            Ok(v)
        } else {
            Err("Value is not a brick".into())
        }
    }
}

#[non_exhaustive]
#[derive(Hash)]
enum DatumRequest {
    Value,
    Brick(BrickPosition),
}

impl DatumRequest {
    fn id(&self) -> Id {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        let v = hasher.finish();
        let hash = bytemuck::bytes_of(&v);
        return Id::from_data(hash);
    }
}

trait Operator {
    fn id(&self) -> OperatorId;
    fn compute<'a>(&'a self, rt: TaskContext<'a>, info: DatumRequest) -> Task<'a>;
}

// TODO look into thiserror/anyhow
type Error = Box<(dyn std::error::Error + 'static)>;

struct RawVolumeSource {
    path: PathBuf,
    _file: File,
    mmap: memmap::Mmap,
    metadata: VolumeMetaData,
}

impl RawVolumeSource {
    fn open(path: PathBuf, metadata: VolumeMetaData) -> Result<Self, Error> {
        let file = File::open(&path)?;
        let mmap = unsafe { memmap::Mmap::map(&file)? };

        Ok(Self {
            path,
            _file: file,
            mmap,
            metadata,
        })
    }
}

fn to_linear(pos: SVec3, dim: SVec3) -> usize {
    let pos = pos.cast::<usize>().unwrap();
    let dim = dim.cast::<usize>().unwrap();
    (pos.z * dim.y + pos.y) * dim.y + pos.x
}

impl Operator for RawVolumeSource {
    fn id(&self) -> OperatorId {
        OperatorId::new::<Self>(&[Id::from_data(self.path.to_string_lossy().as_bytes()).into()])
    }

    fn compute<'a>(&'a self, _rt: TaskContext<'a>, info: DatumRequest) -> Task<'a> {
        async move {
            match info {
                DatumRequest::Value => Ok(Datum::Volume(self.metadata)),
                DatumRequest::Brick(pos) => {
                    let m = &self.metadata;
                    let begin = m.brick_begin(pos);
                    if !(begin.0.x < m.dimensions.0.x
                        && begin.0.y < m.dimensions.0.y
                        && begin.0.z < m.dimensions.0.z)
                    {
                        return Err("Brick position is outside of volume".into());
                    }
                    let brick_dim = m.brick_dim(pos).0;

                    let mut brick = vec![0.0; hmul(m.brick_size.0) as usize];
                    let voxel_size = std::mem::size_of::<f32>();
                    for z in 0..brick_dim.z {
                        for y in 0..brick_dim.y {
                            let bu8 = voxel_size
                                * to_linear(begin.0 + cgmath::vec3(0, y, z), m.dimensions.0);
                            let eu8 = voxel_size
                                * to_linear(
                                    begin.0 + cgmath::vec3(brick_dim.x, y, z),
                                    m.dimensions.0,
                                );

                            let bf32 = to_linear(cgmath::vec3(0, y, z), m.brick_size.0);
                            let ef32 = to_linear(cgmath::vec3(brick_dim.x, y, z), m.brick_size.0);

                            let in_ = &self.mmap[bu8..eu8];
                            let out = &mut brick[bf32..ef32];
                            out.copy_from_slice(bytemuck::cast_slice(in_));
                        }
                    }
                    Ok(Datum::Brick(brick))
                }
                _ => Err("Invalid Request".into()),
            }
        }
        .into()
    }
}

struct Scale {
    vol: OperatorId,
    factor: OperatorId,
}

impl Operator for Scale {
    fn id(&self) -> OperatorId {
        OperatorId::new::<Self>(&[self.vol, self.factor])
    }

    fn compute<'a>(&'a self, rt: TaskContext<'a>, info: DatumRequest) -> Task<'a> {
        async move {
            let this_id = TaskId::new(self.id(), &info);
            match info {
                DatumRequest::Value => {
                    // TODO: Depending on what exactly we store in the VolumeMetaData, we will have to
                    // update this. Maybe see VolumeFilterList in Voreen as a reference for how to
                    // model VolumeMetaData for this.
                    rt.request(this_id, TaskInfo::new(self.vol, DatumRequest::Value))
                        .await
                }
                b_req @ DatumRequest::Brick(_) => {
                    let f = rt
                        .request(this_id, TaskInfo::new(self.factor, DatumRequest::Value))
                        .await?
                        .float()?;
                    let mut b = rt
                        .request(this_id, TaskInfo::new(self.vol, b_req))
                        .await?
                        .brick()?;

                    for v in &mut b {
                        *v *= f;
                    }

                    Ok(Datum::Brick(b))
                }
            }
        }
        .into()
    }
}

struct Mean {
    vol: OperatorId,
}

impl Operator for Mean {
    fn id(&self) -> OperatorId {
        OperatorId::new::<Self>(&[self.vol])
    }

    fn compute<'a>(&'a self, rt: TaskContext<'a>, info: DatumRequest) -> Task<'a> {
        async move {
            let this_id = TaskId::new(self.id(), &info);
            match info {
                DatumRequest::Value => {
                    let mut sum = 0.0;

                    let vol = rt
                        .request(this_id, TaskInfo::new(self.vol, DatumRequest::Value))
                        .await?
                        .volume()?;

                    let bd = vol.dimension_in_bricks();
                    for z in 0..bd.0.z {
                        for y in 0..bd.0.y {
                            for x in 0..bd.0.x {
                                let brick_pos = BrickPosition(cgmath::vec3(x, y, z));
                                let brick_data = rt
                                    .request(
                                        this_id,
                                        TaskInfo::new(self.vol, DatumRequest::Brick(brick_pos)),
                                    )
                                    .await?
                                    .brick()?;

                                let brick = Brick::new(&brick_data, vol.brick_dim(brick_pos));

                                sum += brick.voxels().sum::<f32>();
                            }
                        }
                    }

                    let v = sum / vol.num_voxels() as f32;
                    Ok(Datum::Float(v))
                }
                _ => Err("Invalid Request".into()),
            }
        }
        .into()
    }
}

impl Operator for f32 {
    fn id(&self) -> OperatorId {
        OperatorId::new::<f32>(&[OperatorId(Id::from_data(bytemuck::bytes_of(self)))])
    }

    fn compute<'a>(&'a self, _rt: TaskContext<'a>, info: DatumRequest) -> Task<'a> {
        async move {
            match info {
                DatumRequest::Value => Ok(Datum::Float(*self)),
                _ => Err("Invalid Request".into()),
            }
        }
        .into()
    }
}

struct Network {
    operators: BTreeMap<OperatorId, Box<dyn Operator>>,
}

impl Network {
    fn new() -> Self {
        Network {
            operators: BTreeMap::new(),
        }
    }

    fn add(&mut self, op: impl Operator + 'static) -> OperatorId {
        let id = op.id();
        self.operators.insert(op.id(), Box::new(op));
        id
    }
}

struct Task<'a>(Pin<Box<dyn Future<Output = Result<Datum, Error>> + 'a>>);

impl<'a, F> From<F> for Task<'a>
where
    F: Future<Output = Result<Datum, Error>> + 'a,
{
    fn from(inner: F) -> Self {
        Self(Box::pin(inner))
    }
}

struct TaskGraph<'a> {
    tasks: BTreeMap<TaskId, Task<'a>>,
    deps: BTreeMap<TaskId, BTreeSet<TaskId>>, // key requires values
    rev_deps: BTreeMap<TaskId, BTreeSet<TaskId>>, // values require key
    ready: BTreeSet<TaskId>,
}

impl<'a> TaskGraph<'a> {
    fn new() -> Self {
        Self {
            tasks: BTreeMap::new(),
            deps: BTreeMap::new(),
            rev_deps: BTreeMap::new(),
            ready: BTreeSet::new(),
        }
    }

    fn exists(&self, id: TaskId) -> bool {
        self.tasks.contains_key(&id)
    }

    fn add(&mut self, id: TaskId, task: Task<'a>) {
        let prev = self.tasks.insert(id, task);
        assert!(prev.is_none(), "Tried to insert task twice");
        self.ready.insert(id);
    }

    fn add_dependency(&mut self, wants: TaskId, wanted: TaskId) {
        self.deps.entry(wants).or_default().insert(wanted);
        self.rev_deps.entry(wanted).or_default().insert(wants);
        self.ready.remove(&wants);
    }

    fn resolved(&mut self, id: TaskId) {
        let removed = self.tasks.remove(&id);
        assert!(removed.is_some(), "Task was not present");
        self.ready.remove(&id);

        for rev_dep in self.rev_deps.remove(&id).iter().flatten() {
            let deps_of_rev_dep = self.deps.get_mut(&rev_dep).unwrap();
            let removed = deps_of_rev_dep.remove(&id);
            assert!(removed);
            if deps_of_rev_dep.is_empty() {
                let inserted = self.ready.insert(*rev_dep);
                assert!(inserted);
            }
        }
    }

    fn get_mut(&mut self, id: TaskId) -> &mut Task<'a> {
        self.tasks.get_mut(&id).unwrap() //TODO: make api here nicer to avoid unwraps etc.
    }

    fn ready(&self) -> Vec<TaskId> {
        self.ready.iter().cloned().collect()
    }
}

fn dummy_raw_waker() -> RawWaker {
    fn no_op(_: *const ()) {}
    fn clone(_: *const ()) -> RawWaker {
        dummy_raw_waker()
    }

    let vtable = &RawWakerVTable::new(clone, no_op, no_op, no_op);
    RawWaker::new(0 as *const (), vtable)
}

fn dummy_waker() -> Waker {
    let raw = dummy_raw_waker();
    // Safety: The dummy waker literally does nothing and thus upholds all cantracts of
    // `RawWaker`/`RawWakerVTable`.
    unsafe { Waker::from_raw(raw) }
}

struct Request {
    caller: TaskId,
    info: TaskInfo,
}

struct Storage {
    memory_cache: RefCell<BTreeMap<TaskId, Datum>>,
}

impl Storage {
    fn new() -> Self {
        Self {
            memory_cache: RefCell::new(BTreeMap::new()),
        }
    }
    fn store_ram(&self, key: TaskId, datum: Datum) {
        let prev = self.memory_cache.borrow_mut().insert(key, datum);
        assert!(prev.is_none());
    }
    fn read_ram(&self, key: TaskId) -> Option<Datum> {
        self.memory_cache.borrow().get(&key).cloned()
    }
}

struct RequestQueue {
    buffer: RefCell<VecDeque<Request>>,
}
impl RequestQueue {
    fn new() -> Self {
        Self {
            buffer: RefCell::new(VecDeque::new()),
        }
    }
    fn push(&self, req: Request) {
        self.buffer.borrow_mut().push_back(req)
    }
    fn drain<'a>(&'a self) -> impl Iterator<Item = Request> + 'a {
        self.buffer
            .borrow_mut()
            .drain(..)
            .collect::<Vec<_>>()
            .into_iter()
    }
}

struct Statistics {
    tasks_executed: usize,
}

impl Statistics {
    fn new() -> Self {
        Self { tasks_executed: 0 }
    }
}

struct RunTime<'a> {
    network: &'a Network,
    waker: Waker,
    tasks: TaskGraph<'a>,
    storage: &'a Storage,
    request_queue: &'a RequestQueue,
    statistics: Statistics,
}

impl<'a> RunTime<'a> {
    fn new(network: &'a Network, storage: &'a Storage, request_queue: &'a RequestQueue) -> Self {
        RunTime {
            network,
            waker: dummy_waker(),
            tasks: TaskGraph::new(),
            storage,
            request_queue,
            statistics: Statistics::new(),
        }
    }
    fn context(&self) -> TaskContext<'a> {
        TaskContext {
            requests: self.request_queue,
            storage: self.storage,
        }
    }

    fn run(&mut self) -> Result<(), Error> {
        loop {
            let ready = self.tasks.ready();
            if ready.is_empty() {
                return Ok(());
            }
            for task_id in ready {
                let mut ctx = Context::from_waker(&self.waker);
                let task = self.tasks.get_mut(task_id);
                match task.0.as_mut().poll(&mut ctx) {
                    Poll::Ready(res) => {
                        self.storage.store_ram(task_id, res?);
                        self.tasks.resolved(task_id);
                        self.statistics.tasks_executed += 1;
                    }
                    Poll::Pending => {
                        // TODO: we can get rid of the caller argument in a lot of the functions
                        // above because we implicitly know that these came for precisely this task.
                        for req in self.request_queue.drain() {
                            let Some(op) = self.network.operators.get(&req.info.op) else {
                                return Err("Operator with specified id not found".into());
                            };

                            let task_id = req.info.id();
                            if !self.tasks.exists(task_id) {
                                let task = op.compute(self.context(), req.info.data);

                                self.tasks.add(task_id, task);
                            }
                            self.tasks.add_dependency(req.caller, task_id);
                        }
                    }
                };
            }
        }
    }

    fn request_blocking(&mut self, op: OperatorId, req: DatumRequest) -> Result<Datum, Error> {
        let info = TaskInfo::new(op, req);
        let Some(op) = self.network.operators.get(&op) else {
            return Err("Operator with specified id not found".into());
        };
        let task_id = info.id();
        let task = op.compute(self.context(), info.data);

        self.tasks.add(task_id, task);

        // TODO this can probably be optimized to only compute the values necessary for the
        // requested task. (Although the question is if such a situation ever occurs with the
        // current API...)
        self.run()?;

        Ok(self.storage.read_ram(task_id).unwrap().clone())
    }
}

#[derive(Parser)]
struct CliArgs {
    #[arg()]
    dim_x: u32,
    #[arg()]
    dim_y: u32,
    #[arg()]
    dim_z: u32,

    #[arg()]
    raw_vol: PathBuf,

    #[arg(short, long, default_value = "1.0")]
    factor: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse();

    let mut network = Network::new();

    let vol_info = VolumeMetaData {
        dimensions: VoxelPosition(cgmath::vec3(args.dim_x, args.dim_y, args.dim_z)),
        brick_size: VoxelPosition(cgmath::vec3(32, 32, 32)),
    };

    let vol = network.add(RawVolumeSource::open(args.raw_vol, vol_info)?);

    let factor = network.add(args.factor);

    let scaled1 = network.add(Scale { vol, factor });

    let scaled2 = network.add(Scale {
        vol: scaled1,
        factor,
    });

    let mean = network.add(Mean { vol: scaled2 });
    let mean_unscaled = network.add(Mean { vol });

    let storage = Storage::new();
    let request_queue = RequestQueue::new();
    let mut rt = RunTime::new(&network, &storage, &request_queue);

    let mean_val = rt.request_blocking(mean, DatumRequest::Value)?.float()?;

    let tasks_executed = rt.statistics.tasks_executed;
    println!(
        "Computed scaled mean val: {} ({} tasks)",
        mean_val, tasks_executed
    );
    let mean_val_unscaled = rt
        .request_blocking(mean_unscaled, DatumRequest::Value)?
        .float()?;
    let tasks_executed = rt.statistics.tasks_executed - tasks_executed;
    println!(
        "Computed unscaled mean val: {} ({} tasks)",
        mean_val_unscaled, tasks_executed
    );

    Ok(())
}
