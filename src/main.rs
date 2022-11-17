use clap::Parser;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    fs::File,
    future::Future,
    hash::{Hash, Hasher},
    path::PathBuf,
    pin::Pin,
    task::{Context, RawWaker, RawWakerVTable, Waker},
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

#[non_exhaustive]
enum Datum {
    String(String),
    Float(f32),
    Volume(VolumeMetaData),
    Brick(BrickData),
}

impl Datum {
    fn string(self) -> Result<String, Error> {
        if let Datum::String(v) = self {
            Ok(v)
        } else {
            Err("Value is not a string".into())
        }
    }
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
    fn compute<'a>(
        &'a self,
        rt: &'a RunTime<'a>,
        info: DatumRequest,
    ) -> Box<dyn Future<Output = Result<Datum, Error>> + 'a>;
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

    fn compute<'a>(
        &'a self,
        _rt: &'a RunTime<'a>,
        info: DatumRequest,
    ) -> Box<dyn Future<Output = Result<Datum, Error>> + 'a> {
        Box::new(async move {
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
        })
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

    fn compute<'a>(
        &'a self,
        rt: &'a RunTime<'a>,
        info: DatumRequest,
    ) -> Box<dyn Future<Output = Result<Datum, Error>> + 'a> {
        Box::new(async move {
            match info {
                DatumRequest::Value => {
                    // TODO: Depending on what exactly we store in the VolumeMetaData, we will have to
                    // update this. Maybe see VolumeFilterList in Voreen as a reference for how to
                    // model VolumeMetaData for this.
                    rt.request(self.id(), TaskInfo::new(self.vol, DatumRequest::Value))
                        .await
                }
                b_req @ DatumRequest::Brick(_) => {
                    let f = rt
                        .request(self.id(), TaskInfo::new(self.factor, DatumRequest::Value))
                        .await?
                        .float()?;
                    let mut b = rt
                        .request(self.id(), TaskInfo::new(self.vol, b_req))
                        .await?
                        .brick()?;

                    for v in &mut b {
                        *v *= f;
                    }

                    Ok(Datum::Brick(b))
                }
            }
        })
    }
}

struct Mean {
    vol: OperatorId,
}

impl Operator for Mean {
    fn id(&self) -> OperatorId {
        OperatorId::new::<Self>(&[self.vol])
    }

    fn compute<'a>(
        &'a self,
        rt: &'a RunTime<'a>,
        info: DatumRequest,
    ) -> Box<dyn Future<Output = Result<Datum, Error>> + 'a> {
        Box::new(async move {
            match info {
                DatumRequest::Value => {
                    let mut sum = 0.0;

                    let vol = rt
                        .request(self.id(), TaskInfo::new(self.vol, DatumRequest::Value))
                        .await?
                        .volume()?;

                    let bd = vol.dimension_in_bricks();
                    for z in 0..bd.0.z {
                        for y in 0..bd.0.y {
                            for x in 0..bd.0.x {
                                let brick_pos = BrickPosition(cgmath::vec3(x, y, z));
                                let brick_data = rt
                                    .request(
                                        self.id(),
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
        })
    }
}

impl Operator for f32 {
    fn id(&self) -> OperatorId {
        OperatorId::new::<f32>(&[OperatorId(Id::from_data(bytemuck::bytes_of(self)))])
    }

    fn compute<'a>(
        &'a self,
        _rt: &'a RunTime<'a>,
        info: DatumRequest,
    ) -> Box<dyn Future<Output = Result<Datum, Error>> + 'a> {
        Box::new(async move {
            match info {
                DatumRequest::Value => Ok(Datum::Float(*self)),
                _ => Err("Invalid Request".into()),
            }
        })
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

struct RunTime<'a> {
    network: &'a Network,
    waker: Waker,
    tasks: RefCell<BTreeMap<TaskId, Pin<Box<dyn Future<Output = Result<Datum, Error>> + 'a>>>>,
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

impl<'a> RunTime<'a> {
    fn new(network: &'a Network) -> Self {
        RunTime {
            network,
            waker: dummy_waker(),
            tasks: RefCell::new(BTreeMap::new()),
        }
    }

    async fn request(&'a self, caller: OperatorId, info: TaskInfo) -> Result<Datum, Error> {
        // TODO: here caching
        let task_id = info.id();
        let Some(op) = self.network.operators.get(&info.op) else {
            return Err("Operator with specified id not found".into());
        };
        let mut task = Box::into_pin(op.compute(&self, info.data));
        let mut context = Context::from_waker(&self.waker);

        // TODO: not sure if we want to poll here once or not.
        match task.as_mut().poll(&mut context) {
            std::task::Poll::Ready(res) => return res,
            std::task::Poll::Pending => {}
        }

        self.tasks.borrow_mut().insert(task_id, task);

        loop {
            let mut tasks = self.tasks.borrow_mut();
            let task = tasks.get_mut(&task_id).unwrap();

            match task.as_mut().poll(&mut context) {
                std::task::Poll::Ready(res) => return res,
                std::task::Poll::Pending => {}
            }
        }
        // TODO: figure out waking up tasks, construct a wait graph using caller and info
    }

    fn request_blocking(&self, op: OperatorId, info: DatumRequest) -> Result<Datum, Error> {
        todo!()
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

    let scaled = network.add(Scale { vol, factor });

    let mean = network.add(Mean { vol: scaled });

    let rt = RunTime::new(&network);

    let mean_val = rt.request_blocking(mean, DatumRequest::Value)?.float()?;

    println!("Computed scaled mean val: {}", mean_val);
    Ok(())
}
