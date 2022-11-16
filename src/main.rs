use clap::Parser;
use std::{collections::BTreeMap, fs::File, path::PathBuf};

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

#[derive(Copy, Clone)]
struct VoxelPosition(SVec3);
#[derive(Copy, Clone)]
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
enum DatumRequest {
    Value,
    Brick(BrickPosition),
}

trait Operator {
    fn id(&self) -> OperatorId;
    fn compute(&self, rt: &RunTime, info: DatumRequest) -> Result<Datum, Error>;
}

struct VariableString {
    value: String,
    id: OperatorId,
}

// TODO look into thiserror/anyhow
type Error = Box<(dyn std::error::Error + 'static)>;

impl Operator for VariableString {
    fn id(&self) -> OperatorId {
        self.id
    }
    fn compute(&self, _rt: &RunTime, info: DatumRequest) -> Result<Datum, Error> {
        if let DatumRequest::Value = info {
            Ok(Datum::String(self.value.clone()))
        } else {
            Err("Invalid Request".into())
        }
    }
}

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

    fn compute(&self, _rt: &RunTime, info: DatumRequest) -> Result<Datum, Error> {
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
                        let bu8 =
                            voxel_size * to_linear(begin.0 + cgmath::vec3(0, y, z), m.dimensions.0);
                        let eu8 = voxel_size
                            * to_linear(begin.0 + cgmath::vec3(brick_dim.x, y, z), m.dimensions.0);

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
}

struct Scale {
    vol: OperatorId,
    factor: OperatorId,
}

impl Operator for Scale {
    fn id(&self) -> OperatorId {
        OperatorId::new::<Self>(&[self.vol, self.factor])
    }

    fn compute(&self, rt: &RunTime, info: DatumRequest) -> Result<Datum, Error> {
        match info {
            DatumRequest::Value => {
                // TODO: Depending on what exactly we store in the VolumeMetaData, we will have to
                // update this. Maybe see VolumeFilterList in Voreen as a reference for how to
                // model VolumeMetaData for this.
                rt.request(self.vol, DatumRequest::Value)
            }
            b_req @ DatumRequest::Brick(_) => {
                let f = rt.request(self.factor, DatumRequest::Value)?.float()?;
                let mut b = rt.request(self.vol, b_req)?.brick()?;

                for v in &mut b {
                    *v *= f;
                }

                Ok(Datum::Brick(b))
            }
        }
    }
}

struct Mean {
    vol: OperatorId,
}

impl Operator for Mean {
    fn id(&self) -> OperatorId {
        OperatorId::new::<Self>(&[self.vol])
    }

    fn compute(&self, rt: &RunTime, info: DatumRequest) -> Result<Datum, Error> {
        match info {
            DatumRequest::Value => {
                let mut sum = 0.0;

                let vol = rt.request(self.vol, DatumRequest::Value)?.volume()?;

                let bd = vol.dimension_in_bricks();
                for z in 0..bd.0.z {
                    for y in 0..bd.0.y {
                        for x in 0..bd.0.x {
                            let brick_pos = BrickPosition(cgmath::vec3(x, y, z));
                            let brick_data = rt
                                .request(self.vol, DatumRequest::Brick(brick_pos))?
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
}

impl Operator for f32 {
    fn id(&self) -> OperatorId {
        OperatorId::new::<f32>(&[OperatorId(Id::from_data(bytemuck::bytes_of(self)))])
    }

    fn compute(&self, _rt: &RunTime, info: DatumRequest) -> Result<Datum, Error> {
        match info {
            DatumRequest::Value => Ok(Datum::Float(*self)),
            _ => Err("Invalid Request".into()),
        }
    }
}

struct RunTime {
    operators: BTreeMap<OperatorId, Box<dyn Operator>>,
}

impl RunTime {
    fn new() -> Self {
        RunTime {
            operators: BTreeMap::new(),
        }
    }

    fn add(&mut self, op: impl Operator + 'static) -> OperatorId {
        let id = op.id();
        self.operators.insert(op.id(), Box::new(op));
        id
    }
    fn request(&self, op: OperatorId, info: DatumRequest) -> Result<Datum, Error> {
        // TODO: here caching
        let Some(op) = self.operators.get(&op) else {
            return Err("Operator with specified id not found".into());
        };
        op.compute(&self, info)
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

    let mut rt = RunTime::new();

    let vol_info = VolumeMetaData {
        dimensions: VoxelPosition(cgmath::vec3(args.dim_x, args.dim_y, args.dim_z)),
        brick_size: VoxelPosition(cgmath::vec3(32, 32, 32)),
    };

    let vol = rt.add(RawVolumeSource::open(args.raw_vol, vol_info)?);

    let factor = rt.add(args.factor);

    let scaled = rt.add(Scale { vol, factor });

    let mean = rt.add(Mean { vol: scaled });

    let mean_val = rt.request(mean, DatumRequest::Value)?.float()?;

    println!("Computed scaled mean val: {}", mean_val);
    Ok(())
}
