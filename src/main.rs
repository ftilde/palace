use std::{fs::File, path::PathBuf};

type SVec3 = cgmath::Vector3<usize>;

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
struct VolumeMetaData {
    dimensions: SVec3,
    brick_size: SVec3,
}

enum Datum {
    String(String),
    Float(f32),
    Volume(VolumeMetaData),
    Brick(Vec<f32>),
}

trait Operator {
    fn id(&self) -> OperatorId;
    fn compute(&self, info: DatumRequest) -> Result<Datum, Error>;
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
    fn compute(&self, info: DatumRequest) -> Result<Datum, Error> {
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
    fn new(path: PathBuf, metadata: VolumeMetaData) -> Result<Self, Error> {
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
    (pos.z * dim.y + pos.y) * dim.y + pos.x
}

impl Operator for RawVolumeSource {
    fn id(&self) -> OperatorId {
        OperatorId::new::<Self>(&[Id::from_data(self.path.to_string_lossy().as_bytes()).into()])
    }

    fn compute(&self, info: DatumRequest) -> Result<Datum, Error> {
        match info {
            DatumRequest::Value => Ok(Datum::Volume(self.metadata)),
            DatumRequest::Brick(pos) => {
                let m = &self.metadata;
                let begin = pos.zip(m.brick_size, |a, b| a * b);
                if !(begin.x < m.dimensions.x
                    && begin.y < m.dimensions.y
                    && begin.z < m.dimensions.z)
                {
                    return Err("Brick position is outside of volume".into());
                }
                let end = begin + m.brick_size;
                let end = end.zip(m.dimensions, std::cmp::min);
                let end_brick = end - begin;

                let mut brick = vec![0.0; hmul(m.brick_size)];
                let voxel_size = std::mem::size_of::<f32>();
                for z in 0..end_brick.z {
                    for y in 0..end_brick.y {
                        let bu8 =
                            voxel_size * to_linear(begin + cgmath::vec3(0, y, z), m.dimensions);
                        let eu8 = voxel_size
                            * to_linear(begin + cgmath::vec3(end_brick.x, y, z), m.dimensions);

                        let bf32 = to_linear(cgmath::vec3(0, y, z), m.brick_size);
                        let ef32 = to_linear(cgmath::vec3(end_brick.x, y, z), m.brick_size);

                        brick[bf32..ef32]
                            .copy_from_slice(bytemuck::cast_slice(&self.mmap[bu8..eu8]));
                    }
                }
                Ok(Datum::Brick(brick))
            } //_ => Err("Invalid Request".into()),
        }
    }
}

struct RunTime {}

enum DatumRequest {
    Value,
    Brick(SVec3),
}

impl RunTime {
    fn request(op: OperatorId, info: DatumRequest) -> Datum {
        todo!()
    }
}

fn main() {
    println!("Hello, world!");
}
