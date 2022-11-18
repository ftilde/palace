use derive_more::Constructor;
use std::{fs::File, path::PathBuf};

use crate::{
    data::{hmul, BrickData, BrickPosition, Datum, SVec3, VolumeMetaData, VoxelPosition},
    id::Id,
    operator::{Operator, OperatorId},
    task::{DatumRequest, Task, TaskContext, TaskInfo},
    Error,
};

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

pub struct RawVolumeSource {
    path: PathBuf,
    _file: File,
    mmap: memmap::Mmap,
    metadata: VolumeMetaData,
}

impl RawVolumeSource {
    pub fn open(path: PathBuf, metadata: VolumeMetaData) -> Result<Self, Error> {
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
            }
        }
        .into()
    }
}

pub struct Scale {
    pub vol: OperatorId,
    pub factor: OperatorId,
}

impl Operator for Scale {
    fn id(&self) -> OperatorId {
        OperatorId::new::<Self>(&[self.vol, self.factor])
    }

    fn compute<'a>(&'a self, rt: TaskContext<'a>, info: DatumRequest) -> Task<'a> {
        async move {
            match info {
                DatumRequest::Value => {
                    // TODO: Depending on what exactly we store in the VolumeMetaData, we will have to
                    // update this. Maybe see VolumeFilterList in Voreen as a reference for how to
                    // model VolumeMetaData for this.
                    rt.request(TaskInfo::new(self.vol, DatumRequest::Value))
                        .await
                }
                b_req @ DatumRequest::Brick(_) => {
                    let f = rt
                        .request(TaskInfo::new(self.factor, DatumRequest::Value))
                        .await?
                        .float()?;
                    let mut b = rt.request(TaskInfo::new(self.vol, b_req)).await?.brick()?;

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

#[derive(Constructor)]
pub struct Mean {
    vol: OperatorId,
}

impl Operator for Mean {
    fn id(&self) -> OperatorId {
        OperatorId::new::<Self>(&[self.vol])
    }

    fn compute<'a>(&'a self, rt: TaskContext<'a>, info: DatumRequest) -> Task<'a> {
        async move {
            match info {
                DatumRequest::Value => {
                    let mut sum = 0.0;

                    let vol = rt
                        .request(TaskInfo::new(self.vol, DatumRequest::Value))
                        .await?
                        .volume()?;

                    let bd = vol.dimension_in_bricks();
                    for z in 0..bd.0.z {
                        for y in 0..bd.0.y {
                            for x in 0..bd.0.x {
                                let brick_pos = BrickPosition(cgmath::vec3(x, y, z));
                                let brick_data = rt
                                    .request(TaskInfo::new(
                                        self.vol,
                                        DatumRequest::Brick(brick_pos),
                                    ))
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
        OperatorId::new::<f32>(&[Id::from_data(bytemuck::bytes_of(self)).into()])
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
