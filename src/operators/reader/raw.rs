use std::{fs::File, path::PathBuf};

use crate::{
    data::{hmul, to_linear, Datum, VolumeMetaData},
    id::Id,
    operator::{Operator, OperatorId},
    task::{DatumRequest, Task, TaskContext},
    Error,
};

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
