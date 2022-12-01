use std::{fs::File, path::PathBuf};

use crate::{
    data::{hmul, to_linear, VolumeMetaData},
    id::Id,
    operator::{Operator, OperatorId},
    operators::{VolumeOperator, VolumeTaskContext},
    task::Task,
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
}

impl VolumeOperator for RawVolumeSource {
    fn compute_metadata<'tasks, 'op: 'tasks>(
        &'op self,
        ctx: VolumeTaskContext<'op, 'tasks>,
    ) -> Task<'tasks> {
        async move { ctx.write_metadata(self.metadata) }.into()
    }

    fn compute_brick<'tasks, 'op: 'tasks>(
        &'op self,
        ctx: VolumeTaskContext<'op, 'tasks>,
        pos: crate::data::BrickPosition,
    ) -> Task<'tasks> {
        async move {
            let m = &self.metadata;
            let begin = m.brick_begin(pos);
            if !(begin.0.x < m.dimensions.0.x
                && begin.0.y < m.dimensions.0.y
                && begin.0.z < m.dimensions.0.z)
            {
                return Err("Brick position is outside of volume".into());
            }
            let brick_dim = m.brick_dim(pos).0;
            let num_voxels = hmul(m.brick_size.0) as usize;

            let voxel_size = std::mem::size_of::<f32>();

            // Safety: We are zeroing all brick data in a first step.
            // TODO: We might want to lift this restriction in the future
            let (brick_data, token) = ctx.brick_slot(pos, num_voxels)?;
            ctx.submit(ctx.thread_pool.spawn(move || {
                brick_data.iter_mut().for_each(|v| {
                    v.write(0.0);
                });

                for z in 0..brick_dim.z {
                    for y in 0..brick_dim.y {
                        let bu8 =
                            voxel_size * to_linear(begin.0 + cgmath::vec3(0, y, z), m.dimensions.0);
                        let eu8 = voxel_size
                            * to_linear(begin.0 + cgmath::vec3(brick_dim.x, y, z), m.dimensions.0);

                        let bf32 = to_linear(cgmath::vec3(0, y, z), m.brick_size.0);
                        let ef32 = to_linear(cgmath::vec3(brick_dim.x, y, z), m.brick_size.0);

                        let in_ = &self.mmap[bu8..eu8];
                        let out = &mut brick_data[bf32..ef32];
                        let in_slice: &[f32] = bytemuck::cast_slice(in_);
                        for (in_, out) in in_slice.iter().zip(out.iter_mut()) {
                            out.write(*in_);
                        }
                    }
                }
            }))
            .await;

            // At this point the thread pool job above has finished and has initialized all bytes
            // in the brick.
            unsafe { ctx.storage.mark_initialized(token) };
            Ok(())
        }
        .into()
    }
}
