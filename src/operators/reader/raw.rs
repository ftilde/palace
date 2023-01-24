use std::{fs::File, path::PathBuf};

use crate::{
    data::{to_linear, BrickPosition, VolumeMetaData, VoxelPosition},
    operator::{Operator, OperatorId},
    task::TaskContext,
    Error,
};

pub struct RawVolumeSourceState {
    pub path: PathBuf,
    _file: File,
    mmap: memmap::Mmap,
    pub size: VoxelPosition,
}

impl RawVolumeSourceState {
    pub fn open(path: PathBuf, size: VoxelPosition) -> Result<Self, Error> {
        let file = File::open(&path)?;
        let mmap = unsafe { memmap::Mmap::map(&file)? };

        let byte_size = crate::data::hmul(size.0) as usize * std::mem::size_of::<f32>();
        assert_eq!(file.metadata()?.len(), byte_size as u64);

        Ok(Self {
            path,
            _file: file,
            mmap,
            size,
        })
    }
    pub async fn load_raw_bricks<'cref, 'inv>(
        &self,
        brick_size: VoxelPosition,
        ctx: TaskContext<'cref, 'inv, BrickPosition, f32>,
        positions: Vec<BrickPosition>,
    ) -> Result<(), Error> {
        let m = VolumeMetaData {
            dimensions: self.size,
            brick_size,
        };
        for pos in positions {
            let begin = m.brick_begin(pos);
            if !(begin.0.x < m.dimensions.0.x
                && begin.0.y < m.dimensions.0.y
                && begin.0.z < m.dimensions.0.z)
            {
                return Err("Brick position is outside of volume".into());
            }
            let brick_dim = m.brick_dim(pos).0;
            let num_voxels = crate::data::hmul(m.brick_size.0) as usize;

            let voxel_size = std::mem::size_of::<f32>();

            let mut brick_handle = ctx.alloc_slot(pos, num_voxels)?;
            let brick_data = &mut *brick_handle;
            ctx.submit(ctx.spawn_job(move || {
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

            // Safety: At this point the thread pool job above has finished and has initialized all bytes
            // in the brick.
            unsafe { brick_handle.initialized() };
        }
        Ok(())
    }

    #[allow(unused)] // We probably will use it directly at some point
    pub fn operate<'op>(&'op self, brick_size: VoxelPosition) -> Operator<'op, BrickPosition, f32> {
        Operator::new(
            OperatorId::new("RawVolumeSourceState::operate")
                .dependent_on(self.path.to_string_lossy().as_bytes()),
            move |ctx, positions, _| {
                async move { self.load_raw_bricks(brick_size, ctx, positions).await }.into()
            },
        )
    }
}
