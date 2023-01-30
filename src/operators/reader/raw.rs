use std::{fs::File, path::PathBuf};

use crate::{
    data::{to_linear, BrickPosition, LocalVoxelPosition, VolumeMetaData, VoxelPosition},
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

        let byte_size = crate::data::hmul(size) * std::mem::size_of::<f32>();
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
        brick_size: LocalVoxelPosition,
        ctx: TaskContext<'cref, 'inv, BrickPosition, f32>,
        positions: Vec<BrickPosition>,
    ) -> Result<(), Error> {
        let m = VolumeMetaData {
            dimensions: self.size,
            brick_size,
        };
        for pos in positions {
            let begin = m.brick_begin(pos);
            if !(begin.x() < m.dimensions.x()
                && begin.y() < m.dimensions.y()
                && begin.z() < m.dimensions.z())
            {
                return Err("Brick position is outside of volume".into());
            }
            let brick_dim = m.brick_dim(pos);
            let num_voxels = crate::data::hmul(m.brick_size);

            let voxel_size = std::mem::size_of::<f32>();

            let mut brick_handle = ctx.alloc_slot(pos, num_voxels)?;
            let brick_data = &mut *brick_handle;
            ctx.submit(ctx.spawn_io(move || {
                brick_data.iter_mut().for_each(|v| {
                    v.write(f32::NAN);
                });

                for z in 0..brick_dim.z().raw {
                    for y in 0..brick_dim.y().raw {
                        let pb: LocalVoxelPosition = [z.into(), y.into(), 0.into()].into();
                        let pe: LocalVoxelPosition = [z.into(), y.into(), brick_dim.x()].into();
                        let bu8 = voxel_size * to_linear(begin + pb, m.dimensions);
                        let eu8 = voxel_size * to_linear(begin + pe, m.dimensions);

                        let bf32 = to_linear(pb, m.brick_size);
                        let ef32 = to_linear(pe, m.brick_size);

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
    pub fn operate<'op>(
        &'op self,
        brick_size: LocalVoxelPosition,
    ) -> Operator<'op, BrickPosition, f32> {
        Operator::new(
            OperatorId::new("RawVolumeSourceState::operate")
                .dependent_on(self.path.to_string_lossy().as_bytes()),
            move |ctx, positions, _| {
                async move { self.load_raw_bricks(brick_size, ctx, positions).await }.into()
            },
        )
    }
}
