use std::{fs::File, path::PathBuf};

use futures::StreamExt;

use crate::{
    array::VolumeMetaData,
    data::{BrickPosition, LocalVoxelPosition, VoxelPosition},
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
            chunk_size: brick_size,
        };
        let mut handles = Vec::with_capacity(positions.len());
        for pos in positions {
            let num_voxels = crate::data::hmul(m.chunk_size);

            let brick_handle = ctx.alloc_slot(pos, num_voxels)?;
            handles.push((pos, brick_handle));
        }

        {
            let handle_data = handles.iter_mut().map(|(pos, h)| (pos, &mut **h));

            let iter = handle_data.into_iter().map(|(pos, brick_data)| {
                ctx.spawn_io(move || {
                    let in_: &[f32] = bytemuck::cast_slice(&self.mmap[..]);
                    let in_ = ndarray::ArrayView3::from_shape(
                        crate::data::stride_shape(m.dimensions, m.dimensions),
                        in_,
                    )
                    .unwrap();

                    brick_data.iter_mut().for_each(|v| {
                        v.write(f32::NAN);
                    });

                    let chunk_info = m.chunk_info(*pos);
                    //if !(begin.x() < m.dimensions.x()
                    //    && begin.y() < m.dimensions.y()
                    //    && begin.z() < m.dimensions.z())
                    //{
                    //    return Err("Brick position is outside of volume".into());
                    //}

                    let mut out_chunk = crate::data::chunk_mut(brick_data, &chunk_info);
                    let begin = chunk_info.begin();
                    let end = chunk_info.end();
                    let in_chunk = in_.slice(ndarray::s!(
                        begin.z().raw as usize..end.z().raw as usize,
                        begin.y().raw as usize..end.y().raw as usize,
                        begin.x().raw as usize..end.x().raw as usize,
                    ));

                    ndarray::azip!((o in &mut out_chunk, i in &in_chunk) { o.write(*i); });
                })
            });
            let mut stream = ctx.submit_unordered(iter);
            while let Some(_) = stream.next().await {}
        }

        for (_, handle) in handles {
            // Safety: At this point the thread pool job above has finished and has initialized all bytes
            // in the brick.
            unsafe { handle.initialized() };
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
