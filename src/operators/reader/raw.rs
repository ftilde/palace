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
        mut positions: Vec<BrickPosition>,
    ) -> Result<(), Error> {
        let m = VolumeMetaData {
            dimensions: self.size,
            chunk_size: brick_size,
        };
        let dim_in_bricks = m.dimension_in_bricks();

        positions.sort_by_key(|v| crate::data::to_linear(*v, dim_in_bricks));

        let max_lin_len = 4096; //expected page size

        let chunk_mem_size_x = m.chunk_size.x().raw as usize * std::mem::size_of::<f32>();

        let mut batches = Vec::new();
        let mut current_batch: Vec<BrickPosition> = Vec::new();
        let mut current_pos = 0;
        for pos in positions {
            if !(pos.x() < dim_in_bricks.x()
                && pos.y() < dim_in_bricks.y()
                && pos.z() < dim_in_bricks.z())
            {
                return Err(format!("Brick position {:?} is outside of volume", pos).into());
            }

            current_pos += chunk_mem_size_x;
            if current_pos < max_lin_len {
                if let Some(end) = current_batch.last() {
                    let mut next = *end;
                    next.0[2] = next.0[2] + 1u32;
                    if next == pos {
                        current_batch.push(pos);
                        current_pos = 0;
                        continue;
                    }
                }
            }
            batches.push(std::mem::take(&mut current_batch));
            current_batch.push(pos);
        }
        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        let in_: &[f32] = bytemuck::cast_slice(&self.mmap[..]);
        let in_ = ndarray::ArrayView3::from_shape(crate::data::contiguous_shape(m.dimensions), in_)
            .unwrap();
        let work = batches.into_iter().map(|positions| {
            let num_voxels = crate::data::hmul(m.chunk_size);

            let mut brick_handles = positions
                .into_iter()
                .map(|pos| {
                    (
                        pos,
                        ctx.alloc_slot(pos, num_voxels)
                            .unwrap()
                            .into_thread_handle(),
                    )
                })
                .collect::<Vec<_>>();
            ctx.spawn_io(move || {
                if brick_handles.is_empty() {
                    return brick_handles;
                }
                for (pos, ref mut brick_handle) in &mut brick_handles {
                    let chunk_info = m.chunk_info(*pos);
                    crate::data::init_non_full(brick_handle, &chunk_info, f32::NAN);
                }

                let first = brick_handles.first().unwrap();
                let first_info = m.chunk_info(first.0);
                let global_begin = first_info.begin;

                let last = brick_handles.last().unwrap();
                let last_info = m.chunk_info(last.0);
                let global_end = last_info.end();

                let strip_size_z = first_info.logical_dimensions.z();
                let strip_size_y = first_info.logical_dimensions.y();
                for z in 0..strip_size_z.raw {
                    for y in 0..strip_size_y.raw {
                        // Note: This assumes that all bricks have the same memory size! This may
                        // change in the future
                        let line_begin_brick =
                            crate::data::to_linear(LocalVoxelPosition::from([z, y, 0]), brick_size);

                        let global_line = in_.slice(ndarray::s!(
                            (global_begin.z().raw + z) as usize,
                            (global_begin.y().raw + y) as usize,
                            global_begin.x().raw as usize..global_end.x().raw as usize,
                        ));
                        let global_line = global_line.as_slice().unwrap();

                        let bricks = brick_handles
                            .iter_mut()
                            .map(|(_, handle)| &mut handle[line_begin_brick..]);
                        let mut global_brick_begin = 0;
                        for brick_line in bricks {
                            let global_line_brick = &global_line[global_brick_begin..];
                            for (o, i) in brick_line.iter_mut().zip(global_line_brick.iter()) {
                                o.write(*i);
                            }
                            global_brick_begin += brick_size.x().raw as usize;
                        }
                    }
                }

                brick_handles
            })
        });

        let stream = ctx.submit_unordered(work);

        futures::pin_mut!(stream);
        while let Some(handles) = stream.next().await {
            for (_, handle) in handles {
                let handle = handle.into_main_handle(ctx.storage());
                unsafe { handle.initialized() };
            }
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
