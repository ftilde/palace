use std::{fs::File, mem::MaybeUninit, path::PathBuf, rc::Rc};

use ash::vk;
use derive_more::Deref;
use futures::StreamExt;

use crate::{
    array::VolumeMetaData,
    data::{BrickPosition, LocalVoxelPosition, VoxelPosition},
    dim::D3,
    operator::{Operator, OperatorDescriptor},
    storage::DataLocation,
    task::{RequestStream, TaskContext},
    util::Map,
    vec::Vector,
    Error,
};

#[derive(Clone, Deref)]
pub struct RawVolumeSourceState(Rc<RawVolumeSourceStateInner>);

pub struct RawVolumeSourceStateInner {
    pub path: PathBuf,
    _file: File,
    mmap: memmap::Mmap,
    pub size: VoxelPosition,
}

impl RawVolumeSourceState {
    pub fn open(path: PathBuf, size: VoxelPosition) -> Result<Self, Error> {
        let file = File::open(&path)?;
        let mmap = unsafe { memmap::Mmap::map(&file)? };

        let byte_size = size.hmul() * std::mem::size_of::<f32>();
        assert_eq!(file.metadata()?.len(), byte_size as u64);

        Ok(Self(Rc::new(RawVolumeSourceStateInner {
            path,
            _file: file,
            mmap,
            size,
        })))
    }
    pub async fn load_raw_bricks<'cref, 'inv>(
        &self,
        brick_size: LocalVoxelPosition,
        ctx: TaskContext<'cref, 'inv, BrickPosition, f32>,
        mut positions: Vec<(BrickPosition, DataLocation)>,
    ) -> Result<(), Error> {
        let m = VolumeMetaData {
            dimensions: self.0.size,
            chunk_size: brick_size,
        };
        let dim_in_bricks = m.dimension_in_chunks();

        positions.sort_by_key(|(v, _)| crate::data::to_linear(*v, dim_in_bricks));

        let max_lin_len = 4096; //expected page size

        let chunk_mem_size_x = m.chunk_size.x().raw as usize * std::mem::size_of::<f32>();

        let mut batches_cpu = Vec::new();
        let mut batches_gpus: Map<usize, Vec<Vec<Vector<D3, crate::coordinate::ChunkCoordinate>>>> =
            Default::default();
        let mut current_batch: Vec<BrickPosition> = Vec::new();
        let mut current_pos = 0;
        let mut current_loc = None;
        for (pos, loc) in positions {
            if !(pos.x() < dim_in_bricks.x()
                && pos.y() < dim_in_bricks.y()
                && pos.z() < dim_in_bricks.z())
            {
                return Err(format!("Brick position {:?} is outside of volume", pos).into());
            }

            if current_loc.is_none() {
                current_loc = Some(loc);
            }

            current_pos += chunk_mem_size_x;
            if current_pos < max_lin_len {
                if let Some(end) = current_batch.last() {
                    let mut next = *end;
                    next[2] = next[2] + 1u32;
                    if next == pos && current_loc == Some(loc) {
                        current_batch.push(pos);
                        continue;
                    }
                } else {
                    current_batch.push(pos);
                    continue;
                }
            }
            let finished_batch = std::mem::take(&mut current_batch);
            assert!(!finished_batch.is_empty());
            match loc {
                DataLocation::CPU(_) => {
                    batches_cpu.push(finished_batch);
                }
                DataLocation::GPU(i) => {
                    batches_gpus.entry(i).or_default().push(finished_batch);
                }
            }
            current_batch.push(pos);
            current_pos = 0;
            current_loc = Some(loc);
        }
        if !current_batch.is_empty() {
            match current_loc.unwrap() {
                DataLocation::CPU(_) => {
                    batches_cpu.push(current_batch);
                }
                DataLocation::GPU(i) => {
                    batches_gpus.entry(i).or_default().push(current_batch);
                }
            }
        }

        let in_: &[f32] = bytemuck::cast_slice(&self.0.mmap[..]);
        let in_ = ndarray::ArrayView3::from_shape(crate::data::contiguous_shape(m.dimensions), in_)
            .unwrap();

        {
            let requests = batches_cpu.into_iter().map(|positions| {
                let num_voxels = m.chunk_size.hmul();

                let brick_handles = positions.iter().map(|pos| ctx.alloc_slot(*pos, num_voxels));

                (ctx.group(brick_handles), positions)
            });
            let stream = ctx.submit_unordered_with_data(requests).then_req(
                *ctx,
                |(brick_handles, positions)| {
                    let mut brick_handles = brick_handles
                        .into_iter()
                        .zip(positions)
                        .map(|(h, pos)| (pos, h.into_thread_handle()))
                        .collect::<Vec<_>>();

                    ctx.spawn_io(move || {
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
                                let line_begin_brick = crate::data::to_linear(
                                    LocalVoxelPosition::from([z, y, 0]),
                                    brick_size,
                                );

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
                                    for (o, i) in
                                        brick_line.iter_mut().zip(global_line_brick.iter())
                                    {
                                        o.write(*i);
                                    }
                                    global_brick_begin += brick_size.x().raw as usize;
                                }
                            }
                        }

                        brick_handles
                    })
                },
            );

            futures::pin_mut!(stream);
            while let Some(handles) = stream.next().await {
                for (_, handle) in handles {
                    let handle = handle.into_main_handle(ctx.storage());
                    unsafe { handle.initialized(*ctx) };
                }
            }
        }
        for (id, batches) in batches_gpus {
            let device = &ctx.device_contexts[id];
            let requests = batches.into_iter().map(|positions| {
                let num_voxels = m.chunk_size.hmul();

                let brick_handles = positions
                    .iter()
                    .map(|pos| ctx.alloc_slot_gpu(device, *pos, num_voxels));

                (ctx.group(brick_handles), positions)
            });
            let stream = ctx
                .submit_unordered_with_data(requests)
                .then_req_with_data(*ctx, |(brick_handles, positions)| {
                    let num_voxels = m.chunk_size.hmul();
                    let layout = std::alloc::Layout::array::<f32>(num_voxels).unwrap();

                    let staging_bufs =
                        (0..positions.len()).map(|_| device.staging_to_gpu.request(device, layout));

                    (ctx.group(staging_bufs), (brick_handles, positions))
                })
                .then_req(*ctx, |(staging_bufs, (brick_handles, positions))| {
                    let brick_handles = brick_handles
                        .into_iter()
                        .map(|h| h.into_thread_handle())
                        .collect::<Vec<_>>();
                    let mut staging_bufs_cpu = staging_bufs
                        .iter()
                        .zip(positions.iter())
                        .map(|(buf, pos)| {
                            let float_ptr = buf.mapped_ptr().unwrap().cast::<MaybeUninit<f32>>();
                            let slice = unsafe {
                                std::slice::from_raw_parts_mut(
                                    float_ptr.as_ptr(),
                                    buf.size as usize / std::mem::size_of::<MaybeUninit<f32>>(),
                                )
                            };
                            (*pos, slice)
                        })
                        .collect::<Vec<_>>();

                    ctx.spawn_io(move || {
                        for (pos, ref mut buf) in &mut staging_bufs_cpu {
                            let chunk_info = m.chunk_info(*pos);
                            crate::data::init_non_full(buf, &chunk_info, f32::NAN);
                        }

                        let first = staging_bufs_cpu.first().unwrap();
                        let first_info = m.chunk_info(first.0);
                        let global_begin = first_info.begin;

                        let last = staging_bufs_cpu.last().unwrap();
                        let last_info = m.chunk_info(last.0);
                        let global_end = last_info.end();

                        let strip_size_z = first_info.logical_dimensions.z();
                        let strip_size_y = first_info.logical_dimensions.y();
                        for z in 0..strip_size_z.raw {
                            for y in 0..strip_size_y.raw {
                                // Note: This assumes that all bricks have the same memory size! This may
                                // change in the future
                                let line_begin_brick = crate::data::to_linear(
                                    LocalVoxelPosition::from([z, y, 0]),
                                    brick_size,
                                );

                                let global_line = in_.slice(ndarray::s!(
                                    (global_begin.z().raw + z) as usize,
                                    (global_begin.y().raw + y) as usize,
                                    global_begin.x().raw as usize..global_end.x().raw as usize,
                                ));
                                let global_line = global_line.as_slice().unwrap();

                                let bricks = staging_bufs_cpu
                                    .iter_mut()
                                    .map(|(_, handle)| &mut handle[line_begin_brick..]);
                                let mut global_brick_begin = 0;
                                for brick_line in bricks {
                                    let global_line_brick = &global_line[global_brick_begin..];
                                    for (o, i) in
                                        brick_line.iter_mut().zip(global_line_brick.iter())
                                    {
                                        o.write(*i);
                                    }
                                    global_brick_begin += brick_size.x().raw as usize;
                                }
                            }
                        }

                        std::mem::drop(staging_bufs_cpu);
                        (staging_bufs, brick_handles)
                    })
                });

            futures::pin_mut!(stream);
            while let Some((staging_bufs, brick_handles)) = stream.next().await {
                for (staging_buf, brick_handle) in
                    staging_bufs.into_iter().zip(brick_handles.into_iter())
                {
                    let handle = brick_handle.into_main_handle(device);
                    device.with_cmd_buffer(|cmd| {
                        let copy_info = vk::BufferCopy::builder().size(handle.size as _);
                        unsafe {
                            device.functions().cmd_copy_buffer(
                                cmd.raw(),
                                staging_buf.buffer,
                                handle.buffer,
                                &[*copy_info],
                            );
                        }
                    });

                    unsafe {
                        handle.initialized(
                            *ctx,
                            crate::vulkan::SrcBarrierInfo {
                                stage: vk::PipelineStageFlags2::TRANSFER,
                                access: vk::AccessFlags2::TRANSFER_WRITE,
                            },
                        )
                    };

                    unsafe { device.staging_to_gpu.return_buf(device, staging_buf) };
                }
            }
        }

        Ok(())
    }

    #[allow(unused)] // We probably will use it directly at some point
    pub fn operate(&self, brick_size: LocalVoxelPosition) -> Operator<BrickPosition, f32> {
        Operator::with_state(
            OperatorDescriptor::new("RawVolumeSourceState::operate")
                .dependent_on_data(self.0.path.to_string_lossy().as_bytes()),
            self.clone(),
            move |ctx, positions, this| {
                async move { this.load_raw_bricks(brick_size, ctx, positions).await }.into()
            },
        )
    }
}
