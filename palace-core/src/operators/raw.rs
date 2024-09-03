use std::{fs::File, mem::MaybeUninit, ops::DerefMut, path::PathBuf, rc::Rc};

use ash::vk;
use derive_more::Deref;
use futures::StreamExt;

use crate::{
    array::{ChunkIndex, VolumeMetaData},
    data::{BrickPosition, LocalVoxelPosition, VoxelPosition},
    dim::D3,
    dtypes::{DType, ElementType},
    operator::{DataDescriptor, OperatorDescriptor},
    operators::tensor::TensorOperator,
    storage::DataLocation,
    task::{RequestStream, TaskContext},
    util::Map,
    vec::Vector,
    Error,
};

use super::volume::VolumeOperator;

#[derive(Clone, Deref)]
pub struct RawVolumeSourceState(Rc<RawVolumeSourceStateInner>);

pub struct RawVolumeSourceStateInner {
    pub path: PathBuf,
    _file: File,
    mmap: memmap::Mmap,
    pub size: VoxelPosition,
}

fn copy_chunk_line<S: DerefMut<Target = [MaybeUninit<u8>]>>(
    dtype: DType,
    m: VolumeMetaData,
    in_: ndarray::ArrayView4<u8>,
    chunks_in_line: &mut [(BrickPosition, S)],
) {
    let elm_size = dtype.element_layout().size();
    assert_eq!(elm_size, in_.shape()[3]);
    let brick_size = m.chunk_size.push_dim_small(elm_size.try_into().unwrap());

    //dbg!(chunks_in_line.len());

    for (pos, ref mut buf) in &mut *chunks_in_line {
        let chunk_info = m.chunk_info_vec(pos);
        crate::data::init_non_full_raw(buf.as_mut(), &chunk_info, 0xff);
    }

    let first = chunks_in_line.first().unwrap();
    let first_info = m.chunk_info_vec(&first.0);
    let global_begin = first_info.begin;

    let last = chunks_in_line.last().unwrap();
    let last_info = m.chunk_info_vec(&last.0);
    let global_end = last_info.end();

    let strip_size_z = first_info.logical_dimensions.z();
    let strip_size_y = first_info.logical_dimensions.y();
    for z in 0..strip_size_z.raw {
        for y in 0..strip_size_y.raw {
            // Note: This assumes that all bricks have the same memory size! This may
            // change in the future
            let line_begin_brick = crate::data::to_linear(&Vector::from([z, y, 0, 0]), &brick_size);
            let line_size = brick_size[2].raw as usize * elm_size;

            let global_line = in_.slice(ndarray::s!(
                (global_begin.z().raw + z) as usize,
                (global_begin.y().raw + y) as usize,
                global_begin.x().raw as usize..global_end.x().raw as usize,
                ..,
            ));
            let global_line = global_line.as_slice().unwrap();

            let bricks = chunks_in_line
                .iter_mut()
                .map(|(_, handle)| &mut handle.as_mut()[line_begin_brick..][..line_size]);
            let mut global_brick_begin = 0;
            for brick_line in bricks {
                let global_line_brick = &global_line[global_brick_begin..];
                let actual_line_line = line_size.min(global_line_brick.len());
                let global_line_brick = &global_line_brick[..actual_line_line];
                crate::data::write_slice_uninit(
                    &mut brick_line[..actual_line_line],
                    global_line_brick,
                );
                global_brick_begin += line_size;
            }
        }
    }
}

pub fn open(
    path: PathBuf,
    metadata: VolumeMetaData,
    dtype: DType,
) -> Result<VolumeOperator<DType>, Error> {
    let file = File::open(&path)?;
    let mmap = unsafe { memmap::Mmap::map(&file)? };

    let size = metadata.dimensions;
    let byte_size = dtype.array_layout(size.hmul()).size();
    assert_eq!(file.metadata()?.len(), byte_size as u64);

    let state = RawVolumeSourceState(Rc::new(RawVolumeSourceStateInner {
        path,
        _file: file,
        mmap,
        size,
    }));

    let vol = TensorOperator::with_state(
        OperatorDescriptor::new("raw_volume::open")
            .dependent_on_data(state.path.to_string_lossy().as_bytes()),
        dtype,
        metadata,
        (state, metadata),
        move |ctx, positions, (state, metadata)| {
            async move {
                state
                    .load_raw_bricks(dtype, metadata.chunk_size, ctx, positions)
                    .await
            }
            .into()
        },
    );

    Ok(vol)
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
        dtype: DType,
        brick_size: LocalVoxelPosition,
        ctx: TaskContext<'cref, 'inv, DType>,
        mut positions: Vec<(ChunkIndex, DataLocation)>,
    ) -> Result<(), Error> {
        let m = VolumeMetaData {
            dimensions: self.0.size,
            chunk_size: brick_size,
        };
        let dim_in_bricks = m.dimension_in_chunks();

        positions.sort_by_key(|(v, _)| v.0);

        let max_lin_len = 4096; //expected page size

        let chunk_mem_size_x = dtype.array_layout(m.chunk_size.x().raw as usize).size();

        let mut batches_cpu = Vec::new();
        let mut batches_gpus: Map<usize, Vec<Vec<Vector<D3, crate::coordinate::ChunkCoordinate>>>> =
            Default::default();
        let mut current_batch: Vec<BrickPosition> = Vec::new();
        let mut current_pos = 0;
        let mut current_loc = None;
        for (pos, loc) in positions {
            let pos = m.chunk_pos_from_index(pos);
            if !(pos.x() < dim_in_bricks.x()
                && pos.y() < dim_in_bricks.y()
                && pos.z() < dim_in_bricks.z())
            {
                return Err(format!("Brick position {:?} is outside of volume", pos).into());
            }

            current_pos += chunk_mem_size_x;
            let start_new_line = if current_pos < max_lin_len {
                if let Some(end) = current_batch.last() {
                    let mut next = *end;
                    next[2] = next[2] + 1u32;
                    let adjacent = next == pos && current_loc == Some(loc);

                    !adjacent
                } else {
                    // We have just started a new line anyway
                    false
                }
            } else {
                // Line too long
                true
            };

            if start_new_line {
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
                current_pos = 0;
            }

            current_loc = Some(loc);
            current_batch.push(pos);
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

        let element_layout = dtype.element_layout();
        let in_: &[u8] = &self.0.mmap[..];
        assert_eq!(
            in_.as_ptr() as usize % element_layout.align(),
            0,
            "Slice must be aligned for type"
        );
        let in_ = ndarray::ArrayView4::from_shape(
            crate::data::contiguous_shape(
                &m.dimensions
                    .push_dim_small(element_layout.size().try_into().unwrap()),
            ),
            in_,
        )
        .unwrap();

        {
            let requests = batches_cpu.into_iter().map(|positions| {
                let num_voxels = m.chunk_size.hmul();

                let brick_handles = positions.iter().map(|pos| {
                    let data_id =
                        DataDescriptor::new(ctx.current_op_desc().unwrap(), m.chunk_index(pos));
                    ctx.alloc_raw(data_id, dtype.array_layout(num_voxels))
                });

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
                        copy_chunk_line(dtype, m, in_, &mut brick_handles);

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
                    .map(|pos| ctx.alloc_slot_gpu(device, m.chunk_index(pos), num_voxels));

                (ctx.group(brick_handles), positions)
            });
            let stream = ctx
                .submit_unordered_with_data(requests)
                .then_req_with_data(*ctx, |(brick_handles, positions)| {
                    let num_voxels = m.chunk_size.hmul();
                    let layout = dtype.array_layout(num_voxels);

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
                            let ptr = buf.mapped_ptr().unwrap().cast::<MaybeUninit<u8>>().as_ptr();
                            let slice =
                                unsafe { std::slice::from_raw_parts_mut(ptr, buf.size as usize) };
                            (*pos, slice)
                        })
                        .collect::<Vec<_>>();

                    ctx.spawn_io(move || {
                        copy_chunk_line(dtype, m, in_, &mut staging_bufs_cpu);

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
}
