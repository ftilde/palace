use futures::stream::StreamExt;

use crate::{
    array::VolumeMetaData,
    data::{slice_range, BrickPosition, LocalVoxelCoordinate, LocalVoxelPosition},
    operator::{Operator, OperatorId},
    storage::{InplaceResult, ThreadInplaceResult},
    task::{Task, TaskContext},
};

use super::ScalarOperator;

pub trait VolumeOperatorState {
    fn operate<'a>(&'a self) -> VolumeOperator<'a>;
}

pub struct VolumeOperator<'op> {
    pub metadata: Operator<'op, (), VolumeMetaData>,
    pub bricks: Operator<'op, BrickPosition, f32>,
}

impl<'op> VolumeOperator<'op> {
    pub fn new<
        M: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, (), VolumeMetaData>,
                crate::operator::OutlivesMarker<'op, 'inv>,
            ) -> Task<'cref>
            + 'op,
        B: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, BrickPosition, f32>,
                Vec<BrickPosition>,
                crate::operator::OutlivesMarker<'op, 'inv>,
            ) -> Task<'cref>
            + 'op,
    >(
        base_id: OperatorId,
        metadata: M,
        bricks: B,
    ) -> Self {
        Self {
            metadata: crate::operators::scalar(base_id.slot(0), metadata),
            bricks: Operator::new(base_id.slot(1), bricks),
        }
    }
}

pub fn linear_rescale<'op>(
    input: &'op VolumeOperator<'_>,
    factor: &'op ScalarOperator<'_, f32>,
    offset: &'op ScalarOperator<'_, f32>,
) -> VolumeOperator<'op> {
    VolumeOperator::new(
        OperatorId::new("volume_scale")
            .dependent_on(&input.metadata)
            .dependent_on(&input.bricks)
            .dependent_on(factor)
            .dependent_on(offset),
        move |ctx, _| {
            async move {
                let req = input.metadata.request_scalar();
                let m = ctx.submit(req).await;
                ctx.write(m)
            }
            .into()
        },
        move |ctx, positions, _| {
            async move {
                let (factor, offset) = futures::join! {
                    ctx.submit(factor.request_scalar()),
                    ctx.submit(offset.request_scalar()),
                };

                let requests = positions
                    .into_iter()
                    .map(|pos| (input.bricks.request_inplace(pos, ctx.current_op())));

                let stream = ctx.submit_unordered(requests).then(|brick_handle| {
                    let mut brick_handle = brick_handle.unwrap().into_thread_handle();
                    ctx.submit(ctx.spawn_compute(move || {
                        match &mut brick_handle {
                            ThreadInplaceResult::Inplace(ref mut rw) => {
                                for v in rw.iter_mut() {
                                    *v = factor * *v + offset;
                                }
                            }
                            ThreadInplaceResult::New(r, ref mut w) => {
                                for (i, o) in r.iter().zip(w.iter_mut()) {
                                    o.write(factor * *i + offset);
                                }
                            }
                        }
                        brick_handle
                    }))
                });

                futures::pin_mut!(stream);
                // Drive the stream until completion
                while let Some(brick_handle) = stream.next().await {
                    let brick_handle = brick_handle.into_main_handle(ctx.storage());
                    if let InplaceResult::New(_, w) = brick_handle {
                        // Safety: We have written all values in the above closure executed on
                        // the thread pool.
                        unsafe { w.initialized() };
                    };
                }

                Ok(())
            }
            .into()
        },
    )
}

pub fn mean<'op>(input: &'op VolumeOperator<'_>) -> ScalarOperator<'op, f32> {
    crate::operators::scalar(
        OperatorId::new("volume_mean")
            .dependent_on(&input.metadata)
            .dependent_on(&input.bricks),
        move |ctx, _| {
            async move {
                let vol = ctx.submit(input.metadata.request_scalar()).await;

                let to_request = vol.brick_positions().collect::<Vec<_>>();
                let batch_size = 1024;

                let mut sum = 0.0;
                for chunk in to_request.chunks(batch_size) {
                    let mut stream = ctx.submit_unordered_with_data(chunk.iter().map(|pos|
                            (input.bricks.request(*pos), *pos)));

                    let mut tasks = Vec::new();
                    while let Some((brick_handle, brick_pos)) = stream.next().await {
                        // Second, when any brick arrives, create a compute future...
                        let chunk_info = vol.chunk_info(brick_pos);
                        let brick_handle = brick_handle.into_thread_handle();
                        let mut task = Box::pin(
                            ctx.submit(ctx.spawn_compute(move || {
                                let sum = if chunk_info.is_full() {
                                    brick_handle.iter().sum::<f32>()
                                } else {
                                    let brick = crate::data::chunk(&brick_handle, &chunk_info);
                                    brick.iter().sum::<f32>()
                                };
                                (brick_handle, sum)
                            }))
                        );
                        // ... which is immediately polled once to submit the compute task onto the
                        // pool.
                        match futures::poll!(task.as_mut()) {
                            std::task::Poll::Ready(_) => panic!("Future cannot be ready since it has just submitted it's task to the runtime."),
                            std::task::Poll::Pending => {},
                        }
                        tasks.push(task);
                    }
                    // Third, collect the brick results into a global sum
                    for task in tasks {
                        let (handle, part_sum) = task.await;
                        sum += part_sum;
                        handle.into_main_handle(ctx.storage());
                    }
                }

                let v = sum / vol.num_elements() as f32;

                ctx.write(v)
            }
            .into()
        },
    )
}

pub fn rechunk<'op>(
    input: &'op VolumeOperator<'_>,
    brick_size: LocalVoxelPosition,
) -> VolumeOperator<'op> {
    VolumeOperator::new(
        OperatorId::new("volume_rechunk")
            .dependent_on(&input.metadata)
            .dependent_on(&input.bricks),
        move |ctx, _| {
            async move {
                let req = input.metadata.request_scalar();
                let mut m = ctx.submit(req).await;
                m.chunk_size = brick_size;
                ctx.write(m)
            }
            .into()
        },
        move |ctx, positions, _| {
            // TODO: optimize case where input.brick_size == output.brick_size
            async move {
                let m_in = ctx.submit(input.metadata.request_scalar()).await;
                let m_out = {
                    let mut m_out = m_in;
                    m_out.chunk_size = brick_size;
                    m_out
                };
                for pos in positions {
                    let out_info = m_out.chunk_info(pos);
                    let out_begin = out_info.begin();
                    let out_end = out_info.end();

                    let in_begin_brick = m_in.chunk_pos(out_begin);
                    let in_end_brick = m_in.chunk_pos(out_end.map(|v| v - 1.into()));

                    let mut brick_handle = ctx.alloc_slot(pos, out_info.mem_elements())?;
                    let out_data = &mut *brick_handle;
                    crate::data::init_non_full(out_data, &out_info, f32::NAN);
                    let mut out_chunk = crate::data::chunk_mut(out_data, &out_info);

                    let mut stream = ctx.submit_unordered_with_data(
                        itertools::iproduct! {
                            in_begin_brick.z().raw..=in_end_brick.z().raw,
                            in_begin_brick.y().raw..=in_end_brick.y().raw,
                            in_begin_brick.x().raw..=in_end_brick.x().raw
                        }
                        .map(|(z, y, x)| BrickPosition::from([z, y, x]))
                        .map(|pos| (input.bricks.request(pos), pos)),
                    );
                    while let Some((in_data_handle, in_brick_pos)) = stream.next().await {
                        let in_data = &*in_data_handle;
                        let in_info = m_in.chunk_info(in_brick_pos);

                        let in_chunk = crate::data::chunk(in_data, &in_info);
                        ctx.submit(ctx.spawn_compute(|| {
                            let in_begin = in_info.begin();
                            let in_end = in_info.end();

                            let overlap_begin = in_begin.zip(out_begin, |i, o| i.max(o));
                            let overlap_end = in_end.zip(out_end, |i, o| i.min(o));
                            let overlap_size = (overlap_end - overlap_begin)
                                .map(LocalVoxelCoordinate::interpret_as);

                            let in_chunk_begin = in_info.in_chunk(overlap_begin);
                            let in_chunk_end = in_chunk_begin + overlap_size;

                            let out_chunk_begin = out_info.in_chunk(overlap_begin);
                            let out_chunk_end = out_chunk_begin + overlap_size;

                            let mut o =
                                out_chunk.slice_mut(slice_range(out_chunk_begin, out_chunk_end));
                            let i = in_chunk.slice(slice_range(in_chunk_begin, in_chunk_end));

                            ndarray::azip!((o in &mut o, i in &i) { o.write(*i); });
                        }))
                        .await;
                    }

                    // Safety: We have queried and then copied the data of all overlapping input
                    // bricks.
                    unsafe { brick_handle.initialized() };
                }
                Ok(())
            }
            .into()
        },
    )
}
