use futures::stream::StreamExt;

use crate::{
    data::{Brick, BrickPosition, VolumeMetaData},
    operator::{Operator, OperatorId},
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

                for pos in positions {
                    match ctx
                        .submit(input.bricks.request_inplace(pos, ctx.current_op()))
                        .await
                    {
                        Ok(mut rw) => {
                            let rw = &mut *rw;
                            ctx.submit(ctx.spawn_compute(|| {
                                for v in rw.iter_mut() {
                                    *v = factor * *v + offset;
                                }
                            }))
                            .await;
                        }
                        Err((r, w)) => {
                            let mut w = w?;
                            let r = &*r;
                            let w_ref = &mut *w;
                            ctx.submit(ctx.spawn_compute(|| {
                                for (i, o) in r.iter().zip(w_ref.iter_mut()) {
                                    o.write(factor * *i + offset);
                                }
                            }))
                            .await;
                            unsafe { w.initialized() };
                        }
                    }
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

                // First query all input bricks
                let mut stream = ctx.submit_unordered_with_data(
                    vol.brick_positions()
                        .map(|pos| (input.bricks.request(pos), pos)),
                );

                let mut tasks = Vec::new();
                while let Some((brick_data, brick_pos)) = stream.next().await {
                    // Second, when any brick arrives, create a compute future...
                    let mut task = Box::pin(async move {
                        let brick =
                            Brick::new(&*brick_data, vol.brick_dim(brick_pos), vol.brick_size);
                        ctx.submit(ctx.spawn_compute(|| {
                            let voxels = brick.voxels().collect::<Vec<_>>();
                            voxels.iter().sum::<f32>()
                        }))
                        .await
                    });
                    // ... which is immediately polled once to submit the compute task onto the
                    // pool.
                    match futures::poll!(task.as_mut()) {
                        std::task::Poll::Ready(_) => panic!("Future cannot be ready since it has just submitted it's task to the runtime."),
                        std::task::Poll::Pending => {},
                    }
                    tasks.push(task);
                }
                // Third, collect the brick results into a global sum
                let mut sum = 0.0;
                for task in tasks {
                    sum += task.await;
                }

                let v = sum / vol.num_voxels() as f32;

                ctx.write(v)
            }
            .into()
        },
    )
}
