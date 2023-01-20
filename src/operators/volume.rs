use futures::stream::StreamExt;

use crate::{
    data::{Brick, BrickPosition, VolumeMetaData},
    operator::{ComputeFunction, Operator, OperatorId},
    task::Task,
};

use super::{ScalarOperator, ScalarTaskContext};

//#[derive(Copy, Clone)]
//pub struct VolumeTaskContext<'op, 'op> {
//    inner: TaskContext<'op, 'op>,
//    current_op_id: OperatorId,
//}
//
//impl<'op, 'op> std::ops::Deref for VolumeTaskContext<'op, 'op> {
//    type Target = TaskContext<'op, 'op>;
//
//    fn deref(&self) -> &Self::Target {
//        &self.inner
//    }
//}
//
//impl<'op, 'op> VolumeTaskContext<'op, 'op> {
//    pub fn write_metadata(&self, metadata: VolumeMetaData) -> Result<(), Error> {
//        let id = TaskId::new(self.current_op_id, &DatumRequest::Value);
//        self.inner.storage.write_to_ram(id, metadata)
//    }
//
//    pub fn alloc_brick<'a>(
//        &'a self,
//        pos: BrickPosition,
//        num_voxels: usize,
//    ) -> Result<WriteHandleUninit<'a, [MaybeUninit<f32>]>, Error> {
//        let id = TaskId::new(self.current_op_id, &DatumRequest::Brick(pos));
//        self.inner.storage.alloc_ram_slot_slice(id, num_voxels)
//    }
//}

//pub fn request_metadata<'req, 'op: 'req, 'op: 'op>(
//    vol: &'op dyn VolumeOperator,
//) -> Request<'req, 'op, ReadHandle<'req, VolumeMetaData>> {
//    let op_id = vol.id();
//    let id = TaskId::new(op_id, &DatumRequest::Value); //TODO: revisit
//    Request {
//        id,
//        type_: RequestType::Data(Box::new(move |ctx| {
//            let ctx = VolumeTaskContext {
//                inner: ctx,
//                current_op_id: op_id,
//            };
//            vol.compute_metadata(ctx)
//        })),
//        poll: Box::new(move |ctx| unsafe { ctx.storage.read_ram(id) }),
//        _marker: Default::default(),
//    }
//}
//
//pub fn request_brick<'req, 'op: 'req, 'op: 'op>(
//    vol: &'op dyn VolumeOperator,
//    pos: BrickPosition,
//) -> Request<'req, 'op, ReadHandle<'req, [f32]>> {
//    let req = DatumRequest::Brick(pos);
//    let op_id = vol.id();
//    let id = TaskId::new(op_id, &req); //TODO: revisit
//    Request {
//        id,
//        type_: RequestType::Data(Box::new(move |ctx| {
//            let ctx = VolumeTaskContext {
//                inner: ctx,
//                current_op_id: op_id,
//            };
//            vol.compute_brick(ctx, pos)
//        })),
//        poll: Box::new(move |ctx| unsafe { ctx.storage.read_ram_slice(id) }),
//        _marker: Default::default(),
//    }
//}
//
//pub fn request_inplace_rw_brick<'req, 'op: 'req, 'op: 'op>(
//    read_vol: &'op dyn VolumeOperator,
//    pos: BrickPosition,
//    write_vol: &'op dyn VolumeOperator,
//) -> Request<'req, 'op, InplaceResultSlice<'req, f32>> {
//    let req = DatumRequest::Brick(pos);
//    let op_id = read_vol.id();
//    let read_id = TaskId::new(op_id, &req); //TODO: revisit
//    let write_id = TaskId::new(write_vol.id(), &req);
//    Request {
//        id: read_id,
//        type_: RequestType::Data(Box::new(move |ctx| {
//            let ctx = VolumeTaskContext {
//                inner: ctx,
//                current_op_id: op_id,
//            };
//            read_vol.compute_brick(ctx, pos)
//        })),
//        poll: Box::new(move |ctx| unsafe {
//            ctx.storage.try_update_inplace_slice(read_id, write_id)
//        }),
//        _marker: Default::default(),
//    }
//}

pub struct VolumeOperator<'op> {
    pub metadata: Operator<'op, (), VolumeMetaData>,
    pub bricks: Operator<'op, BrickPosition, f32>,
}

impl<'op> VolumeOperator<'op> {
    pub fn new<
        F: for<'tasks> Fn(
                ScalarTaskContext<'tasks, VolumeMetaData>,
                crate::operator::OutlivesMarker<'op, 'tasks>,
            ) -> Task<'tasks>
            + 'op,
    >(
        base_id: OperatorId,
        metadata: F,
        bricks: ComputeFunction<'op, BrickPosition>,
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
                let req = input.metadata.request(());
                let m = ctx.submit(req).await;
                ctx.write(m[0])
            }
            .into()
        },
        Box::new(move |ctx, positions, _| {
            async move {
                let (factor, offset) = futures::join! {
                    ctx.submit(factor.request(())),
                    ctx.submit(offset.request(())),
                };
                let factor = factor[0];
                let offset = offset[0];

                for pos in positions {
                    match ctx
                        .submit(input.bricks.request_inplace(pos, ctx.current_op()))
                        .await
                    {
                        Ok(mut rw) => {
                            for v in rw.iter_mut() {
                                *v = factor * *v + offset;
                            }
                        }
                        Err((r, w)) => {
                            let mut w = w?;
                            for (i, o) in r.iter().zip(w.iter_mut()) {
                                o.write(factor * *i + offset);
                            }
                            unsafe { w.initialized() };
                        }
                    }
                }
                Ok(())
            }
            .into()
        }),
    )
}

pub fn mean<'op>(input: &'op VolumeOperator<'_>) -> ScalarOperator<'op, f32> {
    crate::operators::scalar(
        OperatorId::new("volume_mean")
            .dependent_on(&input.metadata)
            .dependent_on(&input.bricks),
        move |ctx, _| {
            async move {
                let mut sum = 0.0;

                let vol = ctx.submit(input.metadata.request(())).await[0];

                let mut stream = ctx.submit_unordered_with_data(
                    vol.brick_positions()
                        .map(|pos| (input.bricks.request(pos), pos)),
                );
                while let Some((brick_data, brick_pos)) = stream.next().await {
                    let brick = Brick::new(&*brick_data, vol.brick_dim(brick_pos), vol.brick_size);

                    let voxels = brick.voxels().collect::<Vec<_>>();
                    sum += voxels.iter().sum::<f32>();
                }

                let v = sum / vol.num_voxels() as f32;

                ctx.write(v)
            }
            .into()
        },
    )
}
