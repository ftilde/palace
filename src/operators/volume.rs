use std::mem::MaybeUninit;

use futures::stream::StreamExt;

use derive_more::Constructor;

use crate::{
    data::{hmul, Brick, BrickPosition, VolumeMetaData},
    operator::{Operator, OperatorId},
    storage::{ReadHandle, WriteHandle},
    task::{DatumRequest, Request, RequestType, Task, TaskContext, TaskId},
    Error,
};

use super::{request_value, ScalarOperator, ScalarTaskContext};

#[derive(Copy, Clone)]
pub struct VolumeTaskContext<'op, 'tasks> {
    inner: TaskContext<'op, 'tasks>,
    current_op_id: OperatorId,
}

impl<'op, 'tasks> std::ops::Deref for VolumeTaskContext<'op, 'tasks> {
    type Target = TaskContext<'op, 'tasks>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'op, 'tasks> VolumeTaskContext<'op, 'tasks> {
    pub fn write_metadata(&self, metadata: VolumeMetaData) -> Result<(), Error> {
        let id = TaskId::new(self.current_op_id, &DatumRequest::Value);
        self.inner.storage.write_to_ram(id, metadata)
    }
    pub unsafe fn with_brick_slot<F: FnOnce(&mut [MaybeUninit<f32>]) -> Result<(), Error>>(
        &self,
        pos: BrickPosition,
        num_voxels: usize,
        f: F,
    ) -> Result<(), Error> {
        let id = TaskId::new(self.current_op_id, &DatumRequest::Brick(pos));
        unsafe { self.inner.storage.with_ram_slot_slice(id, num_voxels, f) }
    }

    pub fn brick_slot<'a>(
        &'a self,
        pos: BrickPosition,
        num_voxels: usize,
    ) -> Result<WriteHandle<'a, [MaybeUninit<f32>]>, Error> {
        let id = TaskId::new(self.current_op_id, &DatumRequest::Brick(pos));
        self.inner.storage.alloc_ram_slot_slice(id, num_voxels)
    }
}

pub fn request_metadata<'req, 'tasks: 'req, 'op: 'tasks>(
    vol: &'op dyn VolumeOperator,
) -> Request<'req, 'op, ReadHandle<'req, VolumeMetaData>> {
    let op_id = vol.id();
    let id = TaskId::new(op_id, &DatumRequest::Value); //TODO: revisit
    Request {
        id,
        type_: RequestType::Data(Box::new(move |ctx| {
            let ctx = VolumeTaskContext {
                inner: ctx,
                current_op_id: op_id,
            };
            vol.compute_metadata(ctx)
        })),
        poll: Box::new(move |ctx| unsafe { ctx.storage.read_ram(id) }),
        _marker: Default::default(),
    }
}

pub fn request_brick<'req, 'tasks: 'req, 'op: 'tasks>(
    vol: &'op dyn VolumeOperator,
    metadata: &VolumeMetaData,
    pos: BrickPosition,
) -> Request<'req, 'op, ReadHandle<'req, [f32]>> {
    let num_voxels = hmul(metadata.brick_size.0) as usize;
    let req = DatumRequest::Brick(pos);
    let op_id = vol.id();
    let id = TaskId::new(op_id, &req); //TODO: revisit
    Request {
        id,
        type_: RequestType::Data(Box::new(move |ctx| {
            let ctx = VolumeTaskContext {
                inner: ctx,
                current_op_id: op_id,
            };
            vol.compute_brick(ctx, pos)
        })),
        poll: Box::new(move |ctx| unsafe { ctx.storage.read_ram_slice(id, num_voxels) }),
        _marker: Default::default(),
    }
}

pub trait VolumeOperator: Operator {
    fn compute_metadata<'tasks, 'op: 'tasks>(
        &'op self,
        ctx: VolumeTaskContext<'op, 'tasks>,
    ) -> Task<'tasks>;
    fn compute_brick<'tasks, 'op: 'tasks>(
        &'op self,
        ctx: VolumeTaskContext<'op, 'tasks>,
        position: BrickPosition,
    ) -> Task<'tasks>;
}

pub struct LinearRescale<'op> {
    pub vol: &'op dyn VolumeOperator,
    pub factor: &'op dyn ScalarOperator<f32>,
    pub offset: &'op dyn ScalarOperator<f32>,
}

impl Operator for LinearRescale<'_> {
    fn id(&self) -> OperatorId {
        //TODO: we may want to cache operatorids in the operators themselves
        OperatorId::new::<Self>(&[self.vol.id(), self.factor.id(), self.offset.id()])
    }
}

impl VolumeOperator for LinearRescale<'_> {
    fn compute_metadata<'tasks, 'op: 'tasks>(
        &'op self,
        ctx: VolumeTaskContext<'op, 'tasks>,
    ) -> Task<'tasks> {
        // TODO: Depending on what exactly we store in the VolumeMetaData, we will have to
        // update this. Maybe see VolumeFilterList in Voreen as a reference for how to
        // model VolumeMetaData for this.
        async move {
            let m = ctx.submit(request_metadata(self.vol)).await;
            ctx.write_metadata(*m)
        }
        .into()
    }

    fn compute_brick<'tasks, 'op: 'tasks>(
        &'op self,
        ctx: VolumeTaskContext<'op, 'tasks>,
        position: BrickPosition,
    ) -> Task<'tasks> {
        async move {
            let (v, factor, offset) = futures::join! {
                ctx.submit(request_metadata(self.vol)),
                ctx.submit(request_value(self.factor)),
                ctx.submit(request_value(self.offset)),
            };
            let b = ctx.submit(request_brick(self.vol, &v, position)).await;

            let num_voxels = hmul(v.brick_size.0) as usize;

            unsafe {
                ctx.with_brick_slot(position, num_voxels, |buf| {
                    for (i, o) in b.iter().zip(buf.iter_mut()) {
                        o.write(*factor * *i + *offset);
                    }
                    Ok(())
                })
            }
        }
        .into()
    }
}

#[derive(Constructor)]
pub struct Mean<'op> {
    vol: &'op dyn VolumeOperator,
}

impl Operator for Mean<'_> {
    fn id(&self) -> OperatorId {
        OperatorId::new::<Self>(&[self.vol.id()])
    }
}
impl ScalarOperator<f32> for Mean<'_> {
    fn compute_value<'op, 'tasks>(
        &'op self,
        ctx: ScalarTaskContext<'op, 'tasks, f32>,
    ) -> Task<'tasks> {
        async move {
            let mut sum = 0.0;

            let vol = ctx.submit(request_metadata(self.vol)).await;

            let mut stream = ctx.submit_unordered_with_data(
                vol.brick_positions()
                    .map(|pos| (request_brick(self.vol, &vol, pos), pos)),
            );
            while let Some((brick_data, brick_pos)) = stream.next().await {
                let brick = Brick::new(&*brick_data, vol.brick_dim(brick_pos), vol.brick_size);

                let voxels = brick.voxels().collect::<Vec<_>>();
                sum += voxels.iter().sum::<f32>();
            }

            let v = sum / vol.num_voxels() as f32;

            ctx.write(&v)
        }
        .into()
    }
}
