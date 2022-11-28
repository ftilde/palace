use std::mem::MaybeUninit;

use derive_more::Constructor;

use crate::{
    data::{hmul, Brick, BrickPosition, VolumeMetaData},
    operator::{Operator, OperatorId},
    task::{DatumRequest, Task, TaskContext, TaskId},
    Error,
};

use super::{request_value, ScalarOperator, ScalarTaskContext};

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
        self.inner.write_to_ram(id, metadata)
    }
    pub unsafe fn write_brick<F: FnOnce(&mut [MaybeUninit<f32>]) -> Result<(), Error>>(
        &self,
        pos: BrickPosition,
        num_voxels: usize,
        f: F,
    ) -> Result<(), Error> {
        let id = TaskId::new(self.current_op_id, &DatumRequest::Brick(pos));
        unsafe { self.inner.with_ram_slot_slice(id, num_voxels, f) }
    }
}

pub async fn request_metadata<'op, 'tasks>(
    vol: &'op dyn VolumeOperator,
    ctx: TaskContext<'op, 'tasks>,
) -> Result<&'tasks VolumeMetaData, Error> {
    let op_id = vol.id();
    let id = TaskId::new(op_id, &DatumRequest::Value); //TODO: revisit
    let v: &VolumeMetaData = unsafe {
        ctx.request(
            id,
            Box::new(move |ctx| {
                let ctx = VolumeTaskContext {
                    inner: ctx,
                    current_op_id: op_id,
                };
                vol.compute_metadata(ctx)
            }),
        )
        .await?
    };
    Ok(v)
}

pub async fn request_brick<'op, 'tasks>(
    vol: &'op dyn VolumeOperator,
    ctx: TaskContext<'op, 'tasks>,
    metadata: &VolumeMetaData,
    pos: BrickPosition,
) -> Result<&'tasks [f32], Error> {
    let num_voxels = hmul(metadata.brick_size.0) as usize;
    let req = DatumRequest::Brick(pos);
    let op_id = vol.id();
    let id = TaskId::new(op_id, &req); //TODO: revisit
    unsafe {
        ctx.request_slice::<f32>(
            id,
            Box::new(move |ctx| {
                let ctx = VolumeTaskContext {
                    inner: ctx,
                    current_op_id: op_id,
                };
                vol.compute_brick(ctx, pos)
            }),
            num_voxels,
        )
        .await
    }
}

pub trait VolumeOperator: Operator {
    fn compute_metadata<'op, 'tasks>(
        &'op self,
        ctx: VolumeTaskContext<'op, 'tasks>,
    ) -> Task<'tasks>;
    fn compute_brick<'op, 'tasks>(
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
    fn compute_metadata<'op, 'tasks>(
        &'op self,
        ctx: VolumeTaskContext<'op, 'tasks>,
    ) -> Task<'tasks> {
        // TODO: Depending on what exactly we store in the VolumeMetaData, we will have to
        // update this. Maybe see VolumeFilterList in Voreen as a reference for how to
        // model VolumeMetaData for this.
        async move {
            let m = request_metadata(self.vol, *ctx).await?;
            ctx.write_metadata(*m)
        }
        .into()
    }

    fn compute_brick<'op, 'tasks>(
        &'op self,
        ctx: VolumeTaskContext<'op, 'tasks>,
        position: BrickPosition,
    ) -> Task<'tasks> {
        async move {
            let (v, factor, offset) = futures::try_join! {
                request_metadata(self.vol, *ctx),
                request_value(self.factor, *ctx),
                request_value(self.offset, *ctx),
            }?;
            let b = request_brick(self.vol, *ctx, &v, position).await?;

            let num_voxels = hmul(v.brick_size.0) as usize;

            unsafe {
                ctx.write_brick(position, num_voxels, |buf| {
                    for (i, o) in b.iter().zip(buf.iter_mut()) {
                        o.write(*factor * *i + offset);
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

            let vol = request_metadata(self.vol, *ctx).await?;

            let bd = vol.dimension_in_bricks();
            for z in 0..bd.0.z {
                for y in 0..bd.0.y {
                    for x in 0..bd.0.x {
                        let brick_pos = BrickPosition(cgmath::vec3(x, y, z));
                        let brick_data = request_brick(self.vol, *ctx, &vol, brick_pos).await?;

                        let brick =
                            Brick::new(brick_data, vol.brick_dim(brick_pos), vol.brick_size);

                        let voxels = brick.voxels().collect::<Vec<_>>();
                        sum += voxels.iter().sum::<f32>();
                    }
                }
            }

            let v = sum / vol.num_voxels() as f32;

            ctx.write(&v)
        }
        .into()
    }
}
