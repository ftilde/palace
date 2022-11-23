use std::mem::MaybeUninit;

use derive_more::Constructor;

use crate::{
    data::{hmul, Brick, BrickPosition, VolumeMetaData},
    operator::{Operator, OperatorId},
    task::{DatumRequest, Task, TaskContext, TaskId},
    Error,
};

use super::{PodOperator, PodOperatorWrite};

pub trait VolumeOperatorWrite {
    fn write_metadata<'op, 'tasks>(
        &'op self,
        ctx: TaskContext<'op, 'tasks>,
        metadata: VolumeMetaData,
    ) -> Result<(), Error>;
    unsafe fn write_brick<'op, 'tasks, F: FnOnce(&mut [MaybeUninit<f32>]) -> Result<(), Error>>(
        &'op self,
        ctx: TaskContext<'op, 'tasks>,
        pos: BrickPosition,
        num_voxels: usize,
        f: F,
    ) -> Result<(), Error>;
}

impl<T> VolumeOperatorWrite for T
where
    T: VolumeOperator + Sized,
{
    fn write_metadata<'op, 'tasks>(
        &'op self,
        ctx: TaskContext<'op, 'tasks>,
        metadata: VolumeMetaData,
    ) -> Result<(), Error> {
        let id = TaskId::new(self.id(), &DatumRequest::Value);
        ctx.write_to_ram(id, metadata)
    }
    unsafe fn write_brick<'op, 'tasks, F: FnOnce(&mut [MaybeUninit<f32>]) -> Result<(), Error>>(
        &'op self,
        ctx: TaskContext<'op, 'tasks>,
        pos: BrickPosition,
        num_voxels: usize,
        f: F,
    ) -> Result<(), Error> {
        let id = TaskId::new(self.id(), &DatumRequest::Brick(pos));
        unsafe { ctx.with_ram_slot_slice(id, num_voxels, f) }
    }
}

pub async fn request_metadata<'op, 'tasks>(
    vol: &'op dyn VolumeOperator,
    ctx: TaskContext<'op, 'tasks>,
) -> Result<&'tasks VolumeMetaData, Error> {
    let id = TaskId::new(vol.id(), &DatumRequest::Value); //TODO: revisit
    let v: &VolumeMetaData = unsafe {
        ctx.request(id, Box::new(move |ctx| vol.compute_metadata(ctx)))
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
    let id = TaskId::new(vol.id(), &req); //TODO: revisit
    unsafe {
        ctx.request_slice::<f32>(
            id,
            Box::new(move |ctx| vol.compute_brick(ctx, pos)),
            num_voxels,
        )
        .await
    }
}

pub trait VolumeOperator: Operator {
    fn compute_metadata<'op, 'tasks>(&'op self, ctx: TaskContext<'op, 'tasks>) -> Task<'tasks>;
    fn compute_brick<'op, 'tasks>(
        &'op self,
        ctx: TaskContext<'op, 'tasks>,
        position: BrickPosition,
    ) -> Task<'tasks>;

    //fn request_metadata<'op, 'tasks>(
    //    &'op self,
    //    ctx: TaskContext<'op, 'tasks>,
    //) -> Pin<Box<dyn Future<Output = Result<&'tasks VolumeMetaData, Error>> + 'tasks>> {
    //    Box::pin(async move {
    //        let id = TaskId::new(self.id(), &DatumRequest::Value); //TODO: revisit
    //        let v: &VolumeMetaData = unsafe {
    //            ctx.request(id, Box::new(move |ctx| self.compute_metadata(ctx)))
    //                .await?
    //        };
    //        Ok(v)
    //    })
    //}
    //fn request_brick<'op, 'tasks>(
    //    &'op self,
    //    ctx: TaskContext<'op, 'tasks>,
    //    metadata: &VolumeMetaData,
    //    pos: BrickPosition,
    //) -> Pin<Box<dyn Future<Output = Result<&'tasks [f32], Error>> + 'tasks>> {
    //    let num_voxels = hmul(metadata.brick_size.0) as usize;
    //    Box::pin(async move {
    //        let req = DatumRequest::Brick(pos);
    //        let id = TaskId::new(self.id(), &req); //TODO: revisit
    //        unsafe {
    //            ctx.request_slice::<f32>(
    //                id,
    //                Box::new(move |ctx| self.compute_brick(ctx, pos)),
    //                num_voxels,
    //            )
    //            .await
    //        }
    //    })
    //}
}

pub struct Scale<'op> {
    pub vol: &'op dyn VolumeOperator,
    pub factor: &'op dyn PodOperator<f32>,
}

impl Operator for Scale<'_> {
    fn id(&self) -> OperatorId {
        //TODO: we may want to cache operatorids in the operators themselves
        OperatorId::new::<Self>(&[self.vol.id(), self.factor.id()])
    }
}

impl VolumeOperator for Scale<'_> {
    fn compute_metadata<'op, 'tasks>(&'op self, ctx: TaskContext<'op, 'tasks>) -> Task<'tasks> {
        // TODO: Depending on what exactly we store in the VolumeMetaData, we will have to
        // update this. Maybe see VolumeFilterList in Voreen as a reference for how to
        // model VolumeMetaData for this.
        async move {
            let m = request_metadata(self.vol, ctx).await?;
            self.write_metadata(ctx, *m)
        }
        .into()
    }

    fn compute_brick<'op, 'tasks>(
        &'op self,
        ctx: TaskContext<'op, 'tasks>,
        position: BrickPosition,
    ) -> Task<'tasks> {
        async move {
            let v = request_metadata(self.vol, ctx).await?;
            let f = self.factor.request_value(ctx).await?;
            let b = request_brick(self.vol, ctx, &v, position).await?;

            let num_voxels = hmul(v.brick_size.0) as usize;

            unsafe {
                self.write_brick(ctx, position, num_voxels, |buf| {
                    for (i, o) in b.iter().zip(buf.iter_mut()) {
                        o.write(*f * *i);
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
impl PodOperator<f32> for Mean<'_> {
    fn compute_value<'op, 'tasks>(&'op self, ctx: TaskContext<'op, 'tasks>) -> Task<'tasks> {
        async move {
            let mut sum = 0.0;

            let vol = request_metadata(self.vol, ctx).await?;

            let bd = vol.dimension_in_bricks();
            for z in 0..bd.0.z {
                for y in 0..bd.0.y {
                    for x in 0..bd.0.x {
                        let brick_pos = BrickPosition(cgmath::vec3(x, y, z));
                        let brick_data = request_brick(self.vol, ctx, &vol, brick_pos).await?;

                        let brick = Brick::new(brick_data, vol.brick_dim(brick_pos));

                        sum += brick.voxels().sum::<f32>();
                    }
                }
            }

            let v = sum / vol.num_voxels() as f32;

            self.write(ctx, &v)
        }
        .into()
    }
}
