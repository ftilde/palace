use derive_more::Constructor;

use crate::{
    data::{hmul, Brick, BrickPosition, VolumeMetaData},
    operator::{Operator, OperatorId},
    task::{DatumRequest, Task, TaskContext, TaskInfo},
};

pub struct Scale {
    pub vol: OperatorId,
    pub factor: OperatorId,
}

impl Operator for Scale {
    fn id(&self) -> OperatorId {
        OperatorId::new::<Self>(&[self.vol, self.factor])
    }

    fn compute<'a>(&'a self, ctx: TaskContext<'a>, info: DatumRequest) -> Task<'a> {
        async move {
            match info {
                DatumRequest::Value => {
                    // TODO: Depending on what exactly we store in the VolumeMetaData, we will have to
                    // update this. Maybe see VolumeFilterList in Voreen as a reference for how to
                    // model VolumeMetaData for this.
                    let v: &VolumeMetaData = unsafe {
                        ctx.request(TaskInfo::new(self.vol, DatumRequest::Value))
                            .await?
                    };
                    ctx.write_to_ram(&info, v.clone())
                }
                b_req @ DatumRequest::Brick(_) => {
                    let v: &VolumeMetaData = unsafe {
                        ctx.request(TaskInfo::new(self.vol, DatumRequest::Value))
                            .await?
                    };
                    let f: &f32 = unsafe {
                        ctx.request(TaskInfo::new(self.factor, DatumRequest::Value))
                            .await?
                    };
                    let num_voxels = hmul(v.brick_size.0) as usize;
                    let b: &[f32] = unsafe {
                        ctx.request_slice(TaskInfo::new(self.vol, b_req), num_voxels)
                            .await?
                    };

                    unsafe {
                        ctx.with_ram_slot_slice(&info, num_voxels, |buf| {
                            for (i, o) in b.iter().zip(buf.iter_mut()) {
                                o.write(*f * *i);
                            }
                            Ok(())
                        })
                    }
                }
            }
        }
        .into()
    }
}

#[derive(Constructor)]
pub struct Mean {
    vol: OperatorId,
}

impl Operator for Mean {
    fn id(&self) -> OperatorId {
        OperatorId::new::<Self>(&[self.vol])
    }

    fn compute<'a>(&'a self, ctx: TaskContext<'a>, info: DatumRequest) -> Task<'a> {
        async move {
            if let DatumRequest::Value = info {
                let mut sum = 0.0;

                let vol: &VolumeMetaData = unsafe {
                    ctx.request(TaskInfo::new(self.vol, DatumRequest::Value))
                        .await?
                };

                let bd = vol.dimension_in_bricks();
                let num_voxels = hmul(vol.brick_size.0) as usize;
                for z in 0..bd.0.z {
                    for y in 0..bd.0.y {
                        for x in 0..bd.0.x {
                            let brick_pos = BrickPosition(cgmath::vec3(x, y, z));
                            let brick_data: &[f32] = unsafe {
                                ctx.request_slice(
                                    TaskInfo::new(self.vol, DatumRequest::Brick(brick_pos)),
                                    num_voxels,
                                )
                                .await?
                            };

                            let brick = Brick::new(brick_data, vol.brick_dim(brick_pos));

                            sum += brick.voxels().sum::<f32>();
                        }
                    }
                }

                let v = sum / vol.num_voxels() as f32;

                ctx.write_to_ram(&info, v)
            } else {
                Err("Invalid Request".into())
            }
        }
        .into()
    }
}
