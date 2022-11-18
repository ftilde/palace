use derive_more::Constructor;

use crate::{
    data::{Brick, BrickPosition, Datum},
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

    fn compute<'a>(&'a self, rt: TaskContext<'a>, info: DatumRequest) -> Task<'a> {
        async move {
            match info {
                DatumRequest::Value => {
                    // TODO: Depending on what exactly we store in the VolumeMetaData, we will have to
                    // update this. Maybe see VolumeFilterList in Voreen as a reference for how to
                    // model VolumeMetaData for this.
                    rt.request(TaskInfo::new(self.vol, DatumRequest::Value))
                        .await
                }
                b_req @ DatumRequest::Brick(_) => {
                    let f = rt
                        .request(TaskInfo::new(self.factor, DatumRequest::Value))
                        .await?
                        .float()?;
                    let mut b = rt.request(TaskInfo::new(self.vol, b_req)).await?.brick()?;

                    for v in &mut b {
                        *v *= f;
                    }

                    Ok(Datum::Brick(b))
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

    fn compute<'a>(&'a self, rt: TaskContext<'a>, info: DatumRequest) -> Task<'a> {
        async move {
            if let DatumRequest::Value = info {
                let mut sum = 0.0;

                let vol = rt
                    .request(TaskInfo::new(self.vol, DatumRequest::Value))
                    .await?
                    .volume()?;

                let bd = vol.dimension_in_bricks();
                for z in 0..bd.0.z {
                    for y in 0..bd.0.y {
                        for x in 0..bd.0.x {
                            let brick_pos = BrickPosition(cgmath::vec3(x, y, z));
                            let brick_data = rt
                                .request(TaskInfo::new(self.vol, DatumRequest::Brick(brick_pos)))
                                .await?
                                .brick()?;

                            let brick = Brick::new(&brick_data, vol.brick_dim(brick_pos));

                            sum += brick.voxels().sum::<f32>();
                        }
                    }
                }

                let v = sum / vol.num_voxels() as f32;
                Ok(Datum::Float(v))
            } else {
                Err("Invalid Request".into())
            }
        }
        .into()
    }
}
