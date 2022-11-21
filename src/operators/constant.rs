use crate::{
    id::Id,
    operator::{Operator, OperatorId},
    task::{DatumRequest, Task, TaskContext},
};

impl Operator for f32 {
    fn id(&self) -> OperatorId {
        OperatorId::new::<f32>(&[Id::from_data(bytemuck::bytes_of(self)).into()])
    }

    fn compute<'a>(&'a self, ctx: TaskContext<'a>, info: DatumRequest) -> Task<'a> {
        async move {
            match info {
                DatumRequest::Value => ctx.write_to_ram(&info, *self),
                _ => Err("Invalid Request".into()),
            }
        }
        .into()
    }
}
