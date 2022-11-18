use crate::{
    data::Datum,
    id::Id,
    operator::{Operator, OperatorId},
    task::{DatumRequest, Task, TaskContext},
};

impl Operator for f32 {
    fn id(&self) -> OperatorId {
        OperatorId::new::<f32>(&[Id::from_data(bytemuck::bytes_of(self)).into()])
    }

    fn compute<'a>(&'a self, _rt: TaskContext<'a>, info: DatumRequest) -> Task<'a> {
        async move {
            match info {
                DatumRequest::Value => Ok(Datum::Float(*self)),
                _ => Err("Invalid Request".into()),
            }
        }
        .into()
    }
}
