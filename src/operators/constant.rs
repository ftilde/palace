use std::{future::Future, pin::Pin};

use crate::{
    id::Id,
    operator::{Operator, OperatorId},
    task::{DatumRequest, Task, TaskContext, TaskId},
    Error,
};

impl<T: bytemuck::Pod> Operator for T {
    fn id(&self) -> OperatorId {
        OperatorId::new::<f32>(&[Id::from_data(bytemuck::bytes_of(self)).into()])
    }
}

pub trait PodOperatorWrite<T> {
    fn write<'op, 'tasks>(&self, ctx: TaskContext<'op, 'tasks>, value: &T) -> Result<(), Error>;
}

impl<T, P> PodOperatorWrite<T> for P
where
    P: PodOperator<T> + Sized,
    T: bytemuck::Pod,
{
    fn write<'op, 'tasks>(&self, ctx: TaskContext<'op, 'tasks>, value: &T) -> Result<(), Error> {
        let id = TaskId::new(self.id(), &DatumRequest::Value);
        ctx.write_to_ram(id, *value)
    }
}

pub trait PodOperator<T: bytemuck::Pod>: Operator {
    fn compute_value<'op, 'tasks>(&'op self, ctx: TaskContext<'op, 'tasks>) -> Task<'tasks>;
    fn request_value<'op, 'tasks>(
        &'op self,
        ctx: TaskContext<'op, 'tasks>,
    ) -> Pin<Box<dyn Future<Output = Result<&'tasks T, Error>> + 'tasks>> {
        Box::pin(async move {
            let id = TaskId::new(self.id(), &DatumRequest::Value);
            unsafe {
                ctx.request::<T>(id, Box::new(move |ctx| self.compute_value(ctx)))
                    .await
            }
        })
    }
}

impl<T: bytemuck::Pod> PodOperator<T> for T {
    fn compute_value<'op, 'tasks>(&'op self, ctx: TaskContext<'op, 'tasks>) -> Task<'tasks> {
        async move { self.write(ctx, &*self) }.into()
    }
}
