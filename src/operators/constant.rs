use crate::{
    id::Id,
    operator::{Operator, OperatorId},
    storage::ReadHandle,
    task::{DatumRequest, Request, RequestType, Task, TaskContext, TaskId},
    Error,
};

impl<T: bytemuck::Pod> Operator for T {
    fn id(&self) -> OperatorId {
        OperatorId::new::<f32>(&[Id::from_data(bytemuck::bytes_of(self)).into()])
    }
}

// TODO remove those pub when request_blocking in RunTime is figured out
pub struct ScalarTaskContext<'op, 'tasks, T> {
    pub inner: TaskContext<'op, 'tasks>,
    pub op_id: OperatorId,
    pub marker: std::marker::PhantomData<T>,
}

impl<'op, 'tasks, T> std::ops::Deref for ScalarTaskContext<'op, 'tasks, T> {
    type Target = TaskContext<'op, 'tasks>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: bytemuck::Pod> ScalarTaskContext<'_, '_, T> {
    pub fn write(&self, value: &T) -> Result<(), Error> {
        let id = TaskId::new(self.op_id, &DatumRequest::Value);
        self.inner.storage.write_to_ram(id, *value)
    }
}

pub fn request_value<'req, 'tasks: 'req, 'op: 'tasks, T: bytemuck::Pod + 'req>(
    op: &'op dyn ScalarOperator<T>,
) -> Request<'req, 'op, ReadHandle<'req, T>> {
    let op_id = op.id();
    let id = TaskId::new(op_id, &DatumRequest::Value);
    Request {
        id,
        type_: RequestType::Data(Box::new(move |ctx| {
            let ctx = ScalarTaskContext {
                inner: ctx,
                op_id,
                marker: Default::default(),
            };
            op.compute_value(ctx)
        })),
        poll: Box::new(move |ctx| unsafe { ctx.storage.read_ram(id) }),
        _marker: Default::default(),
    }
}

pub trait ScalarOperator<T: bytemuck::Pod>: Operator {
    fn compute_value<'op, 'tasks>(
        &'op self,
        ctx: ScalarTaskContext<'op, 'tasks, T>,
    ) -> Task<'tasks>;
}

impl<T: bytemuck::Pod> ScalarOperator<T> for T {
    fn compute_value<'op, 'tasks>(
        &'op self,
        ctx: ScalarTaskContext<'op, 'tasks, T>,
    ) -> Task<'tasks> {
        async move { ctx.write(&*self) }.into()
    }
}
