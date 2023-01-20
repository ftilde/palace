use bytemuck::Pod;

use crate::{
    operator::{DataId, Operator, OperatorId},
    task::{Task, TaskContext},
    Error,
};

pub type ScalarOperator<'op, T> = Operator<'op, (), T>;

pub fn scalar<
    'op,
    T: Pod,
    F: for<'tasks> Fn(
            ScalarTaskContext<'tasks, T>,
            crate::operator::OutlivesMarker<'op, 'tasks>,
        ) -> Task<'tasks>
        + 'op,
>(
    id: OperatorId,
    compute: F,
) -> ScalarOperator<'op, T> {
    Operator::new(id, move |ctx, d, m| {
        assert!(d.len() == 1);
        let data_id = DataId::new(ctx.current_op(), &());
        let ctx = ScalarTaskContext {
            inner: ctx,
            data_id,
            marker: Default::default(),
        };
        compute(ctx, m)
    })
}

pub fn constant<'tasks, T: bytemuck::Pod>(val: &'tasks T) -> ScalarOperator<'tasks, T> {
    let op_id = OperatorId::new(std::any::type_name::<T>()).dependent_on(bytemuck::bytes_of(val));
    let val = *val;
    scalar(op_id, move |ctx, _| async move { ctx.write(val) }.into())
}

impl<'tasks, T: bytemuck::Pod> From<&'tasks T> for ScalarOperator<'tasks, T> {
    fn from(value: &'tasks T) -> Self {
        constant(value)
    }
}

pub struct ScalarTaskContext<'op, T> {
    pub inner: TaskContext<'op>,
    pub data_id: DataId,
    pub marker: std::marker::PhantomData<T>,
}

impl<'op, T> std::ops::Deref for ScalarTaskContext<'op, T> {
    type Target = TaskContext<'op>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: bytemuck::Pod> ScalarTaskContext<'_, T> {
    pub fn write(&self, value: T) -> Result<(), Error> {
        self.inner.storage.write_to_ram(self.data_id, value)
    }
}

//pub fn request_value<'req, 'tasks: 'req, 'op: 'tasks, T: bytemuck::Pod + 'req>(
//    op: &'op dyn ScalarOperator<T>,
//) -> Request<'req, 'op, ReadHandle<'req, T>> {
//    let op_id = op.id();
//    let id = TaskId::new(op_id, &DatumRequest::Value);
//    Request {
//        id,
//        type_: RequestType::Data(Box::new(move |ctx| {
//            let ctx = ScalarTaskContext {
//                inner: ctx,
//                op_id,
//                marker: Default::default(),
//            };
//            op.compute_value(ctx)
//        })),
//        poll: Box::new(move |ctx| unsafe { ctx.storage.read_ram(id) }),
//        _marker: Default::default(),
//    }
//}
