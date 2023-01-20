use bytemuck::Pod;

use crate::{
    operator::{Operator, OperatorId},
    task::{Task, TaskContext},
};

pub type ScalarOperator<'op, T> = Operator<'op, (), T>;

pub fn scalar<
    'op,
    T: Pod,
    F: for<'tasks> Fn(
            TaskContext<'tasks, (), T>,
            crate::operator::OutlivesMarker<'op, 'tasks>,
        ) -> Task<'tasks>
        + 'op,
>(
    id: OperatorId,
    compute: F,
) -> ScalarOperator<'op, T> {
    Operator::new(id, move |ctx, d, m| {
        assert!(d.len() == 1);
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
