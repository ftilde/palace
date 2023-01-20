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
