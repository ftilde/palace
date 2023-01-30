use crate::{
    operator::{Operator, OperatorId},
    task::{Task, TaskContext},
};

pub type ScalarOperator<'op, T> = Operator<'op, (), T>;

pub fn scalar<
    'op,
    T: Copy,
    F: for<'cref, 'inv> Fn(
            TaskContext<'cref, 'inv, (), T>,
            crate::operator::OutlivesMarker<'op, 'inv>,
        ) -> Task<'cref>
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

pub fn constant<'op, T: bytemuck::Pod>(val: &'op T) -> ScalarOperator<'op, T> {
    let op_id = OperatorId::new(std::any::type_name::<T>()).dependent_on(bytemuck::bytes_of(val));
    let val = *val;
    scalar(op_id, move |ctx, _| async move { ctx.write(val) }.into())
}

impl<'op, T: bytemuck::Pod> From<&'op T> for ScalarOperator<'op, T> {
    fn from(value: &'op T) -> Self {
        constant(value)
    }
}
