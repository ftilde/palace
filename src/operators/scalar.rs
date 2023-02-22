use crate::{
    operator::{Operator, OperatorId},
    task::{Task, TaskContext},
};

pub type ScalarOperator<'op, T> = Operator<'op, (), T>;

pub fn scalar<
    'op,
    T: Copy,
    S: 'op,
    F: for<'cref, 'inv> Fn(
            TaskContext<'cref, 'inv, (), T>,
            &'inv S,
            crate::operator::OutlivesMarker<'op, 'inv>,
        ) -> Task<'cref>
        + 'op,
>(
    id: OperatorId,
    state: S,
    compute: F,
) -> ScalarOperator<'op, T> {
    Operator::with_state(id, state, move |ctx, d, s, m| {
        assert!(d.len() == 1);
        compute(ctx, s, m)
    })
}

pub fn constant<'op, T: bytemuck::Pod>(val: T) -> ScalarOperator<'op, T> {
    let op_id = OperatorId::new(std::any::type_name::<T>()).dependent_on(bytemuck::bytes_of(&val));
    scalar(op_id, (), move |ctx, _, _| {
        async move { ctx.write(val) }.into()
    })
}

impl<'op, T: bytemuck::Pod> From<T> for ScalarOperator<'op, T> {
    fn from(value: T) -> Self {
        constant(value)
    }
}
