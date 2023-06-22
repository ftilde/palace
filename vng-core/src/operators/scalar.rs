use bytemuck::Pod;

use crate::{
    id::Id,
    operator::{Operator, OperatorId},
    task::{Task, TaskContext},
};
use std::hash::Hash;

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

impl<'op, T: Copy + 'op> ScalarOperator<'op, T> {
    pub fn map<D: Pod, O: Copy + 'op>(self, data: D, f: fn(T, &D) -> O) -> ScalarOperator<'op, O> {
        scalar(
            OperatorId::new("ScalarOperator::map")
                .dependent_on(&self)
                .dependent_on(Id::from_data(bytemuck::bytes_of(&data)))
                .dependent_on(Id::hash(&f)),
            (self, data, f),
            move |ctx, (s, data, f), _| {
                async move {
                    let v = ctx.submit(s.request_scalar()).await;
                    ctx.write(f(v, data))
                }
                .into()
            },
        )
    }
    pub fn zip<O: Copy + 'op>(self, other: ScalarOperator<'op, O>) -> ScalarOperator<'op, (T, O)> {
        scalar(
            OperatorId::new("ScalarOperator::map")
                .dependent_on(&self)
                .dependent_on(&other),
            (self, other),
            move |ctx, (s, other), _| {
                async move {
                    let t = futures::join! {
                        ctx.submit(s.request_scalar()),
                        ctx.submit(other.request_scalar()),
                    };
                    ctx.write(t)
                }
                .into()
            },
        )
    }
}

pub fn constant_pod<'op, T: bytemuck::Pod>(val: T) -> ScalarOperator<'op, T> {
    let op_id = OperatorId::new(std::any::type_name::<T>()).dependent_on(bytemuck::bytes_of(&val));
    scalar(op_id, (), move |ctx, _, _| {
        async move { ctx.write(val) }.into()
    })
}

impl<'op, T: bytemuck::Pod> From<T> for ScalarOperator<'op, T> {
    fn from(value: T) -> Self {
        constant_pod(value)
    }
}

pub fn constant_hash<'op, T: Copy + Hash + 'op>(val: T) -> ScalarOperator<'op, T> {
    let op_id = OperatorId::new(std::any::type_name::<T>()).dependent_on(Id::hash(&val));
    scalar(op_id, (), move |ctx, _, _| {
        async move { ctx.write(val) }.into()
    })
}

// TODO: See if we can find a better way here
pub fn constant_as_array<'op, E: bytemuck::Pod, T: Copy + AsRef<[E; 16]> + 'op>(
    val: T,
) -> ScalarOperator<'op, T> {
    let mut op_id = OperatorId::new(std::any::type_name::<T>());
    for v in val.as_ref() {
        op_id = op_id.dependent_on(bytemuck::bytes_of(v));
    }
    scalar(op_id, (), move |ctx, _, _| {
        async move { ctx.write(val) }.into()
    })
}
