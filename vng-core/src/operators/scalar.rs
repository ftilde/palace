use bytemuck::Pod;

use crate::{
    id::Id,
    operator::{Operator, OperatorId},
    task::{Task, TaskContext},
};
use std::hash::Hash;

pub type ScalarOperator<T> = Operator<(), T>;

pub fn scalar<
    T: Copy,
    S: 'static,
    F: for<'cref, 'inv> Fn(TaskContext<'cref, 'inv, (), T>, &'inv S) -> Task<'cref> + 'static,
>(
    id: OperatorId,
    state: S,
    compute: F,
) -> ScalarOperator<T> {
    Operator::with_state(id, state, move |ctx, d, s| {
        assert!(d.len() == 1);
        compute(ctx, s)
    })
}

impl<T: Copy + 'static> ScalarOperator<T> {
    pub fn map<D: Pod, O: Copy + 'static>(self, data: D, f: fn(T, &D) -> O) -> ScalarOperator<O> {
        scalar(
            OperatorId::new("ScalarOperator::map")
                .dependent_on(&self)
                .dependent_on(Id::from_data(bytemuck::bytes_of(&data)))
                .dependent_on(Id::hash(&f)),
            (self, data, f),
            move |ctx, (s, data, f)| {
                async move {
                    let v = ctx.submit(s.request_scalar()).await;
                    ctx.write(f(v, data))
                }
                .into()
            },
        )
    }
    pub fn zip<O: Copy + 'static>(self, other: ScalarOperator<O>) -> ScalarOperator<(T, O)> {
        scalar(
            OperatorId::new("ScalarOperator::map")
                .dependent_on(&self)
                .dependent_on(&other),
            (self, other),
            move |ctx, (s, other)| {
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

pub fn constant_pod<T: bytemuck::Pod>(val: T) -> ScalarOperator<T> {
    let op_id = OperatorId::new(std::any::type_name::<T>()).dependent_on(bytemuck::bytes_of(&val));
    scalar(op_id, (), move |ctx, _| {
        async move { ctx.write(val) }.into()
    })
}

impl<T: bytemuck::Pod> From<T> for ScalarOperator<T> {
    fn from(value: T) -> Self {
        constant_pod(value)
    }
}

pub fn constant_hash<T: Copy + Hash + 'static>(val: T) -> ScalarOperator<T> {
    let op_id = OperatorId::new(std::any::type_name::<T>()).dependent_on(Id::hash(&val));
    scalar(op_id, (), move |ctx, _| {
        async move { ctx.write(val) }.into()
    })
}

// TODO: See if we can find a better way here
pub fn constant_as_array<E: bytemuck::Pod, T: Copy + AsRef<[E; 16]> + 'static>(
    val: T,
) -> ScalarOperator<T> {
    let mut op_id = OperatorId::new(std::any::type_name::<T>());
    for v in val.as_ref() {
        op_id = op_id.dependent_on(bytemuck::bytes_of(v));
    }
    scalar(op_id, (), move |ctx, _| {
        async move { ctx.write(val) }.into()
    })
}
