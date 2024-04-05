use crate::{
    operator::{Operator, OperatorDescriptor},
    storage::Element,
    task::{Task, TaskContext},
};
use id::{Id, Identify};

pub type ScalarOperator<T> = Operator<(), T>;

pub fn scalar<
    T: Element,
    S: 'static,
    F: for<'cref, 'inv> Fn(TaskContext<'cref, 'inv, (), T>, &'inv S) -> Task<'cref> + 'static,
>(
    descriptor: OperatorDescriptor,
    state: S,
    compute: F,
) -> ScalarOperator<T> {
    Operator::with_state(descriptor, state, move |ctx, d, s| {
        assert!(d.len() == 1);
        compute(ctx, s)
    })
}

impl<T: Element> ScalarOperator<T> {
    pub fn map<D: Identify + 'static, O: Element>(
        self,
        data: D,
        f: fn(T, &D) -> O,
    ) -> ScalarOperator<O> {
        scalar(
            OperatorDescriptor::new("ScalarOperator::map")
                .dependent_on(&self)
                .dependent_on_data(&data)
                .dependent_on_data(&Id::hash(&f)),
            (self, data, f),
            move |ctx, (s, data, f)| {
                async move {
                    let v = ctx.submit(s.request_scalar()).await;
                    ctx.write(f(v, data));
                    Ok(())
                }
                .into()
            },
        )
    }
    // Just an alias, but sometimes it helps to use this to identify error messages
    pub fn map_scalar<D: Identify + 'static, O: Element>(
        self,
        data: D,
        f: fn(T, &D) -> O,
    ) -> ScalarOperator<O> {
        self.map(data, f)
    }
    pub fn zip<O: Element>(
        self,
        other: ScalarOperator<O>,
    ) -> ScalarOperator<crate::storage::P<T, O>> {
        scalar(
            OperatorDescriptor::new("ScalarOperator::zip")
                .dependent_on(&self)
                .dependent_on(&other),
            (self, other),
            move |ctx, (s, other)| {
                async move {
                    let (l, r) = futures::join! {
                        ctx.submit(s.request_scalar()),
                        ctx.submit(other.request_scalar()),
                    };
                    ctx.write(crate::storage::P(l, r));
                    Ok(())
                }
                .into()
            },
        )
    }
}

pub fn constant<T: Element + Identify>(val: T) -> ScalarOperator<T> {
    let op_id = OperatorDescriptor::new(std::any::type_name::<T>()).dependent_on_data(&val);
    scalar(op_id, (), move |ctx, _| {
        async move {
            ctx.write(val);
            Ok(())
        }
        .into()
    })
}

impl<T: Element + Identify> From<T> for ScalarOperator<T> {
    fn from(value: T) -> Self {
        constant(value)
    }
}
