use crate::{
    dtypes::StaticElementType,
    operator::{Operator, OperatorDescriptor},
    storage::Element,
    task::{Task, TaskContext},
};
use id::Identify;

pub type ScalarOperator<T> = Operator<T>;

pub fn scalar<
    T: Element,
    S: 'static,
    F: for<'cref, 'inv> Fn(TaskContext<'cref, 'inv, StaticElementType<T>>, &'inv S) -> Task<'cref>
        + 'static,
>(
    descriptor: OperatorDescriptor,
    state: S,
    compute: F,
) -> ScalarOperator<StaticElementType<T>> {
    Operator::with_state(descriptor, Default::default(), state, move |ctx, d, s| {
        assert!(d.len() == 1);
        compute(ctx, s)
    })
}

impl<T: Element> ScalarOperator<StaticElementType<T>> {
    pub fn map<D: Identify + 'static, O: Element>(
        self,
        data: D,
        f: fn(T, &D) -> O,
    ) -> ScalarOperator<StaticElementType<O>> {
        scalar(
            OperatorDescriptor::new("ScalarOperator::map")
                .dependent_on(&self)
                .dependent_on_data(&data)
                .dependent_on_data(&(f as usize)),
            (self, data, f),
            move |ctx, (s, data, f)| {
                async move {
                    let v = ctx.submit(s.request_scalar()).await;
                    ctx.write_scalar(f(v, data));
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
    ) -> ScalarOperator<StaticElementType<O>> {
        self.map(data, f)
    }
    //pub fn zip<O: Element>(
    //    self,
    //    other: ScalarOperator<StaticElementType<O>>,
    //) -> ScalarOperator<StaticElementType<crate::storage::P<T, O>>> {
    //    scalar(
    //        OperatorDescriptor::new("ScalarOperator::zip")
    //            .dependent_on(&self)
    //            .dependent_on(&other),
    //        (self, other),
    //        move |ctx, (s, other)| {
    //            async move {
    //                let (l, r) = futures::join! {
    //                    ctx.submit(s.request_scalar()),
    //                    ctx.submit(other.request_scalar()),
    //                };
    //                ctx.write_scalar(crate::storage::P(l, r));
    //                Ok(())
    //            }
    //            .into()
    //        },
    //    )
    //}
}

pub fn constant<T: Element + Identify>(val: T) -> ScalarOperator<StaticElementType<T>> {
    let op_id = OperatorDescriptor::new(std::any::type_name::<T>()).dependent_on_data(&val);
    scalar(op_id, (), move |ctx, _| {
        async move {
            ctx.write_scalar(val);
            Ok(())
        }
        .into()
    })
}

impl<T: Element + Identify> From<T> for ScalarOperator<StaticElementType<T>> {
    fn from(value: T) -> Self {
        constant(value)
    }
}
