use crate::{
    dtypes::StaticElementType,
    op_descriptor,
    operator::{DataParam, Operator, OperatorDescriptor},
    storage::Element,
    task::{Task, TaskContext},
};
use id::{Identify, IdentifyHash};

pub type ScalarOperator<T> = Operator<T>;

pub fn scalar<T: Element, S: Identify + 'static>(
    descriptor: OperatorDescriptor,
    state: S,
    compute: for<'cref, 'inv> fn(
        TaskContext<'cref, 'inv, StaticElementType<T>>,
        &'inv S,
    ) -> Task<'cref>,
) -> ScalarOperator<StaticElementType<T>> {
    Operator::with_state(
        descriptor,
        Default::default(),
        (DataParam(state), DataParam(IdentifyHash(compute))),
        |ctx, d, (s, compute)| {
            assert!(d.len() == 1);
            compute(ctx, s)
        },
    )
}

impl<T: Element> ScalarOperator<StaticElementType<T>> {
    pub fn map<D: Identify + 'static, O: Element>(
        self,
        data: D,
        f: fn(T, &D) -> O,
    ) -> ScalarOperator<StaticElementType<O>> {
        scalar(
            op_descriptor!(),
            (self, DataParam(data), DataParam(IdentifyHash(f))),
            |ctx, (s, data, f)| {
                async move {
                    let v = ctx.submit(s.request_scalar()).await;
                    ctx.submit(ctx.write_scalar(f(v, data))).await;
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
    let op_id = op_descriptor!();
    scalar(op_id, DataParam(val), move |ctx, val| {
        async move {
            ctx.submit(ctx.write_scalar(**val)).await;
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
