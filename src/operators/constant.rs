use crate::{
    id::Id,
    operator::{DataId, Operator, OperatorId},
};

pub type ScalarOperator<'tasks, T> = Operator<'tasks, (), T>;

impl<'tasks, T: bytemuck::Pod> From<&'tasks T> for ScalarOperator<'tasks, T> {
    fn from(value: &'tasks T) -> Self {
        scalar(value)
    }
}

fn scalar<'tasks, T: bytemuck::Pod>(val: &'tasks T) -> ScalarOperator<'tasks, T> {
    let op_id = OperatorId::new(
        std::any::type_name::<T>(),
        &[Id::from_data(bytemuck::bytes_of(val)).into()],
    );
    let val = *val;
    Operator::new(
        op_id,
        Box::new(move |ctx, d, _| {
            async move {
                assert!(d.len() <= 1);
                for d in d {
                    let id = DataId::new(ctx.current_op, &d);
                    ctx.storage.write_to_ram(id, val)?;
                }
                Ok(())
            }
            .into()
        }),
    )
}

// TODO remove those pub when request_blocking in RunTime is figured out
//pub struct ScalarTaskContext<'tasks, 'op, T> {
//    pub inner: TaskContext<'tasks, 'op>,
//    pub op_id: OperatorId,
//    pub marker: std::marker::PhantomData<T>,
//}
//
//impl<'tasks, 'op, T> std::ops::Deref for ScalarTaskContext<'tasks, 'op, T> {
//    type Target = TaskContext<'tasks, 'op>;
//
//    fn deref(&self) -> &Self::Target {
//        &self.inner
//    }
//}
//
//impl<T: bytemuck::Pod> ScalarTaskContext<'_, '_, T> {
//    pub fn write(&self, value: &T) -> Result<(), Error> {
//        let id = TaskId::new(self.op_id, &DatumRequest::Value);
//        self.inner.storage.write_to_ram(id, *value)
//    }
//}

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
