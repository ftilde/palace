use bytemuck::AnyBitPattern;

use crate::{
    id::Id,
    storage::{InplaceResult, ReadHandle},
    task::{DataRequest, OpaqueTaskContext, Request, RequestType, Task, TaskContext},
};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct OperatorId(Id);
impl OperatorId {
    pub fn new(name: &'static str) -> Self {
        let id = Id::from_data(name.as_bytes());
        OperatorId(id)
    }
    pub fn dependent_on(self, id: impl Into<Id>) -> Self {
        OperatorId(Id::combine(&[self.0, id.into()]))
    }
    pub fn slot(&self, slot_number: usize) -> Self {
        let id = Id::combine(&[self.0, Id::from_data(bytemuck::bytes_of(&slot_number))]);
        OperatorId(id)
    }
    pub fn inner(&self) -> Id {
        self.0
    }
}

impl Into<Id> for OperatorId {
    fn into(self) -> Id {
        self.0
    }
}
impl<I, O> Into<Id> for &Operator<'_, I, O> {
    fn into(self) -> Id {
        self.id.into()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct DataId(Id);
impl DataId {
    pub fn new(op: OperatorId, descriptor: &impl bytemuck::NoUninit) -> Self {
        let hash = bytemuck::bytes_of(descriptor);
        let data_id = Id::from_data(hash);

        DataId(Id::combine(&[op.inner(), data_id]))
    }
}

impl From<Id> for OperatorId {
    fn from(inner: Id) -> Self {
        Self(inner)
    }
}

pub struct TypeErased(*mut ());
impl TypeErased {
    pub fn pack<T>(v: T) -> Self {
        TypeErased(Box::into_raw(Box::new(v)) as *mut ())
    }
    pub unsafe fn unpack<T>(self) -> T {
        *Box::from_raw(self.0 as *mut T)
    }
}

// Workaround because to limit the lifetime of allowed values in HRTBs. For example, we cannot
// write for<'a: 'b> or restrict 'a in a where clause.
// See https://stackoverflow.com/questions/75147315/rust-returning-this-value-requires-that-op-must-outlive-static-with-hrt
pub type OutlivesMarker<'longer, 'shorter> = &'shorter &'longer ();
pub type ComputeFunction<'op, ItemDescriptor, Output> = Box<
    dyn for<'cref, 'inv> Fn(
            TaskContext<'cref, 'inv, ItemDescriptor, Output>,
            Vec<ItemDescriptor>,
            OutlivesMarker<'op, 'inv>,
        ) -> Task<'cref>
        + 'op,
>;

pub trait OpaqueOperator {
    fn id(&self) -> OperatorId;
    unsafe fn compute<'cref, 'inv>(
        &'inv self,
        context: OpaqueTaskContext<'cref, 'inv>,
        items: Vec<TypeErased>,
    ) -> Task<'cref>;
}

pub struct Operator<'op, ItemDescriptor, Output: ?Sized> {
    id: OperatorId,
    compute: ComputeFunction<'op, ItemDescriptor, Output>,
}

impl<'op, Output: AnyBitPattern> Operator<'op, (), Output> {
    pub fn request_scalar<'req, 'inv: 'req>(
        &'inv self,
    ) -> Request<'req, 'inv, ReadHandle<'req, Output>> {
        let item = ();
        let id = DataId::new(self.id, &item);

        Request {
            type_: RequestType::Data(DataRequest {
                id,
                source: self,
                item: TypeErased::pack(item),
            }),
            poll: Box::new(move |ctx| unsafe {
                ctx.storage.read_ram(id).map(|v| v.map(|a| &a[0]))
            }),
            _marker: Default::default(),
        }
    }
}

impl<'op, ItemDescriptor: bytemuck::NoUninit + 'static, Output: AnyBitPattern>
    Operator<'op, ItemDescriptor, Output>
{
    pub fn new<
        F: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, ItemDescriptor, Output>,
                Vec<ItemDescriptor>,
                OutlivesMarker<'op, 'inv>,
            ) -> Task<'cref>
            + 'op,
    >(
        id: OperatorId,
        compute: F,
    ) -> Self {
        Self {
            id,
            compute: Box::new(compute),
        }
    }

    pub fn request<'req, 'inv: 'req>(
        &'inv self,
        item: ItemDescriptor,
    ) -> Request<'req, 'inv, ReadHandle<'req, [Output]>> {
        let id = DataId::new(self.id, &item);

        Request {
            type_: RequestType::Data(DataRequest {
                id,
                source: self,
                item: TypeErased::pack(item),
            }),
            poll: Box::new(move |ctx| unsafe { ctx.storage.read_ram(id) }),
            _marker: Default::default(),
        }
    }
    pub fn request_inplace<'req, 'inv: 'req>(
        &'inv self,
        item: ItemDescriptor,
        write_id: OperatorId,
    ) -> Request<'req, 'inv, InplaceResult<'req, f32>> {
        let read_id = DataId::new(self.id, &item);
        let write_id = DataId::new(write_id, &item);

        Request {
            type_: RequestType::Data(DataRequest {
                id: read_id,
                source: self,
                item: TypeErased::pack(item),
            }),
            poll: Box::new(move |ctx| unsafe { ctx.storage.try_update_inplace(read_id, write_id) }),
            _marker: Default::default(),
        }
    }
}

impl<'op, ItemDescriptor: bytemuck::NoUninit, Output: bytemuck::AnyBitPattern> OpaqueOperator
    for Operator<'op, ItemDescriptor, Output>
{
    fn id(&self) -> OperatorId {
        self.id
    }
    unsafe fn compute<'cref, 'inv>(
        &'inv self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        items: Vec<TypeErased>,
    ) -> Task<'cref> {
        let items = items
            .into_iter()
            .map(|v| unsafe { v.unpack() })
            .collect::<Vec<_>>();
        let ctx = TaskContext::new(ctx);
        (self.compute)(ctx, items, &&())
    }
}
