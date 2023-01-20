use bytemuck::AnyBitPattern;

use crate::{
    id::Id,
    storage::{InplaceResultSlice, ReadHandle},
    task::{DataRequest, Request, RequestType, Task, TaskContext},
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
    pub fn new(op: OperatorId, descriptor: &impl bytemuck::Pod) -> Self {
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

pub type OutlivesMarker<'longer, 'shorter> = &'shorter &'longer ();
pub type ComputeFunction<'op, ItemDescriptor> = Box<
    dyn for<'tasks> Fn(
            TaskContext<'tasks>,
            Vec<ItemDescriptor>,
            OutlivesMarker<'op, 'tasks>,
        ) -> Task<'tasks>
        + 'op,
>;

pub trait OpaqueOperator {
    fn id(&self) -> OperatorId;
    unsafe fn compute<'tasks>(
        &'tasks self,
        context: TaskContext<'tasks>,
        items: Vec<TypeErased>,
    ) -> Task<'tasks>;
}

pub struct Operator<'op, ItemDescriptor, Output: ?Sized> {
    id: OperatorId,
    compute: ComputeFunction<'op, ItemDescriptor>,
    _marker: std::marker::PhantomData<(ItemDescriptor, Output)>,
}

impl<'op, ItemDescriptor: bytemuck::Pod + 'static, Output: AnyBitPattern>
    Operator<'op, ItemDescriptor, Output>
{
    pub fn new<
        F: for<'tasks> Fn(
                TaskContext<'tasks>,
                Vec<ItemDescriptor>,
                OutlivesMarker<'op, 'tasks>,
            ) -> Task<'tasks>
            + 'op,
    >(
        id: OperatorId,
        compute: F,
    ) -> Self {
        Self {
            id,
            compute: Box::new(compute),
            _marker: Default::default(),
        }
    }

    pub fn request<'req>(
        &'req self,
        item: ItemDescriptor,
    ) -> Request<'req, ReadHandle<'req, [Output]>> {
        let id = DataId::new(self.id, &item);

        // Safety: We make sure to only use objects with appropriate lifetimes when using the
        // pointer.
        let self_static: &Operator<'static, ItemDescriptor, Output> =
            unsafe { std::mem::transmute(self) };
        let self_ptr: *const dyn OpaqueOperator = self_static;
        Request {
            type_: RequestType::Data(DataRequest {
                id,
                source: self_ptr,
                item: TypeErased::pack(item),
            }),
            poll: Box::new(move |ctx| unsafe { ctx.storage.read_ram_slice(id) }),
            _marker: Default::default(),
        }
    }
    pub fn request_inplace<'req>(
        &'req self,
        item: ItemDescriptor,
        write_id: OperatorId,
    ) -> Request<'req, InplaceResultSlice<'req, f32>> {
        let read_id = DataId::new(self.id, &item); //TODO: revisit
        let write_id = DataId::new(write_id, &item);

        // Safety: We make sure to only use objects with appropriate lifetimes when using the
        // pointer.
        let self_static: &Operator<'static, ItemDescriptor, Output> =
            unsafe { std::mem::transmute(self) };
        let self_ptr: *const dyn OpaqueOperator = self_static;
        Request {
            type_: RequestType::Data(DataRequest {
                id: read_id,
                source: self_ptr,
                item: TypeErased::pack(item),
            }),
            poll: Box::new(move |ctx| unsafe {
                ctx.storage.try_update_inplace_slice(read_id, write_id)
            }),
            _marker: Default::default(),
        }
    }
}

impl<'op, ItemDescriptor, Output> OpaqueOperator for Operator<'op, ItemDescriptor, Output> {
    fn id(&self) -> OperatorId {
        self.id
    }
    unsafe fn compute<'tasks>(
        &'tasks self,
        context: TaskContext<'tasks>,
        items: Vec<TypeErased>,
    ) -> Task<'tasks> {
        let items = items
            .into_iter()
            .map(|v| unsafe { v.unpack() })
            .collect::<Vec<_>>();
        (self.compute)(context, items, &&())
    }
}
