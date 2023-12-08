use std::rc::Rc;

use crate::{
    id::{Id, Identify},
    storage::{gpu, ram, DataLocation, Element, VisibleDataLocation},
    task::{DataRequest, OpaqueTaskContext, Request, RequestType, Task, TaskContext},
    task_graph::{LocatedDataId, VisibleDataId},
    vulkan::{DeviceId, DstBarrierInfo},
};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct OperatorId(Id, pub &'static str);
impl OperatorId {
    pub fn new(name: &'static str) -> Self {
        let id = Id::from_data(name.as_bytes());
        OperatorId(id, name)
    }
    pub fn dependent_on(self, v: &(impl Identify + ?Sized)) -> Self {
        OperatorId(Id::combine(&[self.0, v.id()]), self.1)
    }
    pub fn slot(&self, slot_number: usize) -> Self {
        let id = Id::combine(&[self.0, Id::from_data(bytemuck::bytes_of(&slot_number))]);
        OperatorId(id, self.1)
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
impl<I, O> Identify for Operator<I, O> {
    fn id(&self) -> Id {
        self.id.into()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct DataId(pub Id);

impl DataId {
    pub fn new(op: OperatorId, descriptor: &impl std::hash::Hash) -> Self {
        let data_id = Id::hash(descriptor);

        DataId(Id::combine(&[op.inner(), data_id]))
    }

    pub fn in_location(self, location: DataLocation) -> LocatedDataId {
        LocatedDataId { id: self, location }
    }
    pub fn with_visibility(self, location: VisibleDataLocation) -> VisibleDataId {
        VisibleDataId { id: self, location }
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
    pub unsafe fn unpack_ref<T>(&self) -> &T {
        &*(self.0 as *mut T)
    }
}

pub type ComputeFunction<ItemDescriptor, Output> = Rc<
    dyn for<'cref, 'inv> Fn(
        TaskContext<'cref, 'inv, ItemDescriptor, Output>,
        Vec<ItemDescriptor>,
        &'inv TypeErased,
    ) -> Task<'cref>,
>;

#[derive(Copy, Clone)]
pub enum ItemGranularity {
    Single,
    Batched,
}

pub trait OpaqueOperator {
    fn id(&self) -> OperatorId;
    fn granularity(&self) -> ItemGranularity;
    unsafe fn compute<'cref, 'inv>(
        &'inv self,
        context: OpaqueTaskContext<'cref, 'inv>,
        items: Vec<TypeErased>,
    ) -> Task<'cref>;
}

pub struct Operator<ItemDescriptor, Output: ?Sized> {
    id: OperatorId,
    state: Rc<TypeErased>,
    granularity: ItemGranularity,
    compute: ComputeFunction<ItemDescriptor, Output>,
}
impl<ItemDescriptor, Output: ?Sized> Clone for Operator<ItemDescriptor, Output> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            state: self.state.clone(),
            granularity: self.granularity.clone(),
            compute: self.compute.clone(),
        }
    }
}

impl<Output: Element> Operator<(), Output> {
    #[must_use]
    pub fn request_scalar<'req, 'inv: 'req>(&'inv self) -> Request<'req, 'inv, Output> {
        let item = ();
        let id = DataId::new(self.id, &item);

        Request {
            type_: RequestType::Data(DataRequest {
                id,
                location: VisibleDataLocation::Ram,
                source: self,
                item: TypeErased::pack(item),
            }),
            gen_poll: Box::new(move |ctx| {
                let mut access = Some(ctx.storage.register_access(id));
                Box::new(move || unsafe {
                    access = match ctx
                        .storage
                        .read(access.take().unwrap())
                        .map(|v| *v.map(|a| &a[0]))
                    {
                        Ok(v) => return Some(v),
                        Err(access) => Some(access),
                    };
                    None
                })
            }),
            _marker: Default::default(),
        }
    }

    #[must_use]
    pub fn request_scalar_gpu<'req, 'inv: 'req>(
        &'inv self,
        gpu: DeviceId,
        dst_info: DstBarrierInfo,
    ) -> Request<'req, 'inv, gpu::ReadHandle<'req>> {
        self.request_gpu(gpu, (), dst_info)
    }
}

impl<ItemDescriptor: std::hash::Hash + 'static, Output: Element> Operator<ItemDescriptor, Output> {
    pub fn new<
        F: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, ItemDescriptor, Output>,
                Vec<ItemDescriptor>,
                &'inv (),
            ) -> Task<'cref>
            + 'static,
    >(
        id: OperatorId,
        compute: F,
    ) -> Self {
        Self::with_state(id, (), compute)
    }

    pub fn with_state<
        S: 'static,
        F: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, ItemDescriptor, Output>,
                Vec<ItemDescriptor>,
                &'inv S,
            ) -> Task<'cref>
            + 'static,
    >(
        id: OperatorId,
        state: S,
        compute: F,
    ) -> Self {
        Self {
            id,
            state: Rc::new(TypeErased::pack(state)),
            granularity: ItemGranularity::Batched,
            compute: Rc::new(move |ctx, items, state| {
                // Safety: `state` (passed as parameter to this function is precisely `S: 'op`,
                // then packed and then unpacked again here to `S: 'op`.
                let state = unsafe { state.unpack_ref() };
                compute(ctx, items, state)
            }),
        }
    }

    pub fn unbatched<
        S: 'static,
        F: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, ItemDescriptor, Output>,
                ItemDescriptor,
                &'inv S,
            ) -> Task<'cref>
            + 'static,
    >(
        id: OperatorId,
        state: S,
        compute: F,
    ) -> Self {
        Self {
            id,
            state: Rc::new(TypeErased::pack(state)),
            granularity: ItemGranularity::Single,
            compute: Rc::new(move |ctx, items, state| {
                // Safety: `state` (passed as parameter to this function is precisely `S: 'op`,
                // then packed and then unpacked again here to `S: 'op`.
                let state = unsafe { state.unpack_ref() };
                assert_eq!(items.len(), 1);
                let item = items.into_iter().next().unwrap();
                compute(ctx, item, state)
            }),
        }
    }

    #[must_use]
    pub fn request<'req, 'inv: 'req>(
        &'inv self,
        item: ItemDescriptor,
    ) -> Request<'req, 'inv, ram::ReadHandle<'req, [Output]>> {
        let id = DataId::new(self.id, &item);

        Request {
            type_: RequestType::Data(DataRequest {
                id,
                location: VisibleDataLocation::Ram,
                source: self,
                item: TypeErased::pack(item),
            }),
            gen_poll: Box::new(move |ctx| {
                let mut access = Some(ctx.storage.register_access(id));
                Box::new(move || unsafe {
                    access = match ctx.storage.read(access.take().unwrap()) {
                        Ok(v) => return Some(v),
                        Err(access) => Some(access),
                    };
                    None
                })
            }),
            _marker: Default::default(),
        }
    }

    #[must_use]
    pub fn request_inplace<'req, 'inv: 'req>(
        &'inv self,
        o_ctx: OpaqueTaskContext<'req, 'inv>,
        item: ItemDescriptor,
        write_id: OperatorId,
    ) -> Request<'req, 'inv, ram::InplaceResult<'req, Output>> {
        let read_id = DataId::new(self.id, &item);
        let write_id = DataId::new(write_id, &item);

        Request {
            type_: RequestType::Data(DataRequest {
                id: read_id,
                location: VisibleDataLocation::Ram,
                source: self,
                item: TypeErased::pack(item),
            }),
            gen_poll: Box::new(move |ctx| {
                let mut access = Some(ctx.storage.register_access(read_id));
                Box::new(move || unsafe {
                    access = match ctx.storage.try_update_inplace(
                        o_ctx,
                        access.take().unwrap(),
                        write_id,
                    ) {
                        Ok(v) => return Some(v),
                        Err(access) => Some(access),
                    };
                    None
                })
            }),
            _marker: Default::default(),
        }
    }

    #[must_use]
    pub fn request_gpu<'req, 'inv: 'req>(
        &'inv self,
        gpu: DeviceId,
        item: ItemDescriptor,
        dst_info: DstBarrierInfo,
    ) -> Request<'req, 'inv, gpu::ReadHandle<'req>> {
        let id = DataId::new(self.id, &item);

        Request {
            type_: RequestType::Data(DataRequest {
                id,
                location: VisibleDataLocation::VRam(gpu, dst_info),
                source: self,
                item: TypeErased::pack(item),
            }),
            gen_poll: Box::new(move |ctx| {
                let device = &ctx.device_contexts[gpu];
                let mut access = Some(device.storage.register_access(
                    device,
                    ctx.current_frame,
                    id,
                ));
                Box::new(
                    move || match device.storage.read(access.take().unwrap(), dst_info) {
                        Ok(r) => Some(r),
                        Err(t) => {
                            access = Some(t);
                            None
                        }
                    },
                )
            }),
            _marker: Default::default(),
        }
    }

    #[must_use]
    pub fn request_inplace_gpu<'req, 'inv: 'req>(
        &'inv self,
        gpu: DeviceId,
        item: ItemDescriptor,
        write_id: OperatorId,
        dst_info: DstBarrierInfo,
    ) -> Request<'req, 'inv, gpu::InplaceResult<'req>> {
        let read_id = DataId::new(self.id, &item);
        let write_id = DataId::new(write_id, &item);

        Request {
            type_: RequestType::Data(DataRequest {
                id: read_id,
                location: VisibleDataLocation::VRam(gpu, dst_info),
                source: self,
                item: TypeErased::pack(item),
            }),
            gen_poll: Box::new(move |ctx| {
                let device = &ctx.device_contexts[gpu];
                let mut access = Some(device.storage.register_access(
                    device,
                    ctx.current_frame,
                    read_id,
                ));
                Box::new(move || {
                    match device.storage.try_update_inplace(
                        device,
                        ctx.current_frame,
                        access.take().unwrap(),
                        write_id,
                        dst_info,
                    ) {
                        Ok(r) => Some(r),
                        Err(t) => {
                            access = Some(t);
                            None
                        }
                    }
                })
            }),
            _marker: Default::default(),
        }
    }
}

impl<ItemDescriptor: std::hash::Hash, Output: Copy> OpaqueOperator
    for Operator<ItemDescriptor, Output>
{
    fn id(&self) -> OperatorId {
        self.id
    }
    fn granularity(&self) -> ItemGranularity {
        self.granularity
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
        (self.compute)(ctx, items, &self.state)
    }
}
