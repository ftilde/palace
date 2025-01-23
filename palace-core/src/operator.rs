use std::rc::Rc;

use crate::{
    array::ChunkIndex,
    dtypes::{ConversionError, DType, ElementType, StaticElementType},
    storage::{
        gpu,
        ram::{self, RawReadHandle},
        CpuDataLocation, DataLocation, DataLongevity, Element, VisibleDataLocation,
    },
    task::{DataRequest, OpaqueTaskContext, Request, RequestType, Task, TaskContext},
    task_graph::{LocatedDataId, VisibleDataId},
    vulkan::{DeviceId, DstBarrierInfo},
};
use id::{Id, Identify};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct OperatorId(Id, pub &'static str);
impl OperatorId {
    pub fn new(name: &'static str) -> Self {
        let id = Id::from_data(name.as_bytes());
        OperatorId(id, name)
    }
    fn dependent_on(self, v: Id) -> Self {
        OperatorId(Id::combine(&[self.0, v]), self.1)
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

pub trait OperatorParameter: Identify + 'static {
    fn data_longevity(&self) -> DataLongevity;
}

impl<N: OperatorNetworkNode + Identify + 'static> OperatorParameter for N {
    fn data_longevity(&self) -> DataLongevity {
        self.descriptor().data_longevity
    }
}

#[derive(Identify)]
pub struct DataParam<I: Identify>(pub I);

impl<I: Identify + 'static> OperatorParameter for DataParam<I> {
    fn data_longevity(&self) -> DataLongevity {
        DataLongevity::Stable
    }
}

impl<I: Identify> std::ops::Deref for DataParam<I> {
    type Target = I;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl OperatorParameter for () {
    fn data_longevity(&self) -> DataLongevity {
        DataLongevity::Stable
    }
}

impl<const N: usize, E: OperatorParameter> OperatorParameter for [E; N] {
    fn data_longevity(&self) -> DataLongevity {
        self.iter()
            .map(|v| v.data_longevity())
            .min()
            .unwrap_or(DataLongevity::Stable)
    }
}

macro_rules! impl_for_tuples {
    ( ) => {};
    ( $first:ident, $( $rest:ident, )* ) => {
        // Recursion
        impl_for_tuples!($( $rest, )*);

        #[allow(non_snake_case, unused_mut)]
        impl<$first: OperatorParameter, $( $rest: OperatorParameter ),*> OperatorParameter for ($first, $( $rest, )*) {
            fn data_longevity(&self) -> DataLongevity {
                let ($first, $( $rest, )*) = self;
                let mut longevity = $first.data_longevity();
                $( longevity = longevity.min($rest.data_longevity()); )*
                longevity
            }
        }
    };
}

impl_for_tuples!(I1, I2, I3, I4, I5, I6, I7, I8, I9, I10,);

#[derive(Clone, Copy)]
pub struct OperatorDescriptor {
    pub id: OperatorId,
    pub data_longevity: DataLongevity,
    pub cache_results: bool,
}

#[macro_export]
macro_rules! op_descriptor {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let name = type_name_of(f)
            .rsplit("::")
            .find(|&part| part != "f" && part != "{{closure}}")
            .unwrap();
        OperatorDescriptor::with_name(name)
    }};
}
pub use op_descriptor;

impl OperatorDescriptor {
    #[track_caller]
    pub fn with_name(name: &'static str) -> Self {
        let id = Id::combine(&[Id::from_data(name.as_bytes()), Id::source_file_location()]);
        let id = OperatorId(id, name);

        Self {
            id,
            data_longevity: DataLongevity::Stable,
            cache_results: false,
        }
    }
    pub fn dependent_on(self, v: &dyn OperatorNetworkNode) -> Self {
        let d = v.descriptor();
        Self {
            id: self.id.dependent_on(d.id.into()),
            data_longevity: self.data_longevity.min(d.data_longevity),
            cache_results: self.cache_results,
        }
    }
    pub fn dependent_on_data(self, v: &(impl Identify + ?Sized)) -> Self {
        Self {
            id: self.id.dependent_on(v.id()),
            data_longevity: self.data_longevity,
            cache_results: self.cache_results,
        }
    }
    pub fn data_longevity(mut self, data_longevity: DataLongevity) -> Self {
        self.data_longevity = data_longevity;
        self
    }
    pub fn cache_results(mut self, cache_results: bool) -> Self {
        self.cache_results = cache_results;
        self
    }
    pub fn ephemeral(self) -> Self {
        self.data_longevity(DataLongevity::Ephemeral)
    }
    pub fn unstable(self) -> Self {
        self.data_longevity(DataLongevity::Unstable)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct DataId(pub Id);

impl DataId {
    pub fn new(op: OperatorId, chunk: ChunkIndex) -> Self {
        DataId(Id::combine(&[op.inner(), chunk.0.id()]))
    }

    pub fn in_location(self, location: DataLocation) -> LocatedDataId {
        LocatedDataId { id: self, location }
    }
    pub fn with_visibility(self, location: VisibleDataLocation) -> VisibleDataId {
        VisibleDataId { id: self, location }
    }
}

#[derive(Copy, Clone)]
pub struct DataDescriptor {
    pub id: DataId,
    pub longevity: DataLongevity,
}

impl DataDescriptor {
    pub fn new(op: OperatorDescriptor, chunk: ChunkIndex) -> Self {
        Self {
            longevity: op.data_longevity,
            id: DataId::new(op.id, chunk),
        }
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

pub type ComputeFunction<Output> = Rc<
    dyn for<'cref, 'inv> Fn(
        TaskContext<'cref, 'inv, Output>,
        Vec<(ChunkIndex, DataLocation)>,
        &'inv TypeErased,
    ) -> Task<'cref>,
>;

#[derive(Copy, Clone)]
pub enum ItemGranularity {
    Single,
    Batched,
}

pub trait OperatorNetworkNode {
    fn descriptor(&self) -> OperatorDescriptor;
}

impl<O> OperatorNetworkNode for Operator<O> {
    fn descriptor(&self) -> OperatorDescriptor {
        self.descriptor
    }
}

pub trait OpaqueOperator {
    fn op_id(&self) -> OperatorId;
    fn longevity(&self) -> DataLongevity;
    fn cache_results(&self) -> bool;
    fn operator_descriptor(&self) -> OperatorDescriptor {
        OperatorDescriptor {
            id: self.op_id(),
            data_longevity: self.longevity(),
            cache_results: self.cache_results(),
        }
    }
    fn granularity(&self) -> ItemGranularity;
    unsafe fn compute<'cref, 'inv>(
        &'inv self,
        context: OpaqueTaskContext<'cref, 'inv>,
        items: Vec<(ChunkIndex, DataLocation)>,
    ) -> Task<'cref>;
}

pub struct Operator<OutputType> {
    descriptor: OperatorDescriptor,
    state: Rc<TypeErased>,
    granularity: ItemGranularity,
    compute: ComputeFunction<OutputType>,
    dtype: OutputType,
}
impl<OutputType: Clone> Clone for Operator<OutputType> {
    fn clone(&self) -> Self {
        Self {
            descriptor: self.descriptor.clone(),
            state: self.state.clone(),
            granularity: self.granularity.clone(),
            compute: self.compute.clone(),
            dtype: self.dtype.clone(),
        }
    }
}
impl<OutputType: Clone> Operator<OutputType> {
    pub fn dtype(&self) -> OutputType {
        self.dtype.clone()
    }
}

impl<Output: Element> Operator<StaticElementType<Output>> {
    #[must_use]
    pub fn request_scalar<'req, 'inv: 'req>(&'inv self) -> Request<'req, 'inv, Output> {
        let item = ChunkIndex(0);
        let id = DataId::new(self.op_id(), item);

        Request {
            type_: RequestType::Data(DataRequest::new(
                id,
                VisibleDataLocation::CPU(CpuDataLocation::Ram),
                self,
                item,
            )),
            gen_poll: Box::new(move |ctx| {
                let mut access = Some(ctx.storage.register_access(ctx.current_frame, id));
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
        self.request_gpu(gpu, ChunkIndex(0), dst_info)
    }
}

impl<Output: Element> Operator<StaticElementType<Output>> {
    #[must_use]
    pub fn request<'req, 'inv: 'req>(
        &'inv self,
        item: ChunkIndex,
    ) -> Request<'req, 'inv, ram::ReadHandle<'req, [Output]>> {
        let id = DataId::new(self.op_id(), item);

        Request {
            type_: RequestType::Data(DataRequest::new(
                id,
                VisibleDataLocation::CPU(CpuDataLocation::Ram),
                self,
                item,
            )),
            gen_poll: Box::new(move |ctx| {
                let mut access = Some(ctx.storage.register_access(ctx.current_frame, id));
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
        item: ChunkIndex,
        write_id: OperatorDescriptor,
    ) -> Request<'req, 'inv, ram::InplaceResult<'req, 'inv, Output>> {
        let read_id = DataId::new(self.op_id(), item);
        let write_desc = DataDescriptor::new(write_id, item);

        Request {
            type_: RequestType::Data(DataRequest::new(
                read_id,
                VisibleDataLocation::CPU(CpuDataLocation::Ram),
                self,
                item,
            )),
            gen_poll: Box::new(move |ctx| {
                let mut access = Some(ctx.storage.register_access(ctx.current_frame, read_id));
                Box::new(move || unsafe {
                    access = match ctx.storage.try_update_inplace(
                        o_ctx,
                        access.take().unwrap(),
                        write_desc,
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

    // I don't think we actually want to expose this, since it is not guaranteed that we have a
    // disk cache
    //#[must_use]
    //pub fn request_disk<'req, 'inv: 'req>(
    //    &'inv self,
    //    item: ItemDescriptor,
    //) -> Request<'req, 'inv, disk::ReadHandle<'req, [Output]>> {
    //    let id = DataId::new(self.id(), &item);

    //    Request {
    //        type_: RequestType::Data(DataRequest {
    //            id,
    //            location: VisibleDataLocation::CPU(CpuDataLocation::Disk),
    //            source: self,
    //            item: TypeErased::pack(item),
    //        }),
    //        gen_poll: Box::new(move |ctx| {
    //            let mut access = Some(ctx.disk_cache.unwrap().register_access(id));
    //            Box::new(move || unsafe {
    //                access = match ctx.disk_cache.unwrap().read(access.take().unwrap()) {
    //                    Ok(v) => return Some(v),
    //                    Err(access) => Some(access),
    //                };
    //                None
    //            })
    //        }),
    //        _marker: Default::default(),
    //    }
    //}
}

impl<OutputType: ElementType> Operator<OutputType> {
    pub fn new(
        descriptor: OperatorDescriptor,
        dtype: OutputType,
        compute: for<'cref, 'inv> fn(
            TaskContext<'cref, 'inv, OutputType>,
            Vec<(ChunkIndex, DataLocation)>, //DataLocation is only a hint
            &'inv (),
        ) -> Task<'cref>,
    ) -> Self {
        Self::with_state(descriptor, dtype, (), compute)
    }

    pub fn with_state<S: OperatorParameter>(
        mut descriptor: OperatorDescriptor,
        dtype: OutputType,
        state: S,
        compute: for<'cref, 'inv> fn(
            TaskContext<'cref, 'inv, OutputType>,
            Vec<(ChunkIndex, DataLocation)>, //DataLocation is only a hint
            &'inv S,
        ) -> Task<'cref>,
    ) -> Self {
        descriptor.data_longevity = descriptor.data_longevity.min(state.data_longevity());
        descriptor.id = descriptor.id.dependent_on(state.id());
        Self {
            descriptor,
            dtype,
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

    pub fn unbatched<S: OperatorParameter>(
        mut descriptor: OperatorDescriptor,
        dtype: OutputType,
        state: S,
        compute: for<'cref, 'inv> fn(
            TaskContext<'cref, 'inv, OutputType>,
            ChunkIndex,
            DataLocation, //DataLocation is only a hint
            &'inv S,
        ) -> Task<'cref>,
    ) -> Self {
        descriptor.data_longevity = descriptor.data_longevity.min(state.data_longevity());
        descriptor.id = descriptor.id.dependent_on(state.id());
        Self {
            descriptor,
            dtype,
            state: Rc::new(TypeErased::pack(state)),
            granularity: ItemGranularity::Single,
            compute: Rc::new(move |ctx, items, state| {
                // Safety: `state` (passed as parameter to this function is precisely `S: 'op`,
                // then packed and then unpacked again here to `S: 'op`.
                let state = unsafe { state.unpack_ref() };
                assert_eq!(items.len(), 1);
                let item = items.into_iter().next().unwrap();
                compute(ctx, item.0, item.1, state)
            }),
        }
    }

    #[must_use]
    pub fn request_raw<'req, 'inv: 'req>(
        &'inv self,
        item: ChunkIndex,
    ) -> Request<'req, 'inv, RawReadHandle<'req>> {
        let id = DataId::new(self.op_id(), item);

        Request {
            type_: RequestType::Data(DataRequest::new(
                id,
                VisibleDataLocation::CPU(CpuDataLocation::Ram),
                self,
                item,
            )),
            gen_poll: Box::new(move |ctx| {
                let mut access = Some(ctx.storage.register_access(ctx.current_frame, id));
                Box::new(move || {
                    access = match ctx.storage.read_raw(access.take().unwrap()) {
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
        item: ChunkIndex,
        dst_info: DstBarrierInfo,
    ) -> Request<'req, 'inv, gpu::ReadHandle<'req>> {
        let id = DataId::new(self.op_id(), item);

        Request {
            type_: RequestType::Data(DataRequest::new(
                id,
                VisibleDataLocation::GPU(gpu, dst_info),
                self,
                item,
            )),
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
        item: ChunkIndex,
        write_id: OperatorDescriptor,
        write_dtype: DType,
        num_elements: usize,
        dst_info: DstBarrierInfo,
    ) -> Request<'req, 'inv, gpu::InplaceResult<'req, 'inv>> {
        let write_id = DataDescriptor::new(write_id, item);
        let read_id = DataId::new(self.op_id(), item);

        Request {
            type_: RequestType::Data(DataRequest::new(
                read_id,
                VisibleDataLocation::GPU(gpu, dst_info),
                self,
                item,
            )),
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
                        num_elements,
                        dst_info,
                        write_dtype,
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

impl<OutputType> Identify for Operator<OutputType> {
    fn id(&self) -> Id {
        self.descriptor.id.into()
    }
}

impl<T> TryFrom<Operator<DType>> for Operator<StaticElementType<T>>
where
    StaticElementType<T>: TryFrom<DType, Error = ConversionError>,
{
    fn try_from(value: Operator<DType>) -> Result<Self, ConversionError> {
        let new_dtype = value.dtype.try_into()?;
        let old_dtype = value.dtype;
        Ok(Operator {
            descriptor: value.descriptor,
            state: value.state,
            granularity: value.granularity,
            compute: Rc::new(move |ctx, items, tr| {
                (value.compute)(unsafe { TaskContext::new(*ctx, old_dtype) }, items, tr)
            }),
            dtype: new_dtype,
        })
    }

    type Error = ConversionError;
}

impl<T: 'static> From<Operator<StaticElementType<T>>> for Operator<DType>
where
    DType: From<StaticElementType<T>>,
{
    fn from(value: Operator<StaticElementType<T>>) -> Self {
        let new_dtype = value.dtype.into();
        let old_dtype = value.dtype;
        Operator {
            descriptor: value.descriptor,
            state: value.state,
            granularity: value.granularity,
            compute: Rc::new(move |ctx, items, tr| {
                (value.compute)(unsafe { TaskContext::new(*ctx, old_dtype) }, items, tr)
            }),
            dtype: new_dtype,
        }
    }
}

impl Operator<DType> {
    // Note: Panics if alignment of dtypes is not compatible
    pub fn reinterpret_dtype(self, new_dtype: DType) -> Self {
        let old_dtype = self.dtype();
        assert_eq!(
            old_dtype.element_layout().align(),
            new_dtype.element_layout().align()
        );
        Operator {
            descriptor: self.descriptor,
            state: self.state,
            granularity: self.granularity,
            compute: Rc::new(move |ctx, items, tr| {
                (self.compute)(unsafe { TaskContext::new(*ctx, old_dtype) }, items, tr)
            }),
            dtype: new_dtype,
        }
    }
}

impl<OutputType: Clone> OpaqueOperator for Operator<OutputType> {
    fn op_id(&self) -> OperatorId {
        self.descriptor.id
    }
    fn longevity(&self) -> DataLongevity {
        self.descriptor.data_longevity
    }
    fn granularity(&self) -> ItemGranularity {
        self.granularity
    }
    fn cache_results(&self) -> bool {
        self.descriptor.cache_results
    }
    unsafe fn compute<'cref, 'inv>(
        &'inv self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        items: Vec<(ChunkIndex, DataLocation)>,
    ) -> Task<'cref> {
        let ctx = TaskContext::new(ctx, self.dtype.clone());
        (self.compute)(ctx, items, &self.state)
    }
}

pub fn cache<'op, OutputType>(mut input: Operator<OutputType>) -> Operator<OutputType> {
    input.descriptor.cache_results = true;
    input
}
