use std::hash::Hash;
use std::rc::Rc;

use futures::StreamExt;

use crate::{
    array::{ChunkIndex, TensorEmbeddingData, TensorMetaData},
    data::{GlobalCoordinate, LocalCoordinate, Vector},
    dim::*,
    dtypes::{ConversionError, DType, ElementType, StaticElementType},
    op_descriptor,
    operator::{DataParam, Operator, OperatorDescriptor, OperatorNetworkNode, OperatorParameter},
    storage::{
        cpu::{InplaceHandle, ThreadInplaceHandle},
        DataLocation, Element,
    },
    task::{RequestStream, Task, TaskContext},
};
use id::{Identify, IdentifyHash};

#[derive(Clone, Identify)]
pub struct TensorOperator<D: DynDimension, E> {
    pub metadata: TensorMetaData<D>,
    pub chunks: Operator<E>,
}

impl<D: DynDimension, E> PartialEq for TensorOperator<D, E> {
    fn eq(&self, other: &Self) -> bool {
        self.chunks.id() == other.chunks.id()
    }
}
impl<D: DynDimension, E> Eq for TensorOperator<D, E> {}
impl<D: DynDimension, E> Hash for TensorOperator<D, E> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u128(self.chunks.id().raw())
    }
}

impl<D: DynDimension, E> OperatorNetworkNode for TensorOperator<D, E> {
    fn descriptor(&self) -> OperatorDescriptor {
        self.chunks.descriptor().dependent_on_data(&self.metadata)
    }
}

impl<D: DynDimension, E: ElementType> TensorOperator<D, E> {
    pub fn new(
        descriptor: OperatorDescriptor,
        dtype: E,
        metadata: TensorMetaData<D>,
        chunks: for<'cref, 'inv> fn(
            TaskContext<'cref, 'inv, E>,
            Vec<(ChunkIndex, DataLocation)>,
            &'inv (),
        ) -> Task<'cref>,
    ) -> Self {
        Self::with_state(descriptor, dtype, metadata, (), chunks)
    }

    pub fn with_state<SB: OperatorParameter>(
        descriptor: OperatorDescriptor,
        dtype: E,
        metadata: TensorMetaData<D>,
        state_chunks: SB,
        chunks: for<'cref, 'inv> fn(
            TaskContext<'cref, 'inv, E>,
            Vec<(ChunkIndex, DataLocation)>,
            &'inv SB,
        ) -> Task<'cref>,
    ) -> Self {
        Self {
            metadata,
            chunks: Operator::with_state(descriptor, dtype, state_chunks, chunks),
        }
    }

    pub fn unbatched<SB: OperatorParameter>(
        descriptor: OperatorDescriptor,
        dtype: E,
        metadata: TensorMetaData<D>,
        state_chunks: SB,
        chunks: for<'cref, 'inv> fn(
            TaskContext<'cref, 'inv, E>,
            ChunkIndex,
            DataLocation,
            &'inv SB,
        ) -> Task<'cref>,
    ) -> Self {
        Self {
            metadata,
            chunks: Operator::unbatched(descriptor, dtype, state_chunks, chunks),
        }
    }
    pub fn dtype(&self) -> E {
        self.chunks.dtype()
    }
}

impl<D: DynDimension, E> TensorOperator<D, E> {
    pub fn embedded(self, data: TensorEmbeddingData<D>) -> EmbeddedTensorOperator<D, E> {
        EmbeddedTensorOperator {
            inner: self,
            embedding_data: data,
        }
    }

    pub fn cache(self) -> Self {
        Self {
            metadata: self.metadata,
            chunks: crate::operator::cache(self.chunks),
        }
    }

    pub fn dim(&self) -> D {
        self.metadata.dim()
    }
}

impl<D: DynDimension, E> TensorOperator<D, E> {
    pub fn into_dyn(self) -> TensorOperator<DDyn, E> {
        TensorOperator {
            metadata: self.metadata.into_dyn(),
            chunks: self.chunks,
        }
    }

    pub fn try_into_static<DS: Dimension>(self) -> Option<TensorOperator<DS, E>> {
        Some(TensorOperator {
            metadata: self.metadata.try_into_static()?,
            chunks: self.chunks,
        })
    }
}

impl<D: DynDimension, T> TryFrom<TensorOperator<D, DType>>
    for TensorOperator<D, StaticElementType<T>>
where
    StaticElementType<T>: TryFrom<DType, Error = ConversionError>,
{
    fn try_from(value: TensorOperator<D, DType>) -> Result<Self, ConversionError> {
        Ok(Self {
            metadata: value.metadata,
            chunks: value.chunks.try_into()?,
        })
    }

    type Error = ConversionError;
}

impl<D: DynDimension, T: 'static> From<TensorOperator<D, StaticElementType<T>>>
    for TensorOperator<D, DType>
where
    DType: From<StaticElementType<T>>,
{
    fn from(value: TensorOperator<D, StaticElementType<T>>) -> Self {
        Self {
            metadata: value.metadata,
            chunks: value.chunks.into(),
        }
    }
}

impl<D: DynDimension, E: Element + Identify> TensorOperator<D, StaticElementType<E>> {
    pub fn from_static(
        size: Vector<D, GlobalCoordinate>,
        values: &'static [E],
    ) -> Result<TensorOperator<D, StaticElementType<E>>, crate::Error> {
        let m = TensorMetaData {
            dimensions: size.clone(),
            chunk_size: size.map(LocalCoordinate::interpret_as),
        };
        let n_elem = size.hmul();
        if n_elem != values.len() {
            return Err(format!(
                "Tensor ({}) and data ({}) size do not match",
                n_elem,
                values.len()
            )
            .into());
        }
        Ok(TensorOperator::with_state(
            op_descriptor!(),
            Default::default(),
            m,
            DataParam(values),
            move |ctx, _, values| {
                async move {
                    let mut out = ctx
                        .submit(ctx.alloc_slot_num_elements(ChunkIndex(0), values.len()))
                        .await;
                    let mut out_data = &mut *out;
                    let values: &[E] = &values;
                    ctx.submit(ctx.spawn_compute(move || {
                        crate::data::write_slice_uninit(&mut out_data, values);
                    }))
                    .await;

                    // Safety: slot and values are of the exact same size. Thus all values are
                    // initialized.
                    unsafe { out.initialized(*ctx) };
                    Ok(())
                }
                .into()
            },
        ))
    }

    pub fn from_vec(
        size: Vector<D, GlobalCoordinate>,
        values: Vec<E>,
    ) -> Result<TensorOperator<D, StaticElementType<E>>, crate::Error> {
        Self::from_rc(size, values.into())
    }

    pub fn from_rc(
        size: Vector<D, GlobalCoordinate>,
        values: Rc<[E]>,
    ) -> Result<TensorOperator<D, StaticElementType<E>>, crate::Error> {
        let m = TensorMetaData {
            dimensions: size.clone(),
            chunk_size: size.map(LocalCoordinate::interpret_as),
        };
        let n_elem = size.hmul();
        if n_elem != values.len() {
            return Err(format!(
                "Tensor ({}) and data ({}) size do not match",
                n_elem,
                values.len()
            )
            .into());
        }
        Ok(TensorOperator::with_state(
            op_descriptor!(),
            Default::default(),
            m,
            DataParam(values),
            move |ctx, _, values| {
                async move {
                    let mut out = ctx
                        .submit(ctx.alloc_slot_num_elements(ChunkIndex(0), values.len()))
                        .await;
                    let mut out_data = &mut *out;
                    let values: &[E] = &values;
                    ctx.submit(ctx.spawn_compute(move || {
                        crate::data::write_slice_uninit(&mut out_data, values);
                    }))
                    .await;

                    // Safety: slot and values are of the exact same size. Thus all values are
                    // initialized.
                    unsafe { out.initialized(*ctx) };
                    Ok(())
                }
                .into()
            },
        ))
    }
}

impl<D: LargerDim> TensorOperator<D, DType> {
    pub fn unfold_dtype(self) -> Result<TensorOperator<D::Larger, DType>, crate::Error> {
        let dtype = self.dtype();
        if dtype.is_scalar() {
            Err(format!("Tensor already has scalar type {}", dtype).into())
        } else {
            let vec_size = dtype.size;
            let new_md = self
                .metadata
                .clone()
                .push_dim_small(vec_size.into(), vec_size.into());
            let new_dtype = DType::scalar(dtype.scalar);
            Ok(TensorOperator {
                metadata: new_md,
                chunks: self.chunks.reinterpret_dtype(new_dtype),
            })
        }
    }
}

impl<D: SmallerDim> TensorOperator<D, DType> {
    pub fn fold_into_dtype(self) -> Result<TensorOperator<D::Smaller, DType>, crate::Error> {
        let dtype = self.dtype();

        let chunk_dim = self.metadata.dimension_in_chunks().raw();
        let outer_dim = self.metadata.chunk_size.small_dim_element().raw;
        if chunk_dim.small_dim_element() != 1 {
            Err(format!(
                "Tensor consist of single chunk in last dimension, but has dimension-in-chunks {:?}",
                chunk_dim
            )
            .into())
        } else {
            let new_md = self.metadata.clone().pop_dim_small();
            let new_dtype = dtype.vectorize(outer_dim);
            Ok(TensorOperator {
                metadata: new_md,
                chunks: self.chunks.reinterpret_dtype(new_dtype),
            })
        }
    }
}

#[derive(Clone, Identify)]
pub struct EmbeddedTensorOperator<D: DynDimension, E> {
    pub inner: TensorOperator<D, E>,
    pub embedding_data: TensorEmbeddingData<D>,
}

impl<D: DynDimension, E> OperatorNetworkNode for EmbeddedTensorOperator<D, E> {
    fn descriptor(&self) -> OperatorDescriptor {
        self.chunks
            .descriptor()
            .dependent_on_data(&self.embedding_data)
    }
}

impl<D: DynDimension, E> std::ops::Deref for EmbeddedTensorOperator<D, E> {
    type Target = TensorOperator<D, E>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<D: DynDimension, E> std::ops::DerefMut for EmbeddedTensorOperator<D, E> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<D: DynDimension, E> Into<TensorOperator<D, E>> for EmbeddedTensorOperator<D, E> {
    fn into(self) -> TensorOperator<D, E> {
        self.inner
    }
}

impl<D: DynDimension, T> TryFrom<EmbeddedTensorOperator<D, DType>>
    for EmbeddedTensorOperator<D, StaticElementType<T>>
where
    StaticElementType<T>: TryFrom<DType, Error = ConversionError>,
{
    fn try_from(value: EmbeddedTensorOperator<D, DType>) -> Result<Self, ConversionError> {
        Ok(Self {
            inner: value.inner.try_into()?,
            embedding_data: value.embedding_data,
        })
    }

    type Error = ConversionError;
}

impl<D: DynDimension, T: 'static> From<EmbeddedTensorOperator<D, StaticElementType<T>>>
    for EmbeddedTensorOperator<D, DType>
where
    DType: From<StaticElementType<T>>,
{
    fn from(value: EmbeddedTensorOperator<D, StaticElementType<T>>) -> Self {
        Self {
            inner: value.inner.into(),
            embedding_data: value.embedding_data,
        }
    }
}

impl<D: DynDimension, E> EmbeddedTensorOperator<D, E> {
    pub fn single_level_lod(self) -> LODTensorOperator<D, E> {
        LODTensorOperator { levels: vec![self] }
    }

    pub fn map_inner<O>(
        self,
        f: impl FnOnce(TensorOperator<D, E>) -> TensorOperator<D, O>,
    ) -> EmbeddedTensorOperator<D, O> {
        EmbeddedTensorOperator {
            inner: f(self.inner),
            embedding_data: self.embedding_data,
        }
    }

    pub fn cache(self) -> Self {
        self.map_inner(|t| t.cache())
    }
}

impl<D: DynDimension, E> EmbeddedTensorOperator<D, E> {
    pub fn into_dyn(self) -> EmbeddedTensorOperator<DDyn, E> {
        EmbeddedTensorOperator {
            inner: self.inner.into_dyn(),
            embedding_data: self.embedding_data.into_dyn(),
        }
    }

    pub fn try_into_static<DS: Dimension>(self) -> Option<EmbeddedTensorOperator<DS, E>> {
        Some(EmbeddedTensorOperator {
            inner: self.inner.try_into_static()?,
            embedding_data: self.embedding_data.try_into_static()?,
        })
    }
}

impl<D: Dimension, E: ElementType> EmbeddedTensorOperator<D, E> {
    pub fn real_dimensions(&self) -> Vector<D, f32> {
        self.embedding_data.spacing * self.metadata.dimensions.raw().f32()
    }
}

#[derive(Clone, Identify)]
pub struct LODTensorOperator<D: DynDimension, E> {
    pub levels: Vec<EmbeddedTensorOperator<D, E>>,
}

impl<D: DynDimension, E> OperatorNetworkNode for LODTensorOperator<D, E> {
    fn descriptor(&self) -> OperatorDescriptor {
        let mut d = self.levels[0].descriptor();
        for l in &self.levels[1..] {
            d = d.dependent_on(l);
        }
        d
    }
}

impl<D: DynDimension, T> TryFrom<LODTensorOperator<D, DType>>
    for LODTensorOperator<D, StaticElementType<T>>
where
    StaticElementType<T>: TryFrom<DType, Error = ConversionError>,
{
    fn try_from(value: LODTensorOperator<D, DType>) -> Result<Self, ConversionError> {
        Ok(Self {
            levels: value
                .levels
                .into_iter()
                .map(|v| v.try_into())
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    type Error = ConversionError;
}

impl<D: DynDimension, T: 'static> From<LODTensorOperator<D, StaticElementType<T>>>
    for LODTensorOperator<D, DType>
where
    DType: From<StaticElementType<T>>,
{
    fn from(value: LODTensorOperator<D, StaticElementType<T>>) -> Self {
        Self {
            levels: value.levels.into_iter().map(|v| v.into()).collect(),
        }
    }
}

impl<D: DynDimension, E> LODTensorOperator<D, E> {
    pub fn fine_metadata(&self) -> TensorMetaData<D> {
        self.levels[0].metadata.clone()
    }
    pub fn fine_embedding_data(&self) -> TensorEmbeddingData<D> {
        self.levels[0].embedding_data.clone()
    }

    pub fn into_dyn(self) -> LODTensorOperator<DDyn, E> {
        LODTensorOperator {
            levels: self.levels.into_iter().map(|l| l.into_dyn()).collect(),
        }
    }

    pub fn try_into_static<DS: Dimension>(self) -> Option<LODTensorOperator<DS, E>> {
        Some(LODTensorOperator {
            levels: self
                .levels
                .into_iter()
                .map(|l| l.try_into_static::<DS>())
                .collect::<Option<Vec<_>>>()?,
        })
    }
}

impl<D: DynDimension, E> LODTensorOperator<D, E> {
    pub fn map<DO: DynDimension, EO>(
        self,
        f: impl FnMut(EmbeddedTensorOperator<D, E>) -> EmbeddedTensorOperator<DO, EO>,
    ) -> LODTensorOperator<DO, EO> {
        LODTensorOperator {
            levels: self.levels.into_iter().map(f).collect(),
        }
    }

    pub fn cache_coarse_levels(self) -> Self {
        LODTensorOperator {
            levels: self
                .levels
                .into_iter()
                .enumerate()
                .map(|(i, level)| if i != 0 { level.cache() } else { level })
                .collect(),
        }
    }
}

#[allow(unused)]
pub async fn map_values_inplace<
    'op,
    'cref,
    'inv,
    E: Element,
    F: Fn(E) -> E + Send + Copy + 'static,
>(
    ctx: TaskContext<'cref, 'inv, StaticElementType<E>>,
    input: &'op Operator<StaticElementType<E>>,
    positions: Vec<(ChunkIndex, DataLocation)>,
    f: F,
) where
    'op: 'inv,
{
    let requests = positions
        .into_iter()
        .map(|(pos, _)| input.request_inplace(*ctx, pos, ctx.current_op_desc().unwrap()));

    let stream = ctx
        .submit_unordered(requests)
        .then_req(ctx.into(), |brick_handle| brick_handle.alloc())
        .then_req(ctx.into(), |brick_handle| {
            let mut brick_handle = brick_handle.into_thread_handle();
            ctx.spawn_compute(move || {
                match &mut brick_handle {
                    ThreadInplaceHandle::Inplace(ref mut rw) => {
                        for v in rw.iter_mut() {
                            *v = f(*v);
                        }
                    }
                    ThreadInplaceHandle::New(r, ref mut w) => {
                        for (i, o) in r.iter().zip(w.iter_mut()) {
                            o.write(f(*i));
                        }
                    }
                }
                brick_handle
            })
        });

    futures::pin_mut!(stream);
    // Drive the stream until completion
    while let Some(brick_handle) = stream.next().await {
        let brick_handle = brick_handle.into_main_handle(*ctx);
        if let InplaceHandle::New(_, w) = brick_handle {
            // Safety: We have written all values in the above closure executed on
            // the thread pool.
            unsafe { w.initialized(*ctx) };
        };
    }
}

#[allow(unused)]
pub fn map<D: DynDimension, E: Element>(
    input: TensorOperator<D, StaticElementType<E>>,
    f: fn(E) -> E,
) -> TensorOperator<D, StaticElementType<E>> {
    TensorOperator::with_state(
        op_descriptor!(),
        Default::default(),
        input.metadata.clone(),
        (input, DataParam(IdentifyHash(f))),
        move |ctx, positions, (input, f)| {
            async move {
                map_values_inplace(ctx, &input.chunks, positions, ***f).await;

                Ok(())
            }
            .into()
        },
    )
}

#[allow(unused)]
pub fn linear_rescale<D: DynDimension>(
    input: TensorOperator<D, StaticElementType<f32>>,
    factor: f32,
    offset: f32,
) -> TensorOperator<D, StaticElementType<f32>> {
    TensorOperator::with_state(
        op_descriptor!(),
        Default::default(),
        input.metadata.clone(),
        (input, DataParam(factor), DataParam(offset)),
        move |ctx, positions, (input, factor, offset)| {
            let factor = **factor;
            let offset = **offset;
            async move {
                map_values_inplace(ctx, &input.chunks, positions, move |i| i * factor + offset)
                    .await;

                Ok(())
            }
            .into()
        },
    )
}

pub type ImageOperator<E> = TensorOperator<D2, E>;
pub type LODImageOperator<E> = LODTensorOperator<D2, E>;
pub type FrameOperator = ImageOperator<StaticElementType<Vector<D4, u8>>>;
