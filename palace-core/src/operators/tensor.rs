use std::rc::Rc;

use futures::StreamExt;

use crate::{
    array::{TensorEmbeddingData, TensorMetaData},
    data::{ChunkCoordinate, GlobalCoordinate, LocalCoordinate, Vector},
    dim::*,
    dtypes::{ElementType, StaticElementType},
    operator::{Operator, OperatorDescriptor, OperatorNetworkNode},
    storage::{
        cpu::{InplaceHandle, ThreadInplaceHandle},
        DataLocation, Element,
    },
    task::{RequestStream, Task, TaskContext},
};
use id::Identify;

#[derive(Clone, Identify)]
pub struct TensorOperator<D: Dimension, E> {
    pub metadata: TensorMetaData<D>,
    pub chunks: Operator<Vector<D, ChunkCoordinate>, E>,
}

impl<D: Dimension, E> OperatorNetworkNode for TensorOperator<D, E> {
    fn descriptor(&self) -> OperatorDescriptor {
        self.chunks.descriptor().dependent_on_data(&self.metadata)
    }
}

impl<D: Dimension, E: ElementType> TensorOperator<D, E> {
    pub fn new<
        B: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, Vector<D, ChunkCoordinate>, E>,
                Vec<(Vector<D, ChunkCoordinate>, DataLocation)>,
                &'inv (),
            ) -> Task<'cref>
            + 'static,
    >(
        descriptor: OperatorDescriptor,
        dtype: E,
        metadata: TensorMetaData<D>,
        chunks: B,
    ) -> Self {
        Self::with_state(descriptor, dtype, metadata, (), chunks)
    }

    pub fn with_state<
        SB: 'static,
        B: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, Vector<D, ChunkCoordinate>, E>,
                Vec<(Vector<D, ChunkCoordinate>, DataLocation)>,
                &'inv SB,
            ) -> Task<'cref>
            + 'static,
    >(
        descriptor: OperatorDescriptor,
        dtype: E,
        metadata: TensorMetaData<D>,
        state_chunks: SB,
        chunks: B,
    ) -> Self {
        Self {
            metadata,
            chunks: Operator::with_state(descriptor, dtype, state_chunks, chunks),
        }
    }

    pub fn unbatched<
        SB: 'static,
        B: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, Vector<D, ChunkCoordinate>, E>,
                Vector<D, ChunkCoordinate>,
                DataLocation,
                &'inv SB,
            ) -> Task<'cref>
            + 'static,
    >(
        descriptor: OperatorDescriptor,
        dtype: E,
        metadata: TensorMetaData<D>,
        state_chunks: SB,
        chunks: B,
    ) -> Self {
        Self {
            metadata,
            chunks: Operator::unbatched(descriptor, dtype, state_chunks, chunks),
        }
    }

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
}

impl<D: Dimension, E: Element + Identify> TensorOperator<D, StaticElementType<E>> {
    pub fn from_static(
        size: Vector<D, GlobalCoordinate>,
        values: &'static [E],
    ) -> Result<TensorOperator<D, StaticElementType<E>>, crate::Error> {
        let m = TensorMetaData {
            dimensions: size,
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
            OperatorDescriptor::new("tensor_from_static")
                .dependent_on_data(&size)
                .dependent_on_data(values), //TODO: this is a performance problem for
            Default::default(),
            m,
            values,
            move |ctx, _, values| {
                async move {
                    let mut out =
                        ctx.submit(ctx.alloc_slot(
                            Vector::<D, ChunkCoordinate>::fill(0.into()),
                            values.len(),
                        ))
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

    pub fn from_rc(
        size: Vector<D, GlobalCoordinate>,
        values: Rc<[E]>,
    ) -> Result<TensorOperator<D, StaticElementType<E>>, crate::Error> {
        let m = TensorMetaData {
            dimensions: size,
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
            OperatorDescriptor::new("tensor_from_static")
                .dependent_on_data(&size)
                .dependent_on_data(&values[..]), //TODO: this is a performance problem for large arrays
            Default::default(),
            m,
            values,
            move |ctx, _, values| {
                async move {
                    let mut out =
                        ctx.submit(ctx.alloc_slot(
                            Vector::<D, ChunkCoordinate>::fill(0.into()),
                            values.len(),
                        ))
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

#[derive(Clone)]
pub struct EmbeddedTensorOperator<D: Dimension, E> {
    pub inner: TensorOperator<D, E>,
    pub embedding_data: TensorEmbeddingData<D>,
}

impl<D: Dimension, E> OperatorNetworkNode for EmbeddedTensorOperator<D, E> {
    fn descriptor(&self) -> OperatorDescriptor {
        self.chunks
            .descriptor()
            .dependent_on_data(&self.embedding_data)
    }
}

impl<D: Dimension, E> std::ops::Deref for EmbeddedTensorOperator<D, E> {
    type Target = TensorOperator<D, E>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<D: Dimension, E> std::ops::DerefMut for EmbeddedTensorOperator<D, E> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<D: Dimension, E> Into<TensorOperator<D, E>> for EmbeddedTensorOperator<D, E> {
    fn into(self) -> TensorOperator<D, E> {
        self.inner
    }
}

impl<D: Dimension, E: ElementType> EmbeddedTensorOperator<D, E> {
    pub fn single_level_lod(self) -> LODTensorOperator<D, E> {
        LODTensorOperator { levels: vec![self] }
    }

    pub fn map_inner<O: ElementType>(
        self,
        f: impl FnOnce(TensorOperator<D, E>) -> TensorOperator<D, O>,
    ) -> EmbeddedTensorOperator<D, O> {
        EmbeddedTensorOperator {
            inner: f(self.inner),
            embedding_data: self.embedding_data,
        }
    }

    pub fn real_dimensions(&self) -> Vector<D, f32> {
        self.embedding_data.spacing * self.metadata.dimensions.raw().f32()
    }

    pub fn cache(self) -> Self {
        self.map_inner(|t| t.cache())
    }
}

#[derive(Clone)]
pub struct LODTensorOperator<D: Dimension, E> {
    pub levels: Vec<EmbeddedTensorOperator<D, E>>,
}

impl<D: Dimension, E> OperatorNetworkNode for LODTensorOperator<D, E> {
    fn descriptor(&self) -> OperatorDescriptor {
        let mut d = self.levels[0].descriptor();
        for l in &self.levels[1..] {
            d = d.dependent_on(l);
        }
        d
    }
}

impl<D: Dimension, E> LODTensorOperator<D, E> {
    pub fn fine_metadata(&self) -> TensorMetaData<D> {
        self.levels[0].metadata.clone()
    }
    pub fn fine_embedding_data(&self) -> TensorEmbeddingData<D> {
        self.levels[0].embedding_data.clone()
    }
}

impl<D: Dimension, E> LODTensorOperator<D, E> {
    pub fn map(
        self,
        f: impl FnMut(EmbeddedTensorOperator<D, E>) -> EmbeddedTensorOperator<D, E>,
    ) -> Self {
        Self {
            levels: self.levels.into_iter().map(f).collect(),
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
    D: Dimension,
>(
    ctx: TaskContext<'cref, 'inv, Vector<D, ChunkCoordinate>, StaticElementType<E>>,
    input: &'op Operator<Vector<D, ChunkCoordinate>, StaticElementType<E>>,
    positions: Vec<(Vector<D, ChunkCoordinate>, DataLocation)>,
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
pub fn map<D: Dimension, E: Element>(
    input: TensorOperator<D, StaticElementType<E>>,
    f: fn(E) -> E,
) -> TensorOperator<D, StaticElementType<E>> {
    TensorOperator::with_state(
        OperatorDescriptor::new("tensor_map")
            .dependent_on(&input)
            .dependent_on_data(&(f as usize)),
        Default::default(),
        input.metadata,
        input,
        move |ctx, positions, input| {
            async move {
                map_values_inplace(ctx, &input.chunks, positions, f).await;

                Ok(())
            }
            .into()
        },
    )
}

#[allow(unused)]
pub fn linear_rescale<D: Dimension>(
    input: TensorOperator<D, StaticElementType<f32>>,
    factor: f32,
    offset: f32,
) -> TensorOperator<D, StaticElementType<f32>> {
    TensorOperator::with_state(
        OperatorDescriptor::new("tensor_linear_scale")
            .dependent_on(&input)
            .dependent_on_data(&factor)
            .dependent_on_data(&offset),
        Default::default(),
        input.metadata,
        (input, factor, offset),
        move |ctx, positions, (input, factor, offset)| {
            let factor = *factor;
            let offset = *offset;
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
