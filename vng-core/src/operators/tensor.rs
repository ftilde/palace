use std::rc::Rc;

use futures::StreamExt;

use crate::{
    array::{TensorEmbeddingData, TensorMetaData},
    data::{ChunkCoordinate, GlobalCoordinate, LocalCoordinate, Vector},
    id::{Id, Identify},
    operator::{Operator, OperatorId},
    storage::{
        ram::{InplaceResult, ThreadInplaceResult},
        Element,
    },
    task::{RequestStream, Task, TaskContext},
};

use super::scalar::ScalarOperator;

#[derive(Clone)]
pub struct TensorOperator<const N: usize, E> {
    pub metadata: Operator<(), TensorMetaData<N>>,
    pub chunks: Operator<Vector<N, ChunkCoordinate>, E>,
}

impl<const N: usize, E: Element> TensorOperator<N, E> {
    pub fn new<
        M: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, (), TensorMetaData<N>>,
                &'inv (),
            ) -> Task<'cref>
            + 'static,
        B: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, Vector<N, ChunkCoordinate>, E>,
                Vec<Vector<N, ChunkCoordinate>>,
                &'inv (),
            ) -> Task<'cref>
            + 'static,
    >(
        base_id: OperatorId,
        metadata: M,
        chunks: B,
    ) -> Self {
        Self::with_state(base_id, (), (), metadata, chunks)
    }

    pub fn with_state<
        SM: 'static,
        SB: 'static,
        M: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, (), TensorMetaData<N>>,
                &'inv SM,
            ) -> Task<'cref>
            + 'static,
        B: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, Vector<N, ChunkCoordinate>, E>,
                Vec<Vector<N, ChunkCoordinate>>,
                &'inv SB,
            ) -> Task<'cref>
            + 'static,
    >(
        base_id: OperatorId,
        state_metadata: SM,
        state_chunks: SB,
        metadata: M,
        chunks: B,
    ) -> Self {
        Self {
            metadata: crate::operators::scalar::scalar(base_id.slot(0), state_metadata, metadata),
            chunks: Operator::with_state(base_id.slot(1), state_chunks, chunks),
        }
    }

    pub fn unbatched<
        SM: 'static,
        SB: 'static,
        M: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, (), TensorMetaData<N>>,
                &'inv SM,
            ) -> Task<'cref>
            + 'static,
        B: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, Vector<N, ChunkCoordinate>, E>,
                Vector<N, ChunkCoordinate>,
                &'inv SB,
            ) -> Task<'cref>
            + 'static,
    >(
        base_id: OperatorId,
        state_metadata: SM,
        state_chunks: SB,
        metadata: M,
        chunks: B,
    ) -> Self {
        Self {
            metadata: crate::operators::scalar::scalar(base_id.slot(0), state_metadata, metadata),
            chunks: Operator::unbatched(base_id.slot(1), state_chunks, chunks),
        }
    }

    pub fn embedded(self, data: TensorEmbeddingData<N>) -> EmbeddedTensorOperator<N, E> {
        EmbeddedTensorOperator {
            inner: self,
            embedding_data: data.into(),
        }
    }
}

impl<const N: usize, E> Identify for TensorOperator<N, E> {
    fn id(&self) -> Id {
        Id::combine(&[(&self.metadata).id(), (&self.chunks).id()])
    }
}

#[derive(Clone)]
pub struct EmbeddedTensorOperator<const N: usize, E> {
    pub inner: TensorOperator<N, E>,
    pub embedding_data: ScalarOperator<TensorEmbeddingData<N>>,
}

impl<const N: usize, E> Identify for EmbeddedTensorOperator<N, E> {
    fn id(&self) -> Id {
        Id::combine(&[(&self.inner).id(), (&self.embedding_data).id()])
    }
}

impl<const N: usize, E> std::ops::Deref for EmbeddedTensorOperator<N, E> {
    type Target = TensorOperator<N, E>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<const N: usize, E> std::ops::DerefMut for EmbeddedTensorOperator<N, E> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<const N: usize, E> Into<TensorOperator<N, E>> for EmbeddedTensorOperator<N, E> {
    fn into(self) -> TensorOperator<N, E> {
        self.inner
    }
}

impl<const N: usize, E> EmbeddedTensorOperator<N, E> {
    pub fn single_level_lod(self) -> LODTensorOperator<N, E> {
        LODTensorOperator { levels: vec![self] }
    }
}

impl<const N: usize, E> EmbeddedTensorOperator<N, E> {
    pub fn map_inner(self, f: impl FnOnce(TensorOperator<N, E>) -> TensorOperator<N, E>) -> Self {
        EmbeddedTensorOperator {
            inner: f(self.inner),
            embedding_data: self.embedding_data,
        }
    }
}

#[derive(Clone)]
pub struct LODTensorOperator<const N: usize, E> {
    pub levels: Vec<EmbeddedTensorOperator<N, E>>,
}

impl<const N: usize, E> Identify for LODTensorOperator<N, E> {
    fn id(&self) -> Id {
        self.levels.as_slice().id()
    }
}

#[allow(unused)]
pub async fn map_values_inplace<
    'op,
    'cref,
    'inv,
    E: Element,
    F: Fn(E) -> E + Send + Copy + 'static,
    const N: usize,
>(
    ctx: TaskContext<'cref, 'inv, Vector<N, ChunkCoordinate>, E>,
    input: &'op Operator<Vector<N, ChunkCoordinate>, E>,
    positions: Vec<Vector<N, ChunkCoordinate>>,
    f: F,
) where
    'op: 'inv,
{
    let requests = positions
        .into_iter()
        .map(|pos| input.request_inplace(*ctx, pos, ctx.current_op()));

    let stream = ctx
        .submit_unordered(requests)
        .then_req(ctx.into(), |brick_handle| {
            let mut brick_handle = brick_handle.unwrap().into_thread_handle();
            ctx.spawn_compute(move || {
                match &mut brick_handle {
                    ThreadInplaceResult::Inplace(ref mut rw) => {
                        for v in rw.iter_mut() {
                            *v = f(*v);
                        }
                    }
                    ThreadInplaceResult::New(r, ref mut w) => {
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
        if let InplaceResult::New(_, w) = brick_handle {
            // Safety: We have written all values in the above closure executed on
            // the thread pool.
            unsafe { w.initialized(*ctx) };
        };
    }
}

#[allow(unused)]
pub fn map<const N: usize, E: Element>(
    input: TensorOperator<N, E>,
    f: fn(E) -> E,
) -> TensorOperator<N, E> {
    TensorOperator::with_state(
        OperatorId::new("tensor_map")
            .dependent_on(&input)
            .dependent_on(&Id::hash(&f)),
        input.clone(),
        input,
        move |ctx, input| {
            async move {
                let req = input.metadata.request_scalar();
                let m = ctx.submit(req).await;
                ctx.write(m)
            }
            .into()
        },
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
pub fn linear_rescale<const N: usize>(
    input: TensorOperator<N, f32>,
    factor: ScalarOperator<f32>,
    offset: ScalarOperator<f32>,
) -> TensorOperator<N, f32> {
    TensorOperator::with_state(
        OperatorId::new("tensor_linear_scale")
            .dependent_on(&input)
            .dependent_on(&factor)
            .dependent_on(&offset),
        input.clone(),
        (input.clone(), factor, offset),
        move |ctx, input| {
            async move {
                let req = input.metadata.request_scalar();
                let m = ctx.submit(req).await;
                ctx.write(m)
            }
            .into()
        },
        move |ctx, positions, (input, factor, offset)| {
            async move {
                let (factor, offset) = futures::join! {
                    ctx.submit(factor.request_scalar()),
                    ctx.submit(offset.request_scalar()),
                };

                map_values_inplace(ctx, &input.chunks, positions, move |i| i * factor + offset)
                    .await;

                Ok(())
            }
            .into()
        },
    )
}

pub fn from_static<const N: usize, E: Element + Identify>(
    size: Vector<N, GlobalCoordinate>,
    values: &'static [E],
) -> Result<TensorOperator<N, E>, crate::Error> {
    let m = TensorMetaData {
        dimensions: size,
        chunk_size: size.map(LocalCoordinate::interpret_as),
    };
    let n_elem = crate::data::hmul(size);
    if n_elem != values.len() {
        return Err(format!(
            "Tensor ({}) and data ({}) size do not match",
            n_elem,
            values.len()
        )
        .into());
    }
    Ok(TensorOperator::with_state(
        OperatorId::new("tensor_from_static")
            .dependent_on(&size)
            .dependent_on(values), //TODO: this is a performance problem for
        m,
        values,
        move |ctx, m| async move { ctx.write(*m) }.into(),
        move |ctx, _, values| {
            async move {
                let mut out = ctx
                    .alloc_slot(Vector::<N, ChunkCoordinate>::fill(0.into()), values.len())
                    .unwrap();
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

pub fn from_rc<const N: usize, E: Element + Identify>(
    size: Vector<N, GlobalCoordinate>,
    values: Rc<[E]>,
) -> Result<TensorOperator<N, E>, crate::Error> {
    let m = TensorMetaData {
        dimensions: size,
        chunk_size: size.map(LocalCoordinate::interpret_as),
    };
    let n_elem = crate::data::hmul(size);
    if n_elem != values.len() {
        return Err(format!(
            "Tensor ({}) and data ({}) size do not match",
            n_elem,
            values.len()
        )
        .into());
    }
    Ok(TensorOperator::with_state(
        OperatorId::new("tensor_from_static")
            .dependent_on(&size)
            .dependent_on(&values[..]), //TODO: this is a performance problem for large arrays
        m,
        values,
        move |ctx, m| async move { ctx.write(*m) }.into(),
        move |ctx, _, values| {
            async move {
                let mut out = ctx
                    .alloc_slot(Vector::<N, ChunkCoordinate>::fill(0.into()), values.len())
                    .unwrap();
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
