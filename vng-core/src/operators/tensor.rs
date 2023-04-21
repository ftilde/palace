use futures::StreamExt;

use crate::{
    array::TensorMetaData,
    data::{ChunkCoordinate, GlobalCoordinate, LocalCoordinate, Vector},
    id::Id,
    operator::{Operator, OperatorId},
    storage::ram::{InplaceResult, ThreadInplaceResult},
    task::{RequestStream, Task, TaskContext},
};

use super::scalar::ScalarOperator;

#[derive(Clone)]
pub struct TensorOperator<'op, const N: usize> {
    pub metadata: Operator<'op, (), TensorMetaData<N>>,
    pub bricks: Operator<'op, Vector<N, ChunkCoordinate>, f32>,
}

impl<'op, const N: usize> TensorOperator<'op, N> {
    pub fn new<
        M: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, (), TensorMetaData<N>>,
                &'inv (),
                crate::operator::OutlivesMarker<'op, 'inv>,
            ) -> Task<'cref>
            + 'op,
        B: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, Vector<N, ChunkCoordinate>, f32>,
                Vec<Vector<N, ChunkCoordinate>>,
                &'inv (),
                crate::operator::OutlivesMarker<'op, 'inv>,
            ) -> Task<'cref>
            + 'op,
    >(
        base_id: OperatorId,
        metadata: M,
        bricks: B,
    ) -> Self {
        Self::with_state(base_id, (), (), metadata, bricks)
    }

    pub fn with_state<
        SM: 'op,
        SB: 'op,
        M: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, (), TensorMetaData<N>>,
                &'inv SM,
                crate::operator::OutlivesMarker<'op, 'inv>,
            ) -> Task<'cref>
            + 'op,
        B: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, Vector<N, ChunkCoordinate>, f32>,
                Vec<Vector<N, ChunkCoordinate>>,
                &'inv SB,
                crate::operator::OutlivesMarker<'op, 'inv>,
            ) -> Task<'cref>
            + 'op,
    >(
        base_id: OperatorId,
        state_metadata: SM,
        state_bricks: SB,
        metadata: M,
        bricks: B,
    ) -> Self {
        Self {
            metadata: crate::operators::scalar::scalar(base_id.slot(0), state_metadata, metadata),
            bricks: Operator::with_state(base_id.slot(1), state_bricks, bricks),
        }
    }

    pub fn unbatched<
        SM: 'op,
        SB: 'op,
        M: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, (), TensorMetaData<N>>,
                &'inv SM,
                crate::operator::OutlivesMarker<'op, 'inv>,
            ) -> Task<'cref>
            + 'op,
        B: for<'cref, 'inv> Fn(
                TaskContext<'cref, 'inv, Vector<N, ChunkCoordinate>, f32>,
                Vector<N, ChunkCoordinate>,
                &'inv SB,
                crate::operator::OutlivesMarker<'op, 'inv>,
            ) -> Task<'cref>
            + 'op,
    >(
        base_id: OperatorId,
        state_metadata: SM,
        state_bricks: SB,
        metadata: M,
        bricks: B,
    ) -> Self {
        Self {
            metadata: crate::operators::scalar::scalar(base_id.slot(0), state_metadata, metadata),
            bricks: Operator::unbatched(base_id.slot(1), state_bricks, bricks),
        }
    }
}

impl<const N: usize> Into<Id> for &TensorOperator<'_, N> {
    fn into(self) -> Id {
        Id::combine(&[(&self.metadata).into(), (&self.bricks).into()])
    }
}

#[allow(unused)]
pub async fn map_values<
    'op,
    'cref,
    'inv,
    F: Fn(f32) -> f32 + Send + Copy + 'static,
    const N: usize,
>(
    ctx: TaskContext<'cref, 'inv, Vector<N, ChunkCoordinate>, f32>,
    input: &'op Operator<'_, Vector<N, ChunkCoordinate>, f32>,
    positions: Vec<Vector<N, ChunkCoordinate>>,
    f: F,
) where
    'op: 'inv,
{
    let requests = positions
        .into_iter()
        .map(|pos| input.request_inplace(pos, ctx.current_op()));

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
        let brick_handle = brick_handle.into_main_handle(ctx.storage());
        if let InplaceResult::New(_, w) = brick_handle {
            // Safety: We have written all values in the above closure executed on
            // the thread pool.
            unsafe { w.initialized() };
        };
    }
}

#[allow(unused)]
pub fn map<'op, const N: usize>(
    input: TensorOperator<'op, N>,
    f: fn(f32) -> f32,
) -> TensorOperator<'op, N> {
    TensorOperator::with_state(
        OperatorId::new("tensor_map")
            .dependent_on(&input)
            .dependent_on(Id::hash(&f)),
        input.clone(),
        input,
        move |ctx, input, _| {
            async move {
                let req = input.metadata.request_scalar();
                let m = ctx.submit(req).await;
                ctx.write(m)
            }
            .into()
        },
        move |ctx, positions, input, _| {
            async move {
                map_values(ctx, &input.bricks, positions, f).await;

                Ok(())
            }
            .into()
        },
    )
}

#[allow(unused)]
pub fn linear_rescale<'op, const N: usize>(
    input: TensorOperator<'op, N>,
    factor: ScalarOperator<'op, f32>,
    offset: ScalarOperator<'op, f32>,
) -> TensorOperator<'op, N> {
    TensorOperator::with_state(
        OperatorId::new("tensor_linear_scale")
            .dependent_on(&input)
            .dependent_on(&factor)
            .dependent_on(&offset),
        input.clone(),
        (input.clone(), factor, offset),
        move |ctx, input, _| {
            async move {
                let req = input.metadata.request_scalar();
                let m = ctx.submit(req).await;
                ctx.write(m)
            }
            .into()
        },
        move |ctx, positions, (input, factor, offset), _| {
            async move {
                let (factor, offset) = futures::join! {
                    ctx.submit(factor.request_scalar()),
                    ctx.submit(offset.request_scalar()),
                };

                map_values(ctx, &input.bricks, positions, move |i| i * factor + offset).await;

                Ok(())
            }
            .into()
        },
    )
}

pub fn from_static<'op, const N: usize>(
    size: Vector<N, GlobalCoordinate>,
    values: &'op [f32],
) -> Result<TensorOperator<'op, N>, crate::Error> {
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
    Ok(TensorOperator::new(
        OperatorId::new("tensor_from_static")
            .dependent_on(Id::hash(&size))
            .dependent_on(bytemuck::cast_slice(values)), //TODO: this is a performance problem for
        move |ctx, _, _| async move { ctx.write(m) }.into(),
        move |ctx, _, _, _| {
            async move {
                let mut out = ctx
                    .alloc_slot(Vector::<N, ChunkCoordinate>::fill(0.into()), values.len())
                    .unwrap();
                let mut out_data = &mut *out;
                ctx.submit(ctx.spawn_compute(move || {
                    crate::data::write_slice_uninit(&mut out_data, values);
                }))
                .await;

                // Safety: slot and values are of the exact same size. Thus all values are
                // initialized.
                unsafe { out.initialized() };
                Ok(())
            }
            .into()
        },
    ))
}
