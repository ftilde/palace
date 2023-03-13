use crate::{
    array::TensorMetaData,
    data::{ChunkCoordinate, Vector},
    id::Id,
    operator::{Operator, OperatorId},
    task::{Task, TaskContext},
};

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
}

impl<const N: usize> Into<Id> for &TensorOperator<'_, N> {
    fn into(self) -> Id {
        Id::combine(&[(&self.metadata).into(), (&self.bricks).into()])
    }
}
