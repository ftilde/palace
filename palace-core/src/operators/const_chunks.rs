use crate::{
    array::TensorEmbeddingData,
    dim::DynDimension,
    dtypes::{ScalarType, StaticElementType},
    jit::jit,
    vec::Vector,
};

use super::{
    aggregation::{chunk_aggregation, AggretationMethod},
    rechunk::ChunkSize,
    tensor::LODTensorOperator,
};

const MARKER_NOT_CONST_BITS: u32 = 0xff_ff_ab_cd;
//const MARKER_NOT_CONST: f32 = f32::from_bits(MARKER_NOT_CONST_BITS);

/// If max(chunk)-min(chunk) < diff_threshold, then the corresponding voxel value is mean(chunk).
/// Otherwise it is MARKER_NOT_CONST. Use this for operators that can be accelerated by such a
/// metadata structure (e.g. raycaster)
pub fn const_chunk_table<'op, D: DynDimension>(
    input: LODTensorOperator<D, StaticElementType<f32>>,
    out_chunk_size: Vector<D, ChunkSize>,
    diff_threshold: f32,
) -> LODTensorOperator<D, StaticElementType<f32>> {
    LODTensorOperator {
        levels: input
            .levels
            .into_iter()
            .map(|level| {
                let out_chunk_size =
                    out_chunk_size.zip(&level.inner.metadata.dimensions, |cs, d| cs.apply(d));
                let ed = TensorEmbeddingData {
                    spacing: level.embedding_data.spacing
                        * level.inner.metadata.chunk_size.raw().f32(),
                };
                let min = jit(chunk_aggregation(
                    level.inner.clone(),
                    out_chunk_size.clone(),
                    AggretationMethod::Min,
                )
                .into());
                let max = jit(chunk_aggregation(
                    level.inner.clone(),
                    out_chunk_size.clone(),
                    AggretationMethod::Max,
                )
                .into());
                let abs_diff = min.clone().sub(max.clone()).unwrap().abs().unwrap();
                let mean = jit(chunk_aggregation(
                    level.inner.clone(),
                    out_chunk_size.clone(),
                    AggretationMethod::Mean,
                )
                .into());
                abs_diff
                    .lt_eq(diff_threshold.into())
                    .unwrap()
                    .select(
                        mean,
                        crate::jit::scalar(MARKER_NOT_CONST_BITS)
                            .reinterpret(ScalarType::F32.into())
                            .unwrap(),
                    )
                    .unwrap()
                    .compile()
                    .unwrap()
                    .embedded(ed)
                    .try_into()
                    .unwrap()
            })
            .collect(),
    }
}
