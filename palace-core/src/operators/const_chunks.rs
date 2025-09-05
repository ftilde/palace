use crate::{
    array::{TensorEmbeddingData, TensorMetaData},
    dim::DynDimension,
    dtypes::{DType, ElementType, ScalarType, StaticElementType},
    jit::jit,
    vec::Vector,
};

use super::{
    aggregation::{chunk_aggregation, AggretationMethod},
    rechunk::ChunkSize,
    tensor::LODTensorOperator,
};

pub const MARKER_NOT_CONST_BITS: u32 = 0xff_ff_ab_cd;
//const MARKER_NOT_CONST: f32 = f32::from_bits(MARKER_NOT_CONST_BITS);
//
pub fn is_const_chunk_value(val: f32) -> bool {
    val.to_bits() != MARKER_NOT_CONST_BITS
}

pub fn cct_embedding_data<D: DynDimension>(
    md: &TensorMetaData<D>,
    ed: &TensorEmbeddingData<D>,
) -> TensorEmbeddingData<D> {
    TensorEmbeddingData {
        spacing: ed.spacing.clone() * md.chunk_size.raw().f32(),
    }
}

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
                let ed = cct_embedding_data(&level.metadata, &level.embedding_data);
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

pub fn ensure_compatibility<D: DynDimension, E: ElementType>(
    input: &LODTensorOperator<D, E>,
    const_brick_table: &LODTensorOperator<D, E>,
) -> Result<DType, crate::Error> {
    let dtype = input.dtype().into();
    let const_table_dtype: DType = input.dtype().into();
    if dtype.size != 1 {
        return Err(format!(
            "const_brick_table element must be one-dimensional: {:?}",
            dtype
        )
        .into());
    }

    if input.levels.len() != const_brick_table.levels.len() {
        return Err(format!("Input tensor and const_brick_table must have the same number of levels, but have {} and {}", input.levels.len(), const_brick_table.levels.len()).into());
    }
    for (level, (i, c)) in input
        .levels
        .iter()
        .zip(const_brick_table.levels.iter())
        .enumerate()
    {
        if i.metadata.dimension_in_chunks().raw() != c.metadata.dimensions.raw() {
            return Err(format!(
                "Level {}: const_brick_table should have size {:?}, but has size {:?}",
                level,
                i.metadata.dimension_in_chunks().raw(),
                c.metadata.dimensions.raw()
            )
            .into());
        }
    }
    Ok(const_table_dtype)
}
