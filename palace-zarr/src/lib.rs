use std::{path::Path, sync::Arc};

use palace_core::{
    data::{Coordinate, CoordinateType},
    dim::{DDyn, DynDimension},
    dtypes::{DType, ScalarType},
    operators::tensor::TensorOperator,
    task::OpaqueTaskContext,
    vec::Vector,
    Error,
};
use zarrs::{
    array::{ArrayBuilder, DataType, FillValue, ZARR_NAN_F32},
    storage::store::FilesystemStore,
};

//fn dtype_zarr_to_palace(d: DataType) -> Result<DType, Error> {
//    Ok(DType::scalar(match d {
//        DataType::Int8 => ScalarType::I8,
//        DataType::Int16 => ScalarType::I16,
//        DataType::Int32 => ScalarType::I32,
//        DataType::UInt8 => ScalarType::U8,
//        DataType::UInt16 => ScalarType::U16,
//        DataType::UInt32 => ScalarType::U32,
//        DataType::Float32 => ScalarType::F32,
//        _ => Err(format!("No palace correspondence for {:?}", d))?,
//    }))
//}

fn dtype_palace_to_zarr(d: DType) -> Result<DataType, Error> {
    if d.size != 1 {
        Err(format!("No zarr correspondence for {:?}", d))?;
    }

    Ok(match d.scalar {
        ScalarType::U8 => DataType::UInt8,
        ScalarType::I8 => DataType::Int8,
        ScalarType::U16 => DataType::UInt16,
        ScalarType::I16 => DataType::Int16,
        ScalarType::F32 => DataType::Float32,
        ScalarType::U32 => DataType::UInt32,
        ScalarType::I32 => DataType::Int32,
    })
}

fn to_zarr_pos<D: DynDimension, C: CoordinateType>(v: &Vector<D, Coordinate<C>>) -> Vector<D, u64> {
    v.map(|n| n.raw as u64)
}

pub async fn save<'cref, 'inv>(
    ctx: OpaqueTaskContext<'cref, 'inv>,
    path: &Path,
    t: &'inv TensorOperator<DDyn, DType>,
) -> Result<(), palace_core::Error> {
    let store = Arc::new(FilesystemStore::new(&path)?);

    let dtype = t.dtype();
    let md = t.metadata.clone();

    let array = ArrayBuilder::new(
        to_zarr_pos(&md.dimensions).inner(),
        dtype_palace_to_zarr(dtype)?,
        to_zarr_pos(&md.chunk_size).inner().try_into()?,
        FillValue::from(ZARR_NAN_F32),
    )
    .build(store, "/array")?;

    array.store_metadata()?;
    //.bytes_to_bytes_codecs(vec![Box::new(GzipCodec::new(5)?)]);

    for chunk_id in md.chunk_indices() {
        let chunk_pos = md.chunk_pos_from_index(chunk_id);

        let chunk_raw = ctx.submit(t.chunks.request_raw(chunk_id)).await;

        array.store_chunk(to_zarr_pos(&chunk_pos).inner().as_slice(), chunk_raw.data())?;
    }

    Ok(())
}
