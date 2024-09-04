use std::{
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};

use palace_core::{
    array::{TensorEmbeddingData, TensorMetaData},
    data::{Coordinate, CoordinateType},
    dim::{DDyn, DynDimension},
    dtypes::{DType, ElementType, ScalarType},
    operator::{DataDescriptor, OperatorDescriptor},
    operators::tensor::{EmbeddedTensorOperator, TensorOperator},
    task::OpaqueTaskContext,
    vec::Vector,
    Error,
};
use zarrs::{
    array::{Array, ArrayBuilder, ArrayError, DataType, FillValue},
    storage::store::FilesystemStore,
};

const SPACING_KEY: &str = "spacing_us";

fn dtype_zarr_to_palace(d: &DataType) -> Result<DType, Error> {
    Ok(DType::scalar(match d {
        DataType::Int8 => ScalarType::I8,
        DataType::Int16 => ScalarType::I16,
        DataType::Int32 => ScalarType::I32,
        DataType::UInt8 => ScalarType::U8,
        DataType::UInt16 => ScalarType::U16,
        DataType::UInt32 => ScalarType::U32,
        DataType::Float32 => ScalarType::F32,
        _ => Err(format!("No palace correspondence for {:?}", d))?,
    }))
}

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

fn from_zarr_pos(v: &[u64]) -> Vector<DDyn, u32> {
    Vector::try_from_slice(v).unwrap().map(|i| i as u32)
}

#[derive(Clone)]
pub struct ZarrSourceState {
    inner: Rc<ZarrSourceStateInner>,
}

pub struct ZarrSourceStateInner {
    metadata: TensorMetaData<DDyn>,
    embedding_data: TensorEmbeddingData<DDyn>,
    array: Array<FilesystemStore>,
    //dataset: hdf5::Dataset,
    path: PathBuf,
    volume_location: String,
    dtype: DType,
}

pub fn open(
    path: PathBuf,
    volume_location: String,
) -> Result<EmbeddedTensorOperator<DDyn, DType>, Error> {
    let state = ZarrSourceState::open(path, volume_location)?;
    Ok(state.operate())
}

impl ZarrSourceState {
    pub fn open(path: PathBuf, volume_location: String) -> Result<Self, Error> {
        let store = Arc::new(FilesystemStore::new(&path)?);
        let array = Array::open(store, &volume_location)?;
        let dimensions = from_zarr_pos(array.shape()).global();
        let nd = dimensions.len();
        let num_chunks = from_zarr_pos(&array.chunk_grid_shape().unwrap()).chunk();

        // This is pretty ugly: Zarr seems to support irregular grids where each chunk can have a
        // different dimension. We don't, however. So we first assume that the grid is regular and
        // then perform a sanity check against the dimension_in_chunks. Since will not catch all
        // cases, however. We may crash when fetching chunks from such an array later, but I don't
        // know a better solution for now.
        let chunk_size: Vec<u64> = zarrs::array::chunk_shape_to_array_shape(
            &array.chunk_shape(vec![0; nd].as_slice()).unwrap(),
        );
        let chunk_size = from_zarr_pos(&chunk_size).local();

        let metadata = TensorMetaData {
            dimensions,
            chunk_size,
        };
        if metadata.dimension_in_chunks() != num_chunks {
            return Err("Array does not appear to have a regular grid".into());
        }

        let spacing = array
            .attributes()
            .get(SPACING_KEY)
            .and_then(|s| s.as_array())
            .and_then(|s| {
                s.into_iter()
                    .map(|v| v.as_f64().map(|v| v as f32))
                    .collect::<Option<Vec<_>>>()
            })
            .and_then(|s| Vector::try_from_slice(s.as_slice()).ok())
            .unwrap_or_else(|| {
                //eprintln!("Missing spacing information. Using default 1.0");
                Vector::fill_with_len(1.0, nd)
            });
        let embedding_data = TensorEmbeddingData { spacing };

        let dtype = dtype_zarr_to_palace(array.data_type())?;

        Ok(Self {
            inner: ZarrSourceStateInner {
                metadata,
                embedding_data,
                array,
                path,
                volume_location,
                dtype,
            }
            .into(),
        })
    }

    fn operate(&self) -> EmbeddedTensorOperator<DDyn, DType> {
        TensorOperator::with_state(
            OperatorDescriptor::new("ZarrSourceState::operate")
                .dependent_on_data(self.inner.path.to_string_lossy().as_bytes())
                .dependent_on_data(self.inner.volume_location.as_bytes()),
            self.inner.dtype,
            self.inner.metadata.clone(),
            self.clone(),
            move |ctx, positions, this| {
                async move {
                    let metadata = &this.inner.metadata;
                    //NO_PUSH_main make parallel
                    for (chunk_id, _) in positions {
                        let layout = this.inner.dtype.array_layout(metadata.num_chunk_elements());

                        let id = DataDescriptor::new(ctx.current_op_desc().unwrap(), chunk_id);
                        let mut brick_handle = ctx.submit(ctx.alloc_raw(id, layout)).await;
                        let brick_data = brick_handle.data();
                        let array = &this.inner.array;
                        ctx.submit(ctx.spawn_io(|| {
                            let chunk_pos = metadata.chunk_pos_from_index(chunk_id);
                            let bytes = array
                                .retrieve_chunk(to_zarr_pos(&chunk_pos).inner().as_slice())?
                                .into_fixed()?;
                            palace_core::data::write_slice_uninit(brick_data, &bytes);
                            Ok::<(), ArrayError>(())
                        }))
                        .await?;

                        // Safety: At this point the thread pool job above has finished and has initialized all bytes
                        // in the brick.
                        unsafe { brick_handle.initialized(*ctx) };
                    }
                    Ok(())
                }
                .into()
            },
        )
        .embedded(self.inner.embedding_data.clone())
        .into()
    }
}

fn create_array_for_tensor<'cref, 'inv>(
    t: &'inv TensorOperator<DDyn, DType>,
) -> Result<ArrayBuilder, palace_core::Error> {
    let md = &t.metadata;
    let dtype = t.dtype();
    let fill_value = vec![0u8; dtype.element_layout().size()];

    Ok(
        ArrayBuilder::new(
            to_zarr_pos(&md.dimensions).inner(),
            dtype_palace_to_zarr(dtype)?,
            to_zarr_pos(&md.chunk_size).inner().try_into()?,
            FillValue::new(fill_value),
        ), //.bytes_to_bytes_codecs(vec![Box::new(GzipCodec::new(5)?)]);
    )
}

async fn write_tensor<'cref, 'inv>(
    ctx: OpaqueTaskContext<'cref, 'inv>,
    array: &Array<FilesystemStore>,
    t: &'inv TensorOperator<DDyn, DType>,
) -> Result<(), palace_core::Error> {
    let md = &t.metadata;
    for chunk_id in md.chunk_indices() {
        let chunk_pos = md.chunk_pos_from_index(chunk_id);

        let chunk_raw = ctx.submit(t.chunks.request_raw(chunk_id)).await;

        array.store_chunk(to_zarr_pos(&chunk_pos).inner().as_slice(), chunk_raw.data())?;
    }
    Ok(())
}

pub async fn save_tensor<'cref, 'inv>(
    ctx: OpaqueTaskContext<'cref, 'inv>,
    path: &Path,
    t: &'inv TensorOperator<DDyn, DType>,
) -> Result<(), palace_core::Error> {
    let store = Arc::new(FilesystemStore::new(&path)?);
    let array = create_array_for_tensor(t)?.build(store, "/array")?;

    array.store_metadata()?;
    write_tensor(ctx, &array, t).await
}

pub async fn save_embedded_tensor<'cref, 'inv>(
    ctx: OpaqueTaskContext<'cref, 'inv>,
    path: &Path,
    t: &'inv EmbeddedTensorOperator<DDyn, DType>,
) -> Result<(), palace_core::Error> {
    let store = Arc::new(FilesystemStore::new(&path)?);
    let mut attributes = serde_json::Map::new();
    attributes.insert(
        SPACING_KEY.to_owned(),
        serde_json::Value::Array(
            t.embedding_data
                .spacing
                .clone()
                .inner()
                .into_iter()
                .map(|i| i.into())
                .collect(),
        ),
    );
    let array = create_array_for_tensor(t)?
        .attributes(attributes)
        .build(store, "/array")?;

    array.store_metadata()?;

    write_tensor(ctx, &array, t).await
}
