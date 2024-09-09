use futures::StreamExt;
use itertools::Itertools;
use std::{
    mem::MaybeUninit,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};

use palace_core::{
    array::{TensorEmbeddingData, TensorMetaData},
    data::{Coordinate, CoordinateType},
    dim::{DDyn, DynDimension},
    dtypes::{DType, ElementType, ScalarType, StaticElementType},
    operator::{DataDescriptor, OperatorDescriptor},
    operators::{
        resample::smooth_downsample,
        tensor::{EmbeddedTensorOperator, LODTensorOperator, TensorOperator},
    },
    runtime::RunTime,
    storage::DataLocation,
    task::{OpaqueTaskContext, RequestStream},
    util::Map,
    vec::Vector,
    vulkan::{vk, DeviceId},
    Error,
};
use zarrs::{
    array::{codec::ZstdCodec, Array, ArrayBuilder, ArrayError, DataType, FillValue},
    node::{Node, NodePath},
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
    path: PathBuf,
    dtype: DType,
}

pub fn open(
    path: PathBuf,
    volume_location: String,
) -> Result<EmbeddedTensorOperator<DDyn, DType>, Error> {
    let store = Arc::new(FilesystemStore::new(&path)?);
    let array = Array::open(store, &volume_location)?;
    let state = ZarrSourceState::from_array(path, array)?;
    Ok(state.operate())
}

fn collect_leafs(p: &Node, out: &mut Vec<NodePath>) {
    let children = p.children();
    if children.is_empty() {
        out.push(p.path().clone());
    } else {
        for c in children {
            collect_leafs(c, out);
        }
    }
}

pub fn open_lod(
    path: PathBuf,
    level_prefix: String,
) -> Result<LODTensorOperator<DDyn, DType>, Error> {
    let store = Arc::new(FilesystemStore::new(&path)?);
    let mut leafs = Vec::new();
    let root = Node::open(&store, "/").unwrap();
    collect_leafs(&root, &mut leafs);
    let level_array_keys = leafs
        .into_iter()
        .filter(|p| p.as_str().starts_with(&level_prefix))
        .collect::<Vec<_>>();
    if level_array_keys.is_empty() {
        return Err(format!("No tensors with prefix {} found", level_prefix).into());
    }

    let mut levels = Vec::new();
    for key in level_array_keys {
        let array = Array::open(Arc::clone(&store), &key.to_string())?;
        let state = ZarrSourceState::from_array(path.clone(), array)?;
        levels.push(state.operate());
    }

    levels.sort_by(|l, r| {
        l.embedding_data
            .spacing
            .length()
            .total_cmp(&r.embedding_data.spacing.length())
    });

    for pair in levels.windows(2) {
        assert!(pair[0].embedding_data.spacing[0] <= pair[1].embedding_data.spacing[0]);
    }

    Ok(LODTensorOperator { levels })
}

impl ZarrSourceState {
    pub fn from_array(path: PathBuf, array: Array<FilesystemStore>) -> Result<Self, Error> {
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
                dtype,
            }
            .into(),
        })
    }

    fn operate(&self) -> EmbeddedTensorOperator<DDyn, DType> {
        TensorOperator::with_state(
            OperatorDescriptor::new("ZarrSourceState::operate")
                .dependent_on_data(self.inner.path.to_string_lossy().as_bytes())
                .dependent_on_data(self.inner.array.path().as_str().as_bytes()),
            self.inner.dtype,
            self.inner.metadata.clone(),
            self.clone(),
            move |ctx, positions, this| {
                async move {
                    let metadata = &this.inner.metadata;
                    let layout = this.inner.dtype.array_layout(metadata.num_chunk_elements());

                    let mut positions_gpus: Map<DeviceId, Vec<_>> = Map::default();
                    let mut positions_cpu = Vec::new();
                    for (position, location) in positions.into_iter() {
                        match location {
                            DataLocation::CPU(_) => positions_cpu.push(position),
                            DataLocation::GPU(i) => {
                                positions_gpus.entry(i).or_default().push(position)
                            }
                        }
                    }

                    let allocations = positions_cpu.into_iter().map(|chunk_id| {
                        let data_id = DataDescriptor::new(ctx.current_op_desc().unwrap(), chunk_id);
                        (ctx.alloc_raw(data_id, layout), chunk_id)
                    });
                    let stream = ctx.submit_unordered_with_data(allocations).then_req(
                        *ctx,
                        |(chunk_handle, chunk_id)| {
                            let chunk_handle = chunk_handle.into_thread_handle();
                            let array = &this.inner.array;
                            ctx.spawn_io(move || {
                                let chunk_data = chunk_handle.data();
                                let chunk_pos = metadata.chunk_pos_from_index(chunk_id);
                                let bytes = array
                                    .retrieve_chunk(to_zarr_pos(&chunk_pos).inner().as_slice())?
                                    .into_fixed()?;
                                palace_core::data::write_slice_uninit(chunk_data, &bytes);
                                Ok::<_, ArrayError>(chunk_handle)
                            })
                        },
                    );

                    futures::pin_mut!(stream);
                    while let Some(handle) = stream.next().await {
                        let handle = handle?.into_main_handle(ctx.storage());
                        unsafe { handle.initialized(*ctx) };
                    }

                    for (device_id, positions_gpu) in positions_gpus {
                        let device = ctx.device_ctx(device_id);
                        let allocations = positions_gpu.into_iter().map(|chunk_id| {
                            let data_id =
                                DataDescriptor::new(ctx.current_op_desc().unwrap(), chunk_id);
                            (ctx.alloc_raw_gpu(device, data_id, layout), chunk_id)
                        });
                        let stream = ctx
                            .submit_unordered_with_data(allocations)
                            .then_req_with_data(*ctx, |(brick_handle, chunk_id)| {
                                let staging_buf = device.staging_to_gpu.request(device, layout);

                                (staging_buf, (brick_handle, chunk_id))
                            })
                            .then_req(*ctx, |(staging_buf, (chunk_handle, chunk_id))| {
                                let chunk_handle = chunk_handle.into_thread_handle();
                                let array = &this.inner.array;

                                ctx.spawn_io(move || {
                                    let chunk_pos = metadata.chunk_pos_from_index(chunk_id);
                                    let bytes = array
                                        .retrieve_chunk(to_zarr_pos(&chunk_pos).inner().as_slice())?
                                        .into_fixed()?;

                                    let ptr = staging_buf
                                        .mapped_ptr()
                                        .unwrap()
                                        .cast::<MaybeUninit<u8>>()
                                        .as_ptr();
                                    let chunk_data = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            ptr,
                                            staging_buf.size as usize,
                                        )
                                    };

                                    palace_core::data::write_slice_uninit(chunk_data, &bytes);
                                    Ok::<_, ArrayError>((staging_buf, chunk_handle))
                                })
                            });

                        futures::pin_mut!(stream);
                        while let Some(res) = stream.next().await {
                            let (staging_buf, chunk_handle) = res?;
                            let handle = chunk_handle.into_main_handle(&device);

                            device.with_cmd_buffer(|cmd| {
                                let copy_info = vk::BufferCopy::builder().size(handle.size as _);
                                unsafe {
                                    device.functions().cmd_copy_buffer(
                                        cmd.raw(),
                                        staging_buf.buffer,
                                        handle.buffer,
                                        &[*copy_info],
                                    );
                                }
                            });

                            unsafe {
                                handle.initialized(
                                    *ctx,
                                    palace_core::vulkan::SrcBarrierInfo {
                                        stage: vk::PipelineStageFlags2::TRANSFER,
                                        access: vk::AccessFlags2::TRANSFER_WRITE,
                                    },
                                )
                            };

                            unsafe { device.staging_to_gpu.return_buf(device, staging_buf) };
                        }
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
    hints: WriteHints,
) -> Result<ArrayBuilder, palace_core::Error> {
    let md = &t.metadata;
    let dtype = t.dtype();
    let fill_value = vec![0u8; dtype.element_layout().size()];

    let mut builder = ArrayBuilder::new(
        to_zarr_pos(&md.dimensions).inner(),
        dtype_palace_to_zarr(dtype)?,
        to_zarr_pos(&md.chunk_size).inner().try_into()?,
        FillValue::new(fill_value),
    );

    if hints.compression_level > 0 {
        builder.bytes_to_bytes_codecs(vec![Box::new(ZstdCodec::new(
            hints.compression_level,
            false,
        ))]);
    }
    Ok(builder)
}

#[derive(Copy, Clone)]
pub struct WriteHints {
    pub compression_level: i32,
}

impl Default for WriteHints {
    fn default() -> Self {
        Self {
            compression_level: 1,
        }
    }
}

fn create_array_for_embedded_tensor<'cref, 'inv>(
    t: &'inv EmbeddedTensorOperator<DDyn, DType>,
    hints: WriteHints,
) -> Result<ArrayBuilder, palace_core::Error> {
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
    let mut b = create_array_for_tensor(t, hints)?;
    b.attributes(attributes);
    Ok(b)
}

async fn write_tensor<'cref, 'inv>(
    ctx: OpaqueTaskContext<'cref, 'inv>,
    array: &Array<FilesystemStore>,
    t: &'inv TensorOperator<DDyn, DType>,
) -> Result<(), palace_core::Error> {
    let md = &t.metadata;

    let num_total = md.dimension_in_chunks().hmul();
    println!("{} chunks to save", num_total);

    let request_chunk_size = 1024;
    let chunk_ids_in_parts = md.chunk_indices().chunks(request_chunk_size);
    let mut i = 0;
    for chunk_ids in &chunk_ids_in_parts {
        let requests = chunk_ids.map(|chunk_id| (t.chunks.request_raw(chunk_id), chunk_id));
        let stream =
            ctx.submit_unordered_with_data(requests)
                .then_req(ctx, |(chunk_handle, chunk_id)| {
                    let chunk_pos = md.chunk_pos_from_index(chunk_id);

                    let chunk_handle = chunk_handle.into_thread_handle();
                    let array = &array;
                    ctx.spawn_io(move || {
                        array.store_chunk(
                            to_zarr_pos(&chunk_pos).inner().as_slice(),
                            chunk_handle.data(),
                        )?;
                        Ok::<_, ArrayError>(chunk_handle)
                    })
                });
        futures::pin_mut!(stream);
        while let Some(handle) = stream.next().await {
            let _handle = handle?.into_main_handle(ctx.storage());
        }
        i += request_chunk_size;
        println!(
            "{}/{}, {}%",
            i,
            num_total,
            i as f32 / num_total as f32 * 100.0
        );
    }

    Ok(())
}

pub async fn save_tensor<'cref, 'inv>(
    ctx: OpaqueTaskContext<'cref, 'inv>,
    path: &Path,
    t: &'inv TensorOperator<DDyn, DType>,
    hints: WriteHints,
) -> Result<(), palace_core::Error> {
    let store = Arc::new(FilesystemStore::new(&path)?);
    let array = create_array_for_tensor(t, hints)?.build(store, "/array")?;

    array.store_metadata()?;
    write_tensor(ctx, &array, t).await
}

pub async fn save_embedded_tensor<'cref, 'inv>(
    ctx: OpaqueTaskContext<'cref, 'inv>,
    path: &Path,
    t: &'inv EmbeddedTensorOperator<DDyn, DType>,
    hints: WriteHints,
) -> Result<(), palace_core::Error> {
    let store = Arc::new(FilesystemStore::new(&path)?);
    let array = create_array_for_embedded_tensor(t, hints)?.build(store, "/array")?;

    array.store_metadata()?;

    write_tensor(ctx, &array, t).await
}

fn level_path(level: usize) -> String {
    format!("/level{}", level)
}

pub fn save_lod_tensor(
    runtime: &mut RunTime,
    path: &Path,
    t: &LODTensorOperator<DDyn, DType>,
    hints: WriteHints,
    recreate_lod: bool,
) -> Result<(), palace_core::Error> {
    let store = Arc::new(FilesystemStore::new(&path)?);
    let store = &store;

    if recreate_lod {
        let step_factor = 2.0;

        let current = t.levels[0].clone();
        let mut current_level = 0;

        // Cast to float. TODO: Remove once we have resampling for other types
        let mut current = current.clone().map_inner(|input| {
            palace_core::jit::jit(input)
                .cast(ScalarType::F32.into())
                .unwrap()
                .compile()
                .unwrap()
        });
        loop {
            let current_location = level_path(current_level);
            let current_location_ref = &current_location;
            let current_ref = &current;
            runtime.resolve(None, false, |ctx, _| {
                async move {
                    let array = create_array_for_embedded_tensor(current_ref, hints)?
                        .build(Arc::clone(store), current_location_ref)?;

                    array.store_metadata()?;

                    write_tensor(ctx, &array, current_ref).await
                }
                .into()
            })?;

            if current.metadata.dimension_in_chunks().hmul() == 1 {
                break;
            }

            let new_md = palace_core::operators::resample::coarser_lod_md(&current, step_factor);

            current = open(path.into(), current_location)?;
            let current_float: EmbeddedTensorOperator<DDyn, StaticElementType<f32>> =
                current.try_into().unwrap();

            //TODO: Maybe we do not want to hardcode this. It would also be easy to offer something
            //like "cache everything but the highest resolution layer" on LODTensorOperator
            current = smooth_downsample(current_float, new_md.clone()).into();
            current_level += 1;
        }
    } else {
        runtime.resolve(None, false, |ctx, _| {
            async move {
                for (level, tensor) in t.levels.iter().enumerate() {
                    let array = create_array_for_embedded_tensor(tensor, hints)?
                        .build(Arc::clone(store), &level_path(level))?;

                    array.store_metadata()?;

                    write_tensor(ctx, &array, tensor).await?;
                }
                Ok(())
            }
            .into()
        })?;
    }

    Ok(())
}
