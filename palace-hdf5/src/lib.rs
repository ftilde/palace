use futures::StreamExt;
use hidefix::idx::DatasetD;
use id::{Id, Identify};
use itertools::Itertools;
use palace_core::array::{ChunkInfo, TensorEmbeddingData, TensorMetaData};
use palace_core::data::{Coordinate, CoordinateType, GlobalCoordinate, LocalCoordinate};
use palace_core::dim::{DDyn, DynDimension};
use palace_core::dtypes::{DType, ElementType, ScalarType};
use palace_core::op_descriptor;
use palace_core::operator::{DataDescriptor, DataParam};
use palace_core::operators::resample::DownsampleStep;
use palace_core::operators::tensor::{EmbeddedTensorOperator, LODTensorOperator};
use palace_core::storage::DataLocation;
use palace_core::task::{OpaqueTaskContext, RequestStream};
use palace_core::vulkan::vk;
use positioned_io::{RandomAccessFile, ReadAt};
use std::borrow::Cow;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use hdf5::Datatype;
use hidefix::prelude::*;

use palace_core::{
    data::{self, Vector},
    operator::OperatorDescriptor,
    operators::tensor::TensorOperator,
    Error,
};

const SPACING_KEY: &str = "element_size_um";
const SPACING_FACTOR_FILE_TO_MEM: f32 = 0.001;
const SPACING_FACTOR_MEM_TO_FILE: f32 = 1.0 / SPACING_FACTOR_FILE_TO_MEM;

#[derive(Clone)]
pub struct Hdf5TensorSourceState {
    inner: Rc<Hdf5TensorSourceStateInner>,
}

pub struct Hdf5TensorSourceStateInner {
    metadata: TensorMetaData<DDyn>,
    embedding_data: TensorEmbeddingData<DDyn>,
    dataset_index: DatasetD<'static>,
    dataset_file: RandomAccessFile,
    path: PathBuf,
    volume_location: String,
    dtype: DType,
}

impl Identify for Hdf5TensorSourceState {
    fn id(&self) -> id::Id {
        Id::combine(&[self.inner.path.id(), self.inner.volume_location.id()])
    }
}

fn to_size_vector<C: CoordinateType>(value: Vec<hdf5::Ix>) -> Vector<DDyn, Coordinate<C>> {
    to_vector(value.into_iter().map(|v| v as u32).collect())
}

fn to_vector<I: Copy, O: From<I> + Copy>(value: Vec<I>) -> Vector<DDyn, O> {
    Vector::from_fn_and_len(value.len(), |i| value[i].into())
}

fn extent_palace_to_hdf(e: &Vector<DDyn, GlobalCoordinate>) -> hdf5::Extents {
    hdf5::Extents::Simple(hdf5::SimpleExtents::from_vec(
        e.map(|v| hdf5::Extent::fixed(v.raw as usize)).inner(),
    ))
}

fn chunk_size(e: &TensorMetaData<DDyn>) -> Vec<usize> {
    e.dimensions
        .zip(&e.chunk_size, |a, b| a.raw.min(b.raw) as usize)
        .inner()
}

fn chunk_to_slice_arg(i: ChunkInfo<DDyn>) -> hdf5::Hyperslab {
    let begin = i.begin();
    let size = &i.logical_dimensions;
    begin
        .zip(&size, |begin, size| hdf5::SliceOrIndex::SliceCount {
            start: begin.raw as usize,
            step: 1,
            count: size.raw as usize,
            block: 1,
        })
        .inner()
        .into()
}

fn dtype_hdf5_to_palace(d: &Datatype) -> Result<DType, Error> {
    let scalar = if d.is::<i8>() {
        ScalarType::I8
    } else if d.is::<u8>() {
        ScalarType::U8
    } else if d.is::<i16>() {
        ScalarType::I16
    } else if d.is::<u16>() {
        ScalarType::U16
    } else if d.is::<i32>() {
        ScalarType::I32
    } else if d.is::<u32>() {
        ScalarType::U32
    } else if d.is::<i64>() {
        ScalarType::I64
    } else if d.is::<u64>() {
        ScalarType::U64
    } else if d.is::<f32>() {
        ScalarType::F32
    } else {
        return Err(format!("No palace correspondence for {:?}", d).into());
    };
    Ok(DType::scalar(scalar))
}

fn dtype_palace_to_hdf5(d: DType) -> Result<Datatype, Error> {
    if d.size != 1 {
        Err(format!("No hdf5 correspondence for {:?}", d))?;
    }

    Ok(match d.scalar {
        ScalarType::U8 => Datatype::from_type::<u8>()?,
        ScalarType::I8 => Datatype::from_type::<u8>()?,
        ScalarType::U16 => Datatype::from_type::<u16>()?,
        ScalarType::I16 => Datatype::from_type::<i16>()?,
        ScalarType::F32 => Datatype::from_type::<f32>()?,
        ScalarType::U32 => Datatype::from_type::<u32>()?,
        ScalarType::I32 => Datatype::from_type::<i32>()?,
        ScalarType::U64 => Datatype::from_type::<i64>()?,
        ScalarType::I64 => Datatype::from_type::<i64>()?,
    })
}

pub(crate) fn decode_chunk<'a>(
    chunk_bytes: &'a [u8],
    storage_info: &StorageInfo,
) -> Result<Cow<'a, [u8]>, Error> {
    debug_assert!(storage_info.data_size < 16); // unlikely data-size

    // Decompress
    let cache = if storage_info.gzip {
        let mut decache = vec![0; storage_info.chunk_size];

        hidefix::filters::gzip::decompress(&chunk_bytes, &mut decache)?;

        debug_assert_eq!(decache.len(), storage_info.chunk_size);

        Cow::from(decache)
    } else {
        Cow::Borrowed(chunk_bytes)
    };

    // Unshuffle
    let cache = if storage_info.shuffle && storage_info.data_size > 1 {
        Cow::from(hidefix::filters::shuffle::unshuffle_sized(
            &cache,
            storage_info.data_size,
        ))
    } else {
        cache
    };

    Ok(cache)
}

pub fn open(
    path: PathBuf,
    volume_location: Option<String>,
) -> Result<EmbeddedTensorOperator<DDyn, DType>, Error> {
    let volume_location = if let Some(volume_location) = volume_location {
        volume_location
    } else {
        let file = hdf5::File::open(&path)?;
        let datasets = file.datasets()?;
        match datasets.len() {
            0 => return Err(format!("No tensors found in dataset").into()),
            1 => datasets[0].name(),
            n => {
                return Err(format!(
                    "More than ({}) one tensor found and no dataset name given",
                    n
                )
                .into())
            }
        }
    };

    let state = Hdf5TensorSourceState::open(path, volume_location)?;
    Ok(state.operate())
}

pub fn open_lod(
    path: PathBuf,
    level_prefix: String,
) -> Result<LODTensorOperator<DDyn, DType>, Error> {
    let file = hdf5::File::open(&path)?;
    let level_array_keys = file
        .datasets()?
        .into_iter()
        .map(|p| p.name())
        .filter(|p| p.starts_with(&level_prefix))
        .collect::<Vec<_>>();
    if level_array_keys.is_empty() {
        return Err(format!("No tensors with prefix {} found", level_prefix).into());
    }

    let mut levels = Vec::new();
    for key in level_array_keys {
        let state = Hdf5TensorSourceState::open(path.clone(), key)?;
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

struct StorageInfo {
    data_size: usize,
    gzip: bool,
    shuffle: bool,
    chunk_size: usize,
}
fn storage_info_static<const D: usize>(ds: &hidefix::idx::Dataset<'static, D>) -> StorageInfo {
    StorageInfo {
        data_size: ds.dsize,
        gzip: ds.gzip.is_some(),
        shuffle: ds.shuffle,
        chunk_size: ds.chunk_shape().iter().product::<u64>() as usize * ds.dsize,
    }
}
fn storage_info(ds: &hidefix::idx::DatasetD<'static>) -> StorageInfo {
    match ds {
        DatasetD::D0(dataset) => storage_info_static(dataset),
        DatasetD::D1(dataset) => storage_info_static(dataset),
        DatasetD::D2(dataset) => storage_info_static(dataset),
        DatasetD::D3(dataset) => storage_info_static(dataset),
        DatasetD::D4(dataset) => storage_info_static(dataset),
        DatasetD::D5(dataset) => storage_info_static(dataset),
        DatasetD::D6(dataset) => storage_info_static(dataset),
        DatasetD::D7(dataset) => storage_info_static(dataset),
        DatasetD::D8(dataset) => storage_info_static(dataset),
        DatasetD::D9(dataset) => storage_info_static(dataset),
    }
}

struct FileChunkInfo {
    addr: u64,
    size: u64,
}
fn chunk_info_static<const D: usize>(
    ds: &hidefix::idx::Dataset<'static, D>,
    pos: &[u64],
) -> FileChunkInfo {
    let c = ds.chunk_at_coord(pos);
    FileChunkInfo {
        addr: c.addr.into(),
        size: c.size.into(),
    }
}
fn chunk_info(ds: &hidefix::idx::DatasetD<'static>, pos: &[u64]) -> FileChunkInfo {
    match ds {
        DatasetD::D0(dataset) => chunk_info_static(dataset, pos),
        DatasetD::D1(dataset) => chunk_info_static(dataset, pos),
        DatasetD::D2(dataset) => chunk_info_static(dataset, pos),
        DatasetD::D3(dataset) => chunk_info_static(dataset, pos),
        DatasetD::D4(dataset) => chunk_info_static(dataset, pos),
        DatasetD::D5(dataset) => chunk_info_static(dataset, pos),
        DatasetD::D6(dataset) => chunk_info_static(dataset, pos),
        DatasetD::D7(dataset) => chunk_info_static(dataset, pos),
        DatasetD::D8(dataset) => chunk_info_static(dataset, pos),
        DatasetD::D9(dataset) => chunk_info_static(dataset, pos),
    }
}

fn copy_chunk(
    dataset: &DatasetD<'static>,
    file: &RandomAccessFile,
    chunk_data_out: &mut [MaybeUninit<u8>],
    out_info: ChunkInfo<DDyn>,
) -> Result<(), Error> {
    let chunk_addr = chunk_info(
        dataset,
        out_info.begin().map(|v| v.raw as u64).inner().as_slice(),
    );

    let storage_info = storage_info(dataset);

    if !storage_info.gzip && !storage_info.shuffle && out_info.is_full() {
        // This is unsound:
        // let chunk_data_out_ptr = chunk_data_out.as_mut_ptr() as *mut u8;
        //let mut chunk_data_out =
        //    unsafe { std::slice::from_raw_parts_mut(chunk_data_out_ptr, chunk_data_out.len()) };
        // So we do unnecessary initialization for now. (Does not cost much, though).
        let mut chunk_data_out = data::fill_uninit(chunk_data_out, 0u8);
        file.read_exact_at(chunk_addr.addr, &mut chunk_data_out)?;
    } else {
        let mut buf =
            palace_core::util::alloc_vec_aligned_zeroed::<u8>(chunk_addr.size as usize, 4096);
        file.read_exact_at(chunk_addr.addr, &mut buf)?;

        let chunk_data = decode_chunk(&buf, &storage_info)?;
        assert_eq!(chunk_data_out.len(), chunk_data.len());
        data::write_slice_uninit(chunk_data_out, &chunk_data);
    }

    Ok(())
}

impl Hdf5TensorSourceState {
    pub fn open(path: PathBuf, volume_location: String) -> Result<Self, Error> {
        let file = hdf5::File::open(&path)?;
        let vol = file.dataset(&volume_location)?;
        let dset: DatasetD = vol.index()?;
        let dimensions: Vector<DDyn, GlobalCoordinate> = to_size_vector(vol.shape());
        let chunk_size: Vector<DDyn, LocalCoordinate> =
            to_size_vector(vol.chunk().unwrap_or_else(|| vol.shape()));
        //println!("Chunksize {:?}", brick_size);
        let spacing: Result<Vector<DDyn, f32>, Error> = vol
            .attr(SPACING_KEY)
            .and_then(|a| a.read_1d::<f32>())
            .map_err(|e| e.into())
            .map(|s| to_vector(s.to_vec()).scale(SPACING_FACTOR_FILE_TO_MEM));

        let spacing = match spacing {
            Ok(spacing) => spacing,
            Err(e) => {
                eprintln!(
                    "Could not load spacing from dataset: {}\n Using default spacing (1.0/dim[0]).",
                    e
                );
                Vector::fill_with_len(1.0 / dimensions[0].raw as f32, dimensions.len())
            }
        };

        if spacing.len() != dimensions.len() {
            return Err(format!(
                "Spacing dimension ({}) does not match tensor dimension ({})",
                spacing.len(),
                dimensions.len(),
            )
            .into());
        }

        let dtype = dtype_hdf5_to_palace(&vol.dtype()?)?;

        let metadata = TensorMetaData {
            dimensions,
            chunk_size,
        };

        let embedding_data = TensorEmbeddingData { spacing };

        let file = RandomAccessFile::open(&path)?;

        Ok(Hdf5TensorSourceState {
            inner: Rc::new(Hdf5TensorSourceStateInner {
                metadata,
                embedding_data,
                dataset_index: dset,
                dataset_file: file,
                path,
                volume_location,
                dtype,
            }),
        })
    }

    fn operate(&self) -> EmbeddedTensorOperator<DDyn, DType> {
        TensorOperator::with_state(
            op_descriptor!(),
            self.inner.dtype,
            self.inner.metadata.clone(),
            DataParam(self.clone()),
            move |ctx, positions, location, this| {
                //println!("Positions: {}", positions.len());
                async move {
                    let metadata = &this.inner.metadata;
                    let dtype = this.inner.dtype;
                    let num_voxels = this.inner.metadata.chunk_size.hmul();
                    let layout = dtype.array_layout(num_voxels);
                    let dataset = &this.inner.dataset_index;
                    let file = &this.inner.dataset_file;

                    match location {
                        DataLocation::CPU(_) => {
                            let allocations = positions.into_iter().map(|chunk_id| {
                                let data_id =
                                    DataDescriptor::new(ctx.current_op_desc().unwrap(), chunk_id);
                                (ctx.alloc_raw(data_id, layout).unwrap_value(), chunk_id)
                            });
                            let stream = ctx.submit_unordered_with_data(allocations).then_req(
                                *ctx,
                                |(chunk_handle, chunk_id)| {
                                    let chunk_handle = chunk_handle.into_thread_handle();
                                    ctx.spawn_compute(move || {
                                        let chunk_info = metadata.chunk_info(chunk_id);
                                        let chunk_data = chunk_handle.data();
                                        palace_core::data::init_non_full(
                                            chunk_data,
                                            &chunk_info,
                                            0,
                                        );

                                        copy_chunk(&dataset, file, chunk_data, chunk_info).unwrap();
                                        chunk_handle
                                    })
                                },
                            );

                            futures::pin_mut!(stream);
                            while let Some(handle) = stream.next().await {
                                let handle = handle.into_main_handle(ctx.storage());
                                unsafe { handle.initialized(*ctx) };
                            }
                        }
                        DataLocation::GPU(device_id) => {
                            let device = ctx.device_ctx(device_id);
                            let allocations = positions.into_iter().map(|chunk_id| {
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

                                    ctx.spawn_compute(move || {
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

                                        let chunk_info = metadata.chunk_info(chunk_id);
                                        palace_core::data::init_non_full(
                                            chunk_data,
                                            &chunk_info,
                                            0,
                                        );

                                        copy_chunk(&dataset, file, chunk_data, chunk_info).unwrap();
                                        (staging_buf, chunk_handle)
                                    })
                                });

                            futures::pin_mut!(stream);
                            while let Some(res) = stream.next().await {
                                let (staging_buf, chunk_handle) = res;
                                let handle = chunk_handle.into_main_handle(&device);

                                device.with_cmd_buffer(|cmd| {
                                    let copy_info =
                                        vk::BufferCopy::default().size(handle.size as _);
                                    unsafe {
                                        device.functions().cmd_copy_buffer(
                                            cmd.raw(),
                                            staging_buf.buffer,
                                            handle.buffer,
                                            &[copy_info],
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

fn write_chunk_static<T: palace_core::storage::Element + hdf5::H5Type>(
    dataset: &hdf5::Dataset,
    chunk_data: &[u8],
    out_info: ChunkInfo<DDyn>,
) {
    let typed_data: &[T] = bytemuck::cast_slice(chunk_data);
    let chunk = data::chunk(typed_data, &out_info);
    let slice_arg = chunk_to_slice_arg(out_info);
    dataset
        .write_slice(chunk.as_standard_layout().view(), slice_arg)
        .unwrap();
}

fn write_chunk(
    dataset: &hdf5::Dataset,
    chunk_data: &[u8],
    out_info: ChunkInfo<DDyn>,
    dtype: ScalarType,
) {
    match dtype {
        ScalarType::U8 => write_chunk_static::<u8>(dataset, chunk_data, out_info),
        ScalarType::I8 => write_chunk_static::<i8>(dataset, chunk_data, out_info),
        ScalarType::U16 => write_chunk_static::<u16>(dataset, chunk_data, out_info),
        ScalarType::I16 => write_chunk_static::<i16>(dataset, chunk_data, out_info),
        ScalarType::F32 => write_chunk_static::<f32>(dataset, chunk_data, out_info),
        ScalarType::U32 => write_chunk_static::<u32>(dataset, chunk_data, out_info),
        ScalarType::I32 => write_chunk_static::<i32>(dataset, chunk_data, out_info),
        ScalarType::U64 => write_chunk_static::<u64>(dataset, chunk_data, out_info),
        ScalarType::I64 => write_chunk_static::<i64>(dataset, chunk_data, out_info),
    }
}

async fn write_tensor<'cref, 'inv>(
    ctx: OpaqueTaskContext<'cref, 'inv>,
    dataset: &hdf5::Dataset,
    t: &'inv TensorOperator<DDyn, DType>,
) -> Result<(), palace_core::Error> {
    let md = &t.metadata;

    let num_total = md.dimension_in_chunks().hmul();
    println!("{} chunks to save", num_total);

    let scalar_dtype = t.dtype().scalar;
    assert_eq!(t.dtype().size, 1);

    let request_chunk_size = 1024;
    let chunk_ids_in_parts = md.chunk_indices().chunks(request_chunk_size);
    let mut i = 0;
    for chunk_ids in &chunk_ids_in_parts {
        let requests = chunk_ids.map(|chunk_id| (t.chunks.request_raw(chunk_id), chunk_id));
        let stream =
            ctx.submit_unordered_with_data(requests)
                .then_req(ctx, |(chunk_handle, chunk_id)| {
                    let chunk_info = md.chunk_info(chunk_id);

                    let chunk_handle = chunk_handle.into_thread_handle();
                    ctx.spawn_compute(move || {
                        write_chunk(dataset, chunk_handle.data(), chunk_info, scalar_dtype);
                        chunk_handle
                    })
                });
        futures::pin_mut!(stream);
        while let Some(handle) = stream.next().await {
            let _handle = handle.into_main_handle(ctx.storage());
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

fn create_dataset_for_tensor(
    file: &hdf5::File,
    t: &TensorOperator<DDyn, DType>,
    location: &str,
    hints: &WriteHints,
) -> Result<hdf5::Dataset, palace_core::Error> {
    let md = &t.metadata;
    let dtype = dtype_palace_to_hdf5(t.dtype())?.to_descriptor()?;
    let mut file = file
        .new_dataset_builder()
        .empty_as(&dtype)
        .shape(extent_palace_to_hdf(&md.dimensions))
        .chunk(chunk_size(&md));
    if hints.compression_level > 0 {
        file = file.deflate(hints.compression_level);
    }
    Ok(file.create(location)?)
}

fn create_dataset_for_embedded_tensor(
    file: &hdf5::File,
    t: &EmbeddedTensorOperator<DDyn, DType>,
    location: &str,
    hints: &WriteHints,
) -> Result<hdf5::Dataset, palace_core::Error> {
    let dataset = create_dataset_for_tensor(&file, &t.inner, location, hints)?;
    dataset
        .new_attr_builder()
        .with_data(
            &t.embedding_data
                .spacing
                .scale(SPACING_FACTOR_MEM_TO_FILE)
                .inner(),
        )
        .create(SPACING_KEY)?;

    Ok(dataset)
}

pub async fn save_tensor<'cref, 'inv>(
    ctx: OpaqueTaskContext<'cref, 'inv>,
    path: &Path,
    t: &'inv TensorOperator<DDyn, DType>,
    hints: &WriteHints,
) -> Result<(), palace_core::Error> {
    let file = hdf5::FileBuilder::new().create(path)?;

    let dataset = create_dataset_for_tensor(&file, t, "volume", hints)?;

    write_tensor(ctx, &dataset, t).await
}

pub async fn save_embedded_tensor<'cref, 'inv>(
    ctx: OpaqueTaskContext<'cref, 'inv>,
    path: &Path,
    t: &'inv EmbeddedTensorOperator<DDyn, DType>,
    hints: &WriteHints,
) -> Result<(), palace_core::Error> {
    let file = hdf5::FileBuilder::new().create(path)?;

    let dataset = create_dataset_for_embedded_tensor(&file, &t, "volume", hints)?;

    write_tensor(ctx, &dataset, t).await
}

fn level_path(level: usize) -> String {
    format!("/level{}", level)
}

#[derive(Clone)]
pub struct WriteHints {
    pub compression_level: u8,
    pub lod_downsample_steps: Option<Vector<DDyn, DownsampleStep>>,
}

pub fn save_lod_tensor(
    runtime: &mut palace_core::runtime::RunTime,
    path: &Path,
    t: &LODTensorOperator<DDyn, DType>,
    hints: &WriteHints,
    recreate_lod: bool,
) -> Result<(), palace_core::Error> {
    let file = hdf5::FileBuilder::new().create(path)?;

    if recreate_lod {
        let mut current = t.levels[0].clone();
        let mut current_level = 0;

        let steps = hints.lod_downsample_steps.clone().unwrap_or_else(|| {
            Vector::fill_with_len(DownsampleStep::Synchronized(2.0), t.levels[0].dim().n())
        });

        loop {
            let current_location = level_path(current_level);
            let current_location_ref = &current_location;
            let current_ref = &current;

            let file = &file;
            runtime.resolve(None, false, |ctx, _| {
                async move {
                    let dataset = create_dataset_for_embedded_tensor(
                        file,
                        &current_ref,
                        &current_location_ref,
                        hints,
                    )?;

                    write_tensor(ctx, &dataset, current_ref).await
                }
                .into()
            })?;

            if !palace_core::operators::resample::can_reduce_further(
                &current.metadata.dimension_in_chunks(),
                &steps,
            ) {
                break;
            }

            let new_md = palace_core::operators::resample::coarser_lod_md(&current, steps.clone());
            std::mem::drop(current);

            current = open(path.into(), Some(current_location))?;

            current =
                palace_core::operators::resample::smooth_downsample(current, new_md.clone()).into();
            current_level += 1;
        }
    } else {
        runtime.resolve(None, false, |ctx, _| {
            async move {
                for (level, tensor) in t.levels.iter().enumerate() {
                    let dataset = create_dataset_for_embedded_tensor(
                        &file,
                        &tensor,
                        &level_path(level),
                        hints,
                    )?;

                    write_tensor(ctx, &dataset, tensor).await?;
                }
                Ok(())
            }
            .into()
        })?;
    }

    Ok(())
}
