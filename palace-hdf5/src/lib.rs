use hidefix::idx::DatasetD;
use id::{Id, Identify};
use palace_core::array::{ChunkInfo, TensorEmbeddingData, TensorMetaData};
use palace_core::data::{Coordinate, CoordinateType, GlobalCoordinate, LocalCoordinate};
use palace_core::dim::DDyn;
use palace_core::dtypes::{DType, ElementType, ScalarType};
use palace_core::op_descriptor;
use palace_core::operator::{DataDescriptor, DataParam};
use palace_core::operators::tensor::EmbeddedTensorOperator;
use palace_core::storage::DataLocation;
use palace_core::vulkan::vk;
use std::borrow::Cow;
use std::fs::File;
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::rc::Rc;

use hdf5::Datatype;
use hidefix::prelude::*;

use palace_core::{
    data::{self, Vector},
    operator::OperatorDescriptor,
    operators::tensor::TensorOperator,
    Error,
};

#[derive(Clone)]
pub struct Hdf5TensorSourceState {
    inner: Rc<Hdf5TensorSourceStateInner>,
}

pub struct Hdf5TensorSourceStateInner {
    metadata: TensorMetaData<DDyn>,
    embedding_data: TensorEmbeddingData<DDyn>,
    dataset_index: DatasetD<'static>,
    _dataset_file: File,
    dataset_mmap: memmap::Mmap,
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
    } else if d.is::<f32>() {
        ScalarType::F32
    } else {
        return Err(format!("No palace correspondence for {:?}", d).into());
    };
    Ok(DType::scalar(scalar))
}

//fn dtype_palace_to_hdf5(d: DType) -> Result<Datatype, Error> {
//    if d.size != 1 {
//        Err(format!("No zarr correspondence for {:?}", d))?;
//    }
//
//    Ok(match d.scalar {
//        ScalarType::U8 => Datatype::from_type::<u8>()?,
//        ScalarType::I8 => Datatype::from_type::<u8>()?,
//        ScalarType::U16 => Datatype::from_type::<u16>()?,
//        ScalarType::I16 => Datatype::from_type::<i16>()?,
//        ScalarType::F32 => Datatype::from_type::<f32>()?,
//        ScalarType::U32 => Datatype::from_type::<u32>()?,
//        ScalarType::I32 => Datatype::from_type::<i32>()?,
//    })
//}

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
    volume_location: String,
) -> Result<EmbeddedTensorOperator<DDyn, DType>, Error> {
    let state = Hdf5TensorSourceState::open(path, volume_location)?;
    Ok(state.operate())
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
    mmap: &memmap::Mmap,
    chunk_data_out: &mut [MaybeUninit<u8>],
    out_info: ChunkInfo<DDyn>,
) -> Result<(), Error> {
    let chunk_addr = chunk_info(
        dataset,
        out_info.begin().map(|v| v.raw as u64).inner().as_slice(),
    );

    let chunk_data_raw = &mmap[chunk_addr.addr as usize..][..chunk_addr.size as usize];
    let storage_info = storage_info(dataset);
    let chunk_data = decode_chunk(chunk_data_raw, &storage_info)?;
    assert_eq!(chunk_data_out.len(), chunk_data.len());
    data::write_slice_uninit(chunk_data_out, &chunk_data);

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
            .attr("element_size_um")
            .and_then(|a| a.read_1d::<f32>())
            .map_err(|e| e.into())
            .map(|s| to_vector(s.to_vec()).scale(0.001));

        let spacing = match spacing {
            Ok(spacing) => spacing,
            Err(e) => {
                eprintln!(
                    "Could not load spacing from dataset: {}\n Using default spacing.",
                    e
                );
                Vector::fill_with_len(1.0, dimensions.len())
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

        let file = File::open(&path)?;
        let mmap = unsafe { memmap::Mmap::map(&file)? };

        Ok(Hdf5TensorSourceState {
            inner: Rc::new(Hdf5TensorSourceStateInner {
                metadata,
                embedding_data,
                dataset_index: dset,
                dataset_mmap: mmap,
                _dataset_file: file,
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
                async move {
                    let metadata = &this.inner.metadata;

                    match location {
                        DataLocation::CPU(_) => {
                            for pos in positions {
                                let chunk = metadata.chunk_info(pos);

                                let num_voxels = this.inner.metadata.chunk_size.hmul();

                                let dtype = this.inner.dtype;
                                let layout = dtype.array_layout(num_voxels);

                                let data_id =
                                    DataDescriptor::new(ctx.current_op_desc().unwrap(), pos);
                                let mut chunk_handle =
                                    ctx.submit(ctx.alloc_raw(data_id, layout)).await;
                                let chunk_data = chunk_handle.data();
                                let dataset = &this.inner.dataset_index;
                                let mmap = &this.inner.dataset_mmap;
                                ctx.submit(ctx.spawn_io(|| {
                                    palace_core::data::init_non_full(chunk_data, &chunk, 0);
                                    let out_info = metadata.chunk_info(pos);

                                    copy_chunk(&dataset, mmap, chunk_data, out_info).unwrap();
                                }))
                                .await;

                                // Safety: At this point the thread pool job above has finished and has initialized all bytes
                                // in the brick.
                                unsafe { chunk_handle.initialized(*ctx) };
                            }
                        }
                        DataLocation::GPU(device_id) => {
                            let device = ctx.device_ctx(device_id);
                            for pos in positions {
                                let chunk = metadata.chunk_info(pos);

                                let num_voxels = this.inner.metadata.chunk_size.hmul();

                                let dtype = this.inner.dtype;
                                let layout = dtype.array_layout(num_voxels);

                                let data_id =
                                    DataDescriptor::new(ctx.current_op_desc().unwrap(), pos);
                                let brick_handle =
                                    ctx.submit(ctx.alloc_raw_gpu(device, data_id, layout)).await;

                                let staging_buf = ctx
                                    .submit(device.staging_to_gpu.request(device, layout))
                                    .await;

                                let ptr = staging_buf
                                    .mapped_ptr()
                                    .unwrap()
                                    .cast::<MaybeUninit<u8>>()
                                    .as_ptr();
                                let chunk_data = unsafe {
                                    std::slice::from_raw_parts_mut(ptr, staging_buf.size as usize)
                                };

                                let dataset = &this.inner.dataset_index;
                                let mmap = &this.inner.dataset_mmap;
                                ctx.submit(ctx.spawn_io(|| {
                                    palace_core::data::init_non_full(chunk_data, &chunk, 0);
                                    let out_info = metadata.chunk_info(pos);

                                    copy_chunk(&dataset, mmap, chunk_data, out_info).unwrap();
                                }))
                                .await;

                                device.with_cmd_buffer(|cmd| {
                                    let copy_info =
                                        vk::BufferCopy::default().size(brick_handle.size as _);
                                    unsafe {
                                        device.functions().cmd_copy_buffer(
                                            cmd.raw(),
                                            staging_buf.buffer,
                                            brick_handle.buffer,
                                            &[copy_info],
                                        );
                                    }
                                });

                                unsafe {
                                    brick_handle.initialized(
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
