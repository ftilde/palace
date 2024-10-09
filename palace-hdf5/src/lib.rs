use palace_core::array::{ChunkInfo, VolumeEmbeddingData};
use palace_core::data::{Coordinate, CoordinateType};
use palace_core::dim::D3;
use palace_core::dtypes::{DType, ElementType, ScalarType};
use palace_core::operator::DataDescriptor;
use palace_core::storage::DataLocation;
use palace_core::util::Map;
use palace_core::vulkan::{vk, DeviceId};
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::rc::Rc;

use hdf5::Datatype;

use palace_core::{
    array::VolumeMetaData,
    data::{self, LocalVoxelPosition, Vector, VoxelPosition},
    operator::OperatorDescriptor,
    operators::{tensor::TensorOperator, volume::EmbeddedVolumeOperator},
    Error,
};

#[derive(Clone)]
pub struct Hdf5VolumeSourceState {
    inner: Rc<Hdf5VolumeSourceStateInner>,
}

pub struct Hdf5VolumeSourceStateInner {
    metadata: VolumeMetaData,
    embedding_data: VolumeEmbeddingData,
    dataset: hdf5::Dataset,
    path: PathBuf,
    volume_location: String,
    dtype: DType,
}

fn to_size_vector<C: CoordinateType>(
    value: Vec<hdf5::Ix>,
) -> Result<Vector<D3, Coordinate<C>>, palace_core::Error> {
    to_vector(value.into_iter().map(|v| v as u32).collect())
}

fn to_vector<I: Copy, O: From<I> + Copy>(
    value: Vec<I>,
) -> Result<Vector<D3, O>, palace_core::Error> {
    match *value {
        [z, y, x] => Ok([z, y, x].into()),
        _ => Err("Invalid number of dimensions".into()),
    }
}

fn to_hdf5(pos: VoxelPosition) -> [usize; 3] {
    [pos.z().raw as _, pos.y().raw as _, pos.x().raw as _]
}

fn to_hdf5_hyperslab(begin: VoxelPosition, end: VoxelPosition) -> hdf5::Hyperslab {
    let begin = to_hdf5(begin);
    let end = to_hdf5(end);

    (begin[0]..end[0], begin[1]..end[1], begin[2]..end[2]).into()
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

pub fn open(
    path: PathBuf,
    volume_location: String,
) -> Result<EmbeddedVolumeOperator<DType>, Error> {
    let state = Hdf5VolumeSourceState::open(path, volume_location)?;
    Ok(state.operate())
}

fn copy_chunk_inner<T: hdf5::H5Type + Copy>(
    dataset: &hdf5::Container,
    selection: hdf5::Hyperslab,
    brick_data: &mut [MaybeUninit<u8>],
    out_info: ChunkInfo<D3>,
) {
    let byte_slice_len = brick_data.len();
    assert_eq!(byte_slice_len % std::mem::size_of::<T>(), 0);
    let elm_slice_len = byte_slice_len / std::mem::size_of::<T>();
    let elm_slice_ptr: *mut MaybeUninit<T> = brick_data.as_mut_ptr().cast();
    assert!(elm_slice_ptr.is_aligned());
    let brick_data: &mut [MaybeUninit<T>] =
        unsafe { std::slice::from_raw_parts_mut(elm_slice_ptr.cast(), elm_slice_len) };

    let mut out_chunk = crate::data::chunk_mut(brick_data, &out_info);
    let in_chunk = dataset.read_slice::<T, _, ndarray::Ix3>(selection).unwrap();
    ndarray::azip!((o in &mut out_chunk, i in &in_chunk) { o.write(*i); });
}
fn copy_chunk(
    dataset: &hdf5::Container,
    selection: hdf5::Hyperslab,
    dtype: DType,
    brick_data: &mut [MaybeUninit<u8>],
    out_info: ChunkInfo<D3>,
) {
    assert!(dtype.is_scalar());

    match dtype.scalar {
        ScalarType::U8 => copy_chunk_inner::<u8>(dataset, selection, brick_data, out_info),
        ScalarType::I8 => copy_chunk_inner::<i8>(dataset, selection, brick_data, out_info),
        ScalarType::U16 => copy_chunk_inner::<u16>(dataset, selection, brick_data, out_info),
        ScalarType::I16 => copy_chunk_inner::<i16>(dataset, selection, brick_data, out_info),
        ScalarType::F32 => copy_chunk_inner::<f32>(dataset, selection, brick_data, out_info),
        ScalarType::U32 => copy_chunk_inner::<u32>(dataset, selection, brick_data, out_info),
        ScalarType::I32 => copy_chunk_inner::<i32>(dataset, selection, brick_data, out_info),
    }
}

impl Hdf5VolumeSourceState {
    pub fn open(path: PathBuf, volume_location: String) -> Result<Self, Error> {
        let file = hdf5::File::open(&path)?;
        let vol = file.dataset(&volume_location)?;
        let dimensions: VoxelPosition = to_size_vector(vol.shape())?;
        let brick_size: LocalVoxelPosition =
            to_size_vector(vol.chunk().unwrap_or_else(|| vol.shape()))?;
        //println!("Chunksize {:?}", brick_size);
        let spacing: Result<Vector<D3, f32>, Error> = vol
            .attr("element_size_um")
            .and_then(|a| a.read_1d::<f32>())
            .map_err(|e| e.into())
            .and_then(|s| to_vector(s.to_vec()).map(|v| v.scale(0.001)));

        let spacing = match spacing {
            Ok(spacing) => spacing,
            Err(e) => {
                eprintln!(
                    "Could not load spacing from dataset: {}\n Using default spacing.",
                    e
                );
                Vector::fill(1.0)
            }
        };

        let dtype = dtype_hdf5_to_palace(&vol.dtype()?)?;

        let metadata = VolumeMetaData {
            dimensions,
            chunk_size: brick_size,
        };

        let embedding_data = VolumeEmbeddingData { spacing };

        Ok(Hdf5VolumeSourceState {
            inner: Rc::new(Hdf5VolumeSourceStateInner {
                metadata,
                embedding_data,
                dataset: vol,
                path,
                volume_location,
                dtype,
            }),
        })
    }

    fn operate(&self) -> EmbeddedVolumeOperator<DType> {
        TensorOperator::with_state(
            OperatorDescriptor::new("Hdf5VolumeSourceState::operate")
                .dependent_on_data(self.inner.path.to_string_lossy().as_bytes())
                .dependent_on_data(self.inner.volume_location.as_bytes()),
            self.inner.dtype,
            self.inner.metadata,
            self.clone(),
            move |ctx, positions, this| {
                async move {
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

                    let metadata = this.inner.metadata;
                    for pos in positions_cpu {
                        let chunk = metadata.chunk_info(pos);

                        let selection = to_hdf5_hyperslab(*chunk.begin(), chunk.end());

                        let num_voxels = this.inner.metadata.chunk_size.hmul();

                        let dtype = this.inner.dtype;
                        let layout = dtype.array_layout(num_voxels);

                        let data_id = DataDescriptor::new(ctx.current_op_desc().unwrap(), pos);
                        let mut brick_handle = ctx.submit(ctx.alloc_raw(data_id, layout)).await;
                        let brick_data = brick_handle.data();
                        let dataset = &this.inner.dataset;
                        ctx.submit(ctx.spawn_io(|| {
                            palace_core::data::init_non_full(brick_data, &chunk, 0);
                            let out_info = metadata.chunk_info(pos);

                            copy_chunk(&dataset, selection, dtype, brick_data, out_info);
                        }))
                        .await;

                        // Safety: At this point the thread pool job above has finished and has initialized all bytes
                        // in the brick.
                        unsafe { brick_handle.initialized(*ctx) };
                    }

                    for (device_id, positions_gpu) in positions_gpus {
                        let device = ctx.device_ctx(device_id);
                        for pos in positions_gpu {
                            let chunk = metadata.chunk_info(pos);

                            let selection = to_hdf5_hyperslab(*chunk.begin(), chunk.end());

                            let num_voxels = this.inner.metadata.chunk_size.hmul();

                            let dtype = this.inner.dtype;
                            let layout = dtype.array_layout(num_voxels);

                            let data_id = DataDescriptor::new(ctx.current_op_desc().unwrap(), pos);
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

                            let dataset = &this.inner.dataset;
                            ctx.submit(ctx.spawn_io(|| {
                                palace_core::data::init_non_full(chunk_data, &chunk, 0);
                                let out_info = metadata.chunk_info(pos);

                                copy_chunk(&dataset, selection, dtype, chunk_data, out_info);
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
                    Ok(())
                }
                .into()
            },
        )
        .embedded(self.inner.embedding_data)
        .into()
    }
}
