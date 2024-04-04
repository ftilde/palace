use palace_core::array::VolumeEmbeddingData;
use palace_core::data::{Coordinate, CoordinateType};
use palace_core::dim::D3;
use std::path::PathBuf;
use std::rc::Rc;

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

pub fn open(path: PathBuf, volume_location: String) -> Result<EmbeddedVolumeOperator<f32>, Error> {
    let state = Hdf5VolumeSourceState::open(path, volume_location)?;
    Ok(state.operate())
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

        let dtype = vol.dtype()?;
        if !dtype.is::<f32>() {
            return Err("Only f32 volumes are supported".into());
        }

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
            }),
        })
    }

    fn operate(&self) -> EmbeddedVolumeOperator<f32> {
        TensorOperator::with_state(
            OperatorDescriptor::new("Hdf5VolumeSourceState::operate")
                .dependent_on_data(self.inner.path.to_string_lossy().as_bytes())
                .dependent_on_data(self.inner.volume_location.as_bytes()),
            self.inner.metadata,
            self.clone(),
            move |ctx, positions, this| {
                async move {
                    let metadata = this.inner.metadata;
                    for (pos, _) in positions {
                        let chunk = metadata.chunk_info(pos);

                        let selection = to_hdf5_hyperslab(chunk.begin(), chunk.end());

                        let num_voxels = this.inner.metadata.chunk_size.hmul();

                        let mut brick_handle = ctx.submit(ctx.alloc_slot(pos, num_voxels)).await;
                        let brick_data = &mut *brick_handle;
                        let dataset = &this.inner.dataset;
                        ctx.submit(ctx.spawn_io(|| {
                            palace_core::data::init_non_full(brick_data, &chunk, f32::NAN);

                            let out_info = metadata.chunk_info(pos);
                            let mut out_chunk = crate::data::chunk_mut(brick_data, &out_info);
                            let in_chunk = dataset
                                .read_slice::<f32, _, ndarray::Ix3>(selection)
                                .unwrap();
                            ndarray::azip!((o in &mut out_chunk, i in &in_chunk) { o.write(*i); });
                        }))
                        .await;

                        // Safety: At this point the thread pool job above has finished and has initialized all bytes
                        // in the brick.
                        unsafe { brick_handle.initialized(*ctx) };
                    }
                    Ok(())
                }
                .into()
            },
        )
        .embedded(self.inner.embedding_data)
    }
}
