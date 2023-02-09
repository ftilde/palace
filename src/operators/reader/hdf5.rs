use std::path::PathBuf;

use crate::{
    array::VolumeMetaData,
    data::{Coordinate, CoordinateType, LocalVoxelPosition, Vector, VoxelPosition},
    operator::OperatorId,
    operators::volume::{VolumeOperator, VolumeOperatorState},
    Error,
};

pub struct Hdf5VolumeSourceState {
    metadata: VolumeMetaData,
    dataset: hdf5::Dataset,
    path: PathBuf,
    volume_location: String,
}

impl<C: CoordinateType> TryFrom<Vec<hdf5::Ix>> for Vector<3, Coordinate<C>> {
    type Error = crate::Error;

    fn try_from(value: Vec<hdf5::Ix>) -> Result<Self, Self::Error> {
        match *value {
            [z, y, x] => Ok([(z as u32), (y as u32), (x as u32)].into()),
            _ => Err("Invalid number of dimensions".into()),
        }
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

impl VolumeOperatorState for Hdf5VolumeSourceState {
    fn operate<'a>(&'a self) -> VolumeOperator<'a> {
        VolumeOperator::new(
            OperatorId::new("Hdf5VolumeSourceState::operate")
                .dependent_on(self.path.to_string_lossy().as_bytes())
                .dependent_on(self.volume_location.as_bytes()),
            move |ctx, _| async move { ctx.write(self.metadata) }.into(),
            move |ctx, positions, _| {
                async move {
                    for pos in positions {
                        let chunk = self.metadata.chunk_info(pos);

                        let selection = to_hdf5_hyperslab(chunk.begin(), chunk.end());

                        let num_voxels = crate::data::hmul(self.metadata.chunk_size);

                        let mut brick_handle = ctx.alloc_slot(pos, num_voxels)?;
                        let brick_data = &mut *brick_handle;
                        ctx.submit(ctx.spawn_io(|| {
                            crate::data::init_non_full(brick_data, &chunk, f32::NAN);

                            let out_info = self.metadata.chunk_info(pos);
                            let mut out_chunk = crate::data::chunk_mut(brick_data, &out_info);
                            let in_chunk = self
                                .dataset
                                .read_slice::<f32, _, ndarray::Ix3>(selection)
                                .unwrap();
                            ndarray::azip!((o in &mut out_chunk, i in &in_chunk) { o.write(*i); });
                        }))
                        .await;

                        // Safety: At this point the thread pool job above has finished and has initialized all bytes
                        // in the brick.
                        unsafe { brick_handle.initialized() };
                    }
                    Ok(())
                }
                .into()
            },
        )
    }
}

impl Hdf5VolumeSourceState {
    pub fn open(path: PathBuf, volume_location: String) -> Result<Self, Error> {
        let file = hdf5::File::open(&path)?;
        let vol = file.dataset(&volume_location)?;
        let dimensions: VoxelPosition = vol.shape().try_into()?;
        let brick_size: LocalVoxelPosition =
            vol.chunk().unwrap_or_else(|| vol.shape()).try_into()?;

        let dtype = vol.dtype()?;
        if !dtype.is::<f32>() {
            return Err("Only f32 volumes are supported".into());
        }

        let metadata = VolumeMetaData {
            dimensions,
            chunk_size: brick_size,
        };

        Ok(Hdf5VolumeSourceState {
            metadata,
            dataset: vol,
            path,
            volume_location,
        })
    }
}
