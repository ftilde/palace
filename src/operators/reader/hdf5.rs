use std::path::PathBuf;

use crate::{
    data::{to_linear, VolumeMetaData, VoxelPosition},
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

impl TryFrom<Vec<hdf5::Ix>> for VoxelPosition {
    type Error = crate::Error;

    fn try_from(value: Vec<hdf5::Ix>) -> Result<Self, Self::Error> {
        match *value {
            [z, y, x] => Ok(VoxelPosition((x as u32, y as u32, z as u32).into())),
            _ => Err("Invalid number of dimensions".into()),
        }
    }
}

fn to_hdf5(pos: VoxelPosition) -> [usize; 3] {
    [pos.0.z as _, pos.0.y as _, pos.0.x as _]
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
                        let begin = self.metadata.brick_begin(pos);
                        let end = self.metadata.brick_end(pos);

                        let selection = to_hdf5_hyperslab(begin, end);

                        let num_voxels = crate::data::hmul(self.metadata.brick_size.0) as usize;

                        let mut brick_handle = ctx.alloc_slot(pos, num_voxels)?;
                        let brick_data = &mut *brick_handle;
                        let brick_dim = self.metadata.brick_dim(pos);
                        ctx.submit(ctx.spawn_io(|| {
                            brick_data.iter_mut().for_each(|v| {
                                v.write(0.0);
                            });

                            let vals = self
                                .dataset
                                .read_slice::<f32, _, ndarray::Ix3>(selection)
                                .unwrap();

                            for z in 0..brick_dim.0.z {
                                for y in 0..brick_dim.0.y {
                                    let line_begin = 0;
                                    let line_end = brick_dim.0.x;
                                    let in_ = vals.slice(ndarray::s!(
                                        z as usize,
                                        y as usize,
                                        line_begin as usize..line_end as usize
                                    ));

                                    let bf32 = to_linear(
                                        cgmath::vec3(line_begin, y, z),
                                        self.metadata.brick_size.0,
                                    );
                                    let ef32 = to_linear(
                                        cgmath::vec3(line_end, y, z),
                                        self.metadata.brick_size.0,
                                    );

                                    let out = &mut brick_data[bf32..ef32];
                                    for (in_, out) in in_.iter().zip(out.iter_mut()) {
                                        out.write(*in_);
                                    }
                                }
                            }
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
        let brick_size: VoxelPosition = vol.chunk().unwrap_or_else(|| vol.shape()).try_into()?;

        let dtype = vol.dtype()?;
        if !dtype.is::<f32>() {
            return Err("Only f32 volumes are supported".into());
        }

        let metadata = VolumeMetaData {
            dimensions,
            brick_size,
        };

        Ok(Hdf5VolumeSourceState {
            metadata,
            dataset: vol,
            path,
            volume_location,
        })
    }
}
