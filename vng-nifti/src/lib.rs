use std::path::PathBuf;

use nifti::{IntoNdArray, NiftiHeader, NiftiObject};

use vng_core::{
    array::VolumeMetaData,
    data::{self, LocalCoordinate, VoxelPosition},
    operator::OperatorId,
    operators::{
        tensor::TensorOperator,
        volume::{VolumeOperator, VolumeOperatorState},
    },
    Error,
};

enum Type {
    Single(PathBuf),
    Separate { header: PathBuf, data: PathBuf },
}

pub struct NiftiVolumeSourceState {
    metadata: VolumeMetaData,
    type_: Type,
    header: NiftiHeader,
}

fn check_type(header: &NiftiHeader) -> Result<(), Error> {
    match header.data_type()? {
        nifti::NiftiType::Float32 => {}
        o => return Err(format!("Unsupported data type {:?}", o).into()),
    }
    Ok(())
}

fn read_metadata(header: &NiftiHeader) -> Result<VolumeMetaData, Error> {
    let dimensions = match *header.dim()? {
        [x, y, z] => VoxelPosition::from([(z as u32), (y as u32), (x as u32)]),
        _ => return Err("Invalid number of dimensions".into()),
    };

    let chunk_size = [
        1.into(),
        LocalCoordinate::interpret_as(dimensions.y()),
        LocalCoordinate::interpret_as(dimensions.x()),
    ]
    .into();

    let metadata = VolumeMetaData {
        dimensions,
        chunk_size,
    };
    Ok(metadata)
}

impl NiftiVolumeSourceState {
    pub fn open_single(path: PathBuf) -> Result<Self, Error> {
        let obj = nifti::ReaderStreamedOptions::new().read_file(&path)?;
        let header = obj.header();
        check_type(header)?;
        let metadata = read_metadata(header)?;

        Ok(NiftiVolumeSourceState {
            metadata,
            type_: Type::Single(path),
            header: header.clone(),
        })
    }
    pub fn open_separate(header_path: PathBuf, data: PathBuf) -> Result<Self, Error> {
        let obj = nifti::ReaderStreamedOptions::new().read_file_pair(&header_path, &data)?;
        let header = obj.header();
        check_type(header)?;
        let metadata = read_metadata(header)?;

        Ok(NiftiVolumeSourceState {
            metadata,
            type_: Type::Separate {
                header: header_path,
                data,
            },
            header: header.clone(),
        })
    }
}

impl VolumeOperatorState for NiftiVolumeSourceState {
    fn operate<'a>(&'a self) -> VolumeOperator<'a> {
        TensorOperator::new(
            match &self.type_ {
                Type::Single(path) => OperatorId::new("NiftiVolumeSourceState::operate")
                    .dependent_on(path.to_string_lossy().as_bytes()),
                Type::Separate { header, data } => {
                    OperatorId::new("NiftiVolumeSourceState::operate")
                        .dependent_on(header.to_string_lossy().as_bytes())
                        .dependent_on(data.to_string_lossy().as_bytes())
                }
            },
            move |ctx, _, _| async move { ctx.write(self.metadata) }.into(),
            move |ctx, mut positions, _, _| {
                positions.sort_by_key(|p| p.z());
                async move {
                    let obj = match &self.type_ {
                        Type::Single(path) => {
                            nifti::ReaderStreamedOptions::new().read_file(path)?
                        }
                        Type::Separate { header, data } => {
                            nifti::ReaderStreamedOptions::new().read_file_pair(header, data)?
                        }
                    };
                    if self.header != *obj.header() {
                        return Err(format!("File has changed").into());
                    }
                    let mut vol = obj.into_volume();
                    for pos in positions {
                        let z = pos.z().raw as usize;
                        let chunk = self.metadata.chunk_info(pos);
                        let mut brick_handle = ctx.alloc_slot(pos, chunk.mem_elements())?;

                        let brick_data = &mut *brick_handle;
                        ctx.submit(ctx.spawn_io(|| {
                            // Skip unwanted slices
                            while vol.slices_read() != z {
                                // TODO: This is not ideal (to put it midly) since we do a lot of
                                // work while reading these slices. We could (1.) look into maybe
                                // add a seek function to the nifti crate or (2.) somehow encourage
                                // a more "complete" of slices in a single task.
                                let _ = vol.read_slice();
                            }
                            let in_chunk = vol.read_slice().unwrap().into_ndarray::<f32>().unwrap();
                            let s = chunk.logical_dimensions;
                            let s = (s.z().raw as usize, s.y().raw as usize, s.x().raw as usize);
                            let in_chunk = in_chunk.to_shape(s).unwrap();

                            let mut out_chunk = data::chunk_mut(brick_data, &chunk);

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
