use std::{path::PathBuf, rc::Rc};

use id::{Id, Identify};
use nifti::{IntoNdArray, NiftiHeader, NiftiObject};

use palace_core::{
    array::{TensorEmbeddingData, VolumeEmbeddingData, VolumeMetaData},
    data::{self, LocalCoordinate, Vector, VoxelPosition},
    dim::*,
    dtypes::{DType, StaticElementType},
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    operators::{tensor::TensorOperator, volume::EmbeddedVolumeOperator},
    Error,
};

enum Type {
    Single(PathBuf),
    Separate { header: PathBuf, data: PathBuf },
}

#[derive(Clone)]
pub struct NiftiVolumeSourceState(Rc<NiftiVolumeSourceStateInner>);

pub struct NiftiVolumeSourceStateInner {
    metadata: VolumeMetaData,
    embedding_data: TensorEmbeddingData<D3>,
    type_: Type,
    header: NiftiHeader,
}

impl Identify for NiftiVolumeSourceState {
    fn id(&self) -> id::Id {
        match &self.0.type_ {
            Type::Single(path) => Id::from_data(path.to_string_lossy().as_bytes()),
            Type::Separate { header, data } => Id::combine(&[
                Id::from_data(header.to_string_lossy().as_bytes()),
                Id::from_data(data.to_string_lossy().as_bytes()),
            ]),
        }
    }
}

fn check_type(header: &NiftiHeader) -> Result<(), Error> {
    match header.data_type()? {
        nifti::NiftiType::Float32 => {}
        o => return Err(format!("Unsupported data type {:?}", o).into()),
    }
    Ok(())
}

fn read_metadata(header: &NiftiHeader) -> Result<(VolumeMetaData, VolumeEmbeddingData), Error> {
    let dimensions = match *header.dim()? {
        [x, y, z] => VoxelPosition::from([(z as u32), (y as u32), (x as u32)]),
        _ => return Err("Invalid number of dimensions".into()),
    };
    let sx = header.pixdim[1];
    let sy = header.pixdim[2];
    let sz = header.pixdim[3];
    let factor = match header.xyzt_units()?.0 {
        nifti::Unit::Meter => 1000.0,
        nifti::Unit::Mm => 1.0,
        nifti::Unit::Micron => 0.001,
        _ => return Err("Invalid length unit".into()),
    };
    let spacing = Vector::from([sz, sy, sx]).scale(factor);

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
    let embedding_data = VolumeEmbeddingData { spacing };
    Ok((metadata, embedding_data))
}

pub fn open_single(path: PathBuf) -> Result<EmbeddedVolumeOperator<DType>, Error> {
    let vol = NiftiVolumeSourceState::open_single(path)?;
    Ok(vol.operate().into())
}

pub fn open_separate(
    header_path: PathBuf,
    data: PathBuf,
) -> Result<EmbeddedVolumeOperator<DType>, Error> {
    let vol = NiftiVolumeSourceState::open_separate(header_path, data)?;
    Ok(vol.operate().into())
}

impl NiftiVolumeSourceState {
    pub fn open_single(path: PathBuf) -> Result<Self, Error> {
        let obj = nifti::ReaderStreamedOptions::new().read_file(&path)?;
        let header = obj.header();
        check_type(header)?;
        let (metadata, embedding_data) = read_metadata(header)?;

        Ok(Self(Rc::new(NiftiVolumeSourceStateInner {
            metadata,
            embedding_data,
            type_: Type::Single(path),
            header: header.clone(),
        })))
    }
    pub fn open_separate(header_path: PathBuf, data: PathBuf) -> Result<Self, Error> {
        let obj = nifti::ReaderStreamedOptions::new().read_file_pair(&header_path, &data)?;
        let header = obj.header();
        check_type(header)?;
        let (metadata, embedding_data) = read_metadata(header)?;

        Ok(Self(Rc::new(NiftiVolumeSourceStateInner {
            metadata,
            embedding_data,
            type_: Type::Separate {
                header: header_path,
                data,
            },
            header: header.clone(),
        })))
    }

    fn operate(&self) -> EmbeddedVolumeOperator<StaticElementType<f32>> {
        TensorOperator::with_state(
            match &self.0.type_ {
                Type::Single(path) => {
                    op_descriptor!().dependent_on_data(path.to_string_lossy().as_bytes())
                }
                Type::Separate { header, data } => op_descriptor!()
                    .dependent_on_data(header.to_string_lossy().as_bytes())
                    .dependent_on_data(data.to_string_lossy().as_bytes()),
            },
            Default::default(),
            self.0.metadata,
            DataParam(self.clone()),
            move |ctx, positions, this| {
                let this = &this.0 .0;
                let md = this.metadata;
                let mut positions = positions
                    .into_iter()
                    .map(|(p, h)| (md.chunk_pos_from_index(p), h))
                    .collect::<Vec<_>>();
                positions.sort_by_key(|(p, _)| p.z());
                async move {
                    let obj = match &this.type_ {
                        Type::Single(path) => {
                            nifti::ReaderStreamedOptions::new().read_file(path)?
                        }
                        Type::Separate { header, data } => {
                            nifti::ReaderStreamedOptions::new().read_file_pair(header, data)?
                        }
                    };
                    if this.header != *obj.header() {
                        return Err(format!("File has changed").into());
                    }
                    let mut vol = obj.into_volume();
                    for (pos, _) in positions {
                        let z = pos.z().raw as usize;
                        let chunk = this.metadata.chunk_info_vec(&pos);
                        let mut brick_handle = ctx
                            .submit(ctx.alloc_slot(md.chunk_index(&pos), chunk.mem_elements()))
                            .await;

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
                        unsafe { brick_handle.initialized(*ctx) };
                    }
                    Ok(())
                }
                .into()
            },
        )
        .embedded(self.0.embedding_data.clone())
    }
}
