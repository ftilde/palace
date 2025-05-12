use std::path::PathBuf;

use palace_core::{
    dim::{DDyn, DynDimension},
    dtypes::DType,
    operators::{
        rechunk::{rechunk, ChunkSize},
        resample::DownsampleStep,
        tensor::{EmbeddedTensorOperator, LODTensorOperator},
    },
    vec::Vector,
};

#[derive(Clone, Default)]
pub struct Hints {
    pub chunk_size: Option<Vector<DDyn, ChunkSize>>,
    pub location: Option<String>,
    pub rechunk: bool,
    pub lod_downsample_steps: Option<Vector<DDyn, DownsampleStep>>,
}

impl Hints {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn chunk_size(mut self, brick_size: Vector<DDyn, ChunkSize>) -> Self {
        self.chunk_size = Some(brick_size);
        self
    }

    pub fn location(mut self, location: String) -> Self {
        self.location = Some(location);
        self
    }

    pub fn rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }
}

pub fn open_single_level(
    path: PathBuf,
    hints: Hints,
) -> Result<EmbeddedTensorOperator<DDyn, DType>, Box<dyn std::error::Error>> {
    let Some(file) = path.file_name() else {
        return Err("No file name in path".into());
    };
    let file = file.to_string_lossy();
    let segments = file.split('.').collect::<Vec<_>>();

    Ok(match segments[..] {
        [.., "vvd"] => palace_vvd::open(
            &path,
            hints
                .chunk_size
                .unwrap_or(Vector::fill_with_len(ChunkSize::Fixed(64.into()), 3))
                .try_into_static()
                .ok_or_else(|| "Chunk size hint must be 3-dimensional for vvd".to_owned())?,
        )?
        .into_dyn(),
        [.., "nii"] | [.., "nii", "gz"] => palace_nifti::open_single(path)?.into_dyn(),
        [.., "hdr"] => {
            let data = path.with_extension("img");
            palace_nifti::open_separate(path, data)?.into_dyn()
        }
        [.., "h5" | "hdf5"] => {
            palace_hdf5::open(path, hints.location.unwrap_or("/volume".to_owned()))?
        }
        [.., "zarr"] | [.., "zarr", "zip"] => {
            palace_zarr::open(path, hints.location.unwrap_or("/array".to_owned()))?
        }
        [.., "png"] => palace_png::read(path)?
            .embedded(Default::default())
            .into_dyn()
            .try_into()
            .unwrap(),
        _ => {
            return Err(format!("Unknown tensor format for file {}", path.to_string_lossy()).into())
        }
    })
}

pub fn open(
    path: PathBuf,
    hints: Hints,
) -> Result<EmbeddedTensorOperator<DDyn, DType>, Box<dyn std::error::Error>> {
    match open_single_level(path.clone(), hints.clone()) {
        Ok(o) => Ok(o),
        Err(e) => open_lod(path, hints)
            .map_err(|_| e)
            .map(|lod| lod.levels[0].clone()),
    }
}

pub fn open_lod(
    path: PathBuf,
    hints: Hints,
) -> Result<LODTensorOperator<DDyn, DType>, Box<dyn std::error::Error>> {
    let Some(file) = path.file_name() else {
        return Err("No file name in path".into());
    };
    let file = file.to_string_lossy();
    let segments = file.split('.').collect::<Vec<_>>();

    let location_hint = hints.location.unwrap_or("/level".to_owned());

    match segments[..] {
        [.., "zarr"] | [.., "zarr", "zip"] => palace_zarr::open_lod(path, location_hint),
        [.., "h5"] | [.., "hdf5"] => palace_hdf5::open_lod(path, location_hint),
        _ => Err(format!(
            "Unknown lod tensor format for file {}",
            path.to_string_lossy()
        )
        .into()),
    }
}

pub enum LodOrigin {
    Existing,
    Dynamic,
}

pub fn open_or_create_lod(
    path: PathBuf,
    hints: Hints,
) -> Result<(LODTensorOperator<DDyn, DType>, LodOrigin), Box<dyn std::error::Error>> {
    Ok(if let Ok(mut vol) = open_lod(path.clone(), hints.clone()) {
        if hints.rechunk {
            if let Some(chunk_size) = hints.chunk_size {
                vol = vol.map(|v| v.map_inner(|v| rechunk(v, chunk_size.clone()).into_dyn()));
            } else {
                return Err("rechunk hint is set, but no chunk_size specified".into());
            }
        }

        (vol, LodOrigin::Existing)
    } else {
        let mut vol = open(path, hints.clone())?.into_dyn();

        if hints.rechunk {
            if let Some(chunk_size) = hints.chunk_size {
                vol = vol.map_inner(|v| rechunk(v, chunk_size).into_dyn());
            } else {
                return Err("rechunk hint is set, but no chunk_size specified".into());
            }
        }

        let steps = hints.lod_downsample_steps.clone().unwrap_or_else(|| {
            Vector::fill_with_len(DownsampleStep::Synchronized(2.0), vol.dim().n())
        });

        let vol: LODTensorOperator<DDyn, DType> =
            palace_core::operators::resample::create_lod(vol, steps);
        (vol.into(), LodOrigin::Dynamic)
    })
}
