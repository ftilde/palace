use std::path::PathBuf;

use palace_core::{
    dim::D3,
    dtypes::DType,
    operators::{
        volume::{ChunkSize, EmbeddedVolumeOperator, LODVolumeOperator},
        volume_gpu::rechunk,
    },
    vec::LocalVoxelPosition,
};

#[derive(Clone, Default)]
pub struct Hints {
    pub chunk_size: Option<LocalVoxelPosition>,
    pub location: Option<String>,
    pub rechunk: bool,
}

impl Hints {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn brick_size(mut self, brick_size: LocalVoxelPosition) -> Self {
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
) -> Result<EmbeddedVolumeOperator<DType>, Box<dyn std::error::Error>> {
    let Some(file) = path.file_name() else {
        return Err("No file name in path".into());
    };
    let file = file.to_string_lossy();
    let segments = file.split('.').collect::<Vec<_>>();

    match segments[..] {
        [.., "vvd"] => palace_vvd::open(
            &path,
            hints
                .chunk_size
                .unwrap_or(LocalVoxelPosition::fill(64.into())),
        ),
        [.., "nii"] | [.., "nii", "gz"] => palace_nifti::open_single(path),
        [.., "hdr"] => {
            let data = path.with_extension("img");
            palace_nifti::open_separate(path, data)
        }
        [.., "h5"] => palace_hdf5::open(path, hints.location.unwrap_or("/volume".to_owned())),
        [.., "zarr"] | [.., "zarr", "zip"] => {
            palace_zarr::open(path, hints.location.unwrap_or("/array".to_owned()))?
                .try_into_static::<D3>()
                .ok_or_else(|| "Volume is not 3-dimensional".into())
        }
        _ => Err(format!("Unknown volume format for file {}", path.to_string_lossy()).into()),
    }
}

pub fn open(
    path: PathBuf,
    hints: Hints,
) -> Result<EmbeddedVolumeOperator<DType>, Box<dyn std::error::Error>> {
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
) -> Result<LODVolumeOperator<DType>, Box<dyn std::error::Error>> {
    let Some(file) = path.file_name() else {
        return Err("No file name in path".into());
    };
    let file = file.to_string_lossy();
    let segments = file.split('.').collect::<Vec<_>>();

    match segments[..] {
        [.., "zarr"] | [.., "zarr", "zip"] => {
            palace_zarr::open_lod(path, hints.location.unwrap_or("/level".to_owned()))?
                .try_into_static::<D3>()
                .ok_or_else(|| "Volume is not 3-dimensional".into())
        }
        _ => Err(format!(
            "Unknown lod volume format for file {}",
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
) -> Result<(LODVolumeOperator<DType>, LodOrigin), Box<dyn std::error::Error>> {
    Ok(if let Ok(vol) = open_lod(path.clone(), hints.clone()) {
        (vol, LodOrigin::Existing)
    } else {
        let mut vol = open(path, hints.clone())?.into_dyn();

        if hints.rechunk {
            if let Some(chunk_size) = hints.chunk_size {
                vol = vol
                    .map_inner(|v| rechunk(v, chunk_size.map(|s| ChunkSize::Fixed(s)).into_dyn()));
            } else {
                return Err("rechunk hint is set, but no chunk_size specified".into());
            }
        }

        let vol: LODVolumeOperator<DType> = palace_core::operators::resample::create_lod(vol, 2.0)
            .try_into_static::<D3>()
            .unwrap();
        (vol.into(), LodOrigin::Dynamic)
    })
}
