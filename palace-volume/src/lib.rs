use std::path::PathBuf;

use palace_core::{
    dim::D3, dtypes::DType, operators::volume::EmbeddedVolumeOperator, vec::LocalVoxelPosition,
};

#[derive(Clone, Default)]
pub struct Hints {
    pub brick_size: Option<LocalVoxelPosition>,
    pub location: Option<String>,
}

impl Hints {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn brick_size(mut self, brick_size: LocalVoxelPosition) -> Self {
        self.brick_size = Some(brick_size);
        self
    }

    pub fn location(mut self, location: String) -> Self {
        self.location = Some(location);
        self
    }
}

pub fn open(
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
                .brick_size
                .unwrap_or(LocalVoxelPosition::fill(64.into())),
        ),
        [.., "nii"] | [.., "nii", "gz"] => palace_nifti::open_single(path),
        [.., "hdr"] => {
            let data = path.with_extension("img");
            palace_nifti::open_separate(path, data)
        }
        [.., "h5"] => palace_hdf5::open(path, hints.location.unwrap_or("/volume".to_owned())),
        [.., "zarr"] => palace_zarr::open(path, hints.location.unwrap_or("/array".to_owned()))?
            .try_into_static::<D3>()
            .ok_or_else(|| "Volume is not 3-dimensional".into()),
        _ => Err(format!("Unknown volume format for file {}", path.to_string_lossy()).into()),
    }
}
