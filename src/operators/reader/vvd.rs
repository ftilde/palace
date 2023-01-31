use sxd_document::*;
use sxd_xpath::evaluate_xpath;

use std::path::{Path, PathBuf};

use crate::{
    array::VolumeMetaData,
    data::{LocalVoxelPosition, VoxelPosition},
    operator::OperatorId,
    operators::{volume::VolumeOperatorState, VolumeOperator},
    Error,
};

use super::RawVolumeSourceState;

pub struct VvdVolumeSourceState {
    raw: RawVolumeSourceState,
    metadata: VolumeMetaData,
}

fn find_valid_path(base: Option<&Path>, val: &sxd_xpath::Value) -> Option<PathBuf> {
    let sxd_xpath::Value::Nodeset(set) = val else {
        return None;
    };

    for v in set {
        let v = v.string_value();
        let path = PathBuf::from(v);
        if path.exists() {
            return Some(path);
        }
        if let Some(base) = base {
            let from_base = base.join(path);
            if from_base.exists() {
                return Some(from_base);
            }
        }
    }
    None
}

impl VolumeOperatorState for VvdVolumeSourceState {
    fn operate<'op>(&'op self) -> VolumeOperator<'op> {
        VolumeOperator::new(
            OperatorId::new("VvdVolumeSourceState::operate")
                .dependent_on(self.raw.path.to_string_lossy().as_bytes()),
            move |ctx, _| async move { ctx.write(self.metadata) }.into(),
            move |ctx, positions, _| {
                async move {
                    self.raw
                        .load_raw_bricks(self.metadata.chunk_size, ctx, positions)
                        .await
                }
                .into()
            },
        )
    }
}

impl VvdVolumeSourceState {
    pub fn open(path: &Path, brick_size: LocalVoxelPosition) -> Result<Self, Error> {
        let content = std::fs::read_to_string(path)?;
        let package = parser::parse(&content)?;
        let document = package.as_document();

        let x = evaluate_xpath(&document, "/VoreenData/Volumes/Volume/RawData/@x")?.number() as u32;
        let y = evaluate_xpath(&document, "/VoreenData/Volumes/Volume/RawData/@y")?.number() as u32;
        let z = evaluate_xpath(&document, "/VoreenData/Volumes/Volume/RawData/@z")?.number() as u32;
        let format =
            evaluate_xpath(&document, "/VoreenData/Volumes/Volume/RawData/@format")?.string();

        let size = VoxelPosition::from([z, y, x]);

        if format != "float" {
            return Err(format!(
                "Unsupported format '{}'. Only float volumes are supported currently",
                format
            )
            .into());
        }

        let simple_path =
            evaluate_xpath(&document, "/VoreenData/Volumes/Volume/RawData/@filename")?;
        let alternative_paths = evaluate_xpath(
            &document,
            "/VoreenData/Volumes/Volume/RawData/Paths/paths/item/@value",
        )?;

        let base = path.parent();
        let Some(raw_path) = find_valid_path(base, &simple_path).or_else(|| find_valid_path(base, &alternative_paths)) else {
            return Err("No valid .raw file path in file".into());
        };

        let metadata = VolumeMetaData {
            dimensions: size,
            chunk_size: brick_size,
        };

        let raw = RawVolumeSourceState::open(raw_path, size)?;
        Ok(VvdVolumeSourceState { raw, metadata })
    }
}
