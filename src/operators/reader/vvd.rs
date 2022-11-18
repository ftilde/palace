use sxd_document::*;
use sxd_xpath::evaluate_xpath;

use std::path::{Path, PathBuf};

use crate::{
    data::{SVec3, VolumeMetaData, VoxelPosition},
    operator::{Operator, OperatorId},
    task::{DatumRequest, Task, TaskContext},
    Error,
};

use super::RawVolumeSource;

pub struct VvdVolumeSource {
    raw: RawVolumeSource,
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

impl VvdVolumeSource {
    pub fn open(path: &Path, brick_size: VoxelPosition) -> Result<Self, Error> {
        let content = std::fs::read_to_string(path)?;
        let package = parser::parse(&content)?;
        let document = package.as_document();

        let x = evaluate_xpath(&document, "/VoreenData/Volumes/Volume/RawData/@x")?.number() as _;
        let y = evaluate_xpath(&document, "/VoreenData/Volumes/Volume/RawData/@y")?.number() as _;
        let z = evaluate_xpath(&document, "/VoreenData/Volumes/Volume/RawData/@z")?.number() as _;
        let format =
            evaluate_xpath(&document, "/VoreenData/Volumes/Volume/RawData/@format")?.string();

        let size = VoxelPosition(SVec3::new(x, y, z));

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

        let vmd = VolumeMetaData {
            dimensions: size,
            brick_size,
        };

        let raw = RawVolumeSource::open(raw_path, vmd)?;
        Ok(VvdVolumeSource { raw })
    }
}

impl Operator for VvdVolumeSource {
    fn id(&self) -> OperatorId {
        OperatorId::new::<Self>(&[self.raw.id()])
    }

    fn compute<'a>(&'a self, rt: TaskContext<'a>, info: DatumRequest) -> Task<'a> {
        self.raw.compute(rt, info)
    }
}
