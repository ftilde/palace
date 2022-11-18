use hard_xml::XmlRead;
use std::{
    borrow::Cow,
    path::{Path, PathBuf},
};

use crate::{
    data::{SVec3, VolumeMetaData, VoxelPosition},
    operator::{Operator, OperatorId},
    task::{DatumRequest, Task, TaskContext},
    Error,
};

use super::RawVolumeSource;

#[derive(XmlRead, PartialEq, Debug)]
#[xml(tag = "VoreenData")]
struct VoreenData<'a> {
    #[xml(attr = "version")]
    version: Cow<'a, str>,

    #[xml(child = "Volumes")]
    volumes: Vec<Volume<'a>>,
}

#[derive(XmlRead, PartialEq, Debug)]
#[xml(tag = "Volume")]
struct Volume<'a> {
    #[xml(child = "filename")]
    raw_data: RawData<'a>,
}

#[derive(XmlRead, PartialEq, Debug)]
#[xml(tag = "RawData")]
struct RawData<'a> {
    #[xml(attr = "filename")]
    file_name: Cow<'a, str>,

    #[xml(attr = "format")]
    format: Cow<'a, str>,

    #[xml(attr = "x")]
    x: u32,

    #[xml(attr = "y")]
    y: u32,

    #[xml(attr = "z")]
    z: u32,
}

pub struct VvdVolumeSource {
    raw: RawVolumeSource,
}

impl VvdVolumeSource {
    pub fn open(path: &Path, brick_size: VoxelPosition) -> Result<Self, Error> {
        let content = std::fs::read_to_string(path).unwrap();
        let metadata = VoreenData::from_str(&content)?;

        let vol = &metadata.volumes[0];
        let raw_data = &vol.raw_data;

        let size = VoxelPosition(SVec3::new(raw_data.x, raw_data.y, raw_data.z));
        let format = &raw_data.format;
        if format != "float" {
            return Err(format!(
                "Unsupported format '{}'. Only float volumes are supported currently",
                format
            )
            .into());
        }
        let vmd = VolumeMetaData {
            dimensions: size,
            brick_size,
        };

        let raw = RawVolumeSource::open(PathBuf::from(raw_data.file_name.as_ref()), vmd)?;
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
