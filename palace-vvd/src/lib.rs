use sxd_document::*;
use sxd_xpath::evaluate_xpath;

use std::path::{Path, PathBuf};

use palace_core::{
    array::{TensorEmbeddingData, VolumeMetaData},
    data::{LocalVoxelPosition, Vector, VoxelPosition},
    dim::*,
    operator::OperatorDescriptor,
    operators::{
        raw::RawVolumeSourceState,
        tensor::TensorOperator,
        volume::{EmbeddedVolumeOperator, EmbeddedVolumeOperatorState},
    },
    Error,
};

#[derive(Clone)]
pub struct VvdVolumeSourceState {
    raw: RawVolumeSourceState,
    metadata: VolumeMetaData,
    embedding_data: TensorEmbeddingData<D3>,
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

impl EmbeddedVolumeOperatorState for VvdVolumeSourceState {
    fn operate(&self) -> EmbeddedVolumeOperator<f32> {
        TensorOperator::with_state(
            OperatorDescriptor::new("VvdVolumeSourceState::operate")
                .dependent_on_data(self.raw.path.to_string_lossy().as_bytes()),
            self.metadata,
            self.clone(),
            move |ctx, positions, this| {
                async move {
                    this.raw
                        .load_raw_bricks(this.metadata.chunk_size, ctx, positions)
                        .await
                }
                .into()
            },
        )
        .embedded(self.embedding_data)
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

        let spacing_x = evaluate_xpath(
            &document,
            "/VoreenData/Volumes/Volume/MetaData/MetaItem[@name='Spacing']/value/@x",
        )?
        .number() as f32;
        let spacing_y = evaluate_xpath(
            &document,
            "/VoreenData/Volumes/Volume/MetaData/MetaItem[@name='Spacing']/value/@y",
        )?
        .number() as f32;
        let spacing_z = evaluate_xpath(
            &document,
            "/VoreenData/Volumes/Volume/MetaData/MetaItem[@name='Spacing']/value/@z",
        )?
        .number() as f32;

        let spacing = Vector::new([spacing_z, spacing_y, spacing_x]);
        fn is_valid(f: f32) -> bool {
            !f.is_nan() && f.is_finite() && f > 0.0
        }
        if !is_valid(spacing_x) || !is_valid(spacing_y) || !is_valid(spacing_z) {
            return Err(format!("Spacing is not valid: {:?}", spacing).into());
        }

        let embedding_data = TensorEmbeddingData { spacing };

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
        let Some(raw_path) = find_valid_path(base, &simple_path)
            .or_else(|| find_valid_path(base, &alternative_paths))
        else {
            return Err("No valid .raw file path in file".into());
        };

        let metadata = VolumeMetaData {
            dimensions: size,
            chunk_size: brick_size,
        };

        let raw = RawVolumeSourceState::open(raw_path, size)?;
        Ok(VvdVolumeSourceState {
            raw,
            metadata,
            embedding_data,
        })
    }
}
