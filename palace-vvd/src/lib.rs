use sxd_document::*;
use sxd_xpath::evaluate_xpath;

use std::path::{Path, PathBuf};

use palace_core::{
    array::{TensorEmbeddingData, VolumeMetaData},
    data::{LocalVoxelPosition, Vector, VoxelPosition},
    dim::*,
    dtypes::DType,
    operators::{
        raycaster::TransFuncOperator, volume::EmbeddedVolumeOperator, volume_gpu::linear_rescale,
    },
    Error,
};

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

fn default_if_nan(v: f32, default: f32) -> f32 {
    if v.is_nan() {
        default
    } else {
        v
    }
}

pub fn open(
    path: &Path,
    brick_size: LocalVoxelPosition,
) -> Result<EmbeddedVolumeOperator<DType>, Error> {
    let content = std::fs::read_to_string(path)?;
    let package = parser::parse(&content)?;
    let document = package.as_document();

    let x = evaluate_xpath(&document, "/VoreenData/Volumes/Volume/RawData/@x")?.number() as u32;
    let y = evaluate_xpath(&document, "/VoreenData/Volumes/Volume/RawData/@y")?.number() as u32;
    let z = evaluate_xpath(&document, "/VoreenData/Volumes/Volume/RawData/@z")?.number() as u32;
    let format = evaluate_xpath(&document, "/VoreenData/Volumes/Volume/RawData/@format")?.string();
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

    let rwm_offset = evaluate_xpath(
        &document,
        "/VoreenData/Volumes/Volume/MetaData/MetaItem[@name='RealWorldMapping']/value/@offset",
    )?
    .number() as f32;
    let rwm_offset = default_if_nan(rwm_offset, 0.0);

    let rwm_scale = evaluate_xpath(
        &document,
        "/VoreenData/Volumes/Volume/MetaData/MetaItem[@name='RealWorldMapping']/value/@scale",
    )?
    .number() as f32;
    let rwm_scale = default_if_nan(rwm_scale, 1.0);

    let embedding_data = TensorEmbeddingData { spacing };

    if format != "float" {
        return Err(format!(
            "Unsupported format '{}'. Only float volumes are supported currently",
            format
        )
        .into());
    }

    let simple_path = evaluate_xpath(&document, "/VoreenData/Volumes/Volume/RawData/@filename")?;
    let alternative_paths = evaluate_xpath(
        &document,
        "/VoreenData/Volumes/Volume/RawData/Paths/paths/item/@value",
    )?;

    let base = path.parent();
    let Some(raw_path) =
        find_valid_path(base, &simple_path).or_else(|| find_valid_path(base, &alternative_paths))
    else {
        return Err("No valid .raw file path in file".into());
    };

    let metadata = VolumeMetaData {
        dimensions: size,
        chunk_size: brick_size,
    };

    let vol = palace_core::operators::raw::open(raw_path, metadata)?;

    let vol = linear_rescale(vol, rwm_scale, rwm_offset);
    Ok(vol.embedded(embedding_data).into())
}

pub fn load_tfi(path: &Path) -> Result<TransFuncOperator, Error> {
    let content = std::fs::read_to_string(path)?;
    let package = parser::parse(&content)?;
    let document = package.as_document();

    let domain_low =
        evaluate_xpath(&document, "/VoreenData/TransFuncIntensity/domain/@x")?.number() as f32;
    let domain_high =
        evaluate_xpath(&document, "/VoreenData/TransFuncIntensity/domain/@y")?.number() as f32;

    let sxd_xpath::Value::Nodeset(keys) =
        evaluate_xpath(&document, "/VoreenData/TransFuncIntensity/Keys/key")?
    else {
        return Err("Did not find keys".into());
    };

    let parser = sxd_xpath::Factory::new();
    let ctx = sxd_xpath::Context::new();
    let mut keys = keys
        .into_iter()
        .map(|n| {
            let intensity = parser
                .build("intensity/@value")?
                .unwrap()
                .evaluate(&ctx, n)?
                .number() as f32;

            let mut color = Vector::<D4, u8>::fill(0);
            for (i, c) in ["r", "g", "b", "a"].iter().enumerate() {
                let val = parser
                    .build(&format!("colorL/@{}", c))?
                    .unwrap()
                    .evaluate(&ctx, n)?
                    .number() as u8;
                color[i] = val;
            }

            Ok((intensity, color))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    keys.sort_by(|l, r| l.0.total_cmp(&r.0));

    let mut ri = 0;

    Ok(TransFuncOperator::gen_normalized(
        domain_low,
        domain_high,
        256,
        |intensity| {
            while ri < keys.len() && intensity > keys[ri].0 {
                ri += 1;
            }
            let li = ri.saturating_sub(1);
            let ri = ri.min(keys.len() - 1);

            let l = keys[li];
            let r = keys[ri];

            let ret = if li != ri {
                let alpha = (intensity - l.0) / (r.0 - l.0);

                (l.1.f32().scale(1.0 - alpha) + r.1.f32().scale(alpha)).map(|v| v as u8)
            } else {
                l.1
            };
            ret
        },
    ))
}
