use futures::StreamExt;
use itertools::Itertools;
use ndarray::ShapeBuilder;
use sxd_document::*;
use sxd_xpath::evaluate_xpath;

use std::path::{Path, PathBuf};

use palace_core::{
    array::{TensorEmbeddingData, VolumeMetaData},
    data::{GlobalCoordinate, Vector, VoxelPosition},
    dim::*,
    dtypes::{DType, ElementType, ScalarType},
    jit::jit,
    operators::{
        rechunk::ChunkSize,
        tensor::{EmbeddedTensorOperator, EmbeddedVolumeOperator, TensorOperator},
    },
    task::{OpaqueTaskContext, RequestStream},
    transfunc::TransFuncOperator,
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
    brick_size: Vector<D3, ChunkSize>,
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

    let (dtype_mult, scalar_type) = match format.as_str() {
        "float" => (1.0, ScalarType::F32),
        "uint8" => (1.0 / ((1 << 8) as f32), ScalarType::U8),
        "uint16" => (1.0 / ((1 << 16) as f32), ScalarType::U16),
        f => return Err(format!("Unsupported format '{}'.", f).into()),
    };
    let dtype = DType::scalar(scalar_type);

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
    // Voreen behavior is: If there is no RWM, a denormalizing-mapping is applied
    // -> We reverse dtype_mult
    let rwm_scale = default_if_nan(rwm_scale, 1.0 / dtype_mult);

    let embedding_data = TensorEmbeddingData { spacing };

    let scale = rwm_scale * dtype_mult;

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

    let chunk_size = brick_size.zip(&size, |c, s| c.apply(s));

    let metadata = VolumeMetaData {
        dimensions: size,
        chunk_size,
    };

    let vol = palace_core::operators::raw::open(raw_path, metadata, dtype)?;
    let vol = jit(vol);
    let mut vol = vol.cast(DType::scalar(ScalarType::F32))?;

    if scale != 1.0 {
        vol = vol.mul(scale.into())?;
    }

    if rwm_offset != 0.0 {
        vol = vol.add(rwm_offset.into())?;
    }

    let vol: TensorOperator<D3, DType> = vol.try_into().unwrap();

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

fn write_vvd(
    out: &mut dyn std::io::Write,
    raw_file: &str,
    dtype: DType,
    dimensions: Vector<D3, GlobalCoordinate>,
    spacing: Vector<D3, f32>,
) -> Result<(), palace_core::Error> {
    if dtype.size != 1 {
        return Err(format!(
            "Only scalar dtypes are supported, but dtype with size {} given",
            dtype.size
        )
        .into());
    }
    let format = {
        let this = &dtype.scalar;
        match this {
            ScalarType::U8 => "uint8",
            ScalarType::I8 => "int8",
            ScalarType::U16 => "uint16",
            ScalarType::I16 => "int16",
            ScalarType::U32 => "uint32",
            ScalarType::I32 => "int32",
            ScalarType::U64 => "uint64",
            ScalarType::I64 => "int64",
            ScalarType::F32 => "float",
        }
    };

    let dim = dimensions.raw();

    write!(
        out,
        r#"<?xml version="1.0" ?>
<VoreenData version="1">
    <Volumes>
        <Volume>
            <RawData format="{format}" x="{dim_x}" y="{dim_y}" z="{dim_z}">
                <Paths noPathSet="false">
                    <paths>
                        <item value="{raw_file}" />
                    </paths>
                </Paths>
            </RawData>
            <MetaData>
                <MetaItem name="Offset" type="Vec3MetaData">
                    <value x="0" y="0" z="0" />
                </MetaItem>
                <MetaItem name="Spacing" type="Vec3MetaData">
                    <value x="{spacing_x}" y="{spacing_y}" z="{spacing_z}" />
                </MetaItem>
            </MetaData>
        </Volume>
    </Volumes>
</VoreenData>"#,
        format = format,
        dim_x = dim.x(),
        dim_y = dim.y(),
        dim_z = dim.z(),
        spacing_x = spacing.x(),
        spacing_y = spacing.y(),
        spacing_z = spacing.z(),
    )?;

    Ok(())
}

pub async fn save_embedded_tensor<'cref, 'inv>(
    ctx: OpaqueTaskContext<'cref, 'inv>,
    path: &Path,
    t: &'inv EmbeddedTensorOperator<D3, DType>,
) -> Result<(), palace_core::Error> {
    let raw_path = path.with_extension("raw");
    let mut vvd_out = std::fs::File::options()
        .truncate(true)
        .write(true)
        .create(true)
        .open(path)?;

    let raw_file_name = raw_path.file_name().unwrap();
    let raw_file_name = raw_file_name.to_str().unwrap();
    write_vvd(
        &mut vvd_out,
        raw_file_name,
        t.dtype(),
        t.metadata.dimensions,
        t.embedding_data.spacing,
    )?;

    let file_size = t
        .dtype()
        .array_layout(t.metadata.num_tensor_elements())
        .size();
    let mut out_file = std::fs::File::options()
        .truncate(true)
        .write(true)
        .read(true)
        .create(true)
        .open(raw_path)?;
    out_file.set_len(file_size as u64)?;
    let mut out_file = unsafe { memmap::MmapMut::map_mut(&mut out_file) }?;
    let out_file_ptr = out_file.as_mut_ptr();

    let num_dtype_bytes = t.dtype().element_layout().size();
    let out_dim = t
        .metadata
        .dimensions
        .push_dim_small(num_dtype_bytes.try_into().unwrap());
    let stride = palace_core::data::dimension_order_stride(&out_dim);

    let md = &t.metadata.push_dim_small(
        num_dtype_bytes.try_into().unwrap(),
        num_dtype_bytes.try_into().unwrap(),
    );

    let num_total = md.dimension_in_chunks().hmul();
    println!("{} chunks to save", num_total);

    let request_chunk_size = 1024;
    let chunk_ids_in_parts = md.chunk_indices().chunks(request_chunk_size);
    let mut i = 0;
    for chunk_ids in &chunk_ids_in_parts {
        let requests = chunk_ids.map(|chunk_id| (t.chunks.request_raw(chunk_id), chunk_id));
        let stream =
            ctx.submit_unordered_with_data(requests)
                .then_req(ctx, |(chunk_handle, chunk_id)| {
                    let chunk_info = md.chunk_info(chunk_id);

                    let begin = chunk_info.begin();
                    let start_offset = (*begin * stride).hadd();

                    let start_ptr =
                        unsafe { out_file_ptr.offset(start_offset.try_into().unwrap()) };

                    let stride = D4::to_ndarray_dim_dyn(stride.inner());

                    let size: ndarray::Shape<
                        <palace_core::dim::D4 as palace_core::dim::DynDimension>::NDArrayDimDyn,
                    > = chunk_info.logical_dimensions.to_ndarray_dim().into();
                    let shape = size.strides(stride);

                    let mut chunk_view_out =
                        unsafe { ndarray::ArrayViewMut::from_shape_ptr(shape, start_ptr) };

                    let chunk_handle = chunk_handle.into_thread_handle();
                    ctx.spawn_compute(move || {
                        let chunk_view_in =
                            palace_core::data::chunk(chunk_handle.data(), &chunk_info);

                        chunk_view_out.assign(&chunk_view_in);

                        chunk_handle
                    })
                });
        futures::pin_mut!(stream);
        while let Some(handle) = stream.next().await {
            let _handle = handle.into_main_handle(ctx.storage());
        }
        i += request_chunk_size;
        println!(
            "{}/{}, {}%",
            i,
            num_total,
            i as f32 / num_total as f32 * 100.0
        );
    }

    Ok(())
}
