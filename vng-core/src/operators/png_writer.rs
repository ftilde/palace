use std::path::PathBuf;

use std::fs::File;
use std::io::BufWriter;

use crate::{data::Vector, task::OpaqueTaskContext};

use super::tensor::ImageOperator;

pub async fn write<'cref, 'inv: 'cref, 'op: 'inv>(
    ctx: OpaqueTaskContext<'cref, 'inv>,
    input: &'inv ImageOperator<Vector<4, u8>>,
    path: PathBuf,
) -> Result<(), crate::Error> {
    let m = input.metadata;

    if m.dimensions != m.chunk_size.global() {
        return Err("Image must consist of a single chunk".into());
    }

    if m.dimensions.x().raw != 4 {
        return Err("Image must have exactly four channels".into());
    }

    let img = ctx
        .submit(input.chunks.request(Vector::fill(0.into())))
        .await;

    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, m.dimensions.raw()[1], m.dimensions.raw()[0]); // Width is 2 pixels and height is 1.
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    //encoder.set_source_gamma(png::ScaledFloat::from_scaled(45455)); // 1.0 / 2.2, scaled by 100000
    encoder.set_source_gamma(png::ScaledFloat::new(1.0 / 2.2)); // 1.0 / 2.2, unscaled, but rounded
    let source_chromaticities = png::SourceChromaticities::new(
        // Using unscaled instantiation here
        (0.31270, 0.32900),
        (0.64000, 0.33000),
        (0.30000, 0.60000),
        (0.15000, 0.06000),
    );
    encoder.set_source_chromaticities(source_chromaticities);
    let mut writer = encoder.write_header().unwrap();

    let data = img
        .into_iter()
        .map(|v| {
            let res: [u8; 4] = (*v).into();
            res
        })
        .flatten() //NO_PUSH_main test this!
        .collect::<Vec<_>>();
    writer.write_image_data(&data).unwrap();

    Ok(())
}
