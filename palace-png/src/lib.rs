use std::rc::Rc;
use std::{io::BufReader, path::PathBuf};

use id::Id;

use std::fs::File;
use std::io::BufWriter;

use palace_core::dtypes::StaticElementType;
use palace_core::operators::tensor::{FrameOperator, ImageOperator};
use palace_core::{data::Vector, dim::*, task::OpaqueTaskContext};

pub async fn write<'cref, 'inv: 'cref, 'op: 'inv>(
    ctx: OpaqueTaskContext<'cref, 'inv>,
    input: &'inv ImageOperator<StaticElementType<Vector<D4, u8>>>,
    path: PathBuf,
) -> Result<(), palace_core::Error> {
    let m = input.metadata;

    if m.dimensions != m.chunk_size.global() {
        return Err("Image must consist of a single chunk".into());
    }

    let chunk_id = input.metadata.chunk_index(&Vector::fill(0.into()));
    let img = ctx.submit(input.chunks.request(chunk_id)).await;

    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, m.dimensions.raw().x(), m.dimensions.raw().y());
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_gamma(png::ScaledFloat::new(1.0));
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
        .flatten()
        .collect::<Vec<_>>();
    writer.write_image_data(&data).unwrap();

    Ok(())
}

pub fn read<'cref, 'inv: 'cref, 'op: 'inv>(
    path: PathBuf,
) -> Result<FrameOperator, palace_core::Error> {
    let id = Id::from_data(path.to_string_lossy().as_bytes());

    let file = File::open(path)?;
    let r = BufReader::new(file);

    let decoder = png::Decoder::new(r);
    let mut reader = decoder.read_info().unwrap();

    let mut buf = vec![0; reader.output_buffer_size()];

    let info = reader.next_frame(&mut buf)?;
    let dimensions = [info.height, info.width].into();

    let bytes = &buf[..info.buffer_size()];
    assert_eq!(info.bit_depth, png::BitDepth::Eight);

    match info.color_type {
        png::ColorType::Grayscale => todo!(),
        png::ColorType::Indexed => todo!(),
        png::ColorType::GrayscaleAlpha => todo!(),
        png::ColorType::Rgb => {
            assert_eq!(bytes.len() % 3, 0);
            let pixels = bytes
                .chunks(3)
                .map(|v| Vector::<D4, u8>::from([v[0], v[1], v[2], 255]))
                .collect::<Rc<[Vector<D4, u8>]>>();

            FrameOperator::from_rc_with_id(dimensions, pixels, id)
        }
        png::ColorType::Rgba => {
            let pixels = bytemuck::cast_slice::<_, Vector<D4, u8>>(bytes);
            FrameOperator::from_rc_with_id(dimensions, pixels.into(), id)
        }
    }
}
