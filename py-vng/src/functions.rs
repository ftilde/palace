use crate::{conversion::*, map_err, types::*};
use pyo3::prelude::*;
use vng_core::{
    array::ImageMetaData,
    data::{LocalVoxelPosition, Vector},
};
use vng_vvd::VvdVolumeSourceState;

#[pyfunction]
pub fn rechunk(vol: VolumeOperator, size: [ChunkSize; 3]) -> VolumeOperator {
    let size = Vector::from(size).map(|s: ChunkSize| s.0);
    vng_core::operators::volume_gpu::rechunk(vol.into(), size).into()
}

#[pyfunction]
pub fn linear_rescale(
    vol: VolumeOperator,
    scale: ToOperator<ScalarOperatorF32>,
    offset: ToOperator<ScalarOperatorF32>,
) -> PyResult<VolumeOperator> {
    Ok(
        vng_core::operators::volume_gpu::linear_rescale(
            vol.into(),
            scale.0.into(),
            offset.0.into(),
        )
        .into(),
    )
}

#[pyfunction]
pub fn separable_convolution<'py>(
    vol: VolumeOperator,
    zyx: [ToOperator<ArrayOperator>; 3],
) -> PyResult<VolumeOperator> {
    let [z, y, x] = zyx;
    Ok(vng_core::operators::volume_gpu::separable_convolution(
        vol.into(),
        [z.0.into(), y.0.into(), x.0.into()],
    )
    .into())
}

#[pyfunction]
pub fn entry_exit_points(
    input_md: VolumeMetadataOperator,
    output_md: ToOperator<ImageMetadataOperator>,
    projection: ToOperator<Mat4Operator>,
) -> VolumeOperator {
    vng_core::operators::raycaster::entry_exit_points(
        input_md.into(),
        output_md.0.into(),
        projection.0.into(),
    )
    .into()
}

#[pyfunction]
pub fn raycast(vol: VolumeOperator, entry_exit_points: VolumeOperator) -> VolumeOperator {
    vng_core::operators::raycaster::raycast(vol.into(), entry_exit_points.into()).into()
}

fn cgmat4_to_numpy(py: Python, mat: vng_core::cgmath::Matrix4<f32>) -> &numpy::PyArray2<f32> {
    // cgmath matrices are row major, but we want column major, so we flip the indices (i, j) below:
    numpy::PyArray2::from_owned_array(
        py,
        numpy::ndarray::Array::from_shape_fn((4, 4), |(j, i)| mat[i][j]),
    )
}

#[pyfunction]
pub fn look_at(
    py: Python,
    eye: Vector<3, f32>,
    center: Vector<3, f32>,
    up: Vector<3, f32>,
) -> &numpy::PyArray2<f32> {
    let mat = vng_core::cgmath::Matrix4::look_at_rh(eye.into(), center.into(), up.into());
    cgmat4_to_numpy(py, mat)
}

#[pyfunction]
pub fn perspective(
    py: Python,
    md: ImageMetaData,
    fov_degree: f32,
    near: f32,
    far: f32,
) -> &numpy::PyArray2<f32> {
    let size = md.dimensions;
    let mat = vng_core::cgmath::perspective(
        vng_core::cgmath::Deg(fov_degree),
        size.x().raw as f32 / size.y().raw as f32,
        near,
        far,
    );
    cgmat4_to_numpy(py, mat)
}

#[pyfunction]
pub fn slice_projection_mat(
    dim: usize,
    input_data: VolumeMetadataOperator,
    output_data: ToOperator<ImageMetadataOperator>,
    selected_slice: ToOperator<ScalarOperatorU32>,
    offset: ToOperator<ScalarOperatorVec2F>,
    zoom_level: ToOperator<ScalarOperatorF32>,
) -> Mat4Operator {
    vng_core::operators::sliceviewer::slice_projection_mat(
        dim,
        input_data.into(),
        output_data.0.into(),
        selected_slice.0 .0.map((), |v, _| v.into()),
        offset.0.into(),
        zoom_level.0.into(),
    )
    .into()
}

#[pyfunction]
pub fn render_slice(
    input: VolumeOperator,
    result_metadata: ToOperator<ImageMetadataOperator>,
    projection_mat: ToOperator<Mat4Operator>,
) -> VolumeOperator {
    vng_core::operators::sliceviewer::render_slice(
        input.into(),
        result_metadata.0.into(),
        projection_mat.0.into(),
    )
    .into()
}

#[pyfunction]
pub fn mean(vol: VolumeOperator) -> ScalarOperatorF32 {
    vng_core::operators::volume_gpu::mean(vol.into()).into()
}

#[pyfunction]
pub fn open_volume(path: std::path::PathBuf) -> PyResult<VolumeOperator> {
    let brick_size_hint = LocalVoxelPosition::fill(32.into());

    let Some(file) = path.file_name() else {
        return map_err(Err("No file name in path".into()));
    };
    let file = file.to_string_lossy();
    let segments = file.split('.').collect::<Vec<_>>();

    let vol_source = match segments[..] {
        [.., "vvd"] => Box::new(map_err(VvdVolumeSourceState::open(&path, brick_size_hint))?),
        //[.., "nii"] | [.., "nii", "gz"] => {
        //    Box::new(map_err(NiftiVolumeSourceState::open_single(path)?))
        //}
        //[.., "hdr"] => {
        //    let data = path.with_extension("img");
        //    Box::new(NiftiVolumeSourceState::open_separate(path, data)?)
        //}
        //[.., "h5"] => Box::new(Hdf5VolumeSourceState::open(path, "/volume".to_string())?),
        _ => {
            return map_err(Err(format!(
                "Unknown volume format for file {}",
                path.to_string_lossy()
            )
            .into()))
        }
    };

    use vng_core::operators::volume::VolumeOperatorState;
    let vol = vol_source.operate();

    Ok(VolumeOperator {
        chunks: VolumeValueOperator(vol.chunks),
        metadata: VolumeMetadataOperator(vol.metadata),
    })
}
