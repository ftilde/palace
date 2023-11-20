use crate::{map_err, types::*};
use pyo3::prelude::*;
use vng_core::{
    array::ImageMetaData,
    data::{GlobalCoordinate, LocalVoxelPosition, Matrix, Vector},
    operators::sliceviewer::SliceviewState,
};
use vng_vvd::VvdVolumeSourceState;

#[pyfunction]
pub fn rechunk(
    py: Python,
    vol: MaybeEmbeddedTensorOperator,
    size: [ChunkSize; 3],
) -> PyResult<PyObject> {
    let size = Vector::from(size).map(|s: ChunkSize| s.0);
    vol.try_map_inner(
        py,
        |vol: vng_core::operators::volume::VolumeOperator<f32>| {
            //TODO: ndim -> static dispatch
            vng_core::operators::volume_gpu::rechunk(vol, size).into()
        },
    )
}

#[pyfunction]
pub fn linear_rescale(
    py: Python,
    vol: MaybeEmbeddedTensorOperator,
    scale: MaybeConstScalarOperator<f32>,
    offset: MaybeConstScalarOperator<f32>,
) -> PyResult<PyObject> {
    let scale = scale.try_into()?;
    let offset = offset.try_into()?;
    vol.try_map_inner(
        py,
        |vol: vng_core::operators::volume::VolumeOperator<f32>| {
            //TODO: ndim -> static dispatch
            vng_core::operators::volume_gpu::linear_rescale(vol, scale, offset)
        },
    )
}

#[pyfunction]
pub fn separable_convolution<'py>(
    py: Python,
    vol: MaybeEmbeddedTensorOperator,
    zyx: [MaybeConstTensorOperator; 3], //TODO
) -> PyResult<PyObject> {
    let [z, y, x] = zyx;

    let kernels = [z.try_into()?, y.try_into()?, x.try_into()?];
    vol.try_map_inner(py, |vol| {
        vng_core::operators::volume_gpu::separable_convolution(vol, kernels)
    })
}

#[pyfunction]
pub fn entry_exit_points(
    input_md: ScalarOperator,
    embedding_data: ScalarOperator,
    output_md: MaybeConstScalarOperator<ImageMetaData>,
    projection: MaybeConstScalarOperator<Matrix<4, f32>>,
) -> PyResult<TensorOperator> {
    vng_core::operators::raycaster::entry_exit_points(
        input_md.try_into()?,
        embedding_data.try_into()?,
        output_md.try_into()?,
        projection.try_into()?,
    )
    .try_into()
}

#[pyfunction]
pub fn raycast(
    vol: LODTensorOperator,
    entry_exit_points: TensorOperator,
) -> PyResult<TensorOperator> {
    vng_core::operators::raycaster::raycast(vol.try_into()?, entry_exit_points.try_into()?)
        .try_into()
}

#[pyfunction]
pub fn slice_projection_mat(
    state: SliceviewState,
    dim: usize,
    input_data: ScalarOperator,
    embedding_data: ScalarOperator,
    output_data: Vector<2, GlobalCoordinate>,
) -> PyResult<ScalarOperator> {
    Ok(state
        .projection_mat(
            dim,
            input_data.try_into()?,
            embedding_data.try_into()?,
            output_data,
        )
        .into())
}

#[pyfunction]
pub fn render_slice(
    input: LODTensorOperator,
    result_metadata: MaybeConstScalarOperator<vng_core::array::ImageMetaData>,
    projection_mat: MaybeConstScalarOperator<vng_core::data::Matrix<4, f32>>,
) -> PyResult<TensorOperator> {
    vng_core::operators::sliceviewer::render_slice(
        input.try_into()?,
        result_metadata.try_into()?,
        projection_mat.try_into()?,
    )
    .try_into()
}

#[pyfunction]
pub fn mean(vol: TensorOperator) -> PyResult<ScalarOperator> {
    Ok(vng_core::operators::volume_gpu::mean(vol.try_into()?).into())
}

#[pyfunction]
pub fn open_volume(path: std::path::PathBuf) -> PyResult<EmbeddedTensorOperator> {
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

    use vng_core::operators::volume::EmbeddedVolumeOperatorState;
    let vol = vol_source.operate();

    vol.try_into()
}
