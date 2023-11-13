use crate::{conversion::*, map_err, types::*};
use pyo3::prelude::*;
use vng_core::{
    data::{GlobalCoordinate, LocalVoxelPosition, Vector},
    operators::sliceviewer::SliceviewState,
};
use vng_vvd::VvdVolumeSourceState;

#[pyfunction]
pub fn rechunk(vol: VolumeOperator, size: [ChunkSize; 3]) -> VolumeOperator {
    let size = Vector::from(size).map(|s: ChunkSize| s.0);
    vng_core::operators::volume_gpu::rechunk(vol.into(), size).into()
}

#[pyfunction]
pub fn linear_rescale(
    py: Python,
    vol: MaybeEmbeddedVolumeOperator,
    scale: ToOperator<ScalarOperatorF32>,
    offset: ToOperator<ScalarOperatorF32>,
) -> PyObject {
    vol.map_inner(py, |vol| {
        vng_core::operators::volume_gpu::linear_rescale(vol.into(), scale.0.into(), offset.0.into())
    })
}

#[pyfunction]
pub fn separable_convolution<'py>(
    py: Python,
    vol: MaybeEmbeddedVolumeOperator,
    zyx: [ToOperator<ArrayOperator>; 3],
) -> PyObject {
    let [z, y, x] = zyx;
    vol.map_inner(py, |vol| {
        vng_core::operators::volume_gpu::separable_convolution(
            vol,
            [z.0.into(), y.0.into(), x.0.into()],
        )
    })
}

#[pyfunction]
pub fn entry_exit_points(
    input_md: VolumeMetadataOperator,
    embedding_data: VolumeEmbeddingDataOperator,
    output_md: ToOperator<ImageMetadataOperator>,
    projection: ToOperator<Mat4Operator>,
) -> VolumeOperator {
    vng_core::operators::raycaster::entry_exit_points(
        input_md.into(),
        embedding_data.into(),
        output_md.0.into(),
        projection.0.into(),
    )
    .into()
}

#[pyfunction]
pub fn raycast(vol: EmbeddedVolumeOperator, entry_exit_points: VolumeOperator) -> VolumeOperator {
    //NO_PUSH_main fix with maybemultilevel or something like that
    let vol: vng_core::operators::volume::EmbeddedVolumeOperator = vol.into();
    vng_core::operators::raycaster::raycast(vol.single_level_lod(), entry_exit_points.into()).into()
}

#[pyfunction]
pub fn slice_projection_mat(
    state: SliceviewState,
    dim: usize,
    input_data: VolumeMetadataOperator,
    embedding_data: VolumeEmbeddingDataOperator,
    output_data: Vector<2, GlobalCoordinate>,
) -> Mat4Operator {
    state
        .projection_mat(dim, input_data.into(), embedding_data.into(), output_data)
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
pub fn open_volume(path: std::path::PathBuf) -> PyResult<EmbeddedVolumeOperator> {
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

    Ok(vol.into())
}
