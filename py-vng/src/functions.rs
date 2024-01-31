use crate::{map_err, types::*};
use pyo3::{exceptions::PyException, prelude::*};
use vng_core::dim::*;
use vng_core::{
    array::{ImageMetaData, VolumeEmbeddingData, VolumeMetaData},
    data::{LocalVoxelPosition, Matrix, Vector},
};
use vng_vvd::VvdVolumeSourceState;

#[pyfunction]
pub fn rechunk(
    py: Python,
    vol: MaybeEmbeddedTensorOperator,
    size: Vec<ChunkSize>,
) -> PyResult<PyObject> {
    //TODO: We REALLY need to figure out the dispatch here...
    //Macro and some for some selected types and dims?
    //Otherwise code will blow up
    match size.len() {
        2 => {
            let size =
                Vector::from(<[ChunkSize; 2]>::try_from(size).unwrap()).map(|s: ChunkSize| s.0);
            vol.try_map_inner(
                py,
                |vol: vng_core::operators::tensor::TensorOperator<D2, Vector<D4, u8>>| {
                    vng_core::operators::volume_gpu::rechunk(vol, size).into()
                },
            )
        }
        3 => {
            let size =
                Vector::from(<[ChunkSize; 3]>::try_from(size).unwrap()).map(|s: ChunkSize| s.0);
            vol.try_map_inner(
                py,
                |vol: vng_core::operators::tensor::TensorOperator<D3, f32>| {
                    vng_core::operators::volume_gpu::rechunk(vol, size).into()
                },
            )
        }
        n => Err(PyErr::new::<PyException, _>(format!(
            "Rechunk for dim {} not supported, yet",
            n
        ))),
    }
}

#[pyfunction]
pub fn linear_rescale(
    py: Python,
    vol: MaybeEmbeddedTensorOperator,
    scale: f32,
    offset: f32,
) -> PyResult<PyObject> {
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
    let kernel_refs = Vector::<D3, _>::from_fn(|i| &kernels[i]);
    vol.try_map_inner(py, |vol| {
        vng_core::operators::volume_gpu::separable_convolution(vol, kernel_refs)
    })
}

#[pyfunction]
pub fn entry_exit_points(
    input_md: VolumeMetaData,
    embedding_data: VolumeEmbeddingData,
    output_md: ImageMetaData,
    projection: Matrix<D4, f32>,
) -> PyResult<TensorOperator> {
    vng_core::operators::raycaster::entry_exit_points(
        input_md,
        embedding_data,
        output_md,
        projection,
    )
    .try_into()
}

#[pyfunction]
pub fn raycast(
    vol: LODTensorOperator,
    entry_exit_points: TensorOperator,
    lod_coarseness: Option<f32>,
) -> PyResult<TensorOperator> {
    vng_core::operators::raycaster::raycast(
        vol.try_into()?,
        entry_exit_points.try_into()?,
        lod_coarseness.unwrap_or(1.0),
    )
    .try_into()
}

#[pyfunction]
pub fn render_slice(
    input: LODTensorOperator,
    result_metadata: vng_core::array::ImageMetaData,
    projection_mat: vng_core::data::Matrix<D4, f32>,
) -> PyResult<TensorOperator> {
    vng_core::operators::sliceviewer::render_slice(
        input.try_into()?,
        result_metadata.try_into()?,
        projection_mat.try_into()?,
    )
    .try_into()
}

#[pyfunction]
pub fn mean(vol: MaybeEmbeddedTensorOperator) -> PyResult<ScalarOperator> {
    Ok(vng_core::operators::volume_gpu::mean(vol.inner().try_into()?).into())
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
