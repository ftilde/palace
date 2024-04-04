use crate::types::*;
use palace_core::dim::*;
use palace_core::operators::raycaster::{RaycasterConfig, TransFuncOperator};
use palace_core::{
    array::{ImageMetaData, VolumeEmbeddingData, VolumeMetaData},
    data::{LocalVoxelPosition, Matrix, Vector},
};
use pyo3::{exceptions::PyException, prelude::*};

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
                |vol: palace_core::operators::tensor::TensorOperator<D2, Vector<D4, u8>>| {
                    palace_core::operators::volume_gpu::rechunk(vol, size).into()
                },
            )
        }
        3 => {
            let size =
                Vector::from(<[ChunkSize; 3]>::try_from(size).unwrap()).map(|s: ChunkSize| s.0);
            vol.try_map_inner(
                py,
                |vol: palace_core::operators::tensor::TensorOperator<D3, f32>| {
                    palace_core::operators::volume_gpu::rechunk(vol, size).into()
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
        |vol: palace_core::operators::volume::VolumeOperator<f32>| {
            //TODO: ndim -> static dispatch
            palace_core::operators::volume_gpu::linear_rescale(vol, scale, offset)
        },
    )
}

#[pyfunction]
pub fn threshold(
    py: Python,
    vol: MaybeEmbeddedTensorOperator,
    threshold: f32,
) -> PyResult<PyObject> {
    vol.try_map_inner(
        py,
        |vol: palace_core::operators::volume::VolumeOperator<f32>| {
            //TODO: ndim -> static dispatch
            palace_core::operators::volume_gpu::threshold(vol, threshold)
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
        palace_core::operators::volume_gpu::separable_convolution(vol, kernel_refs)
    })
}

#[pyfunction]
pub fn vesselness<'py>(
    vol: EmbeddedTensorOperator,
    min_scale: f32,
    max_scale: f32,
    steps: usize,
) -> PyResult<EmbeddedTensorOperator> {
    palace_core::operators::vesselness::multiscale_vesselness(
        vol.try_into()?,
        min_scale,
        max_scale,
        steps,
    )
    .try_into()
}

#[pyfunction]
pub fn entry_exit_points(
    input_md: VolumeMetaData,
    embedding_data: VolumeEmbeddingData,
    output_md: ImageMetaData,
    projection: Matrix<D4, f32>,
) -> PyResult<TensorOperator> {
    palace_core::operators::raycaster::entry_exit_points(
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
    config: Option<RaycasterConfig>,
) -> PyResult<TensorOperator> {
    let tf = TransFuncOperator::grey_ramp(0.0, 1.0); //TODO make configurable
    palace_core::operators::raycaster::raycast(
        vol.try_into()?,
        entry_exit_points.try_into()?,
        tf,
        config.unwrap_or_default(),
    )
    .try_into()
}

#[pyfunction]
pub fn render_slice(
    input: LODTensorOperator,
    result_metadata: palace_core::array::ImageMetaData,
    projection_mat: palace_core::data::Matrix<D4, f32>,
) -> PyResult<TensorOperator> {
    palace_core::operators::sliceviewer::render_slice(
        input.try_into()?,
        result_metadata.try_into()?,
        projection_mat.try_into()?,
    )
    .try_into()
}

#[pyfunction]
pub fn mean(vol: MaybeEmbeddedTensorOperator) -> PyResult<ScalarOperator> {
    Ok(palace_core::operators::volume_gpu::mean(vol.inner().try_into()?).into())
}

#[pyfunction]
pub fn gauss_kernel(stddev: f32) -> PyResult<TensorOperator> {
    palace_core::operators::kernels::gauss(stddev).try_into()
}

#[pyfunction]
pub fn open_volume(
    path: std::path::PathBuf,
    brick_size_hint: Option<u32>,
    volume_path_hint: Option<String>,
) -> PyResult<EmbeddedTensorOperator> {
    let brick_size_hint = brick_size_hint.map(|h| LocalVoxelPosition::fill(h.into()));

    let hints = palace_volume::Hints {
        brick_size: brick_size_hint,
        location: volume_path_hint,
        ..Default::default()
    };
    let vol = crate::map_err(palace_volume::open(path, hints))?;

    vol.try_into()
}
