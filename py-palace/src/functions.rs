use crate::types::*;
use palace_core::array::{PyTensorEmbeddingData, PyTensorMetaData};
use palace_core::data::{LocalVoxelPosition, Matrix, Vector};
use palace_core::dim::*;
use palace_core::dtypes::{DType, StaticElementType};
use palace_core::jit::JitTensorOperator;
use palace_core::operators::raycaster::RaycasterConfig;
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
                |vol: palace_core::operators::tensor::TensorOperator<D2, DType>| {
                    let vol: palace_core::operators::tensor::TensorOperator<
                        D2,
                        StaticElementType<Vector<D4, u8>>,
                    > = vol.try_into()?;
                    Ok(palace_core::operators::volume_gpu::rechunk(vol, size).into())
                },
            )
        }
        3 => {
            let size =
                Vector::from(<[ChunkSize; 3]>::try_from(size).unwrap()).map(|s: ChunkSize| s.0);
            vol.try_map_inner(
                py,
                |vol: palace_core::operators::tensor::TensorOperator<D3, DType>| {
                    let vol: palace_core::operators::tensor::TensorOperator<
                        D3,
                        StaticElementType<f32>,
                    > = vol.try_into()?; //TODO
                    Ok(palace_core::operators::volume_gpu::rechunk(vol, size).into())
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
        |vol: palace_core::operators::volume::VolumeOperator<DType>| {
            //TODO: ndim -> static dispatch
            Ok(
                palace_core::operators::volume_gpu::linear_rescale(vol.try_into()?, scale, offset)
                    .into(),
            )
        },
    )
}
#[pyfunction]
pub fn abs(py: Python, vol: MaybeEmbeddedTensorOperator) -> PyResult<PyObject> {
    vol.try_map_inner_jit(py, |vol: JitTensorOperator<D3>| {
        //TODO: ndim -> static dispatch
        Ok(crate::map_err(vol.abs())?.into())
    })
}

#[pyfunction]
pub fn add(
    py: Python,
    v1: MaybeEmbeddedTensorOperator,
    v2: MaybeEmbeddedTensorOperator,
) -> PyResult<PyObject> {
    v1.try_map_inner_jit(py, |v1: JitTensorOperator<D3>| {
        Ok(crate::map_err(v1.add(v2.inner().try_into_jit()?))?.into())
    })
}

#[pyfunction]
pub fn threshold(
    py: Python,
    vol: MaybeEmbeddedTensorOperator,
    threshold: f32,
) -> PyResult<PyObject> {
    vol.try_map_inner(
        py,
        |vol: palace_core::operators::volume::VolumeOperator<DType>| {
            //TODO: ndim -> static dispatch
            Ok(palace_core::operators::volume_gpu::threshold(vol.try_into()?, threshold).into())
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

    let kernels = [
        z.try_into_core()?.try_into()?,
        y.try_into_core()?.try_into()?,
        x.try_into_core()?.try_into()?,
    ];
    let kernel_refs = Vector::<D3, _>::from_fn(|i| &kernels[i]);
    vol.try_map_inner(py, |vol| {
        Ok(
            palace_core::operators::volume_gpu::separable_convolution(vol.try_into()?, kernel_refs)
                .into(),
        )
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
    input_md: PyTensorMetaData,
    embedding_data: PyTensorEmbeddingData,
    output_md: PyTensorMetaData,
    projection: Matrix<D4, f32>,
) -> PyResult<TensorOperator> {
    palace_core::operators::raycaster::entry_exit_points(
        input_md.try_into()?,
        embedding_data.try_into()?,
        output_md.try_into()?,
        projection,
    )
    .try_into()
}

#[pyfunction]
pub fn raycast(
    vol: LODTensorOperator,
    entry_exit_points: TensorOperator,
    config: Option<RaycasterConfig>,
    tf: Option<TransFuncOperator>,
) -> PyResult<TensorOperator> {
    let tf = tf
        .map(|tf| tf.try_into())
        .unwrap_or_else(|| Ok(CTransFuncOperator::grey_ramp(0.0, 1.0)))?;
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
    result_metadata: PyTensorMetaData,
    projection_mat: Matrix<D4, f32>,
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

#[pyfunction]
pub fn view_image(
    image: LODTensorOperator,
    result_metadata: PyTensorMetaData,
    view_state: palace_core::operators::imageviewer::ImageViewerState,
) -> PyResult<TensorOperator> {
    palace_core::operators::imageviewer::view_image(
        image.try_into()?,
        result_metadata.try_into()?,
        view_state,
    )
    .try_into()
}

#[pyfunction]
pub fn read_png(path: std::path::PathBuf) -> PyResult<TensorOperator> {
    crate::map_err(palace_core::operators::png::read(path))?.try_into()
}

#[pyfunction]
pub fn apply_tf(
    py: Python,
    input: MaybeEmbeddedTensorOperator,
    tf: TransFuncOperator,
) -> PyResult<PyObject> {
    let nd = input.clone().inner().metadata.dimensions.len();
    if nd != 2 {
        return Err(PyErr::new::<PyException, _>(format!(
            "apply_tf for dim {} not supported, yet",
            nd
        )));
    }
    let tf = tf.try_into()?;
    input.try_map_inner(py, |input| {
        Ok(palace_core::operators::volume_gpu::apply_tf::<D2>(input.try_into()?, tf).into())
    })
}

#[pyfunction]
pub fn mandelbrot(md: PyTensorMetaData) -> PyResult<LODTensorOperator> {
    palace_core::operators::procedural::mandelbrot(md.try_into()?).try_into()
}
