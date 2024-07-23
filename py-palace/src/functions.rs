use crate::types::*;
use palace_core::array::{PyTensorEmbeddingData, PyTensorMetaData};
use palace_core::data::{LocalVoxelPosition, Matrix, Vector};
use palace_core::dim::*;
use palace_core::dtypes::DType;
use palace_core::jit::{BinOp, JitTensorOperator, UnaryOp};
use palace_core::operators::raycaster::RaycasterConfig;
use palace_core::operators::tensor::TensorOperator as CTensorOperator;
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyfunction]
pub fn rechunk(
    py: Python,
    tensor: MaybeEmbeddedTensorOperator,
    size: Vec<ChunkSize>,
) -> PyResult<PyObject> {
    if tensor.inner().nd() != size.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "Chunk size must be {}-dimensional to fit tensor",
            tensor.inner().nd()
        )));
    }

    let size = Vector::<DDyn, _>::new(size).map(|s: ChunkSize| s.0);
    tensor.try_map_inner(
        py,
        |vol: palace_core::operators::tensor::TensorOperator<DDyn, DType>| {
            Ok(palace_core::operators::volume_gpu::rechunk(vol, size).into_dyn())
        },
    )
}

#[pyfunction]
pub fn linear_rescale(
    py: Python,
    vol: MaybeEmbeddedTensorOperator,
    scale: f32,
    offset: f32,
) -> PyResult<PyObject> {
    vol.try_map_inner(py, |vol: CTensorOperator<DDyn, DType>| {
        Ok(
            palace_core::operators::volume_gpu::linear_rescale(vol.try_into()?, scale, offset)
                .into(),
        )
    })
}

fn jit_unary(py: Python, op: UnaryOp, vol: MaybeEmbeddedTensorOperator) -> PyResult<PyObject> {
    vol.try_map_inner_jit(py, |vol: JitTensorOperator<DDyn>| {
        Ok(crate::map_err(JitTensorOperator::<DDyn>::unary_op(op, vol))?.into())
    })
}

#[pyfunction]
pub fn abs(py: Python, vol: MaybeEmbeddedTensorOperator) -> PyResult<PyObject> {
    jit_unary(py, UnaryOp::Abs, vol)
}

#[pyfunction]
pub fn neg(py: Python, vol: MaybeEmbeddedTensorOperator) -> PyResult<PyObject> {
    jit_unary(py, UnaryOp::Neg, vol)
}

fn jit_binary(
    py: Python,
    op: BinOp,
    v1: MaybeEmbeddedTensorOperator,
    v2: MaybeEmbeddedTensorOperator,
) -> PyResult<PyObject> {
    v1.try_map_inner_jit(py, |v1: JitTensorOperator<DDyn>| {
        Ok(crate::map_err(JitTensorOperator::<DDyn>::bin_op(
            op,
            v1,
            v2.into_inner().try_into_jit()?,
        ))?
        .into())
    })
}

#[pyfunction]
pub fn add(
    py: Python,
    v1: MaybeEmbeddedTensorOperator,
    v2: MaybeEmbeddedTensorOperator,
) -> PyResult<PyObject> {
    jit_binary(py, BinOp::Add, v1, v2)
}

#[pyfunction]
pub fn mul(
    py: Python,
    v1: MaybeEmbeddedTensorOperator,
    v2: MaybeEmbeddedTensorOperator,
) -> PyResult<PyObject> {
    jit_binary(py, BinOp::Mul, v1, v2)
}

#[pyfunction]
pub fn threshold(
    py: Python,
    vol: MaybeEmbeddedTensorOperator,
    threshold: f32,
) -> PyResult<PyObject> {
    vol.try_map_inner(py, |vol: CTensorOperator<DDyn, DType>| {
        Ok(palace_core::operators::volume_gpu::threshold(vol.try_into()?, threshold).into())
    })
}

#[pyfunction]
pub fn separable_convolution<'py>(
    py: Python,
    tensor: MaybeEmbeddedTensorOperator,
    kernels: Vec<MaybeConstTensorOperator>, //TODO
) -> PyResult<PyObject> {
    if tensor.inner().nd() != kernels.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "Expected {} kernels for tensor, but got {}",
            tensor.inner().nd(),
            kernels.len()
        )));
    }

    let kernels = kernels
        .into_iter()
        .map(|k| {
            let ret: CTensorOperator<D1, DType> = k.try_into_core()?.try_into()?;
            Ok(ret)
        })
        .collect::<Result<Vec<_>, PyErr>>()?;

    for kernel in &kernels {
        if tensor.inner().dtype != kernel.dtype() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Kernel must have the same type as tensor ({:?}), but has {:?}",
                tensor.inner().dtype,
                kernel.dtype(),
            )));
        }
    }

    let kernel_refs = Vector::<DDyn, &CTensorOperator<D1, DType>>::try_from_fn_and_len(
        |i| &kernels[i],
        kernels.len(),
    )
    .unwrap();

    tensor.try_map_inner(py, |vol: CTensorOperator<DDyn, DType>| {
        Ok(
            palace_core::operators::volume_gpu::separable_convolution(vol, kernel_refs)
                .into_dyn()
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
        vol.try_into_core_static()?.try_into()?,
        min_scale,
        max_scale,
        steps,
    )
    .into_dyn()
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
        input_md.try_into_dim()?,
        embedding_data.try_into_dim()?,
        output_md.try_into_dim()?,
        projection,
    )
    .into_dyn()
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

    let eep = entry_exit_points.try_into_core()?;
    let eep = try_into_static_err(eep)?;
    palace_core::operators::raycaster::raycast(
        vol.try_into()?,
        eep.try_into()?,
        tf,
        config.unwrap_or_default(),
    )
    .into_dyn()
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
        result_metadata.try_into_dim()?,
        projection_mat.try_into()?,
    )
    .into_dyn()
    .try_into()
}

#[pyfunction]
pub fn mean(vol: MaybeEmbeddedTensorOperator) -> PyResult<ScalarOperator> {
    let vol = vol.into_inner().try_into_core()?;
    let vol = try_into_static_err(vol)?;
    let vol = vol.try_into()?;
    Ok(palace_core::operators::volume_gpu::mean(vol).into())
}

#[pyfunction]
pub fn gauss_kernel(stddev: f32) -> PyResult<TensorOperator> {
    palace_core::operators::kernels::gauss(stddev)
        .into_dyn()
        .try_into()
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

    vol.into_dyn().try_into()
}

#[pyfunction]
pub fn view_image(
    image: LODTensorOperator,
    result_metadata: PyTensorMetaData,
    view_state: palace_core::operators::imageviewer::ImageViewerState,
) -> PyResult<TensorOperator> {
    palace_core::operators::imageviewer::view_image(
        image.try_into()?,
        result_metadata.try_into_dim()?,
        view_state,
    )
    .into_dyn()
    .try_into()
}

#[pyfunction]
pub fn read_png(path: std::path::PathBuf) -> PyResult<TensorOperator> {
    crate::map_err(palace_core::operators::png::read(path))?
        .into_dyn()
        .try_into()
}

#[pyfunction]
pub fn apply_tf(
    py: Python,
    input: MaybeEmbeddedTensorOperator,
    tf: TransFuncOperator,
) -> PyResult<PyObject> {
    let tf = tf.try_into()?;
    input.try_map_inner(py, |input| {
        Ok(palace_core::operators::volume_gpu::apply_tf::<DDyn>(input.try_into()?, tf).into())
    })
}

#[pyfunction]
pub fn mandelbrot(md: PyTensorMetaData) -> PyResult<LODTensorOperator> {
    palace_core::operators::procedural::mandelbrot(md.try_into_dim()?).try_into()
}
