use crate::types::*;
use numpy::PyUntypedArray;
use palace_core::array::{PyTensorEmbeddingData, PyTensorMetaData};
use palace_core::data::{LocalVoxelPosition, Matrix, Vector};
use palace_core::dtypes::{DType, ScalarType};
use palace_core::jit::{BinOp, JitTensorOperator, UnaryOp};
use palace_core::operators::raycaster::RaycasterConfig;
use palace_core::operators::tensor::TensorOperator as CTensorOperator;
use palace_core::operators::volume_gpu::SampleMethod;
use palace_core::{dim::*, jit};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_stub_gen::derive::{gen_stub_pyclass_enum, gen_stub_pyfunction};

#[gen_stub_pyfunction]
#[pyfunction]
pub fn rechunk(
    py: Python,
    tensor: MaybeEmbeddedTensorOperatorArg,
    size: Vec<ChunkSize>,
) -> PyResult<PyObject> {
    let tensor = tensor.unpack();
    if tensor.inner_ref().nd()? != size.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "Chunk size must be {}-dimensional to fit tensor",
            tensor.inner_ref().nd()?
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

fn jit_unary(op: UnaryOp, vol: JitArgument) -> PyResult<MaybeEmbeddedTensorOperator> {
    vol.try_map_inner_jit(|vol: JitTensorOperator<DDyn>| {
        Ok(crate::map_result(JitTensorOperator::<DDyn>::unary_op(op, vol))?.into())
    })
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn abs(vol: JitArgument) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_unary(UnaryOp::Abs, vol)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn neg(vol: JitArgument) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_unary(UnaryOp::Neg, vol)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn index(vol: JitArgument, i: u32) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_unary(UnaryOp::Index(i), vol)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn splat(vol: JitArgument, size: u32) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_unary(UnaryOp::Splat(size), vol)
}

#[gen_stub_pyclass_enum]
#[derive(FromPyObject)]
pub enum MaybeScalarDType {
    Scalar(ScalarType),
    DType(DType),
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn cast(vol: JitArgument, to: MaybeScalarDType) -> PyResult<MaybeEmbeddedTensorOperator> {
    let to = match to {
        MaybeScalarDType::Scalar(s) => DType::scalar(s),
        MaybeScalarDType::DType(d) => d,
    };
    jit_unary(UnaryOp::Cast(to), vol)
}

#[gen_stub_pyclass_enum]
#[derive(FromPyObject, Clone)]
pub enum JitArgument {
    Tensor(MaybeEmbeddedTensorOperatorArg),
    Const(f32),
}

impl JitArgument {
    pub fn try_map_inner_jit(
        self,
        f: impl FnOnce(jit::JitTensorOperator<DDyn>) -> PyResult<jit::JitTensorOperator<DDyn>>,
    ) -> PyResult<MaybeEmbeddedTensorOperator> {
        Ok(match self {
            JitArgument::Tensor(t) => t.unpack().try_map_inner_jit(f)?,
            JitArgument::Const(c) => {
                let jit_op = c.into();
                let v = f(jit_op)?;
                let v: TensorOperator = v.try_into()?;
                MaybeEmbeddedTensorOperator::Not { i: v }
            }
        })
    }
    pub fn into_jit(self) -> jit::JitTensorOperator<DDyn> {
        match self {
            JitArgument::Tensor(t) => t.unpack().into_inner().into_jit(),
            JitArgument::Const(c) => c.into(),
        }
    }
}

fn jit_binary(
    op: BinOp,
    v1: JitArgument,
    v2: JitArgument,
) -> PyResult<MaybeEmbeddedTensorOperator> {
    v1.try_map_inner_jit(|v1: JitTensorOperator<DDyn>| {
        Ok(crate::map_result(JitTensorOperator::<DDyn>::bin_op(op, v1, v2.into_jit()))?.into())
    })
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn add(v1: JitArgument, v2: JitArgument) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_binary(BinOp::Add, v1, v2)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn sub(v1: JitArgument, v2: JitArgument) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_binary(BinOp::Sub, v1, v2)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn mul(v1: JitArgument, v2: JitArgument) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_binary(BinOp::Mul, v1, v2)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn div(v1: JitArgument, v2: JitArgument) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_binary(BinOp::Div, v1, v2)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn max(v1: JitArgument, v2: JitArgument) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_binary(BinOp::Max, v1, v2)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn min(v1: JitArgument, v2: JitArgument) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_binary(BinOp::Min, v1, v2)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn threshold(
    py: Python,
    vol: MaybeEmbeddedTensorOperatorArg,
    threshold: f32,
) -> PyResult<PyObject> {
    vol.unpack()
        .try_map_inner(py, |vol: CTensorOperator<DDyn, DType>| {
            Ok(palace_core::operators::volume_gpu::threshold(vol.try_into()?, threshold).into())
        })
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn separable_convolution<'py>(
    py: Python,
    tensor: MaybeEmbeddedTensorOperatorArg,
    kernels: Vec<MaybeConstTensorOperator>,
) -> PyResult<PyObject> {
    let tensor = tensor.unpack();
    if tensor.inner_ref().nd()? != kernels.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "Expected {} kernels for tensor, but got {}",
            tensor.inner_ref().nd()?,
            kernels.len()
        )));
    }

    let kernels = kernels
        .into_iter()
        .map(|k| {
            let ret: CTensorOperator<D1, DType> = try_into_static_err(k.try_into_core()?)?;
            Ok(ret)
        })
        .collect::<Result<Vec<_>, PyErr>>()?;

    for kernel in &kernels {
        if tensor.inner_ref().dtype() != kernel.dtype() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Kernel must have the same type as tensor ({:?}), but has {:?}",
                tensor.inner_ref().dtype(),
                kernel.dtype(),
            )));
        }
    }

    let kernel_refs =
        Vector::<DDyn, &CTensorOperator<D1, DType>>::try_from_fn_and_len(kernels.len(), |i| {
            &kernels[i]
        })
        .unwrap();

    tensor.try_map_inner(py, |vol: CTensorOperator<DDyn, DType>| {
        Ok(
            palace_core::operators::volume_gpu::separable_convolution(vol, kernel_refs)
                .into_dyn()
                .into(),
        )
    })
}

//#[gen_stub_pyfunction] TODO: Not working because PyUntypedArray does not impl the required type
#[pyfunction]
pub fn from_numpy(a: &Bound<PyUntypedArray>) -> PyResult<TensorOperator> {
    Ok(tensor_from_numpy(a)?.into())
}

#[gen_stub_pyfunction]
#[pyfunction]
#[gen_stub_pyfunction]
pub fn vesselness(
    vol: EmbeddedTensorOperator,
    min_scale: f32,
    max_scale: f32,
    steps: usize,
) -> PyResult<EmbeddedTensorOperator> {
    Ok(palace_core::operators::vesselness::multiscale_vesselness(
        vol.try_into_core_static()?.try_into()?,
        min_scale,
        max_scale,
        steps,
    )
    .into_dyn()
    .into())
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn entry_exit_points(
    input_md: PyTensorMetaData,
    embedding_data: PyTensorEmbeddingData,
    output_md: PyTensorMetaData,
    projection: Matrix<D4, f32>,
) -> PyResult<TensorOperator> {
    Ok(palace_core::operators::raycaster::entry_exit_points(
        input_md.try_into_dim()?,
        embedding_data.try_into_dim()?,
        output_md.try_into_dim()?,
        projection,
    )
    .into_dyn()
    .into())
}

#[gen_stub_pyfunction]
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

    let eep = entry_exit_points.into_core();
    let eep = try_into_static_err(eep)?;
    Ok(palace_core::operators::raycaster::raycast(
        vol.try_into_core_static()?.try_into()?,
        eep.try_into()?,
        tf,
        config.unwrap_or_default(),
    )
    .into_dyn()
    .into())
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn render_slice(
    input: LODTensorOperator,
    result_metadata: PyTensorMetaData,
    projection_mat: Matrix<D4, f32>,
    tf: Option<TransFuncOperator>,
) -> PyResult<TensorOperator> {
    let tf = tf
        .map(|tf| tf.try_into())
        .unwrap_or_else(|| Ok(CTransFuncOperator::grey_ramp(0.0, 1.0)))?;

    Ok(palace_core::operators::sliceviewer::render_slice(
        input.try_into_core_static()?.try_into()?,
        result_metadata.try_into_dim()?,
        projection_mat.try_into()?,
        tf,
    )
    .into_dyn()
    .into())
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn mean_value(vol: MaybeEmbeddedTensorOperatorArg) -> PyResult<ScalarOperator> {
    let vol = vol.unpack().into_inner().into_core();
    let vol = try_into_static_err(vol)?;
    let vol = vol.try_into()?;
    Ok(palace_core::operators::volume_gpu::mean(vol).into())
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn min_value(
    vol: MaybeEmbeddedTensorOperatorArg,
    num_samples: Option<usize>,
) -> PyResult<ScalarOperator> {
    let vol = vol.unpack().into_inner().into_core();
    let vol = try_into_static_err(vol)?;
    let vol = vol.try_into()?;
    let sample_method = match num_samples {
        Some(n) => SampleMethod::Subset(n),
        None => SampleMethod::All,
    };
    Ok(palace_core::operators::volume_gpu::min(vol, sample_method).into())
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn max_value(
    vol: MaybeEmbeddedTensorOperatorArg,
    num_samples: Option<usize>,
) -> PyResult<ScalarOperator> {
    let vol = vol.unpack().into_inner().into_core();
    let vol = try_into_static_err(vol)?;
    let vol = vol.try_into()?;
    let sample_method = match num_samples {
        Some(n) => SampleMethod::Subset(n),
        None => SampleMethod::All,
    };
    Ok(palace_core::operators::volume_gpu::max(vol, sample_method).into())
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn gauss_kernel(stddev: f32) -> TensorOperator {
    palace_core::operators::kernels::gauss(stddev)
        .into_dyn()
        .into()
}

#[gen_stub_pyfunction]
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
    let vol = crate::map_result(palace_volume::open(path, hints))?;

    Ok(vol.into_dyn().into())
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn open_or_create_lod(
    path: std::path::PathBuf,
    brick_size_hint: Option<u32>,
    volume_path_hint: Option<String>,
) -> PyResult<LODTensorOperator> {
    let brick_size_hint = brick_size_hint.map(|h| LocalVoxelPosition::fill(h.into()));

    let hints = palace_volume::Hints {
        brick_size: brick_size_hint,
        location: volume_path_hint,
        ..Default::default()
    };
    let (vol, _) = crate::map_result(palace_volume::open_or_create_lod(path, hints))?;

    vol.into_dyn().try_into()
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn view_image(
    image: LODTensorOperator,
    result_metadata: PyTensorMetaData,
    view_state: palace_core::operators::imageviewer::ImageViewerState,
) -> PyResult<TensorOperator> {
    Ok(palace_core::operators::imageviewer::view_image(
        image.try_into_core_static()?.try_into()?,
        result_metadata.try_into_dim()?,
        view_state,
    )
    .into_dyn()
    .into())
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn read_png(path: std::path::PathBuf) -> PyResult<TensorOperator> {
    Ok(crate::map_result(palace_core::operators::png::read(path))?
        .into_dyn()
        .into())
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn apply_tf(
    py: Python,
    input: MaybeEmbeddedTensorOperatorArg,
    tf: TransFuncOperator,
) -> PyResult<PyObject> {
    let tf = tf.try_into()?;
    input.unpack().try_map_inner(py, |input| {
        Ok(palace_core::operators::volume_gpu::apply_tf::<DDyn>(input.try_into()?, tf).into())
    })
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn mandelbrot(md: PyTensorMetaData) -> PyResult<LODTensorOperator> {
    palace_core::operators::procedural::mandelbrot(md.try_into_dim()?).try_into()
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn randomwalker_weights(
    input: MaybeEmbeddedTensorOperatorArg,
    min_edge_weight: f32,
    beta: f32,
) -> PyResult<TensorOperator> {
    let input = input
        .unpack()
        .into_inner()
        .try_into_core_static::<D3>()?
        .try_into()?;
    let res: CTensorOperator<DDyn, DType> =
        palace_core::operators::randomwalker::random_walker_weights(
            input,
            palace_core::operators::randomwalker::WeightFunction::Grady { beta },
            min_edge_weight,
        )
        .into_dyn()
        .into();
    Ok(res.into())
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn randomwalker(
    py: Python,
    weights: TensorOperator,
    seeds: MaybeEmbeddedTensorOperatorArg,
    max_iter: Option<usize>,
    max_residuum_norm: Option<f32>,
) -> PyResult<PyObject> {
    let weights = weights.try_into_core_static::<D4>()?.try_into()?;
    let mut config = palace_core::operators::randomwalker::SolverConfig::default();
    if let Some(max_iter) = max_iter {
        config.max_iterations = max_iter;
    }
    if let Some(max_residuum_norm) = max_residuum_norm {
        config.max_residuum_norm = max_residuum_norm;
    }
    seeds.unpack().try_map_inner(py, |seeds| {
        Ok(palace_core::operators::randomwalker::random_walker_inner(
            weights,
            try_into_static_err(seeds)?.try_into()?,
            config,
        )
        .into_dyn()
        .into())
    })
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn rasterize_seed_points(
    points_fg: TensorOperator,
    points_bg: TensorOperator,
    md: PyTensorMetaData,
    ed: PyTensorEmbeddingData,
) -> PyResult<EmbeddedTensorOperator> {
    let points_fg = points_fg.try_into_core_static()?;
    let points_bg = points_bg.try_into_core_static()?;
    let res: palace_core::operators::tensor::EmbeddedTensorOperator<DDyn, DType> =
        palace_core::operators::randomwalker::rasterize_seed_points(
            points_fg,
            points_bg,
            md.try_into_dim()?,
            ed.try_into_dim()?,
        )
        .into_dyn()
        .try_into()?;
    Ok(res.into())
}
