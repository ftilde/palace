use crate::types::*;
use numpy::PyUntypedArray;
use palace_core::array::{PyTensorEmbeddingData, PyTensorMetaData};
use palace_core::data::{Matrix, Vector};
use palace_core::dtypes::{DType, ScalarType};
use palace_core::jit::{BinOp, JitTensorOperator, TernaryOp, UnaryOp};
use palace_core::operators::raycaster::RaycasterConfig;
use palace_core::operators::tensor::TensorOperator as CTensorOperator;
use palace_core::{dim::*, jit};
use pyo3::types::PySlice;
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
            Ok(palace_core::operators::rechunk::rechunk(vol, size).into_dyn())
        },
    )
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn slice(
    py: Python,
    tensor: MaybeEmbeddedTensorOperatorArg,
    slice_args: Vec<Bound<PySlice>>,
) -> PyResult<PyObject> {
    let tensor = tensor.unpack();
    if tensor.inner_ref().nd()? != slice_args.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "Slice args must be {}-dimensional to fit tensor",
            tensor.inner_ref().nd()?
        )));
    }
    let dimensions = &tensor.inner_ref().metadata()?.dimensions;

    let slice_args = slice_args
        .into_iter()
        .zip(dimensions.iter())
        .map(|(slice_arg, dim)| {
            let slice_arg = slice_arg.indices(*dim as _)?;
            if slice_arg.step != 1 {
                return Err(crate::map_err("Step must be 1".into()));
            }

            Ok(palace_core::operators::slice::Range::FromTo(
                slice_arg.start.try_into().unwrap(),
                slice_arg.stop.try_into().unwrap(),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let slice_args = Vector::<DDyn, _>::new(slice_args);

    tensor.try_map_inner(
        py,
        |vol: palace_core::operators::tensor::TensorOperator<DDyn, DType>| {
            Ok(palace_core::operators::slice::slice(vol, slice_args).into_dyn())
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

fn jit_ternary(
    op: TernaryOp,
    v1: JitArgument,
    v2: JitArgument,
    v3: JitArgument,
) -> PyResult<MaybeEmbeddedTensorOperator> {
    v1.try_map_inner_jit(|v1: JitTensorOperator<DDyn>| {
        Ok(crate::map_result(JitTensorOperator::<DDyn>::ternary_op(
            op,
            v1,
            v2.into_jit(),
            v3.into_jit(),
        ))?
        .into())
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
pub fn lt(v1: JitArgument, v2: JitArgument) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_binary(BinOp::LessThan, v1, v2)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn lt_eq(v1: JitArgument, v2: JitArgument) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_binary(BinOp::LessThanEquals, v1, v2)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn gt(v1: JitArgument, v2: JitArgument) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_binary(BinOp::GreaterThan, v1, v2)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn gt_eq(v1: JitArgument, v2: JitArgument) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_binary(BinOp::GreaterThanEquals, v1, v2)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn eq(v1: JitArgument, v2: JitArgument) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_binary(BinOp::Equals, v1, v2)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn neq(v1: JitArgument, v2: JitArgument) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_binary(BinOp::NotEquals, v1, v2)
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn select(
    if_val: JitArgument,
    then_val: JitArgument,
    else_val: JitArgument,
) -> PyResult<MaybeEmbeddedTensorOperator> {
    jit_ternary(TernaryOp::IfThenElse, if_val, then_val, else_val)
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
            palace_core::operators::conv::separable_convolution(vol, kernel_refs)
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
#[pyo3(signature = (vol, entry_exit_points, config=None, tf=None))]
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
#[pyo3(signature = (input, result_metadata, projection_mat, tf=None))]
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
        Default::default(),
    )
    .into_dyn()
    .into())
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (vol, num_samples=None))]
pub fn mean_value(
    vol: MaybeEmbeddedTensorOperatorArg,
    num_samples: Option<usize>,
) -> PyResult<ScalarOperator> {
    let vol = vol.unpack().into_inner().into_core();
    let vol = vol.try_into()?;
    Ok(palace_core::operators::aggregation::mean(vol, num_samples.into()).into())
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (vol, num_samples=None))]
pub fn min_value(
    vol: MaybeEmbeddedTensorOperatorArg,
    num_samples: Option<usize>,
) -> PyResult<ScalarOperator> {
    let vol = vol.unpack().into_inner().into_core();
    let vol = vol.try_into()?;
    Ok(palace_core::operators::aggregation::min(vol, num_samples.into()).into())
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (vol, num_samples=None))]
pub fn max_value(
    vol: MaybeEmbeddedTensorOperatorArg,
    num_samples: Option<usize>,
) -> PyResult<ScalarOperator> {
    let vol = vol.unpack().into_inner().into_core();
    let vol = vol.try_into()?;
    Ok(palace_core::operators::aggregation::max(vol, num_samples.into()).into())
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
#[pyo3(signature = (path, chunk_size_hint=None, tensor_path_hint=None))]
pub fn open(
    path: std::path::PathBuf,
    chunk_size_hint: Option<Vec<u32>>,
    tensor_path_hint: Option<String>,
) -> PyResult<EmbeddedTensorOperator> {
    let chunk_size_hint =
        chunk_size_hint.map(|h| Vector::from_fn_and_len(h.len(), |i| h[i].into()));

    let hints = palace_io::Hints {
        chunk_size: chunk_size_hint,
        location: tensor_path_hint,
        ..Default::default()
    };
    let vol = crate::map_result(palace_io::open(path, hints))?;

    Ok(vol.into_dyn().into())
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (path, chunk_size_hint=None, tensor_path_hint=None, rechunk=false))]
pub fn open_or_create_lod(
    path: std::path::PathBuf,
    chunk_size_hint: Option<Vec<u32>>,
    tensor_path_hint: Option<String>,
    rechunk: bool,
) -> PyResult<LODTensorOperator> {
    let chunk_size_hint =
        chunk_size_hint.map(|h| Vector::from_fn_and_len(h.len(), |i| h[i].into()));

    let hints = palace_io::Hints {
        chunk_size: chunk_size_hint,
        location: tensor_path_hint,
        rechunk,
        ..Default::default()
    };
    let (vol, _) = crate::map_result(palace_io::open_or_create_lod(path, hints))?;

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
        Default::default(),
    )
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
        Ok(palace_core::transfunc::apply::<DDyn>(input.try_into()?, tf).into())
    })
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn mandelbrot(
    md: PyTensorMetaData,
    embedding_data: PyTensorEmbeddingData,
) -> PyResult<LODTensorOperator> {
    palace_core::operators::procedural::mandelbrot(
        md.try_into_dim()?,
        embedding_data.try_into_dim()?,
    )
    .try_into()
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn randomwalker_weights(
    input: MaybeEmbeddedTensorOperatorArg,
    min_edge_weight: f32,
    beta: f32,
) -> PyResult<TensorOperator> {
    let input = input.unpack().into_inner().try_into()?;
    let res: CTensorOperator<DDyn, DType> =
        palace_core::operators::randomwalker::random_walker_weights_grady(
            input,
            beta,
            min_edge_weight,
        )
        .into_dyn()
        .into();
    Ok(res.into())
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn randomwalker_weights_bian(
    input: MaybeEmbeddedTensorOperatorArg,
    min_edge_weight: f32,
    extent: usize,
) -> PyResult<TensorOperator> {
    let input = input.unpack().into_inner().try_into()?;
    let res: CTensorOperator<DDyn, DType> =
        palace_core::operators::randomwalker::random_walker_weights_bian(
            input,
            extent,
            min_edge_weight,
        )
        .into_dyn()
        .into();
    Ok(res.into())
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (weights, seeds, max_iter=None, max_residuum_norm=None))]
pub fn randomwalker(
    py: Python,
    weights: TensorOperator,
    seeds: MaybeEmbeddedTensorOperatorArg,
    max_iter: Option<usize>,
    max_residuum_norm: Option<f32>,
) -> PyResult<PyObject> {
    let weights = weights.try_into()?;
    let mut config = palace_core::operators::randomwalker::SolverConfig::default();
    if let Some(max_iter) = max_iter {
        config.max_iterations = max_iter;
    }
    if let Some(max_residuum_norm) = max_residuum_norm {
        config.max_residuum_norm = max_residuum_norm;
    }
    seeds.unpack().try_map_inner(py, |seeds| {
        Ok(
            palace_core::operators::randomwalker::random_walker_single_chunk(
                weights,
                seeds.try_into()?,
                config,
            )
            .into_dyn()
            .into(),
        )
    })
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (weights, points_fg, points_bg, max_iter=None, max_residuum_norm=None))]
pub fn hierarchical_randomwalker(
    weights: LODTensorOperator,
    points_fg: TensorOperator,
    points_bg: TensorOperator,
    max_iter: Option<usize>,
    max_residuum_norm: Option<f32>,
) -> PyResult<LODTensorOperator> {
    let weights = weights.try_into()?;
    let points_fg = points_fg.try_into_core_static()?;
    let points_bg = points_bg.try_into_core_static()?;

    let mut config = palace_core::operators::randomwalker::SolverConfig::default();
    if let Some(max_iter) = max_iter {
        config.max_iterations = max_iter;
    }
    if let Some(max_residuum_norm) = max_residuum_norm {
        config.max_residuum_norm = max_residuum_norm;
    }
    let res: palace_core::operators::tensor::LODTensorOperator<DDyn, DType> =
        palace_core::operators::randomwalker::hierarchical_random_walker_solver(
            weights, points_fg, points_bg, config,
        )
        .try_into()?;

    res.try_into()
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
            md.into(),
            ed.into(),
        )
        .into_dyn()
        .try_into()?;
    Ok(res.into())
}
