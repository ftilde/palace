use crate::types::*;
use numpy::PyUntypedArray;
use palace_core::array::{PyTensorEmbeddingData, PyTensorMetaData};
use palace_core::data::{Matrix, Vector};
use palace_core::dim::*;
use palace_core::dtypes::DType;
use palace_core::operators::raycaster::RaycasterConfig;
use palace_core::operators::sliceviewer::RenderConfig2D;
use palace_core::operators::tensor::TensorOperator as CTensorOperator;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass_complex_enum, gen_stub_pyfunction};

#[gen_stub_pyfunction]
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
    Ok(
        crate::map_result(palace_core::operators::raycaster::raycast(
            vol.try_into_core_static()?,
            eep.try_into()?,
            tf,
            config.unwrap_or_default(),
        ))?
        .into_dyn()
        .into(),
    )
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (input, result_metadata, projection_mat, tf=None, coarse_lod_factor=1.0))]
pub fn render_slice(
    input: LODTensorOperator,
    result_metadata: PyTensorMetaData,
    projection_mat: Matrix<D4, f32>,
    tf: Option<TransFuncOperator>,
    coarse_lod_factor: f32,
) -> PyResult<TensorOperator> {
    let tf = tf
        .map(|tf| tf.try_into())
        .unwrap_or_else(|| Ok(CTransFuncOperator::grey_ramp(0.0, 1.0)))?;

    Ok(
        crate::map_result(palace_core::operators::sliceviewer::render_slice(
            input.try_into_core_static()?,
            result_metadata.try_into_dim()?,
            projection_mat.try_into()?,
            tf,
            RenderConfig2D { coarse_lod_factor },
        ))?
        .into_dyn()
        .into(),
    )
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
    chunk_size_hint: Option<Vec<ChunkSize>>,
    tensor_path_hint: Option<String>,
) -> PyResult<EmbeddedTensorOperator> {
    let chunk_size_hint = chunk_size_hint.map(|h| Vector::from_fn_and_len(h.len(), |i| h[i].0));

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
#[pyo3(signature = (path, chunk_size_hint=None, tensor_path_hint=None))]
pub fn open_lod(
    path: std::path::PathBuf,
    chunk_size_hint: Option<Vec<ChunkSize>>,
    tensor_path_hint: Option<String>,
) -> PyResult<LODTensorOperator> {
    let chunk_size_hint = chunk_size_hint.map(|h| Vector::from_fn_and_len(h.len(), |i| h[i].0));

    let hints = palace_io::Hints {
        chunk_size: chunk_size_hint,
        location: tensor_path_hint,
        ..Default::default()
    };
    let vol = crate::map_result(palace_io::open_lod(path, hints))?;

    vol.into_dyn().try_into()
}

#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (path, chunk_size_hint=None, tensor_path_hint=None, rechunk=false))]
pub fn open_or_create_lod(
    path: std::path::PathBuf,
    chunk_size_hint: Option<Vec<ChunkSize>>,
    tensor_path_hint: Option<String>,
    rechunk: bool,
) -> PyResult<LODTensorOperator> {
    let chunk_size_hint = chunk_size_hint.map(|h| Vector::from_fn_and_len(h.len(), |i| h[i].0));

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
#[pyo3(signature = (image, result_metadata, view_state, coarse_lod_factor=1.0))]
pub fn view_image(
    image: LODTensorOperator,
    result_metadata: PyTensorMetaData,
    view_state: palace_core::operators::imageviewer::ImageViewerState,
    coarse_lod_factor: f32,
) -> PyResult<TensorOperator> {
    Ok(palace_core::operators::imageviewer::view_image(
        image.try_into_core_static()?.try_into()?,
        result_metadata.try_into_dim()?,
        view_state,
        RenderConfig2D { coarse_lod_factor },
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
pub fn procedural(
    md: PyTensorMetaData,
    embedding_data: PyTensorEmbeddingData,
    body: String,
) -> PyResult<LODTensorOperator> {
    palace_core::operators::procedural::rasterize_lod(
        md.try_into()?,
        embedding_data.try_into()?,
        body,
    )
    .try_into()
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
pub fn mandelbulb(
    md: PyTensorMetaData,
    embedding_data: PyTensorEmbeddingData,
) -> PyResult<LODTensorOperator> {
    palace_core::operators::procedural::mandelbulb(
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
pub fn randomwalker_weight_pairs(
    input: MaybeEmbeddedTensorOperatorArg,
) -> PyResult<TensorOperator> {
    let input: CTensorOperator<DDyn, DType> = input.unpack().into_inner().try_into()?;
    let res: CTensorOperator<DDyn, DType> =
        palace_core::operators::randomwalker::random_walker_weight_pairs(input)
            .into_dyn()
            .into();
    Ok(res.into())
}

#[derive(FromPyObject)]
#[gen_stub_pyclass_complex_enum]
pub enum MaybeVecUint {
    Splat(u32),
    Vec(Vec<u32>),
}

impl MaybeVecUint {
    fn into_vec(self, nd: usize) -> Vec<u32> {
        match self {
            MaybeVecUint::Splat(s) => vec![s; nd],
            MaybeVecUint::Vec(vec) => vec,
        }
    }
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn randomwalker_weights_bian(
    input: MaybeEmbeddedTensorOperatorArg,
    min_edge_weight: f32,
    extent: MaybeVecUint,
) -> PyResult<TensorOperator> {
    let input: CTensorOperator<DDyn, _> = input.unpack().into_inner().try_into()?;
    let extent = Vector::new(extent.into_vec(input.dim().n()));
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
pub fn randomwalker_weights_bhattacharyya_var_gaussian(
    input: MaybeEmbeddedTensorOperatorArg,
    min_edge_weight: f32,
    extent: MaybeVecUint,
) -> PyResult<TensorOperator> {
    let input: CTensorOperator<DDyn, _> = input.unpack().into_inner().try_into()?;
    let extent = Vector::new(extent.into_vec(input.dim().n()));
    let res: CTensorOperator<DDyn, DType> =
        palace_core::operators::randomwalker::random_walker_weights_bhattacharyya_var_gaussian(
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
pub fn randomwalker_weights_ttest(
    input: MaybeEmbeddedTensorOperatorArg,
    min_edge_weight: f32,
    extent: MaybeVecUint,
) -> PyResult<TensorOperator> {
    let input: CTensorOperator<DDyn, _> = input.unpack().into_inner().try_into()?;
    let extent = Vector::new(extent.into_vec(input.dim().n()));
    let res: CTensorOperator<DDyn, DType> =
        palace_core::operators::randomwalker::random_walker_weights_ttest(
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
