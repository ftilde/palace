use palace_core::{
    array::{PyTensorEmbeddingData, PyTensorMetaData},
    dtypes::{DType, ScalarType},
};
use pyo3::{exceptions::PyException, prelude::*};

mod functions;
mod jit;
mod types;

fn map_result<T>(e: Result<T, palace_core::Error>) -> PyResult<T> {
    e.map_err(map_err)
}

fn map_err(e: palace_core::Error) -> PyErr {
    PyErr::new::<PyException, _>(format!("{}", e))
}

#[pymodule]
fn palace(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    use crate::functions::*;
    use crate::types::*;

    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(open_lod, m)?)?;
    m.add_function(wrap_pyfunction!(open_or_create_lod, m)?)?;
    m.add_function(wrap_pyfunction!(load_tf, m)?)?;
    m.add_function(wrap_pyfunction!(grey_ramp_tf, m)?)?;
    m.add_function(wrap_pyfunction!(tf_from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(entry_exit_points, m)?)?;
    m.add_function(wrap_pyfunction!(raycast, m)?)?;
    m.add_function(wrap_pyfunction!(render_slice, m)?)?;
    m.add_function(wrap_pyfunction!(gauss_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(vesselness, m)?)?;
    m.add_function(wrap_pyfunction!(view_image, m)?)?;
    m.add_function(wrap_pyfunction!(apply_tf, m)?)?;
    m.add_function(wrap_pyfunction!(mandelbrot, m)?)?;
    m.add_function(wrap_pyfunction!(from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(randomwalker_weights, m)?)?;
    m.add_function(wrap_pyfunction!(randomwalker_weights_bian, m)?)?;
    m.add_function(wrap_pyfunction!(randomwalker, m)?)?;
    m.add_function(wrap_pyfunction!(hierarchical_randomwalker, m)?)?;
    m.add_function(wrap_pyfunction!(rasterize_seed_points, m)?)?;
    m.add_function(wrap_pyfunction!(crate::jit::jit, m)?)?;
    m.add("chunk_size_full", ChunkSizeFull)?;
    m.add_class::<palace_core::operators::sliceviewer::SliceviewState>()?;
    m.add_class::<palace_core::operators::splitter::SplitDirection>()?;
    m.add_class::<palace_core::operators::raycaster::CameraState>()?;
    m.add_class::<palace_core::operators::raycaster::RaycasterConfig>()?;
    m.add_class::<palace_core::operators::raycaster::TrackballState>()?;
    m.add_class::<palace_core::operators::imageviewer::ImageViewerState>()?;
    m.add_class::<TransFuncOperator>()?;
    m.add_class::<PyTensorMetaData>()?;
    m.add_class::<PyTensorEmbeddingData>()?;
    m.add_class::<RunTime>()?;
    m.add_class::<Events>()?;
    m.add_class::<MouseButton>()?;
    m.add_class::<OnMouseDrag>()?;
    m.add_class::<OnMouseClick>()?;
    m.add_class::<OnWheelMove>()?;
    m.add_class::<OnKeyPress>()?;
    m.add_class::<GuiState>()?;
    m.add_class::<Horizontal>()?;
    m.add_class::<Vertical>()?;
    m.add_class::<Slider>()?;
    m.add_class::<Button>()?;
    m.add_class::<Label>()?;
    m.add_class::<Splitter>()?;
    m.add_class::<ComboBox>()?;
    m.add_class::<DType>()?;
    m.add_class::<ScalarType>()?;
    m.add_class::<FixedStep>()?;

    m.add_class::<state_link::py::Store>()?;
    Ok(())
}

pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
