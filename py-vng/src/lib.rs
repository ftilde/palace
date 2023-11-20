use pyo3::{exceptions::PyException, prelude::*};

mod functions;
mod types;

fn map_err<T>(e: Result<T, vng_core::Error>) -> PyResult<T> {
    e.map_err(|e| PyErr::new::<PyException, _>(format!("{}", e)))
}

#[pymodule]
fn vng(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    use crate::functions::*;
    use crate::types::*;

    m.add_function(wrap_pyfunction!(open_volume, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(linear_rescale, m)?)?;
    m.add_function(wrap_pyfunction!(rechunk, m)?)?;
    m.add_function(wrap_pyfunction!(separable_convolution, m)?)?;
    m.add_function(wrap_pyfunction!(entry_exit_points, m)?)?;
    m.add_function(wrap_pyfunction!(raycast, m)?)?;
    m.add_function(wrap_pyfunction!(slice_projection_mat, m)?)?;
    m.add_function(wrap_pyfunction!(render_slice, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_metadata, m)?)?;
    m.add("chunk_size_full", ChunkSizeFull)?;
    m.add_class::<vng_core::operators::sliceviewer::SliceviewState>()?;
    m.add_class::<vng_core::operators::splitter::SplitDirection>()?;
    m.add_class::<vng_core::operators::raycaster::CameraState>()?;
    m.add_class::<vng_core::operators::raycaster::TrackballState>()?;
    m.add_class::<RunTime>()?;
    m.add_class::<Window>()?;
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

    m.add_class::<state_link::py::Store>()?;
    Ok(())
}
