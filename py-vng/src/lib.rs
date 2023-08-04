use pyo3::{exceptions::PyException, prelude::*};

mod conversion;
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
    m.add_function(wrap_pyfunction!(look_at, m)?)?;
    m.add_function(wrap_pyfunction!(perspective, m)?)?;
    m.add("chunk_size_full", ChunkSizeFull)?;
    m.add_class::<ScalarOperatorF32>()?;
    m.add_class::<ImageMetadata>()?;
    m.add_class::<RunTime>()?;
    m.add_class::<Window>()?;
    Ok(())
}
