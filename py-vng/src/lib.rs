use std::{cell::RefCell, rc::Rc};

use pyo3::{exceptions::PyIOError, prelude::*};
use vng_core::operator::Operator;

#[pyclass(unsendable)]
#[derive(Clone)]
struct RunTime {
    inner: Rc<RefCell<vng_core::runtime::RunTime>>,
}

#[pymethods]
impl RunTime {
    #[new]
    pub fn new(
        storage_size: usize,
        gpu_storage_size: Option<u64>,
        num_compute_threads: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Rc::new(RefCell::new(
                vng_core::runtime::RunTime::new(
                    storage_size,
                    gpu_storage_size,
                    num_compute_threads,
                )
                .map_err(|e| PyErr::new::<PyIOError, _>(format!("{}", e)))?,
            )),
        })
    }

    fn resolve(&self, v: &F32Operator) -> PyResult<f32> {
        let mut inner = self.inner.borrow_mut();
        inner
            .resolve(None, |ctx, _| {
                async move { Ok(ctx.submit(v.0.request_scalar()).await) }.into()
            })
            .map_err(|e| PyErr::new::<PyIOError, _>(format!("{}", e)))
    }
}

//let mut runtime = RunTime::new(storage_size, gpu_storage_size, args.compute_pool_size)?;

#[pyclass(unsendable)]
#[derive(Clone)]
struct F32Operator(Operator<(), f32>);

//#[pyfunction]
//fn add(a: F32Operator, b: F32Operator) -> F32Operator {
//    F32Operator(Operator {
//        eval: Rc::new(move || a.0.eval() + b.0.eval()),
//    })
//}
//

#[pyfunction]
fn constant(val: f32) -> F32Operator {
    F32Operator(vng_core::operators::scalar::constant_pod(val))
}

//#[pyfunction]
//fn resolve(v: F32Operator) -> f32 {
//    v.0.eval()
//}

/// A Python module implemented in Rust.
#[pymodule]
fn vng(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(constant, m)?)?;
    m.add_class::<F32Operator>()?;
    m.add_class::<RunTime>()?;
    Ok(())
}
