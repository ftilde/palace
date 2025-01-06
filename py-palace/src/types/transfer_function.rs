use crate::types::*;
use pyo3::prelude::*;

pub use palace_core::transfunc::TransFuncOperator as CTransFuncOperator;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct TransFuncOperator {
    #[pyo3(get, set)]
    pub min: f32,
    #[pyo3(get, set)]
    pub max: f32,
    #[pyo3(get, set)]
    pub table: TensorOperator,
}

#[pymethods]
impl TransFuncOperator {
    #[new]
    fn new(min: f32, max: f32, table: TensorOperator) -> Self {
        Self { min, max, table }
    }
}

impl From<CTransFuncOperator> for TransFuncOperator {
    fn from(t: CTransFuncOperator) -> Self {
        TransFuncOperator {
            min: t.min,
            max: t.max,
            table: t.table.into_dyn().into(),
        }
    }
}

impl TryInto<CTransFuncOperator> for TransFuncOperator {
    type Error = PyErr;

    fn try_into(self) -> Result<CTransFuncOperator, Self::Error> {
        Ok(CTransFuncOperator {
            min: self.min,
            max: self.max,
            table: self.table.try_into_core_static()?.try_into()?,
        })
    }
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn load_tf(path: std::path::PathBuf) -> PyResult<TransFuncOperator> {
    let raw_tf = crate::map_result(palace_vvd::load_tfi(&path))?;
    Ok(raw_tf.into())
}

#[gen_stub_pyfunction]
#[pyfunction]
pub fn grey_ramp_tf(min: f32, max: f32) -> TransFuncOperator {
    CTransFuncOperator::grey_ramp(min, max).into()
}

//#[gen_stub_pyfunction] TODO: Not working because PyUntypedArray does not impl the required type
#[pyfunction]
pub fn tf_from_numpy(
    min: f32,
    max: f32,
    values: &Bound<numpy::PyUntypedArray>,
) -> PyResult<TransFuncOperator> {
    let values = tensor_from_numpy(values)?;
    Ok(CTransFuncOperator {
        table: try_into_static_err(values)?.try_into()?,
        min,
        max,
    }
    .into())
}
