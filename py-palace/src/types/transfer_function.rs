use crate::types::*;
use pyo3::prelude::*;

pub use palace_core::operators::raycaster::TransFuncOperator as CTransFuncOperator;
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
