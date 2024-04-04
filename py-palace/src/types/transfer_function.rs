use crate::types::*;
use pyo3::prelude::*;

pub use palace_core::operators::raycaster::TransFuncOperator as CTransFuncOperator;

#[pyclass(unsendable)]
#[derive(Clone, Debug)]
pub struct TransFuncOperator {
    #[pyo3(get, set)]
    pub min: f32,
    #[pyo3(get, set)]
    pub max: f32,
    #[pyo3(get, set)]
    pub table: TensorOperator,
}

impl TryFrom<CTransFuncOperator> for TransFuncOperator {
    type Error = PyErr;

    fn try_from(t: CTransFuncOperator) -> Result<Self, Self::Error> {
        Ok(TransFuncOperator {
            min: t.min,
            max: t.max,
            table: t.table.try_into()?,
        })
    }
}

impl TryInto<CTransFuncOperator> for TransFuncOperator {
    type Error = PyErr;

    fn try_into(self) -> Result<CTransFuncOperator, Self::Error> {
        Ok(CTransFuncOperator {
            min: self.min,
            max: self.max,
            table: self.table.try_into()?,
        })
    }
}

#[pyfunction]
pub fn load_tf(path: std::path::PathBuf) -> PyResult<TransFuncOperator> {
    let raw_tf = crate::map_err(palace_vvd::load_tfi(&path))?;
    raw_tf.try_into()
}
