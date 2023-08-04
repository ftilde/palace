mod core;
mod event;
mod tensor;

pub use self::core::*;
pub use self::event::*;
pub use self::tensor::*;

use crate::conversion;
use derive_more::From;
use derive_more::Into;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use vng_core::cgmath::Matrix;
use vng_core::operator::Operator;
use vng_core::operators::volume::ChunkSize as CChunkSize;

#[pyclass]
#[derive(Clone)]
pub struct ChunkSizeFull;

#[derive(Copy, Clone)]
pub struct ChunkSize(pub CChunkSize);

impl<'source> FromPyObject<'source> for ChunkSize {
    fn extract(val: &'source PyAny) -> PyResult<Self> {
        Ok(ChunkSize(if let Ok(_) = val.extract::<ChunkSizeFull>() {
            CChunkSize::Full
        } else {
            CChunkSize::Fixed(val.extract::<u32>()?.into())
        }))
    }
}

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
pub struct Mat4Operator(pub Operator<(), vng_core::cgmath::Matrix4<f32>>);

impl<'a> conversion::FromPyValue<PyReadonlyArray2<'a, f32>> for Mat4Operator {
    fn from_py(v: PyReadonlyArray2<'a, f32>) -> PyResult<Self> {
        if v.shape() != [4, 4] {
            return Err(PyException::new_err(format!(
                "Array must be of size [4, 4], but is of size {:?}",
                v.shape()
            )));
        }

        let vals: [f32; 16] = v.as_slice()?.try_into().unwrap();
        let mat: &vng_core::cgmath::Matrix4<f32> = (&vals).into();

        assert!(v.is_c_contiguous());
        // Array is in row major order, but cgmath matrices are column major, so we need to
        // transpose
        let mat = mat.transpose();
        Ok(Mat4Operator(
            vng_core::operators::scalar::constant_as_array(mat),
        ))
    }
}
impl<'source> conversion::FromPyValues<'source> for Mat4Operator {
    type Converter = conversion::ToOperatorFrom<Self, (PyReadonlyArray2<'source, f32>,)>;
}
