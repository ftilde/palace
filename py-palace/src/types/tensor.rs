use id::Identify;
use palace_core::array::{PyTensorEmbeddingData, PyTensorMetaData};
use palace_core::dtypes::{DType, ElementType, StaticElementType};
use palace_core::jit::{self, JitTensorOperator};
use palace_core::vec::Vector;
use palace_core::{dim::*, storage::Element};
use pyo3::types::PyFunction;
use pyo3::{exceptions::PyException, prelude::*};

use palace_core::operators::scalar::ScalarOperator as CScalarOperator;
use palace_core::operators::tensor::EmbeddedTensorOperator as CEmbeddedTensorOperator;
use palace_core::operators::tensor::LODTensorOperator as CLODTensorOperator;
use palace_core::operators::tensor::TensorOperator as CTensorOperator;

#[pyclass(unsendable)]
pub struct ScalarOperator {
    pub inner: Box<dyn std::any::Any>,
    clone: fn(&Self) -> Self,
}

impl Clone for ScalarOperator {
    fn clone(&self) -> Self {
        (self.clone)(self)
    }
}

impl<T: 'static + Clone> From<CScalarOperator<StaticElementType<T>>> for ScalarOperator {
    fn from(value: CScalarOperator<StaticElementType<T>>) -> Self {
        Self {
            inner: Box::new(value),
            clone: |i| Self {
                inner: Box::new(
                    i.inner
                        .downcast_ref::<CScalarOperator<StaticElementType<T>>>()
                        .unwrap()
                        .clone(),
                ),
                clone: i.clone,
            },
        }
    }
}

impl<T: 'static + Clone> TryInto<CScalarOperator<StaticElementType<T>>> for ScalarOperator {
    type Error = PyErr;

    fn try_into(self) -> Result<CScalarOperator<StaticElementType<T>>, Self::Error> {
        Ok(self.try_unpack()?.clone())
    }
}

impl ScalarOperator {
    pub fn try_unpack<T: 'static>(&self) -> PyResult<&CScalarOperator<StaticElementType<T>>> {
        self.inner
            .downcast_ref::<CScalarOperator<StaticElementType<T>>>()
            .ok_or_else(|| {
                PyErr::new::<PyException, _>(format!(
                    "Expected ScalarOperator<{}>, but got something else",
                    std::any::type_name::<T>()
                ))
            })
    }
    pub fn try_unpack_mut<T: 'static>(
        &mut self,
    ) -> PyResult<&mut CScalarOperator<StaticElementType<T>>> {
        self.inner
            .downcast_mut::<CScalarOperator<StaticElementType<T>>>()
            .ok_or_else(|| {
                PyErr::new::<PyException, _>(format!(
                    "Expected ScalarOperator<{}>, but got something else",
                    std::any::type_name::<T>()
                ))
            })
    }
}

#[derive(FromPyObject)]
pub enum MaybeConstScalarOperator<T> {
    Const(T),
    Operator(ScalarOperator),
}

impl<T: Identify + palace_core::storage::Element> TryInto<CScalarOperator<StaticElementType<T>>>
    for MaybeConstScalarOperator<T>
{
    type Error = PyErr;

    fn try_into(self) -> Result<CScalarOperator<StaticElementType<T>>, Self::Error> {
        match self {
            MaybeConstScalarOperator::Const(c) => Ok(palace_core::operators::scalar::constant(c)),
            MaybeConstScalarOperator::Operator(o) => o.try_into(),
        }
    }
}

#[derive(Clone)]
enum MaybeJitTensorOperator {
    Jit(JitTensorOperator<DDyn>),
    Tensor(CTensorOperator<DDyn, DType>),
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct TensorOperator {
    inner: MaybeJitTensorOperator,
}

impl TensorOperator {
    pub fn nd(&self) -> usize {
        let nd = self.metadata().dimensions.len();
        assert_eq!(nd, self.metadata().chunk_size.len());
        nd
    }
}

#[pymethods]
impl TensorOperator {
    #[getter]
    pub fn dtype(&self) -> DType {
        match &self.inner {
            MaybeJitTensorOperator::Jit(i) => i.dtype(),
            MaybeJitTensorOperator::Tensor(i) => i.dtype(),
        }
    }
    #[getter]
    pub fn metadata(&self) -> PyTensorMetaData {
        match &self.inner {
            MaybeJitTensorOperator::Jit(i) => i.metadata().unwrap().into(),
            MaybeJitTensorOperator::Tensor(i) => i.metadata.clone().into(),
        }
    }
    fn embedded(&self, embedding_data: PyTensorEmbeddingData) -> EmbeddedTensorOperator {
        EmbeddedTensorOperator {
            inner: self.clone(),
            embedding_data,
        }
    }

    fn unfold_into_vec_dtype(&self) -> PyResult<Self> {
        Ok(self
            .clone()
            .into_core()
            .unfold_into_vec_dtype()
            .map_err(crate::map_err)?
            .into())
    }
    fn fold_vec_dtype(&self) -> PyResult<Self> {
        Ok(self
            .clone()
            .into_core()
            .fold_vec_dtype()
            .map_err(crate::map_err)?
            .into())
    }
}

impl From<CTensorOperator<DDyn, DType>> for TensorOperator {
    fn from(t: CTensorOperator<DDyn, DType>) -> Self {
        Self {
            inner: MaybeJitTensorOperator::Tensor(t),
        }
    }
}

impl TryFrom<jit::JitTensorOperator<DDyn>> for TensorOperator {
    type Error = PyErr;

    fn try_from(t: jit::JitTensorOperator<DDyn>) -> Result<Self, Self::Error> {
        let Some(_) = t.metadata() else {
            return crate::map_result(Err("Jit operator does not contain metadata".into()));
        };

        Ok(Self {
            inner: MaybeJitTensorOperator::Jit(t),
        })
    }
}

impl<T: 'static> From<CTensorOperator<DDyn, StaticElementType<T>>> for TensorOperator
where
    DType: From<StaticElementType<T>>,
{
    fn from(t: CTensorOperator<DDyn, StaticElementType<T>>) -> Self {
        CTensorOperator::<DDyn, DType>::from(t).into()
    }
}

impl Into<CTensorOperator<DDyn, DType>> for TensorOperator {
    fn into(self) -> CTensorOperator<DDyn, DType> {
        match self.inner {
            MaybeJitTensorOperator::Jit(j) => j.clone().compile().unwrap(),
            MaybeJitTensorOperator::Tensor(t) => t,
        }
    }
}

impl<T> TryInto<CTensorOperator<DDyn, StaticElementType<T>>> for TensorOperator
where
    StaticElementType<T>: TryFrom<DType, Error = palace_core::dtypes::ConversionError>,
{
    type Error = PyErr;
    fn try_into(self) -> Result<CTensorOperator<DDyn, StaticElementType<T>>, Self::Error> {
        let t: CTensorOperator<DDyn, DType> = self.into();
        Ok(t.try_into()?)
    }
}

impl Into<jit::JitTensorOperator<DDyn>> for TensorOperator {
    fn into(self) -> jit::JitTensorOperator<DDyn> {
        match self.inner {
            MaybeJitTensorOperator::Jit(j) => j,
            MaybeJitTensorOperator::Tensor(t) => t.into(),
        }
    }
}

pub fn try_into_static_err<D: Dimension, T: ElementType>(
    vol: CTensorOperator<DDyn, T>,
) -> Result<CTensorOperator<D, T>, PyErr> {
    let n = vol.dim().n();
    vol.try_into_static().ok_or_else(|| {
        PyErr::new::<PyException, _>(format!(
            "Unable to convert dynamic dimension {} into static dimension {}",
            n,
            D::N
        ))
    })
}

impl TensorOperator {
    pub fn into_core(self) -> CTensorOperator<DDyn, DType> {
        self.into()
    }
    pub fn try_into_core_static<D: Dimension>(self) -> Result<CTensorOperator<D, DType>, PyErr> {
        let s = self.into_core();
        try_into_static_err(s)
    }
    pub fn into_jit(self) -> jit::JitTensorOperator<DDyn> {
        self.into()
    }
}

fn try_tensor_from_numpy<T: Element + numpy::Element + id::Identify>(
    c: &numpy::PyUntypedArray,
) -> PyResult<CTensorOperator<DDyn, DType>> {
    let arr: &numpy::PyArrayDyn<T> = c.downcast()?;
    let dim = Vector::<DDyn, usize>::try_from_slice(arr.shape()).unwrap();

    let values = if arr.is_contiguous() {
        // Safety: There are no other references (not sure how they would even exist?)
        arr.to_vec().unwrap()
    } else {
        let arr = arr
            .reshape_with_order(dim.clone().inner(), numpy::npyffi::NPY_ORDER::NPY_CORDER)
            .unwrap();
        arr.to_vec().unwrap()
    };
    let dim = dim.map(|v| v as u32).global();
    let op = CTensorOperator::from_vec(dim, values).unwrap();
    Ok(op.into())
}
pub fn tensor_from_numpy(c: &numpy::PyUntypedArray) -> PyResult<CTensorOperator<DDyn, DType>> {
    let fns = [
        try_tensor_from_numpy::<i8>,
        try_tensor_from_numpy::<u8>,
        try_tensor_from_numpy::<i16>,
        try_tensor_from_numpy::<u16>,
        try_tensor_from_numpy::<i32>,
        try_tensor_from_numpy::<u32>,
        try_tensor_from_numpy::<f32>,
    ];

    for f in fns {
        if let Ok(t) = f(c) {
            return Ok(t);
        }
    }

    Err(PyErr::new::<PyException, _>(format!(
        "Unable to convert ndarray of type {} to tensor",
        c.dtype()
    )))
}

#[derive(FromPyObject)]
pub enum MaybeConstTensorOperator<'a> {
    Numpy(&'a numpy::PyUntypedArray),
    Operator(TensorOperator),
}

impl<'a> TryInto<CTensorOperator<DDyn, DType>> for MaybeConstTensorOperator<'a> {
    type Error = PyErr;

    fn try_into(self) -> Result<CTensorOperator<DDyn, DType>, Self::Error> {
        match self {
            MaybeConstTensorOperator::Numpy(c) => tensor_from_numpy(c),
            MaybeConstTensorOperator::Operator(o) => Ok(o.into_core()),
        }
    }
}

impl MaybeConstTensorOperator<'_> {
    pub fn try_into_core(self) -> Result<CTensorOperator<DDyn, DType>, PyErr> {
        self.try_into()
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct EmbeddedTensorOperator {
    #[pyo3(get, set)]
    pub inner: TensorOperator,
    #[pyo3(get, set)]
    pub embedding_data: PyTensorEmbeddingData,
}

impl TryFrom<CEmbeddedTensorOperator<DDyn, DType>> for EmbeddedTensorOperator {
    type Error = PyErr;

    fn try_from(t: CEmbeddedTensorOperator<DDyn, DType>) -> Result<Self, Self::Error> {
        Ok(Self {
            inner: t.inner.try_into()?,
            embedding_data: t.embedding_data.into(),
        })
    }
}
impl<T: 'static> TryFrom<CEmbeddedTensorOperator<DDyn, StaticElementType<T>>>
    for EmbeddedTensorOperator
where
    DType: From<StaticElementType<T>>,
{
    type Error = PyErr;

    fn try_from(
        t: CEmbeddedTensorOperator<DDyn, StaticElementType<T>>,
    ) -> Result<Self, Self::Error> {
        CEmbeddedTensorOperator::<DDyn, DType>::from(t).try_into()
    }
}

impl Into<CEmbeddedTensorOperator<DDyn, DType>> for EmbeddedTensorOperator {
    fn into(self) -> CEmbeddedTensorOperator<DDyn, DType> {
        let inner = self.inner.into();
        CEmbeddedTensorOperator {
            inner,
            embedding_data: self.embedding_data.try_into().unwrap(),
        }
    }
}
impl<T> TryInto<CEmbeddedTensorOperator<DDyn, StaticElementType<T>>> for EmbeddedTensorOperator
where
    StaticElementType<T>: TryFrom<DType, Error = palace_core::dtypes::ConversionError>,
{
    type Error = PyErr;
    fn try_into(self) -> Result<CEmbeddedTensorOperator<DDyn, StaticElementType<T>>, Self::Error> {
        let t: CEmbeddedTensorOperator<DDyn, DType> = self.try_into()?;
        Ok(t.try_into()?)
    }
}

#[pymethods]
impl EmbeddedTensorOperator {
    fn single_level_lod(&self) -> PyResult<LODTensorOperator> {
        Ok(LODTensorOperator {
            levels: vec![self.clone()],
        })
    }
    fn create_lod(&self, step_factor: f32) -> PyResult<LODTensorOperator> {
        Ok(
            palace_core::operators::resample::create_lod(self.clone().into_core(), step_factor)
                .try_into()?,
        )
    }
}

impl EmbeddedTensorOperator {
    pub fn into_core(self) -> CEmbeddedTensorOperator<DDyn, DType> {
        self.into()
    }
    pub fn try_into_core_static<D: Dimension>(self) -> PyResult<CEmbeddedTensorOperator<D, DType>> {
        Ok(CEmbeddedTensorOperator {
            inner: self.inner.try_into_core_static()?,
            embedding_data: self.embedding_data.try_into_dim()?,
        })
    }
}

#[derive(FromPyObject, Clone)]
pub enum MaybeEmbeddedTensorOperator {
    Not(TensorOperator),
    Embedded(EmbeddedTensorOperator),
}

impl MaybeEmbeddedTensorOperator {
    pub fn inner(&self) -> &TensorOperator {
        match self {
            MaybeEmbeddedTensorOperator::Not(i) => i,
            MaybeEmbeddedTensorOperator::Embedded(e) => &e.inner,
        }
    }
    pub fn into_inner(self) -> TensorOperator {
        match self {
            MaybeEmbeddedTensorOperator::Not(i) => i,
            MaybeEmbeddedTensorOperator::Embedded(e) => e.inner,
        }
    }
    pub fn try_map_inner(
        self,
        py: Python,
        f: impl FnOnce(CTensorOperator<DDyn, DType>) -> PyResult<CTensorOperator<DDyn, DType>>,
    ) -> PyResult<PyObject> {
        Ok(match self {
            MaybeEmbeddedTensorOperator::Not(v) => {
                let v: CTensorOperator<DDyn, DType> = v.into_core();
                let v = f(v)?;
                let v: TensorOperator = v.into();
                v.into_py(py)
            }
            MaybeEmbeddedTensorOperator::Embedded(orig) => {
                let v: CTensorOperator<DDyn, DType> = orig.inner.into_core();
                let v = f(v)?;
                let v: TensorOperator = v.into();
                EmbeddedTensorOperator {
                    inner: v,
                    embedding_data: orig.embedding_data,
                }
                .into_py(py)
            }
        })
    }

    pub fn try_map_inner_jit(
        self,
        py: Python,
        f: impl FnOnce(jit::JitTensorOperator<DDyn>) -> PyResult<jit::JitTensorOperator<DDyn>>,
    ) -> PyResult<PyObject> {
        Ok(match self {
            MaybeEmbeddedTensorOperator::Not(v) => {
                let v = v.into_jit();
                let v = f(v)?;
                let v: TensorOperator = v.try_into()?;
                v.into_py(py)
            }
            MaybeEmbeddedTensorOperator::Embedded(orig) => {
                let v = orig.inner.into_jit();
                let v = f(v)?;
                let v: TensorOperator = v.try_into()?;
                EmbeddedTensorOperator {
                    inner: v,
                    embedding_data: orig.embedding_data,
                }
                .into_py(py)
            }
        })
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct LODTensorOperator {
    #[pyo3(get, set)]
    pub levels: Vec<EmbeddedTensorOperator>,
}
impl<D: DynDimension> TryFrom<CLODTensorOperator<D, DType>> for LODTensorOperator {
    type Error = PyErr;

    fn try_from(t: CLODTensorOperator<D, DType>) -> Result<Self, Self::Error> {
        Ok(Self {
            levels: t
                .levels
                .into_iter()
                .map(|v| v.into_dyn().try_into())
                .collect::<Result<_, _>>()?,
        })
    }
}

impl<D: DynDimension, T: std::any::Any + Clone> TryFrom<CLODTensorOperator<D, StaticElementType<T>>>
    for LODTensorOperator
where
    DType: From<StaticElementType<T>>,
{
    type Error = PyErr;

    fn try_from(t: CLODTensorOperator<D, StaticElementType<T>>) -> Result<Self, Self::Error> {
        Ok(Self {
            levels: t
                .levels
                .into_iter()
                .map(|v| v.into_dyn().try_into())
                .collect::<Result<_, _>>()?,
        })
    }
}
impl Into<CLODTensorOperator<DDyn, DType>> for LODTensorOperator {
    fn into(self) -> CLODTensorOperator<DDyn, DType> {
        CLODTensorOperator {
            levels: self
                .levels
                .into_iter()
                .map(|v| v.into_core())
                .collect::<Vec<_>>(),
        }
    }
}
impl<T: std::any::Any + Element> TryInto<CLODTensorOperator<DDyn, StaticElementType<T>>>
    for LODTensorOperator
where
    StaticElementType<T>: TryFrom<DType, Error = palace_core::dtypes::ConversionError>,
{
    type Error = PyErr;

    fn try_into(self) -> Result<CLODTensorOperator<DDyn, StaticElementType<T>>, Self::Error> {
        Ok(CLODTensorOperator {
            levels: self
                .levels
                .into_iter()
                .map(|v| v.into_core().try_into())
                .collect::<Result<_, _>>()?,
        })
    }
}
impl LODTensorOperator {
    pub fn into_core(self) -> CLODTensorOperator<DDyn, DType> {
        self.into()
    }

    pub fn try_into_core_static<D: Dimension>(self) -> Result<CLODTensorOperator<D, DType>, PyErr> {
        Ok(CLODTensorOperator {
            levels: self
                .levels
                .into_iter()
                .map(|v| v.try_into_core_static())
                .collect::<Result<_, _>>()?,
        })
    }
}

#[pymethods]
impl LODTensorOperator {
    pub fn fine_metadata(&self) -> PyTensorMetaData {
        self.levels[0].inner.metadata().clone()
    }
    pub fn fine_embedding_data(&self) -> PyTensorEmbeddingData {
        self.levels[0].embedding_data.clone()
    }

    pub fn map(&self, py: Python, f: Py<PyFunction>) -> PyResult<Self> {
        Ok(Self {
            levels: self
                .levels
                .iter()
                .map(|l| {
                    f.call1(py, (l.clone().into_py(py),))
                        .and_then(|v| v.extract::<EmbeddedTensorOperator>(py))
                })
                .collect::<PyResult<Vec<_>>>()?,
        })
    }
}
