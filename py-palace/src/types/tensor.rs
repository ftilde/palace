use id::Identify;
use palace_core::array::{PyTensorEmbeddingData, PyTensorMetaData};
use palace_core::dtypes::StaticElementType;
use palace_core::{
    data::{ChunkCoordinate, Vector},
    dim::*,
    operator::Operator as COperator,
    storage::Element,
};
use pyo3::types::PyFunction;
use pyo3::{exceptions::PyException, prelude::*};

use palace_core::operators::scalar::ScalarOperator as CScalarOperator;
use palace_core::operators::tensor::EmbeddedTensorOperator as CEmbeddedTensorOperator;
use palace_core::operators::tensor::LODTensorOperator as CLODTensorOperator;
use palace_core::operators::tensor::TensorOperator as CTensorOperator;

#[pyclass(unsendable)]
#[derive(Clone, Copy, Debug)]
pub enum DType {
    F32,
    U8Vec4,
    F32Vec4A2,
}

impl DType {
    fn from<T: 'static>() -> PyResult<Self> {
        fn is<L: std::any::Any, R: std::any::Any>() -> bool {
            std::any::TypeId::of::<L>() == std::any::TypeId::of::<R>()
        }
        if is::<f32, T>() {
            Ok(DType::F32)
        } else if is::<Vector<D4, u8>, T>() {
            Ok(DType::U8Vec4)
        } else if is::<[Vector<D4, f32>; 2], T>() {
            Ok(DType::F32Vec4A2)
        } else {
            //TODO: Not sure if we actually NEED to error out
            Err(PyErr::new::<PyException, _>(format!(
                "{} is not a registered DType",
                std::any::type_name::<T>()
            )))
        }
    }
}

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

#[pyclass(unsendable)]
#[derive(Debug)]
pub struct TensorOperator {
    pub inner: Box<dyn std::any::Any>,
    #[pyo3(get)]
    pub dtype: DType,
    #[pyo3(get)]
    pub metadata: PyTensorMetaData,
    clone: fn(&Self) -> Self,
}

impl Clone for TensorOperator {
    fn clone(&self) -> Self {
        (self.clone)(self)
    }
}

impl<D: Dimension, T: std::any::Any + Clone> TryFrom<CTensorOperator<D, StaticElementType<T>>>
    for TensorOperator
{
    type Error = PyErr;

    fn try_from(t: CTensorOperator<D, StaticElementType<T>>) -> Result<Self, Self::Error> {
        Ok(Self {
            inner: Box::new(t.chunks),
            dtype: DType::from::<T>()?,
            metadata: t.metadata.into(),
            clone: |i| Self {
                inner: Box::new(
                    i.inner
                        .downcast_ref::<COperator<Vector<D, ChunkCoordinate>, T>>()
                        .unwrap()
                        .clone(),
                ),
                dtype: i.dtype,
                metadata: i.metadata.clone(),
                clone: i.clone,
            },
        })
    }
}

impl<D: Dimension, T: std::any::Any + Element> TryInto<CTensorOperator<D, StaticElementType<T>>>
    for TensorOperator
{
    type Error = PyErr;

    fn try_into(self) -> Result<CTensorOperator<D, StaticElementType<T>>, Self::Error> {
        let inner = self
            .inner
            .downcast_ref::<COperator<Vector<D, ChunkCoordinate>, StaticElementType<T>>>()
            .ok_or_else(|| {
                PyErr::new::<PyException, _>(format!(
                    "Expected Operator<Vector<{}, ChunkCoordinate>, {}>, but got something else",
                    D::N,
                    std::any::type_name::<T>()
                ))
            })?;

        Ok(CTensorOperator {
            chunks: inner.clone(),
            metadata: self.metadata.try_into()?,
        })
    }
}

#[pymethods]
impl TensorOperator {
    fn embedded(&self, embedding_data: PyTensorEmbeddingData) -> EmbeddedTensorOperator {
        EmbeddedTensorOperator {
            inner: self.clone(),
            embedding_data,
        }
    }
}

#[derive(FromPyObject)] //TODO: Derive macro appears to be broken when we use generics here??
pub enum MaybeConstTensorOperator<'a> {
    ConstD1(numpy::borrow::PyReadonlyArray1<'a, f32>),
    Operator(TensorOperator),
}

impl<'a> TryInto<CTensorOperator<D1, StaticElementType<f32>>> for MaybeConstTensorOperator<'a> {
    type Error = PyErr;

    fn try_into(self) -> Result<CTensorOperator<D1, StaticElementType<f32>>, Self::Error> {
        match self {
            MaybeConstTensorOperator::ConstD1(c) => {
                Ok(palace_core::operators::array::from_rc(c.as_slice()?.into()))
            }
            MaybeConstTensorOperator::Operator(o) => o.try_into(),
        }
    }
}

#[pyclass(unsendable)]
#[derive(Clone, Debug)]
pub struct EmbeddedTensorOperator {
    #[pyo3(get, set)]
    pub inner: TensorOperator,
    #[pyo3(get, set)]
    pub embedding_data: PyTensorEmbeddingData,
}

impl<D: Dimension, T: std::any::Any + Clone>
    TryFrom<CEmbeddedTensorOperator<D, StaticElementType<T>>> for EmbeddedTensorOperator
{
    type Error = PyErr;

    fn try_from(t: CEmbeddedTensorOperator<D, StaticElementType<T>>) -> Result<Self, Self::Error> {
        Ok(Self {
            inner: t.inner.try_into()?,
            embedding_data: t.embedding_data.into(),
        })
    }
}

impl<D: Dimension, T: std::any::Any + Element>
    TryInto<CEmbeddedTensorOperator<D, StaticElementType<T>>> for EmbeddedTensorOperator
{
    type Error = PyErr;

    fn try_into(self) -> Result<CEmbeddedTensorOperator<D, StaticElementType<T>>, Self::Error> {
        let inner = self.inner.try_into()?;
        Ok(CEmbeddedTensorOperator {
            inner,
            embedding_data: self.embedding_data.try_into()?,
        })
    }
}

#[pymethods]
impl EmbeddedTensorOperator {
    //TODO: Generalize for other dims and maybe datatypes
    fn single_level_lod(&self) -> PyResult<LODTensorOperator> {
        Ok(LODTensorOperator {
            levels: vec![self.clone()],
        })
    }
    fn create_lod(&self, step_factor: f32) -> PyResult<LODTensorOperator> {
        let nd = self.inner.metadata.dimensions.len();

        match nd {
            2 => {
                let vol: CEmbeddedTensorOperator<D2, StaticElementType<f32>> =
                    self.clone().try_into()?;
                palace_core::operators::resample::create_lod(vol, step_factor).try_into()
            }
            3 => {
                let vol: CEmbeddedTensorOperator<D3, StaticElementType<f32>> =
                    self.clone().try_into()?;
                palace_core::operators::resample::create_lod(vol, step_factor).try_into()
            }
            n => {
                return Err(PyErr::new::<PyException, _>(format!(
                    "{}-dimensional operation not yet implemented.",
                    n
                )))
            }
        }
    }
}

#[derive(FromPyObject, Debug, Clone)]
pub enum MaybeEmbeddedTensorOperator {
    Not(TensorOperator),
    Embedded(EmbeddedTensorOperator),
}

impl MaybeEmbeddedTensorOperator {
    pub fn inner(self) -> TensorOperator {
        match self {
            MaybeEmbeddedTensorOperator::Not(i) => i,
            MaybeEmbeddedTensorOperator::Embedded(e) => e.inner,
        }
    }
    pub fn try_map_inner<D: Dimension, I: Element + 'static, O: Element + 'static>(
        self,
        py: Python,
        f: impl FnOnce(
            CTensorOperator<D, StaticElementType<I>>,
        ) -> CTensorOperator<D, StaticElementType<O>>,
    ) -> PyResult<PyObject> {
        Ok(match self {
            MaybeEmbeddedTensorOperator::Not(v) => {
                let v: CTensorOperator<D, StaticElementType<I>> = v.try_into()?;
                let v = f(v);
                let v: TensorOperator = v.try_into()?;
                v.into_py(py)
            }
            MaybeEmbeddedTensorOperator::Embedded(v) => {
                let v: CEmbeddedTensorOperator<D, StaticElementType<I>> = v.try_into()?;
                let v = v.map_inner(f);
                let v: EmbeddedTensorOperator = v.try_into()?;
                v.into_py(py)
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

impl<D: Dimension, T: std::any::Any + Clone> TryFrom<CLODTensorOperator<D, StaticElementType<T>>>
    for LODTensorOperator
{
    type Error = PyErr;

    fn try_from(t: CLODTensorOperator<D, StaticElementType<T>>) -> Result<Self, Self::Error> {
        Ok(Self {
            levels: t
                .levels
                .into_iter()
                .map(|v| v.try_into())
                .collect::<Result<_, _>>()?,
        })
    }
}
impl<D: Dimension, T: std::any::Any + Element> TryInto<CLODTensorOperator<D, StaticElementType<T>>>
    for LODTensorOperator
{
    type Error = PyErr;

    fn try_into(self) -> Result<CLODTensorOperator<D, StaticElementType<T>>, Self::Error> {
        Ok(CLODTensorOperator {
            levels: self
                .levels
                .into_iter()
                .map(|v| v.try_into())
                .collect::<Result<_, _>>()?,
        })
    }
}
#[pymethods]
impl LODTensorOperator {
    pub fn fine_metadata(&self) -> PyTensorMetaData {
        self.levels[0].inner.metadata.clone()
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
