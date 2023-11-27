use pyo3::{exceptions::PyException, prelude::*};
use vng_core::array::TensorEmbeddingData as CTensorEmbeddingData;
use vng_core::array::TensorMetaData as CTensorMetaData;
use vng_core::{
    array::{ArrayMetaData, VolumeMetaData},
    data::{ChunkCoordinate, Vector},
    id::Identify,
    operator::Operator as COperator,
    storage::Element,
};

use vng_core::array::ImageMetaData;
use vng_core::operators::scalar::ScalarOperator as CScalarOperator;
use vng_core::operators::tensor::EmbeddedTensorOperator as CEmbeddedTensorOperator;
use vng_core::operators::tensor::LODTensorOperator as CLODTensorOperator;
use vng_core::operators::tensor::TensorOperator as CTensorOperator;

#[pyfunction]
pub fn tensor_metadata(
    py: Python,
    dimensions: PyObject,
    chunk_size: PyObject,
) -> PyResult<PyObject> {
    let ld = dimensions.as_ref(py).len()?;
    let lc = chunk_size.as_ref(py).len()?;
    if ld == lc {
        Ok(match ld {
            1 => ArrayMetaData {
                dimensions: dimensions.extract(py)?,
                chunk_size: chunk_size.extract(py)?,
            }
            .into_py(py),
            2 => ImageMetaData {
                dimensions: dimensions.extract(py)?,
                chunk_size: chunk_size.extract(py)?,
            }
            .into_py(py),
            3 => VolumeMetaData {
                dimensions: dimensions.extract(py)?,
                chunk_size: chunk_size.extract(py)?,
            }
            .into_py(py),
            n => {
                return Err(PyErr::new::<PyException, _>(format!(
                    "{}-dimensional tensor metadata not yet implemented.",
                    n
                )))
            }
        })
    } else {
        Err(PyErr::new::<PyException, _>(format!(
            "Len missmatch between dimensions and chunk_size ({} vs {})",
            ld, lc
        )))
    }
}

#[pyclass(unsendable)]
#[derive(Clone, Copy)]
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
        } else if is::<Vector<4, u8>, T>() {
            Ok(DType::U8Vec4)
        } else if is::<[Vector<4, f32>; 2], T>() {
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

impl<T: 'static> From<CScalarOperator<T>> for ScalarOperator {
    fn from(value: CScalarOperator<T>) -> Self {
        Self {
            inner: Box::new(value),
            clone: |i| Self {
                inner: Box::new(
                    i.inner
                        .downcast_ref::<CScalarOperator<T>>()
                        .unwrap()
                        .clone(),
                ),
                clone: i.clone,
            },
        }
    }
}

impl<T: 'static> TryInto<CScalarOperator<T>> for ScalarOperator {
    type Error = PyErr;

    fn try_into(self) -> Result<CScalarOperator<T>, Self::Error> {
        Ok(self.try_unpack()?.clone())
    }
}

impl ScalarOperator {
    pub fn try_unpack<T: 'static>(&self) -> PyResult<&CScalarOperator<T>> {
        self.inner
            .downcast_ref::<CScalarOperator<T>>()
            .ok_or_else(|| {
                PyErr::new::<PyException, _>(format!(
                    "Expected ScalarOperator<{}>, but got something else",
                    std::any::type_name::<T>()
                ))
            })
    }
    pub fn try_unpack_mut<T: 'static>(&mut self) -> PyResult<&mut CScalarOperator<T>> {
        self.inner
            .downcast_mut::<CScalarOperator<T>>()
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

impl<T: Identify + vng_core::storage::Element> TryInto<CScalarOperator<T>>
    for MaybeConstScalarOperator<T>
{
    type Error = PyErr;

    fn try_into(self) -> Result<CScalarOperator<T>, Self::Error> {
        match self {
            MaybeConstScalarOperator::Const(c) => Ok(vng_core::operators::scalar::constant(c)),
            MaybeConstScalarOperator::Operator(o) => o.try_into(),
        }
    }
}

type CTensorDataOperator<const N: usize, T> = COperator<Vector<N, ChunkCoordinate>, T>;

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct TensorMetaData {
    #[pyo3(get)]
    pub dimensions: Vec<u32>,
    #[pyo3(get)]
    pub chunk_size: Vec<u32>,
}

impl<const N: usize> From<CTensorMetaData<N>> for TensorMetaData {
    fn from(t: CTensorMetaData<N>) -> Self {
        Self {
            dimensions: t.dimensions.raw().into_iter().collect(),
            chunk_size: t.chunk_size.raw().into_iter().collect(),
        }
    }
}

impl<const N: usize> TryInto<CTensorMetaData<N>> for TensorMetaData {
    type Error = PyErr;

    fn try_into(self) -> Result<CTensorMetaData<N>, Self::Error> {
        if N != self.dimensions.len() {
            return Err(PyErr::new::<PyException, _>(format!(
                "Expected TensorMetaData<{}>, but got TensorMetaData<{}>",
                N,
                self.dimensions.len()
            )));
        }

        assert_eq!(self.dimensions.len(), self.chunk_size.len());

        Ok(CTensorMetaData {
            dimensions: self.dimensions.try_into().unwrap(),
            chunk_size: self.chunk_size.try_into().unwrap(),
        })
    }
}

#[pyclass(unsendable)]
pub struct TensorOperator {
    pub inner: Box<dyn std::any::Any>,
    #[pyo3(get)]
    pub dtype: DType,
    #[pyo3(get)]
    metadata: TensorMetaData,
    clone: fn(&Self) -> Self,
}

impl Clone for TensorOperator {
    fn clone(&self) -> Self {
        (self.clone)(self)
    }
}

impl<const N: usize, T: std::any::Any> TryFrom<CTensorOperator<N, T>> for TensorOperator {
    type Error = PyErr;

    fn try_from(t: CTensorOperator<N, T>) -> Result<Self, Self::Error> {
        Ok(Self {
            inner: Box::new(t.chunks),
            dtype: DType::from::<T>()?,
            metadata: t.metadata.into(),
            clone: |i| Self {
                inner: Box::new(
                    i.inner
                        .downcast_ref::<COperator<Vector<N, ChunkCoordinate>, T>>()
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

impl<const N: usize, T: std::any::Any> TryInto<CTensorOperator<N, T>> for TensorOperator {
    type Error = PyErr;

    fn try_into(self) -> Result<CTensorOperator<N, T>, Self::Error> {
        let inner = self
            .inner
            .downcast_ref::<CTensorDataOperator<N, T>>()
            .ok_or_else(|| {
                PyErr::new::<PyException, _>(format!(
                    "Expected Operator<Vector<{}, ChunkCoordinate>, {}>, but got something else",
                    N,
                    std::any::type_name::<T>()
                ))
            })?;

        Ok(CTensorOperator {
            chunks: inner.clone(),
            metadata: self.metadata.try_into()?,
        })
    }
}

#[derive(FromPyObject)] //TODO: Derive macro appears to be broken when we use generics here??
pub enum MaybeConstTensorOperator<'a> {
    ConstD1(numpy::borrow::PyReadonlyArray1<'a, f32>),
    Operator(TensorOperator),
}

impl<'a> TryInto<CTensorOperator<1, f32>> for MaybeConstTensorOperator<'a> {
    type Error = PyErr;

    fn try_into(self) -> Result<CTensorOperator<1, f32>, Self::Error> {
        match self {
            MaybeConstTensorOperator::ConstD1(c) => {
                Ok(vng_core::operators::array::from_rc(c.as_slice()?.into()))
            }
            MaybeConstTensorOperator::Operator(o) => o.try_into(),
        }
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct TensorEmbeddingData {
    #[pyo3(get)]
    pub spacing: Vec<f32>,
}

impl<const N: usize> From<CTensorEmbeddingData<N>> for TensorEmbeddingData {
    fn from(t: CTensorEmbeddingData<N>) -> Self {
        Self {
            spacing: t.spacing.into_iter().collect(),
        }
    }
}

impl<const N: usize> TryInto<CTensorEmbeddingData<N>> for TensorEmbeddingData {
    type Error = PyErr;

    fn try_into(self) -> Result<CTensorEmbeddingData<N>, Self::Error> {
        if N != self.spacing.len() {
            return Err(PyErr::new::<PyException, _>(format!(
                "Expected TensorEmbeddingData<{}>, but got TensorEmbeddingData<{}>",
                N,
                self.spacing.len()
            )));
        }

        Ok(CTensorEmbeddingData {
            spacing: self.spacing.try_into().unwrap(),
        })
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct EmbeddedTensorOperator {
    #[pyo3(get, set)]
    pub inner: TensorOperator,
    #[pyo3(get, set)]
    pub embedding_data: TensorEmbeddingData,
}

impl<const N: usize, T: std::any::Any> TryFrom<CEmbeddedTensorOperator<N, T>>
    for EmbeddedTensorOperator
{
    type Error = PyErr;

    fn try_from(t: CEmbeddedTensorOperator<N, T>) -> Result<Self, Self::Error> {
        Ok(Self {
            inner: t.inner.try_into()?,
            embedding_data: t.embedding_data.into(),
        })
    }
}

impl<const N: usize, T: std::any::Any> TryInto<CEmbeddedTensorOperator<N, T>>
    for EmbeddedTensorOperator
{
    type Error = PyErr;

    fn try_into(self) -> Result<CEmbeddedTensorOperator<N, T>, Self::Error> {
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
    fn create_lod(&self, step_factor: f32) -> PyResult<LODTensorOperator> {
        let vol: CEmbeddedTensorOperator<3, f32> = self.clone().try_into()?;
        vng_core::operators::resample::create_lod(vol, step_factor).try_into()
    }
}

#[derive(FromPyObject)]
pub enum MaybeEmbeddedTensorOperator {
    Not(TensorOperator),
    Embedded(EmbeddedTensorOperator),
}

impl MaybeEmbeddedTensorOperator {
    pub fn try_map_inner<const N: usize, T: Element + 'static>(
        self,
        py: Python,
        f: impl FnOnce(CTensorOperator<N, T>) -> CTensorOperator<N, T>,
    ) -> PyResult<PyObject> {
        Ok(match self {
            MaybeEmbeddedTensorOperator::Not(v) => {
                let v: CTensorOperator<N, T> = v.try_into()?;
                let v = f(v);
                let v: TensorOperator = v.try_into()?;
                v.into_py(py)
            }
            MaybeEmbeddedTensorOperator::Embedded(v) => {
                let v: CEmbeddedTensorOperator<N, T> = v.try_into()?;
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

impl<const N: usize, T: std::any::Any> TryFrom<CLODTensorOperator<N, T>> for LODTensorOperator {
    type Error = PyErr;

    fn try_from(t: CLODTensorOperator<N, T>) -> Result<Self, Self::Error> {
        Ok(Self {
            levels: t
                .levels
                .into_iter()
                .map(|v| v.try_into())
                .collect::<Result<_, _>>()?,
        })
    }
}
impl<const N: usize, T: std::any::Any> TryInto<CLODTensorOperator<N, T>> for LODTensorOperator {
    type Error = PyErr;

    fn try_into(self) -> Result<CLODTensorOperator<N, T>, Self::Error> {
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
    pub fn fine_metadata(&self) -> TensorMetaData {
        self.levels[0].inner.metadata.clone()
    }
    pub fn fine_embedding_data(&self) -> TensorEmbeddingData {
        self.levels[0].embedding_data.clone()
    }
}
