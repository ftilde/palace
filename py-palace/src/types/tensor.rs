use id::Identify;
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use palace_core::array::{PyTensorEmbeddingData, PyTensorMetaData};
use palace_core::dtypes::{DType, ElementType, ScalarType, StaticElementType};
use palace_core::jit::{self, BinOp, JitTensorOperator, TernaryOp, UnaryOp};
use palace_core::mat::Matrix;
use palace_core::operators::conv::BorderHandling;
use palace_core::vec::Vector;
use palace_core::{dim::*, storage::Element};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyFunction, PySlice};
use pyo3::IntoPyObjectExt;
use pyo3::{exceptions::PyException, prelude::*};

use palace_core::operators::scalar::ScalarOperator as CScalarOperator;
use palace_core::operators::tensor::EmbeddedTensorOperator as CEmbeddedTensorOperator;
use palace_core::operators::tensor::LODTensorOperator as CLODTensorOperator;
use palace_core::operators::tensor::TensorOperator as CTensorOperator;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

use crate::jit::{jit_binary, jit_ternary, jit_unary, JitArgument};
use crate::types::DeviceId;

use super::ChunkSize;

#[gen_stub_pyclass]
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

#[gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct TensorOperator {
    inner: MaybeJitTensorOperator,
}

impl TensorOperator {
    pub fn nd(&self) -> PyResult<usize> {
        let nd = self.metadata()?.dimensions.len();
        assert_eq!(nd, self.metadata()?.chunk_size.len());
        Ok(nd)
    }
}

impl From<CTensorOperator<DDyn, DType>> for TensorOperator {
    fn from(t: CTensorOperator<DDyn, DType>) -> Self {
        Self {
            inner: MaybeJitTensorOperator::Tensor(t),
        }
    }
}

impl From<jit::JitTensorOperator<DDyn>> for TensorOperator {
    fn from(t: jit::JitTensorOperator<DDyn>) -> Self {
        Self {
            inner: MaybeJitTensorOperator::Jit(t),
        }
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
    pub fn map_core(
        self,
        f: impl FnOnce(CTensorOperator<DDyn, DType>) -> PyResult<CTensorOperator<DDyn, DType>>,
    ) -> PyResult<Self> {
        let core = self.into_core();
        let res = f(core)?;
        Ok(res.into())
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
    c: &Bound<numpy::PyUntypedArray>,
) -> PyResult<CTensorOperator<DDyn, DType>> {
    let arr: &Bound<numpy::PyArrayDyn<T>> = c.downcast()?;
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
pub fn tensor_from_numpy(
    c: &Bound<numpy::PyUntypedArray>,
) -> PyResult<CTensorOperator<DDyn, DType>> {
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
    Numpy(Bound<'a, numpy::PyUntypedArray>),
    Operator(TensorOperator),
}

impl<'a> pyo3_stub_gen::PyStubType for MaybeConstTensorOperator<'a> {
    fn type_output() -> pyo3_stub_gen::TypeInfo {
        pyo3_stub_gen::TypeInfo {
            name: format!("MaybeConstTensorOperator"),
            import: Default::default(),
        }
    }
}

impl<'a> TryInto<CTensorOperator<DDyn, DType>> for MaybeConstTensorOperator<'a> {
    type Error = PyErr;

    fn try_into(self) -> Result<CTensorOperator<DDyn, DType>, Self::Error> {
        match self {
            MaybeConstTensorOperator::Numpy(c) => tensor_from_numpy(&c),
            MaybeConstTensorOperator::Operator(o) => Ok(o.into_core()),
        }
    }
}

impl MaybeConstTensorOperator<'_> {
    pub fn try_into_core(self) -> Result<CTensorOperator<DDyn, DType>, PyErr> {
        self.try_into()
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct EmbeddedTensorOperator {
    #[pyo3(get, set)]
    pub inner: TensorOperator,
    #[pyo3(get, set)]
    pub embedding_data: PyTensorEmbeddingData,
}

impl From<CEmbeddedTensorOperator<DDyn, DType>> for EmbeddedTensorOperator {
    fn from(t: CEmbeddedTensorOperator<DDyn, DType>) -> Self {
        Self {
            inner: t.inner.into(),
            embedding_data: t.embedding_data.into(),
        }
    }
}
impl<T: 'static> From<CEmbeddedTensorOperator<DDyn, StaticElementType<T>>>
    for EmbeddedTensorOperator
where
    DType: From<StaticElementType<T>>,
{
    fn from(t: CEmbeddedTensorOperator<DDyn, StaticElementType<T>>) -> Self {
        CEmbeddedTensorOperator::<DDyn, DType>::from(t).into()
    }
}

impl Into<CEmbeddedTensorOperator<DDyn, DType>> for EmbeddedTensorOperator {
    fn into(self) -> CEmbeddedTensorOperator<DDyn, DType> {
        let inner = self.inner.into();
        let ret = CEmbeddedTensorOperator {
            inner,
            embedding_data: self.embedding_data.try_into().unwrap(),
        };
        assert_eq!(ret.inner.dim().n(), ret.embedding_data.spacing.len());
        ret
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
    pub fn map_inner(
        &self,
        f: impl FnOnce(&TensorOperator) -> PyResult<TensorOperator>,
    ) -> PyResult<Self> {
        Ok(f(&self.inner)?.embedded(self.embedding_data.clone())?)
    }
    pub fn map_core_inner(
        self,
        f: impl FnOnce(CTensorOperator<DDyn, DType>) -> PyResult<CTensorOperator<DDyn, DType>>,
    ) -> PyResult<Self> {
        let core = self.into_core();
        let res = f(core.inner)?;
        Ok(res.embedded(core.embedding_data).into())
    }
}

#[gen_stub_pyclass_enum]
#[derive(FromPyObject, Clone)]
pub enum MaybeEmbeddedTensorOperatorArg {
    Not(TensorOperator),
    Embedded(EmbeddedTensorOperator),
    Maybe(MaybeEmbeddedTensorOperator),
}

impl MaybeEmbeddedTensorOperatorArg {
    pub fn unpack(self) -> MaybeEmbeddedTensorOperator {
        match self {
            MaybeEmbeddedTensorOperatorArg::Not(i) => MaybeEmbeddedTensorOperator::Not { i },
            MaybeEmbeddedTensorOperatorArg::Embedded(e) => {
                MaybeEmbeddedTensorOperator::Embedded { e }
            }
            MaybeEmbeddedTensorOperatorArg::Maybe(e) => e,
        }
    }
}

impl From<MaybeEmbeddedTensorOperatorArg> for MaybeEmbeddedTensorOperator {
    fn from(value: MaybeEmbeddedTensorOperatorArg) -> Self {
        value.unpack()
    }
}

#[gen_stub_pyclass_enum]
#[pyclass(unsendable)]
#[derive(Clone)]
pub enum MaybeEmbeddedTensorOperator {
    Not { i: TensorOperator },
    Embedded { e: EmbeddedTensorOperator },
}

//#[gen_stub_pymethods] results in internal error: entered unreachable code
#[pymethods]
impl MaybeEmbeddedTensorOperator {
    fn embedded(&self, embedding_data: PyTensorEmbeddingData) -> PyResult<EmbeddedTensorOperator> {
        Ok(match self {
            MaybeEmbeddedTensorOperator::Not { i } => i.embedded(embedding_data)?,
            MaybeEmbeddedTensorOperator::Embedded { e } => e.clone(),
        })
    }
    pub fn inner(&self) -> TensorOperator {
        self.inner_ref().clone()
    }
}

impl MaybeEmbeddedTensorOperator {
    pub fn inner_ref(&self) -> &TensorOperator {
        match self {
            MaybeEmbeddedTensorOperator::Not { i } => i,
            MaybeEmbeddedTensorOperator::Embedded { e } => &e.inner,
        }
    }
    pub fn into_inner(self) -> TensorOperator {
        match self {
            MaybeEmbeddedTensorOperator::Not { i } => i,
            MaybeEmbeddedTensorOperator::Embedded { e } => e.inner,
        }
    }
    pub fn try_map_inner(
        self,
        py: Python,
        f: impl FnOnce(CTensorOperator<DDyn, DType>) -> PyResult<CTensorOperator<DDyn, DType>>,
    ) -> PyResult<PyObject> {
        match self {
            MaybeEmbeddedTensorOperator::Not { i } => {
                let v: CTensorOperator<DDyn, DType> = i.into_core();
                let v = f(v)?;
                let v: TensorOperator = v.into();
                v.into_py_any(py)
            }
            MaybeEmbeddedTensorOperator::Embedded { e } => {
                let v: CTensorOperator<DDyn, DType> = e.inner.into_core();
                let v = f(v)?;
                let v: TensorOperator = v.into();
                EmbeddedTensorOperator {
                    inner: v,
                    embedding_data: e.embedding_data,
                }
                .into_py_any(py)
            }
        }
    }

    pub fn try_map_inner_jit(
        self,
        f: impl FnOnce(jit::JitTensorOperator<DDyn>) -> PyResult<jit::JitTensorOperator<DDyn>>,
    ) -> PyResult<MaybeEmbeddedTensorOperator> {
        Ok(match self {
            MaybeEmbeddedTensorOperator::Not { i } => {
                let v = i.into_jit();
                let v = f(v)?;
                let v: TensorOperator = v.try_into()?;
                MaybeEmbeddedTensorOperator::Not { i: v }
            }
            MaybeEmbeddedTensorOperator::Embedded { e } => {
                let v = e.inner.into_jit();
                let v = f(v)?;
                let v: TensorOperator = v.try_into()?;
                MaybeEmbeddedTensorOperator::Embedded {
                    e: EmbeddedTensorOperator {
                        inner: v,
                        embedding_data: e.embedding_data,
                    },
                }
            }
        })
    }
}

#[gen_stub_pyclass]
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
    pub fn fine_metadata(&self) -> PyResult<PyTensorMetaData> {
        Ok(self.levels[0].inner.metadata()?.clone())
    }
    pub fn fine_embedding_data(&self) -> PyTensorEmbeddingData {
        self.levels[0].embedding_data.clone()
    }

    pub fn nd(&self) -> PyResult<usize> {
        let nd = self.levels[0].nd()?;
        for level in &self.levels {
            assert_eq!(level.nd()?, nd);
        }
        Ok(nd)
    }

    pub fn map(&self, py: Python, f: Py<PyFunction>) -> PyResult<Self> {
        Ok(Self {
            levels: self
                .levels
                .iter()
                .map(|l| {
                    f.call1(py, (l.clone().into_bound_py_any(py).unwrap(),))
                        .and_then(|v| v.extract::<EmbeddedTensorOperator>(py))
                })
                .collect::<PyResult<Vec<_>>>()?,
        })
    }

    pub fn cache_coarse_levels(&self) -> Self {
        Self {
            levels: self
                .levels
                .iter()
                .enumerate()
                .map(|(i, l)| if i != 0 { l.cache() } else { l.clone() })
                .collect(),
        }
    }

    pub fn cache(&self) -> Self {
        Self {
            levels: self.levels.iter().map(|l| l.cache()).collect(),
        }
    }

    fn distribute_on_gpus(&self, devices: Vec<DeviceId>) -> Self {
        Self {
            levels: self
                .levels
                .iter()
                .map(|l| l.distribute_on_gpus(devices.clone()))
                .collect(),
        }
    }
}

#[pyclass]
#[derive(Copy, Clone)]
pub struct FixedStep(f32);

#[pymethods]
impl FixedStep {
    #[new]
    fn new(v: f32) -> Self {
        Self(v)
    }
}

#[gen_stub_pyclass_enum]
pub enum DownsampleStep {
    Ignore,
    Fixed(f32),
    Synchronized(f32),
}

impl<'py> FromPyObject<'py> for DownsampleStep {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(if let Ok(FixedStep(f)) = ob.extract::<FixedStep>() {
            DownsampleStep::Fixed(f)
        } else {
            let v = ob.extract::<Option<f32>>()?;
            match v {
                Some(f) => DownsampleStep::Synchronized(f),
                None => DownsampleStep::Ignore,
            }
        })
    }
}

// Methods for [Embedded]TensorOperator

macro_rules! impl_embedded_tensor_operator_with_delegate {
    ($($fn_name:ident ( $( $arg:ident : $type:ty),*),)*) => {
        #[gen_stub_pymethods]
        #[pymethods]
        impl EmbeddedTensorOperator {
            #[getter]
            pub fn dtype(&self) -> DType {
                self.inner.dtype()
            }
            #[getter]
            pub fn metadata(&self) -> PyResult<PyTensorMetaData> {
                self.inner.metadata()
            }
            fn single_level_lod(&self) -> PyResult<LODTensorOperator> {
                Ok(LODTensorOperator {
                    levels: vec![self.clone()],
                })
            }
            fn create_lod(&self, steps: Vec<DownsampleStep>) -> PyResult<LODTensorOperator> {
                use palace_core::operators::resample::DownsampleStep as CDownsampleStep;
                let steps = Vector::from_fn_and_len(steps.len(), |i| match steps[i] {
                    DownsampleStep::Ignore => CDownsampleStep::Ignore,
                    DownsampleStep::Fixed(f) => CDownsampleStep::Fixed(f),
                    DownsampleStep::Synchronized(f) => CDownsampleStep::Synchronized(f),
                });
                Ok(palace_core::operators::resample::create_lod(
                    self.clone().into_core(),
                    steps,
                )
                .try_into()?)
            }
            fn cache(&self) -> EmbeddedTensorOperator {
                EmbeddedTensorOperator {
                    inner: self.inner.cache(),
                    embedding_data: self.embedding_data.clone(),
                }
            }
            fn distribute_on_gpus(&self, devices: Vec<DeviceId>) -> Self {
                EmbeddedTensorOperator {
                    inner: self.inner.distribute_on_gpus(devices),
                    embedding_data: self.embedding_data.clone(),
                }
            }
            pub fn nd(&self) -> PyResult<usize> {
                self.inner.nd()
            }

            #[pyo3(signature = (num_samples=None))]
            pub fn max_value(&self, num_samples: Option<usize>) -> PyResult<ScalarOperator> {
                self.inner.clone().max_value(num_samples)
            }

            #[pyo3(signature = (num_samples=None))]
            pub fn min_value(&self, num_samples: Option<usize>) -> PyResult<ScalarOperator> {
                self.inner.clone().min_value(num_samples)
            }

            #[pyo3(signature = (num_samples=None))]
            pub fn mean_value(&self, num_samples: Option<usize>) -> PyResult<ScalarOperator> {
                self.inner.clone().mean_value(num_samples)
            }

            fn fold_into_dtype(&self) -> PyResult<Self> {
                Ok(self.clone().into_core().fold_into_dtype()
                    .map_err(crate::map_err)?.into())
            }
            fn unfold_dtype(&self, new_spacing: f32) -> PyResult<Self> {
                Ok(self.clone().into_core().unfold_dtype(new_spacing)
                    .map_err(crate::map_err)?.into())
            }

            //TODO: Quite annoying that we cannot annotate the default argument with the macro
            //expansion below. Don't really know how to do that... We could maybe add the
            //#[pyo3(...)] annotation into the list with $fn_name etc., but that seems like a lot
            //of trouble...
            #[pyo3(signature = (kernels, border_handling="repeat"))]
            fn separable_convolution(
                &self,
                kernels: Vec<MaybeConstTensorOperator>,
                border_handling: &str,
            ) -> PyResult<Self> {
                self.map_inner(|v| v.separable_convolution(kernels, border_handling))
            }

            fn __getitem__(&self, py: Python, slice_args: Vec<SliceArg>) -> PyResult<Self> {
                let slice_args = convert_slice_args(py, slice_args, &self.inner.metadata()?.dimensions)?;

                let t = self.clone().into_core();
                Ok(palace_core::operators::slice::slice_and_squash_embedded(t, slice_args).map_err(crate::map_err)?.into())
            }

            $(
            fn $fn_name(&self, $($arg : $type,)* ) -> PyResult<Self> {
                self.map_inner(|i| i.$fn_name($($arg,)*))
            }
            )*
        }
    };
}

#[gen_stub_pyclass_enum]
#[derive(FromPyObject)]
pub enum MaybeScalarDType {
    Scalar(ScalarType),
    DType(DType),
}

#[gen_stub_pyclass_enum]
#[derive(FromPyObject)]
pub enum SliceArg {
    Scalar(u32),
    Range(Py<PySlice>),
}

fn convert_slice_args(
    py: Python,
    slice_args: Vec<SliceArg>,
    tensor_dim: &Vec<u32>,
) -> PyResult<Vector<DDyn, palace_core::operators::slice::Range>> {
    if tensor_dim.len() != slice_args.len() {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "Slice args must be {}-dimensional to fit tensor",
            tensor_dim.len(),
        )));
    }

    let slice_args = slice_args
        .into_iter()
        .zip(tensor_dim.iter())
        .map(|(slice_arg, dim)| {
            Ok(match slice_arg {
                SliceArg::Scalar(s) => palace_core::operators::slice::Range::Scalar(s),
                SliceArg::Range(slice_arg) => {
                    let slice_arg = slice_arg.into_bound(py);
                    let slice_arg = slice_arg.indices(*dim as _)?;
                    if slice_arg.step != 1 {
                        return Err(crate::map_err("Step must be 1".into()));
                    }
                    palace_core::operators::slice::Range::FromTo(
                        slice_arg.start.try_into().unwrap(),
                        slice_arg.stop.try_into().unwrap(),
                    )
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Vector::<DDyn, _>::new(slice_args))
}

#[gen_stub_pymethods]
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
    pub fn metadata(&self) -> PyResult<PyTensorMetaData> {
        Ok(match &self.inner {
            MaybeJitTensorOperator::Jit(i) => i
                .metadata()
                .ok_or_else(|| {
                    crate::map_err(format!("Jit operator does not have metadata").into())
                })?
                .into(),
            MaybeJitTensorOperator::Tensor(i) => i.metadata.clone().into(),
        })
    }
    fn embedded(&self, embedding_data: PyTensorEmbeddingData) -> PyResult<EmbeddedTensorOperator> {
        if self.nd()? != embedding_data.nd() {
            return Err(crate::map_err(
                format!(
                    "Embedding data dimension mismatch: Must be {}, but is {}",
                    self.nd()?,
                    embedding_data.nd()
                )
                .into(),
            ));
        }
        Ok(EmbeddedTensorOperator {
            inner: self.clone(),
            embedding_data,
        })
    }

    fn cache(&self) -> TensorOperator {
        let t = match &self.inner {
            MaybeJitTensorOperator::Jit(j) => j.clone().compile().unwrap(),
            MaybeJitTensorOperator::Tensor(t) => t.clone(),
        }
        .cache();
        TensorOperator {
            inner: MaybeJitTensorOperator::Tensor(t),
        }
    }

    fn fold_into_dtype(&self) -> PyResult<Self> {
        Ok(self
            .clone()
            .into_core()
            .fold_into_dtype()
            .map_err(crate::map_err)?
            .into())
    }
    fn unfold_dtype(&self) -> PyResult<Self> {
        Ok(self
            .clone()
            .into_core()
            .unfold_dtype()
            .map_err(crate::map_err)?
            .into())
    }
    fn rechunk(&self, size: Vec<ChunkSize>) -> PyResult<Self> {
        if self.nd()? != size.len() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Chunk size must be {}-dimensional to fit tensor",
                self.nd()?
            )));
        }

        let size = Vector::<DDyn, _>::new(size).map(|s: ChunkSize| s.0);
        self.clone().map_core(
            |vol: palace_core::operators::tensor::TensorOperator<DDyn, DType>| {
                Ok(palace_core::operators::rechunk::rechunk(vol, size).into_dyn())
            },
        )
    }
    fn distribute_on_gpus(&self, devices: Vec<DeviceId>) -> Self {
        let t = match &self.inner {
            MaybeJitTensorOperator::Jit(j) => j.clone().compile().unwrap(),
            MaybeJitTensorOperator::Tensor(t) => t.clone(),
        }
        .distribute_on_gpus(devices.into_iter().map(|v| v.0).collect());
        TensorOperator {
            inner: MaybeJitTensorOperator::Tensor(t),
        }
    }

    #[pyo3(signature = (kernels, border_handling="repeat"))]
    fn separable_convolution(
        &self,
        kernels: Vec<MaybeConstTensorOperator>,
        border_handling: &str,
    ) -> PyResult<Self> {
        let border_handling = match border_handling {
            "repeat" => BorderHandling::Repeat,
            "pad0" => BorderHandling::Pad0,
            o => {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "Invalid border handling strategy {}",
                    o
                )))
            }
        };

        if self.nd()? != kernels.len() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Expected {} kernels for tensor, but got {}",
                self.nd()?,
                kernels.len()
            )));
        }

        let kernels = kernels
            .into_iter()
            .map(|k| {
                let ret: CTensorOperator<D1, DType> = try_into_static_err(k.try_into_core()?)?;
                Ok(ret)
            })
            .collect::<Result<Vec<_>, PyErr>>()?;

        let kernel_refs =
            Vector::<DDyn, &CTensorOperator<D1, DType>>::try_from_fn_and_len(kernels.len(), |i| {
                &kernels[i]
            })
            .unwrap();

        self.clone().map_core(|vol: CTensorOperator<DDyn, DType>| {
            Ok(palace_core::operators::conv::separable_convolution(
                vol,
                kernel_refs,
                border_handling,
            )
            .into_dyn()
            .into())
        })
    }

    #[pyo3(signature = (output_size, transform, border_handling="repeat"))]
    fn resample_transform(
        &self,
        output_size: PyTensorMetaData,
        transform: Matrix<DDyn, f32>,
        border_handling: &str,
    ) -> PyResult<Self> {
        if self.nd()? + 1 != transform.dim().n() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Expected {} transform for tensor, but got {}",
                self.nd()? + 1,
                transform.dim().n()
            )));
        }

        if self.nd()? != output_size.dimensions.len() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Expected output size of dim {}, but got {}",
                self.nd()?,
                output_size.dimensions.len()
            )));
        }
        let border_handling = match border_handling {
            "repeat" => BorderHandling::Repeat,
            "pad0" => BorderHandling::Pad0,
            o => {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "Invalid border handling strategy {}",
                    o
                )))
            }
        };

        self.clone().map_core(|vol: CTensorOperator<DDyn, DType>| {
            Ok(palace_core::operators::resample::resample_transform(
                vol,
                output_size.into(),
                transform,
                border_handling,
            )
            .into_dyn()
            .into())
        })
    }

    fn __getitem__(&self, py: Python, slice_args: Vec<SliceArg>) -> PyResult<Self> {
        let tensor = self;
        let dimensions = &tensor.metadata()?.dimensions;

        let slice_args = convert_slice_args(py, slice_args, dimensions)?;

        tensor.clone().map_core(
            |vol: palace_core::operators::tensor::TensorOperator<DDyn, DType>| {
                Ok(
                    palace_core::operators::slice::slice_and_squash(vol, slice_args)
                        .map_err(crate::map_err)?
                        .into_dyn(),
                )
            },
        )
    }

    fn abs(&self) -> PyResult<Self> {
        jit_unary(UnaryOp::Abs, self)
    }
    fn __neg__(&self) -> PyResult<Self> {
        jit_unary(UnaryOp::Neg, self)
    }
    fn log(&self) -> PyResult<Self> {
        jit_unary(UnaryOp::Log, self)
    }
    fn exp(&self) -> PyResult<Self> {
        jit_unary(UnaryOp::Exp, self)
    }
    fn cast(&self, to: MaybeScalarDType) -> PyResult<Self> {
        let to = match to {
            MaybeScalarDType::Scalar(s) => DType::scalar(s),
            MaybeScalarDType::DType(d) => d,
        };
        jit_unary(UnaryOp::Cast(to), self)
    }
    fn index(&self, index: u32) -> PyResult<Self> {
        jit_unary(UnaryOp::Index(index), self)
    }
    fn index_range(&self, from: u32, to: u32) -> PyResult<Self> {
        jit_unary(UnaryOp::IndexRange(from, to), self)
    }
    fn splat(&self, size: u32) -> PyResult<Self> {
        jit_unary(UnaryOp::Splat(size), self)
    }
    fn hsum(&self) -> PyResult<Self> {
        jit_unary(UnaryOp::Fold(jit::FoldOp::Sum), self)
    }
    fn hmul(&self) -> PyResult<Self> {
        jit_unary(UnaryOp::Fold(jit::FoldOp::Mul), self)
    }
    fn hmin(&self) -> PyResult<Self> {
        jit_unary(UnaryOp::Fold(jit::FoldOp::Min), self)
    }
    fn hmax(&self) -> PyResult<Self> {
        jit_unary(UnaryOp::Fold(jit::FoldOp::Max), self)
    }

    fn __add__(&self, a: JitArgument) -> PyResult<Self> {
        jit_binary(BinOp::Add, self, a)
    }
    fn __sub__(&self, a: JitArgument) -> PyResult<Self> {
        jit_binary(BinOp::Sub, self, a)
    }
    fn __mul__(&self, a: JitArgument) -> PyResult<Self> {
        jit_binary(BinOp::Mul, self, a)
    }
    fn __truediv__(&self, a: JitArgument) -> PyResult<Self> {
        jit_binary(BinOp::Div, self, a)
    }
    fn min(&self, a: JitArgument) -> PyResult<Self> {
        jit_binary(BinOp::Min, self, a)
    }
    fn max(&self, a: JitArgument) -> PyResult<Self> {
        jit_binary(BinOp::Max, self, a)
    }
    fn concat(&self, a: JitArgument) -> PyResult<Self> {
        jit_binary(BinOp::Concat, self, a)
    }
    fn __lt__(&self, a: JitArgument) -> PyResult<Self> {
        jit_binary(BinOp::LessThan, self, a)
    }
    fn __le__(&self, a: JitArgument) -> PyResult<Self> {
        jit_binary(BinOp::LessThanEquals, self, a)
    }
    fn __gt__(&self, a: JitArgument) -> PyResult<Self> {
        jit_binary(BinOp::GreaterThan, self, a)
    }
    fn __ge__(&self, a: JitArgument) -> PyResult<Self> {
        jit_binary(BinOp::GreaterThanEquals, self, a)
    }
    fn __eq__(&self, a: JitArgument) -> PyResult<Self> {
        jit_binary(BinOp::Equals, self, a)
    }
    fn __ne__(&self, a: JitArgument) -> PyResult<Self> {
        jit_binary(BinOp::NotEquals, self, a)
    }

    fn select(&self, then_val: JitArgument, else_val: JitArgument) -> PyResult<Self> {
        jit_ternary(TernaryOp::IfThenElse, self, then_val, else_val)
    }

    #[pyo3(signature = (num_samples=None))]
    pub fn mean_value(&self, num_samples: Option<usize>) -> PyResult<ScalarOperator> {
        let vol = self.clone().into_core();
        let vol = vol.try_into()?;
        Ok(palace_core::operators::aggregation::mean(vol, num_samples.into()).into())
    }

    #[pyo3(signature = (num_samples=None))]
    pub fn min_value(&self, num_samples: Option<usize>) -> PyResult<ScalarOperator> {
        let vol = self.clone().into_core();
        let vol = vol.try_into()?;
        Ok(palace_core::operators::aggregation::min(vol, num_samples.into()).into())
    }

    #[pyo3(signature = (num_samples=None))]
    pub fn max_value(&self, num_samples: Option<usize>) -> PyResult<ScalarOperator> {
        let vol = self.clone().into_core();
        let vol = vol.try_into()?;
        Ok(palace_core::operators::aggregation::max(vol, num_samples.into()).into())
    }
}

impl_embedded_tensor_operator_with_delegate!(
    rechunk(size: Vec<ChunkSize>),

    __neg__(),
    log(),
    exp(),
    abs(),
    cast(to: MaybeScalarDType),
    index(index: u32),
    index_range(from: u32, to: u32),
    splat(size: u32),

    __add__(a: JitArgument),
    __sub__(a: JitArgument),
    __mul__(a: JitArgument),
    __truediv__(a: JitArgument),
    min(a: JitArgument),
    max(a: JitArgument),
    concat(a: JitArgument),
    __lt__(a: JitArgument),
    __le__(a: JitArgument),
    __gt__(a: JitArgument),
    __ge__(a: JitArgument),
    __eq__(a: JitArgument),
    __ne__(a: JitArgument),

    select(then_val: JitArgument, else_val: JitArgument),
);
