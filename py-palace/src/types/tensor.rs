use id::Identify;
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use palace_core::array::{PyTensorEmbeddingData, PyTensorMetaData};
use palace_core::dtypes::{DType, ElementType, ScalarType, StaticElementType};
use palace_core::jit::{self, BinOp, JitTensorOperator, TernaryOp, UnaryOp};
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
        Ok(f(&self.inner)?.embedded(self.embedding_data.clone()))
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
    fn embedded(&self, embedding_data: PyTensorEmbeddingData) -> EmbeddedTensorOperator {
        match self {
            MaybeEmbeddedTensorOperator::Not { i } => i.embedded(embedding_data),
            MaybeEmbeddedTensorOperator::Embedded { e } => e.clone(),
        }
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
}

// Methods for [Embedded]TensorOperator

macro_rules! impl_embedded_tensor_operator_with_delegate {
    ($($fn_name:ident ( $( $arg:ident : $type:ty),*),)*) => {
        #[gen_stub_pymethods]
        #[pymethods]
        impl EmbeddedTensorOperator {
            fn single_level_lod(&self) -> PyResult<LODTensorOperator> {
                Ok(LODTensorOperator {
                    levels: vec![self.clone()],
                })
            }
            fn create_lod(&self, step_factor: f32) -> PyResult<LODTensorOperator> {
                Ok(palace_core::operators::resample::create_lod(
                    self.clone().into_core(),
                    step_factor,
                )
                .try_into()?)
            }
            fn cache(&self) -> EmbeddedTensorOperator {
                EmbeddedTensorOperator {
                    inner: self.inner.cache(),
                    embedding_data: self.embedding_data.clone(),
                }
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
    fn embedded(&self, embedding_data: PyTensorEmbeddingData) -> EmbeddedTensorOperator {
        EmbeddedTensorOperator {
            inner: self.clone(),
            embedding_data,
        }
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

    fn __getitem__(&self, slice_args: Vec<Bound<PySlice>>) -> PyResult<Self> {
        let tensor = self;
        if tensor.nd()? != slice_args.len() {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "Slice args must be {}-dimensional to fit tensor",
                tensor.nd()?
            )));
        }
        let dimensions = &tensor.metadata()?.dimensions;

        let slice_args = slice_args
            .into_iter()
            .zip(dimensions.iter())
            .map(|(slice_arg, dim)| {
                let slice_arg = slice_arg.indices(*dim as _)?;
                if slice_arg.step != 1 {
                    return Err(crate::map_err("Step must be 1".into()));
                }

                Ok(palace_core::operators::slice::Range::FromTo(
                    slice_arg.start.try_into().unwrap(),
                    slice_arg.stop.try_into().unwrap(),
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let slice_args = Vector::<DDyn, _>::new(slice_args);

        tensor.clone().map_core(
            |vol: palace_core::operators::tensor::TensorOperator<DDyn, DType>| {
                Ok(palace_core::operators::slice::slice(vol, slice_args).into_dyn())
            },
        )
    }

    fn abs(&self) -> PyResult<Self> {
        jit_unary(UnaryOp::Abs, self)
    }
    fn __neg__(&self) -> PyResult<Self> {
        jit_unary(UnaryOp::Neg, self)
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
    fn splat(&self, size: u32) -> PyResult<Self> {
        jit_unary(UnaryOp::Splat(size), self)
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
}

impl_embedded_tensor_operator_with_delegate!(
    __getitem__(slice_args: Vec<Bound<PySlice>>),

    unfold_dtype(),
    fold_into_dtype(),
    __neg__(),
    abs(),
    cast(to: MaybeScalarDType),
    index(index: u32),
    splat(size: u32),

    __add__(a: JitArgument),
    __sub__(a: JitArgument),
    __mul__(a: JitArgument),
    __truediv__(a: JitArgument),
    min(a: JitArgument),
    max(a: JitArgument),
    __lt__(a: JitArgument),
    __le__(a: JitArgument),
    __gt__(a: JitArgument),
    __ge__(a: JitArgument),
    __eq__(a: JitArgument),
    __ne__(a: JitArgument),

    select(then_val: JitArgument, else_val: JitArgument),
);
