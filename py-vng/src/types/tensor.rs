use derive_more::{From, Into};

use crate::conversion;
use pyo3::{exceptions::PyException, prelude::*};
use vng_core::{
    array::{ArrayMetaData, VolumeEmbeddingData, VolumeMetaData},
    data::{ChunkCoordinate, Vector},
    operator::Operator,
};

use vng_core::array::ImageMetaData;
use vng_core::operators::array::ArrayOperator as CArrayOperator;
use vng_core::operators::volume::EmbeddedVolumeOperator as CEmbeddedVolumeOperator;
use vng_core::operators::volume::LODVolumeOperator as CLODVolumeOperator;
use vng_core::operators::volume::VolumeOperator as CVolumeOperator;

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
#[derive(Clone, From, Into)]
pub struct ScalarOperatorF32(pub Operator<(), f32>);
impl conversion::FromPyValue<f32> for ScalarOperatorF32 {
    fn from_py(v: f32) -> PyResult<Self> {
        Ok(ScalarOperatorF32(vng_core::operators::scalar::constant(v)))
    }
}
impl<'source> conversion::FromPyValues<'source> for ScalarOperatorF32 {
    type Converter = conversion::ToOperatorFrom<Self, (f32,)>;
}

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
pub struct ScalarOperatorVec2F(pub Operator<(), Vector<2, f32>>);
impl conversion::FromPyValue<[f32; 2]> for ScalarOperatorVec2F {
    fn from_py(v: [f32; 2]) -> PyResult<Self> {
        Ok(ScalarOperatorVec2F(vng_core::operators::scalar::constant(
            v.into(),
        )))
    }
}
impl<'source> conversion::FromPyValues<'source> for ScalarOperatorVec2F {
    type Converter = conversion::ToOperatorFrom<Self, ([f32; 2],)>;
}

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
pub struct ScalarOperatorU32(pub Operator<(), u32>);
impl conversion::FromPyValue<u32> for ScalarOperatorU32 {
    fn from_py(v: u32) -> PyResult<Self> {
        Ok(ScalarOperatorU32(vng_core::operators::scalar::constant(v)))
    }
}
impl<'source> conversion::FromPyValues<'source> for ScalarOperatorU32 {
    type Converter = conversion::ToOperatorFrom<Self, (u32,)>;
}

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
pub struct ArrayMetadataOperator(pub Operator<(), ArrayMetaData>);

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
pub struct VolumeMetadataOperator(pub Operator<(), VolumeMetaData>);

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
pub struct VolumeEmbeddingDataOperator(pub Operator<(), VolumeEmbeddingData>);

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
pub struct ImageMetadataOperator(pub Operator<(), ImageMetaData>);

impl<'a> conversion::FromPyValue<ImageMetaData> for ImageMetadataOperator {
    fn from_py(v: ImageMetaData) -> PyResult<Self> {
        Ok(vng_core::operators::scalar::constant(v).into())
    }
}
impl<'source> conversion::FromPyValues<'source> for ImageMetadataOperator {
    type Converter = conversion::ToOperatorFrom<Self, (ImageMetaData,)>;
}

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
pub struct ArrayValueOperator(pub Operator<Vector<1, ChunkCoordinate>, f32>);

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
pub struct VolumeValueOperator(pub Operator<Vector<3, ChunkCoordinate>, f32>);

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct VolumeOperator {
    #[pyo3(get, set)]
    pub metadata: VolumeMetadataOperator,
    #[pyo3(get, set)]
    pub chunks: VolumeValueOperator,
}

impl Into<CVolumeOperator<f32>> for VolumeOperator {
    fn into(self) -> CVolumeOperator<f32> {
        CVolumeOperator {
            metadata: self.metadata.into(),
            chunks: self.chunks.into(),
        }
    }
}

impl From<CVolumeOperator<f32>> for VolumeOperator {
    fn from(value: CVolumeOperator<f32>) -> Self {
        Self {
            metadata: value.metadata.into(),
            chunks: value.chunks.into(),
        }
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct EmbeddedVolumeOperator {
    #[pyo3(get, set)]
    pub inner: VolumeOperator,
    #[pyo3(get, set)]
    pub embedding_data: VolumeEmbeddingDataOperator,
}

impl Into<CEmbeddedVolumeOperator<f32>> for EmbeddedVolumeOperator {
    fn into(self) -> CEmbeddedVolumeOperator<f32> {
        CEmbeddedVolumeOperator {
            inner: self.inner.into(),
            embedding_data: self.embedding_data.into(),
        }
    }
}

impl From<CEmbeddedVolumeOperator<f32>> for EmbeddedVolumeOperator {
    fn from(value: CEmbeddedVolumeOperator<f32>) -> Self {
        Self {
            inner: value.inner.into(),
            embedding_data: value.embedding_data.into(),
        }
    }
}

#[pymethods]
impl EmbeddedVolumeOperator {
    fn create_lod(&self, step_factor: f32, num_levels: usize) -> LODVolumeOperator {
        vng_core::operators::resample::create_lod(self.clone().into(), step_factor, num_levels)
            .into()
    }
}

#[derive(FromPyObject)]
pub enum MaybeEmbeddedVolumeOperator {
    Not(VolumeOperator),
    Embedded(EmbeddedVolumeOperator),
}

impl MaybeEmbeddedVolumeOperator {
    pub fn map_inner(
        self,
        py: Python,
        f: impl FnOnce(CVolumeOperator<f32>) -> CVolumeOperator<f32>,
    ) -> PyObject {
        match self {
            MaybeEmbeddedVolumeOperator::Not(v) => VolumeOperator::from(f(v.into())).into_py(py),
            MaybeEmbeddedVolumeOperator::Embedded(v) => EmbeddedVolumeOperator::from(
                Into::<CEmbeddedVolumeOperator<f32>>::into(v).map_inner(f),
            )
            .into_py(py),
        }
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct LODVolumeOperator {
    levels: Vec<EmbeddedVolumeOperator>,
}

#[pymethods]
impl LODVolumeOperator {
    pub fn fine_metadata(&self) -> VolumeMetadataOperator {
        self.levels[0].inner.metadata.clone()
    }
    pub fn fine_embedding_data(&self) -> VolumeEmbeddingDataOperator {
        self.levels[0].embedding_data.clone()
    }
}

impl Into<CLODVolumeOperator<f32>> for LODVolumeOperator {
    fn into(self) -> CLODVolumeOperator<f32> {
        CLODVolumeOperator {
            levels: self
                .levels
                .into_iter()
                .map(|v| v.into())
                .collect::<Vec<_>>(),
        }
    }
}

impl From<CLODVolumeOperator<f32>> for LODVolumeOperator {
    fn from(value: CLODVolumeOperator<f32>) -> Self {
        Self {
            levels: value
                .levels
                .into_iter()
                .map(|v| v.into())
                .collect::<Vec<_>>(),
        }
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct ArrayOperator {
    pub metadata: ArrayMetadataOperator,
    pub chunks: ArrayValueOperator,
}

impl Into<CArrayOperator<f32>> for ArrayOperator {
    fn into(self) -> CArrayOperator<f32> {
        CArrayOperator {
            metadata: self.metadata.into(),
            chunks: self.chunks.into(),
        }
    }
}

impl From<CArrayOperator<f32>> for ArrayOperator {
    fn from(value: CArrayOperator<f32>) -> Self {
        Self {
            metadata: value.metadata.into(),
            chunks: value.chunks.into(),
        }
    }
}

impl<'a> conversion::FromPyValue<numpy::borrow::PyReadonlyArray1<'a, f32>> for ArrayOperator {
    fn from_py(v: numpy::borrow::PyReadonlyArray1<'a, f32>) -> PyResult<Self> {
        Ok(vng_core::operators::array::from_rc(v.as_slice()?.into()).into())
    }
}
impl<'source> conversion::FromPyValues<'source> for ArrayOperator {
    type Converter =
        conversion::ToOperatorFrom<Self, (numpy::borrow::PyReadonlyArray1<'source, f32>,)>;
}
