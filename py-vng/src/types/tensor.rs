use derive_more::{From, Into};

use crate::conversion;
use pyo3::prelude::*;
use vng_core::{
    array::{ArrayMetaData, VolumeMetaData},
    data::{ChunkCoordinate, Vector},
    operator::Operator,
};

use vng_core::array::ImageMetaData as CImageMetaData;
use vng_core::operators::array::ArrayOperator as CArrayOperator;
use vng_core::operators::volume::VolumeOperator as CVolumeOperator;

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
pub struct ScalarOperatorF32(pub Operator<(), f32>);
impl conversion::FromPyValue<f32> for ScalarOperatorF32 {
    fn from_py(v: f32) -> PyResult<Self> {
        Ok(ScalarOperatorF32(
            vng_core::operators::scalar::constant_pod(v),
        ))
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
        Ok(ScalarOperatorVec2F(
            vng_core::operators::scalar::constant_pod(v.into()),
        ))
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
        Ok(ScalarOperatorU32(
            vng_core::operators::scalar::constant_pod(v),
        ))
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
pub struct ImageMetadataOperator(pub Operator<(), CImageMetaData>);

impl<'a> conversion::FromPyValue<ImageMetadata> for ImageMetadataOperator {
    fn from_py(v: ImageMetadata) -> PyResult<Self> {
        Ok(vng_core::operators::scalar::constant_hash(v.0).into())
    }
}
impl<'source> conversion::FromPyValues<'source> for ImageMetadataOperator {
    type Converter = conversion::ToOperatorFrom<Self, (ImageMetadata,)>;
}

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
pub struct ImageMetadata(pub CImageMetaData);

#[pymethods]
impl ImageMetadata {
    #[new]
    fn new(dimensions: [u32; 2], chunk_size: [u32; 2]) -> Self {
        Self(CImageMetaData {
            dimensions: dimensions.into(),
            chunk_size: chunk_size.into(),
        })
    }

    #[getter]
    fn get_dimensions(&self) -> [u32; 2] {
        self.0.dimensions.raw().into()
    }
    #[setter]
    fn set_dimensions(&mut self, dim: [u32; 2]) {
        self.0.dimensions = Vector::from(dim);
    }

    #[getter]
    fn get_chunk_size(&self) -> [u32; 2] {
        self.0.chunk_size.raw().into()
    }
    #[setter]
    fn set_chunk_size(&mut self, size: [u32; 2]) {
        self.0.chunk_size = Vector::from(size);
    }
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

impl Into<CVolumeOperator> for VolumeOperator {
    fn into(self) -> CVolumeOperator {
        CVolumeOperator {
            metadata: self.metadata.into(),
            chunks: self.chunks.into(),
        }
    }
}

impl From<CVolumeOperator> for VolumeOperator {
    fn from(value: CVolumeOperator) -> Self {
        Self {
            metadata: value.metadata.into(),
            chunks: value.chunks.into(),
        }
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct ArrayOperator {
    pub metadata: ArrayMetadataOperator,
    pub chunks: ArrayValueOperator,
}

impl Into<CArrayOperator> for ArrayOperator {
    fn into(self) -> CArrayOperator {
        CArrayOperator {
            metadata: self.metadata.into(),
            chunks: self.chunks.into(),
        }
    }
}

impl From<CArrayOperator> for ArrayOperator {
    fn from(value: CArrayOperator) -> Self {
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
