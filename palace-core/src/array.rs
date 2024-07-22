use crate::data::{ChunkCoordinate, GlobalCoordinate, LocalCoordinate, Matrix, Vector};
use crate::dim::*;

pub struct ChunkInfo<D: DynDimension> {
    pub mem_dimensions: Vector<D, LocalCoordinate>,
    pub logical_dimensions: Vector<D, LocalCoordinate>,
    pub begin: Vector<D, GlobalCoordinate>,
}
impl<D: DynDimension> ChunkInfo<D> {
    pub fn is_contiguous(&self) -> bool {
        for i in 1..self.mem_dimensions.len() {
            if self.mem_dimensions[i] != self.logical_dimensions[i] {
                return false;
            }
        }
        true
    }
    pub fn is_full(&self) -> bool {
        self.mem_dimensions == self.logical_dimensions
    }
    pub fn mem_elements(&self) -> usize {
        self.mem_dimensions.hmul()
    }

    pub fn in_chunk(&self, pos: &Vector<D, GlobalCoordinate>) -> Vector<D, LocalCoordinate> {
        (pos - &self.begin).map(LocalCoordinate::interpret_as)
    }

    pub fn begin(&self) -> &Vector<D, GlobalCoordinate> {
        &self.begin
    }
    pub fn end(&self) -> Vector<D, GlobalCoordinate> {
        &self.begin + &self.logical_dimensions
    }
}

#[repr(C)]
#[derive(Hash, Debug, id::Identify)]
pub struct TensorMetaData<D: DynDimension> {
    pub dimensions: Vector<D, GlobalCoordinate>,
    pub chunk_size: Vector<D, LocalCoordinate>,
}

//TODO: Why the hell do we need to do this manually?
impl<D: DynDimension> Clone for TensorMetaData<D> {
    fn clone(&self) -> Self {
        Self {
            dimensions: self.dimensions.clone(),
            chunk_size: self.chunk_size.clone(),
        }
    }
}

impl<D: Dimension> Copy for TensorMetaData<D> {}

impl<D: DynDimension> PartialEq for TensorMetaData<D> {
    fn eq(&self, other: &Self) -> bool {
        self.dimensions == other.dimensions && self.chunk_size == other.chunk_size
    }
}
impl<D: DynDimension> Eq for TensorMetaData<D> {}

impl<D: DynDimension> TensorMetaData<D> {
    pub fn try_into_static<DF: Dimension>(self) -> Option<TensorMetaData<DF>> {
        Some(TensorMetaData {
            dimensions: self.dimensions.try_into_static()?,
            chunk_size: self.chunk_size.try_into_static()?,
        })
    }

    pub fn into_dyn(self) -> TensorMetaData<DDyn> {
        TensorMetaData {
            dimensions: self.dimensions.into_dyn(),
            chunk_size: self.chunk_size.into_dyn(),
        }
    }
}

// We have to do this manually since bytemuck cannot verify this in general due to the const
// parameter N. It is fine though (as long as we don't change anything on TensorMetaData).
//unsafe impl<D: Dimension> bytemuck::Pod for TensorMetaData<D> {}

impl<D: LargerDim> TensorMetaData<D> {
    pub fn norm_to_voxel(&self) -> Matrix<D::Larger, f32> {
        Matrix::from_translation(Vector::fill(-0.5))
            * Matrix::from_scale(self.dimensions.raw().f32()).to_homogeneous()
    }
    pub fn voxel_to_norm(&self) -> Matrix<D::Larger, f32> {
        Matrix::from_scale(self.dimensions.raw().f32().map(|v| 1.0 / v)).to_homogeneous()
            * Matrix::from_translation(Vector::fill(0.5))
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, id::Identify)]
pub struct TensorEmbeddingData<D: DynDimension> {
    pub spacing: Vector<D, f32>,
    // NOTE: need to change identify impl if we want to add members
}

impl<D: Dimension> Copy for TensorEmbeddingData<D> {}

impl<D: DynDimension> TensorEmbeddingData<D> {
    pub fn try_into_static<DF: Dimension>(self) -> Option<TensorEmbeddingData<DF>> {
        Some(TensorEmbeddingData {
            spacing: self.spacing.try_into_static()?,
        })
    }

    pub fn into_dyn(self) -> TensorEmbeddingData<DDyn> {
        TensorEmbeddingData {
            spacing: self.spacing.into_dyn(),
        }
    }
}

impl<D: LargerDim> TensorEmbeddingData<D> {
    pub fn voxel_to_physical(&self) -> Matrix<D::Larger, f32> {
        Matrix::from_scale(self.spacing).to_homogeneous()
    }
    pub fn physical_to_voxel(&self) -> Matrix<D::Larger, f32> {
        Matrix::from_scale(self.spacing.map(|v| 1.0 / v)).to_homogeneous()
    }
}

pub fn norm_to_physical<D: LargerDim>(
    md: &TensorMetaData<D>,
    emd: &TensorEmbeddingData<D>,
) -> Matrix<D::Larger, f32> {
    emd.voxel_to_physical() * md.norm_to_voxel()
}

pub fn physical_to_voxel<D: LargerDim>(
    md: &TensorMetaData<D>,
    emd: &TensorEmbeddingData<D>,
) -> Matrix<D::Larger, f32> {
    md.voxel_to_norm() * emd.physical_to_voxel()
}

//TODO: Revisit this. This is definitely fine as long as we don't add other members
//unsafe impl<D: Dimension + bytemuck::Zeroable> bytemuck::Pod for TensorEmbeddingData<D> {}

#[cfg(feature = "python")]
pub use py::TensorEmbeddingData as PyTensorEmbeddingData;
#[cfg(feature = "python")]
pub use py::TensorMetaData as PyTensorMetaData;

#[cfg(feature = "python")]
mod py {
    use numpy::PyArray1;
    use pyo3::{exceptions::PyException, prelude::*};

    use super::*;

    #[pyclass(unsendable)]
    #[derive(Clone, Debug)]
    pub struct TensorMetaData {
        pub dimensions: Vec<u32>,
        pub chunk_size: Vec<u32>,
    }

    #[pymethods]
    impl TensorMetaData {
        #[new]
        fn new(dimensions: Vec<u32>, chunk_size: Vec<u32>) -> PyResult<Self> {
            if dimensions.len() != chunk_size.len() {
                return Err(PyErr::new::<PyException, _>(format!(
                    "dimensions ({:?}) and chunk_size ({:?}) must be of the same length",
                    dimensions, chunk_size,
                )));
            }
            Ok(Self {
                dimensions,
                chunk_size,
            })
        }

        #[getter]
        fn dimensions<'a>(&self, py: Python<'a>) -> &'a PyArray1<u32> {
            PyArray1::from_vec(py, self.dimensions.clone())
        }
        #[getter]
        fn chunk_size<'a>(&self, py: Python<'a>) -> &'a PyArray1<u32> {
            PyArray1::from_vec(py, self.chunk_size.clone())
        }
    }

    impl From<super::TensorMetaData<DDyn>> for TensorMetaData {
        fn from(value: super::TensorMetaData<DDyn>) -> Self {
            assert_eq!(value.dimensions.len(), value.chunk_size.len());
            Self {
                dimensions: value.dimensions.into_iter().map(|v| v.raw).collect(),
                chunk_size: value.chunk_size.into_iter().map(|v| v.raw).collect(),
            }
        }
    }
    impl From<TensorMetaData> for super::TensorMetaData<DDyn> {
        fn from(value: TensorMetaData) -> Self {
            assert_eq!(value.dimensions.len(), value.chunk_size.len());
            Self {
                dimensions: Vector::new(value.dimensions.into_iter().map(|v| v.into()).collect()),
                chunk_size: Vector::new(value.chunk_size.into_iter().map(|v| v.into()).collect()),
            }
        }
    }
    impl<D: Dimension> From<super::TensorMetaData<D>> for TensorMetaData {
        fn from(t: super::TensorMetaData<D>) -> Self {
            Self {
                dimensions: t.dimensions.raw().into_iter().collect(),
                chunk_size: t.chunk_size.raw().into_iter().collect(),
            }
        }
    }
    impl TensorMetaData {
        pub fn try_into_dim<D: Dimension>(self) -> Result<super::TensorMetaData<D>, PyErr> {
            let md: super::TensorMetaData<DDyn> = self.into();
            let l = md.dimensions.len();
            md.try_into_static().ok_or_else(|| {
                PyErr::new::<PyException, _>(format!(
                    "Expected TensorMetaData<{}>, but got TensorMetaData<{}>",
                    D::N,
                    l,
                ))
            })
        }
    }

    #[pyclass(unsendable)]
    #[derive(Clone, Debug)]
    pub struct TensorEmbeddingData {
        spacing: Vec<f32>,
    }

    #[pymethods]
    impl TensorEmbeddingData {
        #[new]
        fn new(value: Vec<f32>) -> Self {
            Self { spacing: value }
        }

        #[getter]
        fn get_spacing<'a>(&self, py: Python<'a>) -> &'a PyArray1<f32> {
            PyArray1::from_vec(py, self.spacing.clone())
        }

        #[setter]
        fn set_spacing(&mut self, value: &PyArray1<f32>) {
            self.spacing = value.to_vec().unwrap();
        }
    }

    impl From<super::TensorEmbeddingData<DDyn>> for TensorEmbeddingData {
        fn from(t: super::TensorEmbeddingData<DDyn>) -> Self {
            Self {
                spacing: t.spacing.inner(),
            }
        }
    }

    impl From<TensorEmbeddingData> for super::TensorEmbeddingData<DDyn> {
        fn from(t: TensorEmbeddingData) -> Self {
            Self {
                spacing: Vector::new(t.spacing),
            }
        }
    }

    impl TensorEmbeddingData {
        pub fn try_into_dim<D: Dimension>(self) -> Result<super::TensorEmbeddingData<D>, PyErr> {
            let md: super::TensorEmbeddingData<DDyn> = self.into();
            let l = md.spacing.len();
            md.try_into_static().ok_or_else(|| {
                PyErr::new::<PyException, _>(format!(
                    "Expected TensorMetaData<{}>, but got TensorMetaData<{}>",
                    D::N,
                    l,
                ))
            })
        }
    }
}

impl<D: DynDimension> TensorMetaData<D> {
    pub fn single_chunk(dimensions: Vector<D, GlobalCoordinate>) -> Self {
        Self {
            dimensions: dimensions.clone(),
            chunk_size: dimensions.local(),
        }
    }

    pub fn chunk_pos_from_index(&self, index: ChunkIndex) -> Vector<D, ChunkCoordinate> {
        crate::vec::from_linear(index.0 as usize, &self.dimension_in_chunks())
    }
    pub fn num_tensor_elements(&self) -> usize {
        self.dimensions.hmul()
    }
    pub fn num_chunk_elements(&self) -> usize {
        self.chunk_size.hmul()
    }
    pub fn dimension_in_chunks(&self) -> Vector<D, ChunkCoordinate> {
        self.dimensions.zip(&self.chunk_size, |a, b| {
            crate::util::div_round_up(a.raw, b.raw).into()
        })
    }
    pub fn chunk_pos(&self, pos: &Vector<D, GlobalCoordinate>) -> Vector<D, ChunkCoordinate> {
        pos.zip(&self.chunk_size, |a, b| (a.raw / b.raw).into())
    }
    fn chunk_begin(&self, pos: &Vector<D, ChunkCoordinate>) -> Vector<D, GlobalCoordinate> {
        pos.zip(&self.chunk_size, |a, b| (a.raw * b.raw).into())
    }
    fn chunk_end(&self, pos: &Vector<D, ChunkCoordinate>) -> Vector<D, GlobalCoordinate> {
        let next_pos = pos + &Vector::fill_with_len(1u32, pos.len());
        let raw_end = self.chunk_begin(&next_pos);
        raw_end.zip(&self.dimensions, std::cmp::min)
    }
    pub fn chunk_index(&self, pos: &Vector<D, ChunkCoordinate>) -> ChunkIndex {
        ChunkIndex(crate::vec::to_linear(&pos, &self.dimension_in_chunks()) as u64)
    }
    pub fn chunk_info(&self, index: ChunkIndex) -> ChunkInfo<D> {
        self.chunk_info_vec(&self.chunk_pos_from_index(index))
    }
    pub fn chunk_info_vec(&self, pos: &Vector<D, ChunkCoordinate>) -> ChunkInfo<D> {
        let begin = self.chunk_begin(pos);
        let end = self.chunk_end(pos);
        let logical_dim = (&end - &begin).map(LocalCoordinate::interpret_as);
        ChunkInfo {
            mem_dimensions: self.chunk_size.clone(),
            logical_dimensions: logical_dim,
            begin,
        }
    }
}
impl<D: LargerDim> TensorMetaData<D> {
    pub fn push_dim_small(
        self,
        dim: GlobalCoordinate,
        chunk_size: LocalCoordinate,
    ) -> TensorMetaData<D::Larger> {
        TensorMetaData {
            dimensions: self.dimensions.push_dim_small(dim),
            chunk_size: self.chunk_size.push_dim_small(chunk_size),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ChunkIndex(pub(crate) u64);

pub type VolumeMetaData = TensorMetaData<D3>;
pub type VolumeEmbeddingData = TensorEmbeddingData<D3>;
impl VolumeMetaData {
    pub fn brick_positions(&self) -> impl Iterator<Item = Vector<D3, ChunkCoordinate>> {
        let bp = self.dimension_in_chunks();
        itertools::iproduct! { 0..bp.z().raw, 0..bp.y().raw, 0..bp.x().raw }
            .map(|(z, y, x)| [z, y, x].into())
    }
}

pub type ImageMetaData = TensorMetaData<D2>;
pub type ImageEmbeddingData = TensorEmbeddingData<D2>;
impl ImageMetaData {
    pub fn brick_positions(&self) -> impl Iterator<Item = Vector<D2, ChunkCoordinate>> {
        let bp = self.dimension_in_chunks();
        itertools::iproduct! { 0..bp.y().raw, 0..bp.x().raw }.map(|(y, x)| [y, x].into())
    }
    pub fn as_vol(&self) -> VolumeMetaData {
        VolumeMetaData {
            dimensions: [1.into(), self.dimensions.y(), self.dimensions.x()].into(),
            chunk_size: [1.into(), self.chunk_size.y(), self.chunk_size.x()].into(),
        }
    }
}

pub type ArrayMetaData = TensorMetaData<D1>;
impl ArrayMetaData {
    pub fn brick_positions(&self) -> impl Iterator<Item = Vector<D1, ChunkCoordinate>> {
        let bp = self.dimension_in_chunks();
        (0..bp[0].raw).into_iter().map(|x| Vector::from([x]))
    }
    pub fn as_image(&self) -> ImageMetaData {
        ImageMetaData {
            dimensions: [1.into(), self.dimensions[0]].into(),
            chunk_size: [1.into(), self.chunk_size[0]].into(),
        }
    }
}
