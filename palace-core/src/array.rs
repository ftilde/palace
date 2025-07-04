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

impl<D: Dimension> Copy for TensorMetaData<D>
where
    Vector<D, GlobalCoordinate>: Copy,
    Vector<D, LocalCoordinate>: Copy,
{
}

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

    pub fn dim(&self) -> D {
        let d = self.dimensions.dim();
        assert_eq!(d, self.chunk_size.dim());
        d
    }

    pub fn num_chunks(&self) -> usize {
        self.dimension_in_chunks().hmul()
    }

    pub fn is_single_chunk(&self) -> bool {
        self.dimension_in_chunks().raw() == Vector::fill_with_len(1u32, self.dimensions.len())
    }
}

// We have to do this manually since bytemuck cannot verify this in general due to the const
// parameter N. It is fine though (as long as we don't change anything on TensorMetaData).
//unsafe impl<D: Dimension> bytemuck::Pod for TensorMetaData<D> {}

impl<D: LargerDim> TensorMetaData<D> {
    pub fn norm_to_voxel(&self) -> Matrix<D::Larger, f32> {
        Matrix::from_translation(Vector::fill_with_len(-0.5, self.dimensions.len()))
            * &Matrix::from_scale(&self.dimensions.raw().f32()).to_homogeneous()
    }
    pub fn voxel_to_norm(&self) -> Matrix<D::Larger, f32> {
        Matrix::from_scale(&self.dimensions.raw().f32().map(|v| 1.0 / v)).to_homogeneous()
            * &Matrix::from_translation(Vector::fill_with_len(0.5, self.dimensions.len()))
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, id::Identify)]
pub struct TensorEmbeddingData<D: DynDimension> {
    pub spacing: Vector<D, f32>,
    // NOTE: need to change identify impl if we want to add members
}

impl<D: Dimension> Default for TensorEmbeddingData<D> {
    fn default() -> Self {
        Self {
            spacing: Vector::fill(1.0),
        }
    }
}

impl<D: Dimension> Copy for TensorEmbeddingData<D> where Vector<D, f32>: Copy {}

impl<D: DynDimension> TensorEmbeddingData<D> {
    pub fn dim(&self) -> D {
        self.spacing.dim()
    }

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
impl<D: DynDimension + SmallerDim> TensorEmbeddingData<D> {
    pub fn pop_dim_small(self) -> TensorEmbeddingData<D::Smaller> {
        TensorEmbeddingData {
            spacing: self.spacing.pop_dim_small(),
        }
    }
    pub fn drop_dim(self, dim: usize) -> TensorEmbeddingData<D::Smaller> {
        TensorEmbeddingData {
            spacing: self.spacing.drop_dim(dim),
        }
    }
}

impl<D: LargerDim> TensorEmbeddingData<D> {
    pub fn voxel_to_physical(&self) -> Matrix<D::Larger, f32> {
        Matrix::from_scale(&self.spacing).to_homogeneous()
    }
    pub fn physical_to_voxel(&self) -> Matrix<D::Larger, f32> {
        Matrix::from_scale(&self.spacing.map(|v| 1.0 / v)).to_homogeneous()
    }
    pub fn push_dim_small(self, spacing: f32) -> TensorEmbeddingData<D::Larger> {
        TensorEmbeddingData {
            spacing: self.spacing.push_dim_large(spacing),
        }
    }
}

pub fn norm_to_physical<D: LargerDim>(
    md: &TensorMetaData<D>,
    emd: &TensorEmbeddingData<D>,
) -> Matrix<D::Larger, f32> {
    emd.voxel_to_physical() * &md.norm_to_voxel()
}

pub fn physical_to_voxel<D: LargerDim>(
    md: &TensorMetaData<D>,
    emd: &TensorEmbeddingData<D>,
) -> Matrix<D::Larger, f32> {
    md.voxel_to_norm() * &emd.physical_to_voxel()
}

//TODO: Revisit this. This is definitely fine as long as we don't add other members
//unsafe impl<D: Dimension + bytemuck::Zeroable> bytemuck::Pod for TensorEmbeddingData<D> {}

use bytemuck::{Pod, Zeroable};
use id::Identify;
#[cfg(feature = "python")]
pub use py::TensorEmbeddingData as PyTensorEmbeddingData;
#[cfg(feature = "python")]
pub use py::TensorMetaData as PyTensorMetaData;

#[cfg(feature = "python")]
mod py {
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::{exceptions::PyException, prelude::*, IntoPyObjectExt};

    use super::*;

    #[pyo3_stub_gen::derive::gen_stub_pyclass]
    #[pyclass(unsendable)]
    #[derive(Clone, Debug)]
    pub struct TensorMetaData {
        pub dimensions: Vec<u32>,
        pub chunk_size: Vec<u32>,
    }

    #[pyo3_stub_gen::derive::gen_stub_pymethods]
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
        fn dimensions<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray1<u32>> {
            PyArray1::from_vec(py, self.dimensions.clone())
        }
        #[getter]
        fn chunk_size<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray1<u32>> {
            PyArray1::from_vec(py, self.chunk_size.clone())
        }

        fn chunk_pos<'a>(&self, pos: Vec<u32>, py: Python<'a>) -> Bound<'a, PyArray1<u32>> {
            PyArray1::from_vec(
                py,
                super::TensorMetaData::<DDyn>::from(self.clone())
                    .chunk_pos(&Vector::<DDyn, u32>::new(pos).global())
                    .raw()
                    .inner(),
            )
        }

        fn pos_in_chunk<'a>(
            &self,
            chunk_pos: Vec<u32>,
            global_pos: Vec<u32>,
            py: Python<'a>,
        ) -> Bound<'a, PyArray1<u32>> {
            PyArray1::from_vec(
                py,
                super::TensorMetaData::<DDyn>::from(self.clone())
                    .chunk_info_vec(&Vector::<DDyn, u32>::new(chunk_pos).chunk())
                    .in_chunk(&Vector::<DDyn, u32>::new(global_pos).global())
                    .raw()
                    .inner(),
            )
        }

        pub fn norm_to_voxel(&self, py: Python) -> PyResult<PyObject> {
            super::TensorMetaData::<DDyn>::from(self.clone())
                .norm_to_voxel()
                .into_py_any(py)
        }
        pub fn voxel_to_norm(&self, py: Python) -> PyResult<PyObject> {
            super::TensorMetaData::<DDyn>::from(self.clone())
                .voxel_to_norm()
                .into_py_any(py)
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

    #[pyo3_stub_gen::derive::gen_stub_pyclass]
    #[pyclass(unsendable)]
    #[derive(Clone, Debug)]
    pub struct TensorEmbeddingData {
        pub spacing: Vec<f32>,
    }

    #[pyo3_stub_gen::derive::gen_stub_pymethods]
    #[pymethods]
    impl TensorEmbeddingData {
        #[new]
        pub fn new(value: Vec<f32>) -> Self {
            Self { spacing: value }
        }

        #[getter]
        fn get_spacing<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray1<f32>> {
            PyArray1::from_vec(py, self.spacing.clone())
        }

        #[setter]
        fn set_spacing(&mut self, value: Bound<numpy::PyArray1<f32>>) -> PyResult<()> {
            if self.spacing.len() != value.len()? {
                return Err(PyErr::new::<PyException, _>(format!(
                    "Expected spacing vec of len {}, but got one of len {}",
                    self.spacing.len(),
                    value.len()?,
                )));
            }

            self.spacing = value.to_vec()?;
            Ok(())
        }

        pub fn nd(&self) -> usize {
            self.spacing.len()
        }

        pub fn voxel_to_physical(&self, py: Python) -> PyResult<PyObject> {
            super::TensorEmbeddingData::<DDyn>::from(self.clone())
                .voxel_to_physical()
                .into_py_any(py)
        }
        pub fn physical_to_voxel(&self, py: Python) -> PyResult<PyObject> {
            super::TensorEmbeddingData::<DDyn>::from(self.clone())
                .physical_to_voxel()
                .into_py_any(py)
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
                    "Expected TensorEmbeddingData<{}>, but got TensorEmbeddingData<{}>",
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
            if a.raw == 0 {
                0.into()
            } else {
                crate::util::div_round_up(a.raw, b.raw).into()
            }
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
        let dim_in_chunks = &self.dimension_in_chunks();
        assert!(pos.zip(dim_in_chunks, |l, r| l < r).hand());
        ChunkIndex(crate::vec::to_linear(&pos, &dim_in_chunks) as u64)
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
    pub fn chunk_indices(&self) -> impl Iterator<Item = ChunkIndex> {
        let bp = self.dimension_in_chunks();
        (0..bp.hmul() as u64).into_iter().map(ChunkIndex)
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

impl<D: SmallerDim> TensorMetaData<D> {
    pub fn pop_dim_small(self) -> TensorMetaData<D::Smaller> {
        TensorMetaData {
            dimensions: self.dimensions.pop_dim_small(),
            chunk_size: self.chunk_size.pop_dim_small(),
        }
    }
}

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Identify, Pod, Zeroable)]
pub struct ChunkIndex(pub(crate) u64);

impl ChunkIndex {
    pub fn raw(&self) -> u64 {
        self.0
    }
}

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
