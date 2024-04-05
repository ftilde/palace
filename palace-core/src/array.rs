use crate::data::{ChunkCoordinate, GlobalCoordinate, LocalCoordinate, Matrix, Vector};
use crate::dim::*;

pub struct ChunkInfo<D: Dimension> {
    pub mem_dimensions: Vector<D, LocalCoordinate>,
    pub logical_dimensions: Vector<D, LocalCoordinate>,
    pub begin: Vector<D, GlobalCoordinate>,
}
impl<D: Dimension> ChunkInfo<D> {
    pub fn is_contiguous(&self) -> bool {
        for i in 1..D::N {
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

    pub fn in_chunk(&self, pos: Vector<D, GlobalCoordinate>) -> Vector<D, LocalCoordinate> {
        (pos - self.begin).map(LocalCoordinate::interpret_as)
    }

    pub fn begin(&self) -> Vector<D, GlobalCoordinate> {
        self.begin
    }
    pub fn end(&self) -> Vector<D, GlobalCoordinate> {
        self.begin + self.logical_dimensions
    }
}

#[repr(C)]
#[derive(Copy, Clone, Hash, Debug, PartialEq, Eq, bytemuck::AnyBitPattern, id::Identify)]
pub struct TensorMetaData<D: Dimension> {
    pub dimensions: Vector<D, GlobalCoordinate>,
    pub chunk_size: Vector<D, LocalCoordinate>,
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
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Zeroable, id::Identify)]
pub struct TensorEmbeddingData<D: Dimension> {
    pub spacing: Vector<D, f32>,
    // NOTE: need to change identify impl if we want to add members
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
unsafe impl<D: Dimension + bytemuck::Zeroable> bytemuck::Pod for TensorEmbeddingData<D> {}

#[cfg(feature = "python")]
mod py {
    use super::*;
    use pyo3::prelude::*;

    impl<'source, D: Dimension> FromPyObject<'source> for TensorMetaData<D>
    where
        Vector<D, LocalCoordinate>: FromPyObject<'source>,
        Vector<D, GlobalCoordinate>: FromPyObject<'source>,
    {
        fn extract(ob: &'source PyAny) -> PyResult<Self> {
            Ok(TensorMetaData {
                dimensions: ob.getattr("dimensions")?.extract()?,
                chunk_size: ob.getattr("chunk_size")?.extract()?,
            })
        }
    }

    impl<D: Dimension> IntoPy<PyObject> for TensorMetaData<D> {
        fn into_py(self, py: Python<'_>) -> PyObject {
            let m = py.import("collections").unwrap();
            let ty = m
                .getattr("namedtuple")
                .unwrap()
                .call(("TensorMetaData", ["dimensions", "chunk_size"]), None)
                .unwrap();
            let v = ty
                .call(
                    (self.dimensions.into_py(py), self.chunk_size.into_py(py)),
                    None,
                )
                .unwrap();
            v.into_py(py)
        }
    }

    impl<'source, D: Dimension> FromPyObject<'source> for TensorEmbeddingData<D>
    where
        Vector<D, f32>: FromPyObject<'source>,
    {
        fn extract(ob: &'source PyAny) -> PyResult<Self> {
            Ok(TensorEmbeddingData {
                spacing: ob.getattr("spacing")?.extract()?,
            })
        }
    }

    impl<D: Dimension> IntoPy<PyObject> for TensorEmbeddingData<D> {
        fn into_py(self, py: Python<'_>) -> PyObject {
            let m = py.import("collections").unwrap();
            let ty = m
                .getattr("namedtuple")
                .unwrap()
                .call(("TensorEmbeddingData", ["spacing"]), None)
                .unwrap();
            let v = ty.call((self.spacing.into_py(py),), None).unwrap();
            v.into_py(py)
        }
    }
}

impl<D: Dimension> TensorMetaData<D> {
    pub fn num_tensor_elements(&self) -> usize {
        self.dimensions.hmul()
    }
    pub fn num_chunk_elements(&self) -> usize {
        self.chunk_size.hmul()
    }
    pub fn dimension_in_chunks(&self) -> Vector<D, ChunkCoordinate> {
        self.dimensions.zip(self.chunk_size, |a, b| {
            crate::util::div_round_up(a.raw, b.raw).into()
        })
    }
    pub fn chunk_pos(&self, pos: Vector<D, GlobalCoordinate>) -> Vector<D, ChunkCoordinate> {
        pos.zip(self.chunk_size, |a, b| (a.raw / b.raw).into())
    }
    fn chunk_begin(&self, pos: Vector<D, ChunkCoordinate>) -> Vector<D, GlobalCoordinate> {
        pos.zip(self.chunk_size, |a, b| (a.raw * b.raw).into())
    }
    fn chunk_end(&self, pos: Vector<D, ChunkCoordinate>) -> Vector<D, GlobalCoordinate> {
        let next_pos = pos + Vector::fill(1u32);
        let raw_end = self.chunk_begin(next_pos);
        raw_end.zip(self.dimensions, std::cmp::min)
    }
    pub fn chunk_info(&self, pos: Vector<D, ChunkCoordinate>) -> ChunkInfo<D> {
        let begin = self.chunk_begin(pos);
        let end = self.chunk_end(pos);
        let logical_dim = (end - begin).map(LocalCoordinate::interpret_as);
        ChunkInfo {
            mem_dimensions: self.chunk_size,
            logical_dimensions: logical_dim,
            begin,
        }
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
