use crate::data::{hmul, ChunkCoordinate, GlobalCoordinate, LocalCoordinate, Matrix, Vector};

pub struct ChunkInfo<const N: usize> {
    pub mem_dimensions: Vector<N, LocalCoordinate>,
    pub logical_dimensions: Vector<N, LocalCoordinate>,
    pub begin: Vector<N, GlobalCoordinate>,
}
impl<const N: usize> ChunkInfo<N> {
    pub fn is_contiguous(&self) -> bool {
        for i in 1..N {
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
        hmul(self.mem_dimensions)
    }

    pub fn in_chunk(&self, pos: Vector<N, GlobalCoordinate>) -> Vector<N, LocalCoordinate> {
        (pos - self.begin).map(LocalCoordinate::interpret_as)
    }

    pub fn begin(&self) -> Vector<N, GlobalCoordinate> {
        self.begin
    }
    pub fn end(&self) -> Vector<N, GlobalCoordinate> {
        self.begin + self.logical_dimensions
    }
}

#[repr(C)]
#[derive(Copy, Clone, Hash, Debug, PartialEq, Eq, bytemuck::AnyBitPattern)]
pub struct TensorMetaData<const N: usize> {
    pub dimensions: Vector<N, GlobalCoordinate>,
    pub chunk_size: Vector<N, LocalCoordinate>,
}

impl<const N: usize> crate::id::Identify for TensorMetaData<N> {
    fn id(&self) -> crate::id::Id {
        crate::id::Id::hash(self)
    }
}

// We have to do this manually since bytemuck cannot verify this in general due to the const
// parameter N. It is fine though (as long as we don't change anything on TensorMetaData).
//unsafe impl<const N: usize> bytemuck::Pod for TensorMetaData<N> {}

//TODO: generalize
impl TensorMetaData<3> {
    pub fn norm_to_voxel(&self) -> Matrix<4, f32> {
        Matrix::from_translation(Vector::fill(-0.5))
            * Matrix::from_scale(self.dimensions.raw().f32()).to_homogeneuous()
    }
    pub fn voxel_to_norm(&self) -> Matrix<4, f32> {
        Matrix::from_scale(self.dimensions.raw().f32().map(|v| 1.0 / v)).to_homogeneuous()
            * Matrix::from_translation(Vector::fill(0.5))
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Zeroable)]
pub struct TensorEmbeddingData<const N: usize> {
    pub spacing: Vector<N, f32>,
    // NOTE: need to change identify impl if we want to add members
}

impl<const N: usize> crate::id::Identify for TensorEmbeddingData<N> {
    fn id(&self) -> crate::id::Id {
        self.spacing.id()
    }
}

//TODO: generalize
impl TensorEmbeddingData<3> {
    pub fn voxel_to_physical(&self) -> Matrix<4, f32> {
        Matrix::from_scale(self.spacing).to_homogeneuous()
    }
    pub fn physical_to_voxel(&self) -> Matrix<4, f32> {
        Matrix::from_scale(self.spacing.map(|v| 1.0 / v)).to_homogeneuous()
    }
}

pub fn norm_to_physical(md: &TensorMetaData<3>, emd: &TensorEmbeddingData<3>) -> Matrix<4, f32> {
    emd.voxel_to_physical() * md.norm_to_voxel()
}
pub fn physical_to_voxel(md: &TensorMetaData<3>, emd: &TensorEmbeddingData<3>) -> Matrix<4, f32> {
    md.voxel_to_norm() * emd.physical_to_voxel()
}

//TODO: Revisit this. This is definitely fine as long as we don't add other members
unsafe impl<const N: usize> bytemuck::Pod for TensorEmbeddingData<N> {}

#[cfg(feature = "python")]
mod py {
    use super::*;
    use pyo3::prelude::*;

    impl<'source, const N: usize> FromPyObject<'source> for TensorMetaData<N> {
        fn extract(ob: &'source PyAny) -> PyResult<Self> {
            Ok(TensorMetaData {
                dimensions: ob.getattr("dimensions")?.extract()?,
                chunk_size: ob.getattr("chunk_size")?.extract()?,
            })
        }
    }

    impl<const N: usize> IntoPy<PyObject> for TensorMetaData<N> {
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
}

impl<const N: usize> TensorMetaData<N> {
    pub fn num_elements(&self) -> usize {
        hmul(self.dimensions)
    }
    pub fn dimension_in_chunks(&self) -> Vector<N, ChunkCoordinate> {
        self.dimensions.zip(self.chunk_size, |a, b| {
            crate::util::div_round_up(a.raw, b.raw).into()
        })
    }
    pub fn chunk_pos(&self, pos: Vector<N, GlobalCoordinate>) -> Vector<N, ChunkCoordinate> {
        pos.zip(self.chunk_size, |a, b| (a.raw / b.raw).into())
    }
    fn chunk_begin(&self, pos: Vector<N, ChunkCoordinate>) -> Vector<N, GlobalCoordinate> {
        pos.zip(self.chunk_size, |a, b| (a.raw * b.raw).into())
    }
    fn chunk_end(&self, pos: Vector<N, ChunkCoordinate>) -> Vector<N, GlobalCoordinate> {
        let next_pos = pos + Vector::fill(1u32);
        let raw_end = self.chunk_begin(next_pos);
        raw_end.zip(self.dimensions, std::cmp::min)
    }
    pub fn chunk_info(&self, pos: Vector<N, ChunkCoordinate>) -> ChunkInfo<N> {
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

pub type VolumeMetaData = TensorMetaData<3>;
pub type VolumeEmbeddingData = TensorEmbeddingData<3>;
impl VolumeMetaData {
    pub fn brick_positions(&self) -> impl Iterator<Item = Vector<3, ChunkCoordinate>> {
        let bp = self.dimension_in_chunks();
        itertools::iproduct! { 0..bp.z().raw, 0..bp.y().raw, 0..bp.x().raw }
            .map(|(z, y, x)| [z, y, x].into())
    }
}

pub type ImageMetaData = TensorMetaData<2>;
impl ImageMetaData {
    pub fn brick_positions(&self) -> impl Iterator<Item = Vector<2, ChunkCoordinate>> {
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

pub type ArrayMetaData = TensorMetaData<1>;
impl ArrayMetaData {
    pub fn brick_positions(&self) -> impl Iterator<Item = Vector<1, ChunkCoordinate>> {
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
