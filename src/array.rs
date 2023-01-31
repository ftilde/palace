use crate::data::{hmul, ChunkCoordinate, GlobalVoxelCoordinate, LocalVoxelCoordinate, Vector};

pub struct ChunkMemInfo<const N: usize> {
    pub mem_dimensions: Vector<N, LocalVoxelCoordinate>,
    pub logical_dimensions: Vector<N, LocalVoxelCoordinate>,
}
impl<const N: usize> ChunkMemInfo<N> {
    pub fn is_contiguous(&self) -> bool {
        for i in 1..N {
            if self.mem_dimensions.0[i] != self.logical_dimensions.0[i] {
                return false;
            }
        }
        true
    }
    pub fn mem_size(&self) -> usize {
        hmul(self.mem_dimensions)
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ArrayMetaData<const N: usize> {
    pub dimensions: Vector<N, GlobalVoxelCoordinate>,
    pub chunk_size: Vector<N, LocalVoxelCoordinate>,
}

impl<const N: usize> ArrayMetaData<N> {
    pub fn num_voxels(&self) -> usize {
        hmul(self.dimensions)
    }
    pub fn dimension_in_bricks(&self) -> Vector<N, ChunkCoordinate> {
        self.dimensions.zip(self.chunk_size, |a, b| {
            crate::util::div_round_up(a.raw, b.raw).into()
        })
    }
    pub fn chunk_pos(&self, pos: Vector<N, GlobalVoxelCoordinate>) -> Vector<N, ChunkCoordinate> {
        pos.zip(self.chunk_size, |a, b| (a.raw / b.raw).into())
    }
    pub fn chunk_begin(&self, pos: Vector<N, ChunkCoordinate>) -> Vector<N, GlobalVoxelCoordinate> {
        pos.zip(self.chunk_size, |a, b| (a.raw * b.raw).into())
    }
    pub fn chunk_end(&self, pos: Vector<N, ChunkCoordinate>) -> Vector<N, GlobalVoxelCoordinate> {
        let next_pos = pos + Vector::fill(1.into());
        let raw_end = self.chunk_begin(next_pos);
        raw_end.zip(self.dimensions, std::cmp::min)
    }
    pub fn chunk_info(&self, pos: Vector<N, ChunkCoordinate>) -> ChunkMemInfo<N> {
        ChunkMemInfo {
            mem_dimensions: self.chunk_size,
            logical_dimensions: self.logical_chunk_dim(pos),
        }
    }
    fn logical_chunk_dim(
        &self,
        pos: Vector<N, ChunkCoordinate>,
    ) -> Vector<N, LocalVoxelCoordinate> {
        (self.chunk_end(pos) - self.chunk_begin(pos)).map(LocalVoxelCoordinate::interpret_as)
    }
}

pub type VolumeMetaData = ArrayMetaData<3>;
impl VolumeMetaData {
    pub fn brick_positions(&self) -> impl Iterator<Item = Vector<3, ChunkCoordinate>> {
        let bp = self.dimension_in_bricks();
        itertools::iproduct! { 0..bp.z().raw, 0..bp.y().raw, 0..bp.x().raw }
            .map(|(z, y, x)| [z, y, x].into())
    }
}

#[allow(unused)]
mod unused {
    use super::*;

    pub type ImageMetaData = ArrayMetaData<2>;
    impl ImageMetaData {
        pub fn brick_positions(&self) -> impl Iterator<Item = Vector<2, ChunkCoordinate>> {
            let bp = self.dimension_in_bricks();
            itertools::iproduct! { 0..bp.y().raw, 0..bp.x().raw }.map(|(y, x)| [y, x].into())
        }
        pub fn as_vol(&self) -> VolumeMetaData {
            VolumeMetaData {
                dimensions: [1.into(), self.dimensions.y(), self.dimensions.x()].into(),
                chunk_size: [1.into(), self.chunk_size.y(), self.chunk_size.x()].into(),
            }
        }
    }
    pub type ArrayMetaData1D = ArrayMetaData<1>;
    impl ArrayMetaData1D {
        pub fn brick_positions(&self) -> impl Iterator<Item = Vector<1, ChunkCoordinate>> {
            let bp = self.dimension_in_bricks();
            (0..bp.0[0].raw).into_iter().map(|x| Vector::from([x]))
        }
        pub fn as_image(&self) -> ImageMetaData {
            ImageMetaData {
                dimensions: [1.into(), self.dimensions.0[0]].into(),
                chunk_size: [1.into(), self.chunk_size.0[0]].into(),
            }
        }
    }
}
