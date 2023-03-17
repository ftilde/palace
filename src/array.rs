use crate::data::{hmul, ChunkCoordinate, GlobalCoordinate, LocalCoordinate, Vector};

pub struct ChunkInfo<const N: usize> {
    pub mem_dimensions: Vector<N, LocalCoordinate>,
    pub logical_dimensions: Vector<N, LocalCoordinate>,
    pub begin: Vector<N, GlobalCoordinate>,
}
impl<const N: usize> ChunkInfo<N> {
    pub fn is_contiguous(&self) -> bool {
        for i in 1..N {
            if self.mem_dimensions.0[i] != self.logical_dimensions.0[i] {
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
#[derive(Copy, Clone, Hash)]
pub struct TensorMetaData<const N: usize> {
    pub dimensions: Vector<N, GlobalCoordinate>,
    pub chunk_size: Vector<N, LocalCoordinate>,
}

impl<const N: usize> TensorMetaData<N> {
    pub fn num_elements(&self) -> usize {
        hmul(self.dimensions)
    }
    pub fn dimension_in_bricks(&self) -> Vector<N, ChunkCoordinate> {
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
impl VolumeMetaData {
    pub fn brick_positions(&self) -> impl Iterator<Item = Vector<3, ChunkCoordinate>> {
        let bp = self.dimension_in_bricks();
        itertools::iproduct! { 0..bp.z().raw, 0..bp.y().raw, 0..bp.x().raw }
            .map(|(z, y, x)| [z, y, x].into())
    }
}

pub type ImageMetaData = TensorMetaData<2>;
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

#[allow(unused)]
mod unused {
    use super::*;
    pub type ArrayMetaData = TensorMetaData<1>;
    impl ArrayMetaData {
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
