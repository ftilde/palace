use bytemuck::{AnyBitPattern, Zeroable};

pub fn hmul<S>(s: cgmath::Vector3<S>) -> S
where
    S: std::ops::Mul<S, Output = S>,
{
    s.x * s.y * s.z
}

pub fn to_linear(pos: SVec3, dim: SVec3) -> usize {
    let pos = pos.cast::<usize>().unwrap();
    let dim = dim.cast::<usize>().unwrap();
    (pos.z * dim.y + pos.y) * dim.y + pos.x
}

pub type SVec3 = cgmath::Vector3<u32>;

#[derive(Copy, Clone, Hash)]
pub struct VoxelPosition(pub SVec3);
#[derive(Copy, Clone, Hash)]
pub struct BrickPosition(pub SVec3);

// TODO: Check if this is actually valid...
unsafe impl Zeroable for VoxelPosition {}
unsafe impl AnyBitPattern for VoxelPosition {}

// TODO: Maybe we don't want this to be copy if it gets too large.
#[derive(Copy, Clone, AnyBitPattern)]
pub struct VolumeMetaData {
    pub dimensions: VoxelPosition,
    pub brick_size: VoxelPosition,
}

fn div_round_up(v1: u32, v2: u32) -> u32 {
    (v1 + v2 - 1) / v2
}

impl VolumeMetaData {
    pub fn num_voxels(&self) -> u64 {
        hmul(self.dimensions.0.cast::<u64>().unwrap())
    }
    pub fn dimension_in_bricks(&self) -> BrickPosition {
        BrickPosition(self.dimensions.0.zip(self.brick_size.0, div_round_up))
    }
    //pub fn brick_pos(&self, pos: VoxelPosition) -> BrickPosition {
    //    BrickPosition(pos.0.zip(self.brick_size.0, |a, b| a / b))
    //}
    pub fn brick_begin(&self, pos: BrickPosition) -> VoxelPosition {
        VoxelPosition(pos.0.zip(self.brick_size.0, |a, b| a * b))
    }
    pub fn brick_end(&self, pos: BrickPosition) -> VoxelPosition {
        let next_pos = pos.0 + cgmath::vec3(1, 1, 1);
        let raw_end = next_pos.zip(self.brick_size.0, |a, b| a * b);
        VoxelPosition(raw_end.zip(self.dimensions.0, std::cmp::min))
    }
    pub fn brick_dim(&self, pos: BrickPosition) -> VoxelPosition {
        VoxelPosition(self.brick_end(pos).0 - self.brick_begin(pos).0)
    }
}

pub struct Brick<'a> {
    size: VoxelPosition,
    data: &'a [f32],
}

impl<'a> Brick<'a> {
    pub fn new(data: &'a [f32], size: VoxelPosition) -> Self {
        Self { data, size }
    }
    pub fn voxels(&'a self) -> impl Iterator<Item = f32> + 'a {
        itertools::iproduct! { 0..self.size.0.z, 0..self.size.0.y, 0..self.size.0.x }
            .map(|(z, y, x)| to_linear(cgmath::vec3(x, y, z), self.size.0))
            .map(|i| self.data[i])
    }
}
