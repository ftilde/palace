use crate::task::TaskId;
use crate::Error;
use std::{cell::RefCell, collections::BTreeMap};

pub type BrickData = Vec<f32>;

#[derive(Clone)] //TODO remove clone bound
#[non_exhaustive]
pub enum Datum {
    Float(f32),
    Volume(VolumeMetaData),
    Brick(BrickData),
}

impl Datum {
    pub fn float(self) -> Result<f32, Error> {
        if let Datum::Float(v) = self {
            Ok(v)
        } else {
            Err("Value is not a float".into())
        }
    }
    pub fn volume(self) -> Result<VolumeMetaData, Error> {
        if let Datum::Volume(v) = self {
            Ok(v)
        } else {
            Err("Value is not a volume".into())
        }
    }
    pub fn brick(self) -> Result<BrickData, Error> {
        if let Datum::Brick(v) = self {
            Ok(v)
        } else {
            Err("Value is not a brick".into())
        }
    }
}

pub struct Storage {
    memory_cache: RefCell<BTreeMap<TaskId, Datum>>,
}

impl Storage {
    pub fn new() -> Self {
        Self {
            memory_cache: RefCell::new(BTreeMap::new()),
        }
    }
    pub fn store_ram(&self, key: TaskId, datum: Datum) {
        let prev = self.memory_cache.borrow_mut().insert(key, datum);
        assert!(prev.is_none());
    }
    pub fn read_ram(&self, key: TaskId) -> Option<Datum> {
        self.memory_cache.borrow().get(&key).cloned()
    }
}

pub fn hmul<S>(s: cgmath::Vector3<S>) -> S
where
    S: std::ops::Mul<S, Output = S>,
{
    s.x * s.y * s.z
}

pub type SVec3 = cgmath::Vector3<u32>;

#[derive(Copy, Clone, Hash)]
pub struct VoxelPosition(pub SVec3);
#[derive(Copy, Clone, Hash)]
pub struct BrickPosition(pub SVec3);

// TODO: Maybe we don't want this to be copy if it gets too large.
#[derive(Copy, Clone)]
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
