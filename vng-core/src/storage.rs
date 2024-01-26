pub mod cpu;
pub mod disk;
pub mod gpu;
pub mod ram;

use std::{cell::RefCell, collections::BTreeMap};

use crate::{
    operator::DataId,
    runtime::FrameNumber,
    util::Map,
    vulkan::{DeviceId, DstBarrierInfo},
};

pub trait Element: Send + Sync + bytemuck::AnyBitPattern {}

impl<T: bytemuck::AnyBitPattern + Send + Sync> Element for T {}

#[derive(Copy, Clone, bytemuck::AnyBitPattern)]
#[repr(C)]
pub struct P<L, R>(pub L, pub R);

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum DataLocation {
    CPU(CpuDataLocation),
    GPU(DeviceId),
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum CpuDataLocation {
    Ram,
    Disk,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum VisibleDataLocation {
    CPU(CpuDataLocation),
    GPU(DeviceId, DstBarrierInfo),
}

pub const GARBAGE_COLLECT_GOAL_FRACTION: u64 = 10;

type LRUIndexInner = u64;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum DataVersion {
    Final,
    Preview(FrameNumber),
}

impl DataVersion {
    pub fn of_frame(&self) -> Option<FrameNumber> {
        match self {
            DataVersion::Final => None,
            DataVersion::Preview(v) => Some(*v),
        }
    }
}

impl Ord for DataVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self, other) {
            (DataVersion::Final, DataVersion::Final) => Ordering::Equal,
            (DataVersion::Final, DataVersion::Preview(_)) => Ordering::Greater,
            (DataVersion::Preview(_), DataVersion::Final) => Ordering::Less,
            (DataVersion::Preview(l), DataVersion::Preview(r)) => l.cmp(r),
        }
    }
}

impl PartialOrd for DataVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl DataVersion {
    pub fn type_(&self) -> DataVersionType {
        match self {
            DataVersion::Final => DataVersionType::Final,
            DataVersion::Preview(_) => DataVersionType::Preview,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum DataVersionType {
    Final,
    Preview,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum DataLongevity {
    Ephemeral = 0,
    Unstable = 1,
    Stable = 2,
}

impl TryFrom<usize> for DataLongevity {
    type Error = &'static str;

    fn try_from(raw: usize) -> Result<Self, Self::Error> {
        match raw {
            0 => Ok(DataLongevity::Ephemeral),
            1 => Ok(DataLongevity::Unstable),
            2 => Ok(DataLongevity::Stable),
            _ => Err("Invalid DataLongevity value"),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct LRUIndex(u64);
const NUM_BITS_FOR_LONGEVITY: u64 = 2;
const ID_MASK: u64 = u64::MAX >> NUM_BITS_FOR_LONGEVITY;
//const LONGEVITY_MASK: u64 = !ID_MASK;
const LONGEVITY_BASE: u64 = ID_MASK + 1;

impl LRUIndex {
    fn new(inner: LRUIndexInner, longevity: DataLongevity) -> Self {
        assert_eq!(inner & ID_MASK, inner);
        Self(inner | (LONGEVITY_BASE * longevity as u64))
    }

    fn longevity(&self) -> DataLongevity {
        let raw = self.0 >> (64 - NUM_BITS_FOR_LONGEVITY);
        DataLongevity::try_from(raw as usize).unwrap()
    }
    fn inner_id(&self) -> LRUIndexInner {
        self.0 & ID_MASK
    }
}

struct LRUManager<T> {
    inner: [LRUManagerInner<T>; 3],
}

impl<T> Default for LRUManager<T> {
    fn default() -> Self {
        Self {
            inner: std::array::from_fn(|_| Default::default()),
        }
    }
}

impl<T: Clone> LRUManager<T> {
    fn remove(&mut self, old: LRUIndex) {
        let longevity = old.longevity();
        let raw = old.inner_id();
        self.inner[longevity as usize].remove(raw);
    }

    #[must_use]
    fn add(&mut self, data: T, longevity: DataLongevity) -> LRUIndex {
        let raw = self.inner[longevity as usize].add(data);
        LRUIndex::new(raw, longevity)
    }

    fn inner_mut<'a>(
        &'a mut self,
    ) -> impl Iterator<Item = (DataLongevity, &'a mut LRUManagerInner<T>)> + 'a {
        self.inner
            .iter_mut()
            .enumerate()
            .map(|(i, v)| (DataLongevity::try_from(i).unwrap(), v))
    }
}

struct LRUManagerInner<T> {
    list: BTreeMap<LRUIndexInner, T>,
    current: LRUIndexInner,
}

impl<T> Default for LRUManagerInner<T> {
    fn default() -> Self {
        Self {
            list: Default::default(),
            current: 0,
        }
    }
}

impl<T: Clone> LRUManagerInner<T> {
    fn remove(&mut self, old: LRUIndexInner) {
        self.list.remove(&old).unwrap();
    }

    #[must_use]
    fn add(&mut self, data: T) -> LRUIndexInner {
        let new = self
            .current
            .checked_add(1)
            .expect("Looks like we need to handle wrapping here...");
        self.current = new;

        self.list.insert(new, data);

        new
    }

    fn get_next(&self) -> Option<T> {
        self.list.first_key_value().map(|(_, d)| d.clone())
    }
    fn pop_next(&mut self) {
        self.list.pop_first();
    }

    fn drain_lru<'a>(&'a mut self) -> impl Iterator<Item = T> + 'a {
        std::iter::from_fn(move || {
            return self.list.pop_first().map(|(_, d)| d);
        })
    }
}

#[derive(Default)]
pub struct NewDataManager {
    inner: RefCell<Map<DataId, DataVersionType>>,
}

impl NewDataManager {
    fn add(&self, key: DataId, version: DataVersionType) {
        self.inner.borrow_mut().insert(key, version);
    }
    fn remove(&self, key: DataId) {
        self.inner.borrow_mut().remove(&key);
    }
    fn drain(&self) -> impl Iterator<Item = (DataId, DataVersionType)> {
        let mut m = self.inner.borrow_mut();
        let ret = std::mem::take(&mut *m);
        ret.into_iter()
    }
}

#[derive(Copy, Clone, derive_more::From, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GarbageCollectId(u64);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_longevity_cmp() {
        assert!(DataLongevity::Ephemeral < DataLongevity::Unstable);
        assert!(DataLongevity::Unstable < DataLongevity::Stable);
    }
}
