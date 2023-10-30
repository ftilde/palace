pub mod gpu;
pub mod ram;

use std::{cell::RefCell, collections::BTreeMap};

use crate::{
    operator::DataId,
    runtime::FrameNumber,
    util::Map,
    vulkan::{DeviceId, DstBarrierInfo},
};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum DataLocation {
    Ram,
    VRam(DeviceId),
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum VisibleDataLocation {
    Ram,
    VRam(DeviceId, DstBarrierInfo),
}

const GARBAGE_COLLECT_GOAL_FRACTION: u64 = 10;

type LRUIndex = u64;

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

struct LRUManager<T> {
    list: BTreeMap<LRUIndex, T>,
    current: LRUIndex,
}

impl<T> Default for LRUManager<T> {
    fn default() -> Self {
        Self {
            list: Default::default(),
            current: 0,
        }
    }
}

impl<T: Clone> LRUManager<T> {
    fn remove(&mut self, old: LRUIndex) {
        self.list.remove(&old).unwrap();
    }

    #[must_use]
    fn add(&mut self, data: T) -> LRUIndex {
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
