pub mod gpu;
pub mod ram;

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
};

use crate::{
    operator::DataId,
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

type LRUIndex = u64;

#[derive(Default)]
struct LRUManager {
    list: BTreeMap<LRUIndex, DataId>,
    current: LRUIndex,
}

impl LRUManager {
    fn remove(&mut self, old: LRUIndex) {
        self.list.remove(&old).unwrap();
    }

    #[must_use]
    fn add(&mut self, data: DataId) -> LRUIndex {
        let new = self
            .current
            .checked_add(1)
            .expect("Looks like we need to handle wrapping here...");
        self.current = new;

        self.list.insert(new, data);

        new
    }

    fn get_next(&self) -> Option<DataId> {
        self.list.first_key_value().map(|(_, d)| *d)
    }
    fn pop_next(&mut self) {
        self.list.pop_first();
    }

    fn drain_lru<'a>(&'a mut self) -> impl Iterator<Item = DataId> + 'a {
        std::iter::from_fn(move || {
            return self.list.pop_first().map(|(_, d)| d);
        })
    }
}

#[derive(Default)]
pub struct NewDataManager {
    inner: RefCell<BTreeSet<DataId>>,
}

impl NewDataManager {
    fn add(&self, key: DataId) {
        self.inner.borrow_mut().insert(key);
    }
    fn remove(&self, key: DataId) {
        self.inner.borrow_mut().remove(&key);
    }
    fn drain(&self) -> impl Iterator<Item = DataId> {
        let mut m = self.inner.borrow_mut();
        let ret = std::mem::take(&mut *m);
        ret.into_iter()
    }
}
