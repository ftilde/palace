use std::{fs::File, os::unix::fs::MetadataExt, path::Path, sync::Mutex};

use itertools::Itertools;
use zarrs::{
    byte_range::extract_byte_ranges_read,
    storage::{
        Bytes, ListableStorageTraits, ReadableStorageTraits, StorageError, StoreKey, StoreKeys,
        StoreKeysPrefixes, StorePrefix, StorePrefixes,
    },
};
use zip::{read::ZipArchive, result::ZipError};

pub struct Reader {
    archive: Mutex<ZipArchive<File>>,
    size: u64,
}

impl Reader {
    pub fn open(path: &Path) -> Result<Self, palace_core::Error> {
        let file = File::open(path)?;
        let size = file.metadata()?.size();
        let archive = ZipArchive::new(file)?;
        Ok(Reader {
            archive: archive.into(),
            size,
        }
        .into())
    }
}

impl ReadableStorageTraits for Reader {
    fn get_partial_values_key(
        &self,
        key: &zarrs::storage::StoreKey,
        byte_ranges: &[zarrs::byte_range::ByteRange],
    ) -> Result<Option<Vec<zarrs::storage::Bytes>>, StorageError> {
        let mut r = self.archive.lock().unwrap();
        let mut file = match r.by_name(key.as_str()) {
            Ok(f) => f,
            Err(err) => match err {
                ZipError::FileNotFound => return Ok(None),
                _ => return Err(StorageError::Other(err.to_string())),
            },
        };

        let size = file.size();

        let out = extract_byte_ranges_read(&mut file, size, byte_ranges)?
            .into_iter()
            .map(Bytes::from)
            .collect();
        Ok(Some(out))
    }

    fn size_key(&self, key: &zarrs::storage::StoreKey) -> Result<Option<u64>, StorageError> {
        let mut r = self.archive.lock().unwrap();
        let file = match r.by_name(key.as_str()) {
            Ok(f) => f,
            Err(err) => match err {
                ZipError::FileNotFound => return Ok(None),
                _ => return Err(StorageError::Other(err.to_string())),
            },
        };
        Ok(Some(file.size()))
    }
}

impl ListableStorageTraits for Reader {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        Ok(self
            .archive
            .lock()
            .unwrap()
            .file_names()
            .filter_map(|v| StoreKey::try_from(v).ok())
            .sorted()
            .collect())
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        let mut zip_archive = self.archive.lock().unwrap();
        let file_names: Vec<String> = zip_archive
            .file_names()
            .map(std::string::ToString::to_string)
            .collect();
        Ok(file_names
            .into_iter()
            .filter_map(|name| {
                if name.starts_with(prefix.as_str()) {
                    if let Ok(file) = zip_archive.by_name(&name) {
                        if file.is_file() {
                            let name = name.strip_suffix('/').unwrap_or(&name);
                            if let Ok(store_key) = StoreKey::try_from(name) {
                                return Some(store_key);
                            }
                        }
                    }
                }
                None
            })
            .sorted()
            .collect())
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let zip_archive = self.archive.lock().unwrap();
        let mut keys: StoreKeys = vec![];
        let mut prefixes: StorePrefixes = vec![];
        for name in zip_archive.file_names() {
            if name.starts_with(prefix.as_str()) {
                if name.ends_with('/') {
                    if let Ok(store_prefix) = StorePrefix::try_from(name) {
                        if let Some(parent) = store_prefix.parent() {
                            if &parent == prefix {
                                prefixes.push(store_prefix);
                            }
                        }
                    }
                } else if let Ok(store_key) = StoreKey::try_from(name) {
                    let parent = store_key.parent();
                    if &parent == prefix {
                        keys.push(store_key);
                    }
                }
            }
        }
        keys.sort();
        prefixes.sort();

        Ok(StoreKeysPrefixes::new(keys, prefixes))
    }

    fn size(&self) -> Result<u64, StorageError> {
        Ok(self.size)
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        let mut size = 0;
        for key in self.list_prefix(prefix)? {
            if let Some(size_key) = self.size_key(&key)? {
                size += size_key;
            }
        }
        Ok(size)
    }
}
