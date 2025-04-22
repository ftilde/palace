use std::{fs::File, path::PathBuf, sync::Mutex};

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
    main_archive: ZipArchive<CloneFile>,
    clones: Mutex<Vec<ZipArchive<CloneFile>>>,
}

struct CloneFile {
    file: File,
    path: PathBuf,
}

impl Clone for CloneFile {
    fn clone(&self) -> Self {
        let file = File::open(&self.path).unwrap();
        Self {
            file,
            path: self.path.clone(),
        }
    }
}

impl std::io::Read for CloneFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.file.read(buf)
    }
}

impl std::io::Seek for CloneFile {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.file.seek(pos)
    }
}

fn open_zip(path: PathBuf) -> Result<ZipArchive<CloneFile>, palace_core::Error> {
    let file = File::open(&path)?;
    let file = CloneFile { file, path };
    Ok(ZipArchive::new(file)?)
}

impl Reader {
    pub fn open(path: PathBuf) -> Result<Self, palace_core::Error> {
        let archive = open_zip(path)?;
        Ok(Reader {
            main_archive: archive.into(),
            clones: Default::default(),
        }
        .into())
    }

    fn with_archive<R>(&self, f: impl FnOnce(&mut ZipArchive<CloneFile>) -> R) -> R {
        let mut archive = self
            .clones
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| self.main_archive.clone());

        let res = f(&mut archive);

        self.clones.lock().unwrap().push(archive);
        res
    }
}

impl ReadableStorageTraits for Reader {
    fn get_partial_values_key(
        &self,
        key: &zarrs::storage::StoreKey,
        byte_ranges: &[zarrs::byte_range::ByteRange],
    ) -> Result<Option<Vec<zarrs::storage::Bytes>>, StorageError> {
        self.with_archive(|r| {
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
        })
    }

    fn size_key(&self, key: &zarrs::storage::StoreKey) -> Result<Option<u64>, StorageError> {
        self.with_archive(|r| {
            let file = match r.by_name(key.as_str()) {
                Ok(f) => f,
                Err(err) => match err {
                    ZipError::FileNotFound => return Ok(None),
                    _ => return Err(StorageError::Other(err.to_string())),
                },
            };
            Ok(Some(file.size()))
        })
    }
}

impl ListableStorageTraits for Reader {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.with_archive(|r| {
            Ok(r.file_names()
                .filter_map(|v| StoreKey::try_from(v).ok())
                .sorted()
                .collect())
        })
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.with_archive(|r| {
            let file_names: Vec<String> = r
                .file_names()
                .map(std::string::ToString::to_string)
                .collect();
            Ok(file_names
                .into_iter()
                .filter_map(|name| {
                    if name.starts_with(prefix.as_str()) {
                        if let Ok(file) = r.by_name(&name) {
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
        })
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.with_archive(|r| {
            let mut keys: StoreKeys = vec![];
            let mut prefixes: StorePrefixes = vec![];
            for name in r.file_names() {
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
        })
    }

    //fn size(&self) -> Result<u64, StorageError> {
    //    Ok(self.size)
    //}

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
