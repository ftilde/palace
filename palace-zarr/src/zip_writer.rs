use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    sync::Mutex,
};

use palace_core::Error;
use zarrs::storage::WritableStorageTraits;
use zip::write::{FileOptions, SimpleFileOptions, ZipWriter};

pub struct ZipWriterStore {
    path: PathBuf,
    inner: Mutex<Option<ZipWriter<File>>>,
    write_options: SimpleFileOptions,
}

impl ZipWriterStore {
    pub fn new(path: &Path) -> Result<Self, Error> {
        let file = File::create(path)?;
        //let file = BufWriter::new(file);
        let writer = ZipWriter::new(file);
        let write_options =
            FileOptions::default().compression_method(zip::CompressionMethod::Stored);
        Ok(Self {
            path: path.to_owned(),
            inner: Mutex::new(Some(writer)),
            write_options,
        })
    }
}

pub trait FlushWriteStore: WritableStorageTraits {
    fn flush(&self) -> Result<(), Error>;
}

impl FlushWriteStore for zarrs::filesystem::FilesystemStore {
    fn flush(&self) -> Result<(), Error> {
        Ok(())
    }
}

impl FlushWriteStore for ZipWriterStore {
    fn flush(&self) -> Result<(), Error> {
        let mut writer = self.inner.lock().unwrap();
        let reader = writer.take().unwrap().finish_into_readable()?;
        let archive_offset = reader.offset();
        std::mem::drop(reader);

        let new_file = File::options().read(true).write(true).open(&self.path)?;
        let new_writer = ZipWriter::new_append_with_config(
            zip::read::Config {
                archive_offset: zip::read::ArchiveOffset::Known(archive_offset),
            },
            new_file,
        )?;

        *writer = Some(new_writer);
        Ok(())
    }
}

impl WritableStorageTraits for ZipWriterStore {
    fn set(
        &self,
        key: &zarrs::storage::StoreKey,
        value: zarrs::storage::Bytes,
    ) -> Result<(), zarrs::storage::StorageError> {
        let mut writer = self.inner.lock().unwrap();
        let writer = writer.as_mut().unwrap();

        let path = PathBuf::from(key.as_str());
        if let Some(parent) = path.parent() {
            let _ = writer.add_directory_from_path(parent, self.write_options);
        }

        writer
            .start_file_from_path(path, self.write_options)
            .unwrap();

        writer.write_all(value.iter().as_slice()).unwrap();

        Ok(())
    }

    fn set_partial_values(
        &self,
        _key_offset_values: &[zarrs::storage::StoreKeyOffsetValue],
    ) -> Result<(), zarrs::storage::StorageError> {
        todo!()
    }

    fn erase(&self, _key: &zarrs::storage::StoreKey) -> Result<(), zarrs::storage::StorageError> {
        todo!()
    }

    fn erase_prefix(
        &self,
        _prefix: &zarrs::storage::StorePrefix,
    ) -> Result<(), zarrs::storage::StorageError> {
        todo!()
    }
}
