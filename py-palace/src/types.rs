mod core;
mod event;
mod gui;
mod splitter;
mod tensor;

pub use self::core::*;
pub use self::event::*;
pub use self::gui::*;
pub use self::splitter::*;
pub use self::tensor::*;

use pyo3::prelude::*;
use palace_core::operators::volume::ChunkSize as CChunkSize;

#[pyclass]
#[derive(Clone)]
pub struct ChunkSizeFull;

#[derive(Copy, Clone, Debug)]
pub struct ChunkSize(pub CChunkSize);

impl<'source> FromPyObject<'source> for ChunkSize {
    fn extract(val: &'source PyAny) -> PyResult<Self> {
        Ok(ChunkSize(if let Ok(_) = val.extract::<ChunkSizeFull>() {
            CChunkSize::Full
        } else {
            CChunkSize::Fixed(val.extract::<u32>()?.into())
        }))
    }
}
