mod core;
mod event;
mod gui;
mod splitter;
mod tensor;
mod transfer_function;

pub use self::core::*;
pub use self::event::*;
pub use self::gui::*;
pub use self::splitter::*;
pub use self::tensor::*;
pub use self::transfer_function::*;

use palace_core::operators::volume::ChunkSize as CChunkSize;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyclass;

#[gen_stub_pyclass]
#[pyclass]
#[derive(Clone)]
pub struct ChunkSizeFull;

#[gen_stub_pyclass]
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
