use palace_core::{array::PyTensorMetaData, data::Vector, operators::splitter as c};
use pyo3::prelude::*;

use super::{Events, TensorOperator};

#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(unsendable)]
pub struct Splitter(c::Splitter);

//#[pyo3_stub_gen::derive::gen_stub_pymethods] doesn't work because of mutable reference to events
#[pymethods]
impl Splitter {
    #[new]
    fn new(size: [u32; 2], split_pos: f32, split_dir: c::SplitDirection) -> Self {
        Splitter(c::Splitter::new(
            Vector::from(size).into(),
            split_pos,
            split_dir,
        ))
    }

    fn split_events(&mut self, e: &mut Events) -> (Events, Events) {
        let (l, r) = self.0.split_events(&mut e.0);

        (Events(l), Events(r))
    }

    fn render(&self, input_l: TensorOperator, input_r: TensorOperator) -> PyResult<TensorOperator> {
        Ok(self
            .0
            .clone()
            .render(
                input_l.try_into_core_static()?.try_into()?,
                input_r.try_into_core_static()?.try_into()?,
            )
            .into_dyn()
            .into())
    }

    fn metadata_l(&self) -> PyTensorMetaData {
        self.0.metadata_first().into()
    }

    fn metadata_r(&self) -> PyTensorMetaData {
        self.0.metadata_last().into()
    }
}
