use pyo3::prelude::*;
use vng_core::{array::ImageMetaData, data::Vector, operators::splitter as c};

use super::{Events, VolumeOperator};

#[pyclass(unsendable)]
pub struct Splitter(c::Splitter);

#[pymethods]
impl Splitter {
    #[new]
    fn new(size: [u32; 2], split_pos: f32) -> Self {
        Splitter(c::Splitter::new(Vector::from(size).into(), split_pos))
    }

    fn split_events(&mut self, e: &mut Events) -> (Events, Events) {
        let (l, r) = self.0.split_events(&mut e.0);

        (Events(l), Events(r))
    }

    fn render(&self, input_l: VolumeOperator, input_r: VolumeOperator) -> VolumeOperator {
        self.0.clone().render(input_l.into(), input_r.into()).into()
    }

    fn metadata_l(&self) -> ImageMetaData {
        self.0.metadata_l().into()
    }

    fn metadata_r(&self) -> ImageMetaData {
        self.0.metadata_r().into()
    }
}
