use derive_more::{From, Into};
use pyo3::prelude::*;

#[pyclass(unsendable)]
#[derive(From, Into)]
pub struct Events(pub vng_core::event::EventStream);
