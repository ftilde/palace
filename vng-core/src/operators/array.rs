use std::rc::Rc;

use crate::{id::Identify, storage::Element};

use super::tensor::TensorOperator;

pub type ArrayOperator<E> = TensorOperator<1, E>;

pub fn from_static<E: Element + Identify>(values: &'static [E]) -> ArrayOperator<E> {
    super::tensor::from_static([values.len() as u32].into(), values).unwrap()
}

pub fn from_rc<E: Element + Identify>(values: Rc<[E]>) -> ArrayOperator<E> {
    super::tensor::from_rc([values.len() as u32].into(), values).unwrap()
}
