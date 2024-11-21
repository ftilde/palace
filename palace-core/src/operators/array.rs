use std::rc::Rc;

use crate::{dim::*, dtypes::StaticElementType, storage::Element};
use id::Identify;

use super::tensor::TensorOperator;

pub type ArrayOperator<E> = TensorOperator<D1, E>;

pub fn from_static<E: Element + Identify>(
    values: &'static [E],
) -> ArrayOperator<StaticElementType<E>> {
    ArrayOperator::from_static([values.len() as u32].into(), values).unwrap()
}

pub fn from_rc<E: Element + Identify>(values: Rc<[E]>) -> ArrayOperator<StaticElementType<E>> {
    ArrayOperator::from_rc([values.len() as u32].into(), values).unwrap()
}

pub fn from_vec<E: Element + Identify>(values: Vec<E>) -> ArrayOperator<StaticElementType<E>> {
    from_rc(Rc::from(values))
}
