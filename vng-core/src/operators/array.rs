use std::rc::Rc;

use super::tensor::TensorOperator;

pub type ArrayOperator = TensorOperator<1>;

pub fn from_static(values: &'static [f32]) -> ArrayOperator {
    super::tensor::from_static([values.len() as u32].into(), values).unwrap()
}

pub fn from_rc(values: Rc<[f32]>) -> ArrayOperator {
    super::tensor::from_rc([values.len() as u32].into(), values).unwrap()
}
