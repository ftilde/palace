use super::tensor::TensorOperator;

pub type ArrayOperator<'op> = TensorOperator<'op, 1>;

pub fn from_static<'op>(values: &'op [f32]) -> ArrayOperator<'op> {
    super::tensor::from_static([values.len() as u32].into(), values).unwrap()
}
