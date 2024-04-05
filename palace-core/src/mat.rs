use std::ops::{Add, Mul, Sub};

use crate::{dim::*, vec::Vector};
use id::{Id, Identify};

// A column major matrix
#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Matrix<D: Dimension, T: Copy>(Vector<D, Vector<D, T>>);

impl<D: Dimension, T: Copy + Identify> Identify for Matrix<D, T> {
    fn id(&self) -> Id {
        Id::combine_it(self.0.into_iter().map(|v| v.id()))
    }
}

impl<D: Dimension, T: Copy> Matrix<D, T> {
    pub fn new(v: D::Array<Vector<D, T>>) -> Self {
        Self::from_col_fn(|i| v[i])
    }
    pub fn from_col_fn(f: impl FnMut(usize) -> Vector<D, T>) -> Self {
        Self(Vector::from_fn(f))
    }
    pub fn from_fn(mut f: impl FnMut(usize, usize) -> T) -> Self {
        Self(Vector::from_fn(|col| Vector::from_fn(|row| f(row, col))))
    }

    pub fn at(&self, row: usize, col: usize) -> &T {
        &self.0[col][row]
    }
    pub fn col(&self, col: usize) -> &Vector<D, T> {
        &self.0[col]
    }
    pub fn zip<U: Copy, O: Copy>(
        &self,
        other: &Matrix<D, U>,
        mut f: impl FnMut(&T, &U) -> O,
    ) -> Matrix<D, O> {
        Matrix::from_fn(|row, col| f(self.at(row, col), other.at(row, col)))
    }
    pub fn map<O: Copy>(&self, mut f: impl FnMut(&T) -> O) -> Matrix<D, O> {
        Matrix::from_fn(|row, col| f(self.at(row, col)))
    }
}
impl<D: Dimension, T: Copy + Mul<Output = T>> Matrix<D, T> {
    pub fn scaled_by(&self, by: T) -> Self {
        self.map(|v| *v * by)
    }
}
impl<D: Dimension, T: Copy> Matrix<D, T> {
    pub fn transposed(&self) -> Self {
        Self::from_fn(|row, col| *self.at(col, row))
    }
}
impl<D: Dimension, T: num::Zero + Copy> Matrix<D, T> {
    pub fn from_scale(scale: Vector<D, T>) -> Self {
        Self::from_col_fn(|i| {
            let m = scale[i];
            let mut v = Vector::fill(num::zero::<T>());
            v[i] = m;
            v
        })
    }
}

impl<T: num::Zero + Copy> Matrix<D4, T> {
    pub fn from_hom_scale(scale: Vector<D3, T>) -> Self {
        Self::from_col_fn(|i| {
            let m = scale[i];
            let mut v = Vector::fill(num::zero::<T>());
            v[i] = m;
            v
        })
    }
}

impl<D: Dimension, T: num::Zero + num::One + Copy> Matrix<D, T> {
    pub fn identity() -> Self {
        Self::from_col_fn(|i| {
            let mut v = Vector::fill(num::zero::<T>());
            v[i] = T::one();
            v
        })
    }
}

impl<D: SmallerDim, T: num::Zero + num::One + Copy> Matrix<D, T> {
    pub fn from_translation(translation_vec: Vector<D::Smaller, T>) -> Self {
        Self::from_col_fn(|i| {
            if i == 0 {
                translation_vec.push_dim_large(T::one())
            } else {
                Vector::fill(num::zero::<T>()).map_element(i, |_| T::one())
            }
        })
    }
}

impl<D: LargerDim, T: num::Zero + num::One + Copy> Matrix<D, T> {
    pub fn to_homogeneous(self) -> Matrix<D::Larger, T> {
        Matrix::<D::Larger, T>::from_col_fn(|i| {
            if i == 0 {
                let v = Vector::<D, T>::fill(T::zero());
                v.to_homogeneous_coord()
            } else {
                let v = self
                    .col(i - 1)
                    .to_homogeneous_coord()
                    .map_element(0, |_| num::zero());
                v
            }
        })
    }
}

impl<D: SmallerDim, T: num::Zero + num::One + Copy> Matrix<D, T> {
    pub fn to_scaling_part(self) -> Matrix<D::Smaller, T> {
        Matrix::<D::Smaller, T>::from_col_fn(|i| self.col(i + 1).drop_dim(0))
    }
}

impl Matrix<D4, f32> {
    pub fn invert(self) -> Option<Self> {
        use cgmath::SquareMatrix;
        let m: cgmath::Matrix4<f32> = self.into();
        m.invert().map(|m| m.into())
    }

    pub fn from_angle_y(angle_rad: f32) -> Self {
        cgmath::Matrix4::from_angle_y(cgmath::Rad(angle_rad)).into()
    }
}

impl<D: Dimension, T: Copy + num::Zero + Add<Output = T> + Mul<Output = T>> Mul<Vector<D, T>>
    for Matrix<D, T>
{
    type Output = Vector<D, T>;

    fn mul(self, rhs: Vector<D, T>) -> Self::Output {
        let m = self.transposed();
        Self::Output::from_fn(|i| m.col(i).dot(&rhs))
    }
}

impl<D: SmallerDim, T: Copy + num::Zero + Add<Output = T> + Mul<Output = T> + num::One>
    Matrix<D, T>
{
    pub fn transform(self, rhs: Vector<D::Smaller, T>) -> Vector<D::Smaller, T> {
        (self * rhs.to_homogeneous_coord()).to_non_homogeneous_coord()
    }
}

impl<D: Dimension, T: Copy + num::Zero + Add<Output = T> + Mul<Output = T>> Mul<Matrix<D, T>>
    for Matrix<D, T>
{
    type Output = Matrix<D, T>;

    fn mul(self, rhs: Matrix<D, T>) -> Self::Output {
        let lhs = self.transposed();
        Self::Output::from_fn(|row, col| lhs.col(row).dot(rhs.col(col)))
    }
}

impl<D: Dimension, T: Copy + Add<Output = T>> Add<Matrix<D, T>> for Matrix<D, T> {
    type Output = Matrix<D, T>;

    fn add(self, rhs: Matrix<D, T>) -> Self::Output {
        self.zip(&rhs, |l, r| *l + *r)
    }
}

impl<D: Dimension, T: Copy + Sub<Output = T>> Sub<Matrix<D, T>> for Matrix<D, T> {
    type Output = Matrix<D, T>;

    fn sub(self, rhs: Matrix<D, T>) -> Self::Output {
        self.zip(&rhs, |l, r| *l - *r)
    }
}

impl<T: Copy> From<Matrix<D3, T>> for cgmath::Matrix3<T> {
    fn from(value: Matrix<D3, T>) -> Self {
        cgmath::Matrix3 {
            x: (*value.col(2)).into(),
            y: (*value.col(1)).into(),
            z: (*value.col(0)).into(),
        }
    }
}

impl<T: Copy> From<cgmath::Matrix3<T>> for Matrix<D3, T> {
    fn from(value: cgmath::Matrix3<T>) -> Self {
        Self::new([value.z.into(), value.y.into(), value.x.into()])
    }
}

impl<T: Copy> From<Matrix<D4, T>> for cgmath::Matrix4<T> {
    fn from(value: Matrix<D4, T>) -> Self {
        cgmath::Matrix4 {
            x: (*value.col(3)).into(),
            y: (*value.col(2)).into(),
            z: (*value.col(1)).into(),
            w: (*value.col(0)).into(),
        }
    }
}

impl<T: Copy> From<cgmath::Matrix4<T>> for Matrix<D4, T> {
    fn from(value: cgmath::Matrix4<T>) -> Self {
        Self::new([
            value.w.into(),
            value.z.into(),
            value.y.into(),
            value.x.into(),
        ])
    }
}

#[cfg(feature = "python")]
mod py {
    use super::*;
    use pyo3::prelude::*;

    impl<'source, D: Dimension, T: Copy + numpy::Element> FromPyObject<'source> for Matrix<D, T> {
        fn extract(ob: &'source PyAny) -> PyResult<Self> {
            let np = ob.extract::<numpy::borrow::PyReadonlyArray2<T>>()?;
            let shape = np.shape();
            let n = D::N;
            if shape != [n, n] {
                let s0 = shape[0];
                let s1 = shape[1];
                return Err(PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "Expected a matrix of shape ({n},{n}), but got one of shape ({s0},{s1})"
                )));
            }
            Ok(Matrix::from_fn(|i, j| np.get((i, j)).unwrap().clone()))
        }
    }

    impl<D: Dimension, T: Copy + numpy::Element> IntoPy<PyObject> for Matrix<D, T> {
        fn into_py(self, py: Python<'_>) -> PyObject {
            numpy::PyArray2::from_owned_array(
                py,
                numpy::ndarray::Array::from_shape_fn((4, 4), |(i, j)| self.at(i, j).clone()),
            )
            .into_py(py)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn dot() {
        let v1 = Vector::<D5, usize>::from_fn(|i| (i + 1) * 5);
        let v2 = Vector::<D5, usize>::new([1, 0, 1, 0, 2]);
        assert_eq!(v1.dot(&v2), 5 + 15 + 50);
    }

    #[test]
    fn mul_mat_vec() {
        let v = Vector::<D2, usize>::new([5, 2]);
        let m = Matrix::<D2, usize>::new([[1usize, 2].into(), [3usize, 4].into()]);
        let r = Vector::<D2, usize>::new([5 + 6, 10 + 8]);
        assert_eq!(m * v, r);
    }

    #[test]
    fn mul_mat_mat() {
        let m1 = Matrix::<D2, i32>::new([[1, 2].into(), [3, 4].into()]);
        let m2 = Matrix::<D2, i32>::new([[9, 8].into(), [0, -1].into()]);
        let r = Matrix::<D2, i32>::new([[9 + 24, 18 + 32].into(), [-3, -4].into()]);
        assert_eq!(m1 * m2, r);

        let m1 = Matrix::<D2, i32>::identity();
        let m2 = Matrix::<D2, i32>::new([[1, 2].into(), [3, 4].into()]);
        assert_eq!(m1 * m2, m2);
        assert_eq!(m2 * m1, m2);
        assert_eq!(m1 * m1, m1);

        let m1 = Matrix::<D2, i32>::new([[1, 0].into(), [0, 0].into()]);
        let m2 = Matrix::<D2, i32>::new([[1, 2].into(), [3, 4].into()]);
        let r = Matrix::<D2, i32>::new([[1, 0].into(), [3, 0].into()]);
        assert_eq!(m1 * m2, r);
    }

    #[test]
    fn add_mat_mat() {
        let m1 = Matrix::<D2, i32>::new([[1, 2].into(), [3, 4].into()]);
        let m2 = Matrix::<D2, i32>::new([[9, 8].into(), [0, -1].into()]);
        let r = Matrix::<D2, i32>::new([[1 + 9, 2 + 8].into(), [3, 3].into()]);
        assert_eq!(m1 + m2, r);
    }
}
