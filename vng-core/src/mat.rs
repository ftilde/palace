use std::ops::{Add, Mul, Sub};

use crate::{
    id::{Id, Identify},
    vec::Vector,
};

// A column major matrix
#[repr(transparent)]
#[derive(
    Copy, Clone, Hash, PartialEq, Eq, Debug, bytemuck::Pod, bytemuck::Zeroable, state_link::State,
)]
pub struct Matrix<const N: usize, T>([Vector<N, T>; N]);

impl<const N: usize, T: Identify> Identify for Matrix<N, T> {
    fn id(&self) -> Id {
        (&self.0[..]).id()
    }
}

impl<const N: usize, T> Matrix<N, T> {
    pub fn new(v: [Vector<N, T>; N]) -> Self {
        Self(v)
    }
    pub fn from_col_fn(f: impl FnMut(usize) -> Vector<N, T>) -> Self {
        Self(std::array::from_fn(f))
    }
    pub fn from_fn(mut f: impl FnMut(usize, usize) -> T) -> Self {
        Self(std::array::from_fn(|col| {
            Vector::from_fn(|row| f(row, col))
        }))
    }

    pub fn at(&self, row: usize, col: usize) -> &T {
        &self.0[col][row]
    }
    pub fn col(&self, col: usize) -> &Vector<N, T> {
        &self.0[col]
    }
    pub fn zip<U, O>(&self, other: &Matrix<N, U>, mut f: impl FnMut(&T, &U) -> O) -> Matrix<N, O> {
        Matrix::from_fn(|row, col| f(self.at(row, col), other.at(row, col)))
    }
    pub fn map<O>(&self, mut f: impl FnMut(&T) -> O) -> Matrix<N, O> {
        Matrix::from_fn(|row, col| f(self.at(row, col)))
    }
}
impl<const N: usize, T: Copy + Mul<Output = T>> Matrix<N, T> {
    pub fn scaled_by(&self, by: T) -> Self {
        self.map(|v| *v * by)
    }
}
impl<const N: usize, T: Copy> Matrix<N, T> {
    pub fn transposed(&self) -> Self {
        Self::from_fn(|row, col| *self.at(col, row))
    }
}
impl<const N: usize, T: num::Zero + Copy> Matrix<N, T> {
    pub fn from_scale(scale: Vector<N, T>) -> Self {
        Self::from_col_fn(|i| {
            let m = scale[i];
            let mut v = Vector::fill(num::zero::<T>());
            v[i] = m;
            v
        })
    }
}

impl<T: num::Zero + Copy> Matrix<4, T> {
    pub fn from_hom_scale(scale: Vector<3, T>) -> Self {
        Self::from_col_fn(|i| {
            let m = scale[i];
            let mut v = Vector::fill(num::zero::<T>());
            v[i] = m;
            v
        })
    }
}

impl<const N: usize, T: num::Zero + num::One + Copy> Matrix<N, T> {
    pub fn identity() -> Self {
        Self::from_col_fn(|i| {
            let mut v = Vector::fill(num::zero::<T>());
            v[i] = T::one();
            v
        })
    }
}

impl<T: num::Zero + num::One + Copy> Matrix<4, T> {
    pub fn from_translation(translation_vec: Vector<3, T>) -> Self {
        Self::from_col_fn(|i| {
            if i == 0 {
                translation_vec.push_dim_large(T::one())
            } else {
                Vector::fill(num::zero::<T>()).map_element(i, |_| T::one())
            }
        })
    }
}

impl<T: num::Zero + num::One + Copy> Matrix<3, T> {
    pub fn to_homogeneuous(self) -> Matrix<4, T> {
        Matrix::<4, T>::from_col_fn(|i| {
            if i == 0 {
                let v = Vector::<3, T>::fill(T::zero());
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

impl<T: num::Zero + num::One + Copy> Matrix<4, T> {
    pub fn to_scaling_part(self) -> Matrix<3, T> {
        Matrix::<3, T>::from_col_fn(|i| self.col(i + 1).drop_dim(0))
    }
}

impl Matrix<4, f32> {
    pub fn invert(self) -> Option<Self> {
        use cgmath::SquareMatrix;
        let m: cgmath::Matrix4<f32> = self.into();
        m.invert().map(|m| m.into())
    }

    pub fn from_angle_y(angle_rad: f32) -> Self {
        cgmath::Matrix4::from_angle_y(cgmath::Rad(angle_rad)).into()
    }
}

impl<const N: usize, T: Copy + Add<Output = T> + Mul<Output = T>> Mul<Vector<N, T>>
    for Matrix<N, T>
{
    type Output = Vector<N, T>;

    fn mul(self, rhs: Vector<N, T>) -> Self::Output {
        let m = self.transposed();
        Self::Output::from_fn(|i| m.col(i).dot(&rhs))
    }
}

impl<T: Copy + Add<Output = T> + Mul<Output = T> + num::One> Matrix<4, T> {
    pub fn transform(self, rhs: Vector<3, T>) -> Vector<3, T> {
        (self * rhs.to_homogeneous_coord()).to_non_homogeneous_coord()
    }
}

impl<const N: usize, T: Copy + Add<Output = T> + Mul<Output = T>> Mul<Matrix<N, T>>
    for Matrix<N, T>
{
    type Output = Matrix<N, T>;

    fn mul(self, rhs: Matrix<N, T>) -> Self::Output {
        let lhs = self.transposed();
        Self::Output::from_fn(|row, col| lhs.col(row).dot(rhs.col(col)))
    }
}

impl<const N: usize, T: Copy + Add<Output = T>> Add<Matrix<N, T>> for Matrix<N, T> {
    type Output = Matrix<N, T>;

    fn add(self, rhs: Matrix<N, T>) -> Self::Output {
        self.zip(&rhs, |l, r| *l + *r)
    }
}

impl<const N: usize, T: Copy + Sub<Output = T>> Sub<Matrix<N, T>> for Matrix<N, T> {
    type Output = Matrix<N, T>;

    fn sub(self, rhs: Matrix<N, T>) -> Self::Output {
        self.zip(&rhs, |l, r| *l - *r)
    }
}

impl<T: Copy> From<Matrix<4, T>> for cgmath::Matrix4<T> {
    fn from(value: Matrix<4, T>) -> Self {
        cgmath::Matrix4 {
            x: (*value.col(3)).into(),
            y: (*value.col(2)).into(),
            z: (*value.col(1)).into(),
            w: (*value.col(0)).into(),
        }
    }
}

impl<T: Copy> From<cgmath::Matrix4<T>> for Matrix<4, T> {
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

    // Matrix
    impl<'source, const N: usize, T: numpy::Element> FromPyObject<'source> for Matrix<N, T> {
        fn extract(ob: &'source PyAny) -> PyResult<Self> {
            let np = ob.extract::<numpy::borrow::PyReadonlyArray2<T>>()?;
            let shape = np.shape();
            if shape != [N, N] {
                let s0 = shape[0];
                let s1 = shape[1];
                return Err(PyErr::new::<pyo3::exceptions::PyException, _>(format!(
                    "Expected a matrix of shape ({N},{N}), but got one of shape ({s0},{s1})"
                )));
            }
            Ok(Matrix::from_fn(|i, j| np.get((i, j)).unwrap().clone()))
        }
    }

    impl<const N: usize, T: numpy::Element> IntoPy<PyObject> for Matrix<N, T> {
        fn into_py(self, py: Python<'_>) -> PyObject {
            numpy::PyArray2::from_owned_array(
                py,
                numpy::ndarray::Array::from_shape_fn((4, 4), |(i, j)| self.at(i, j).clone()),
            )
            .into_py(py)
        }
    }
}
