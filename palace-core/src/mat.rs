use std::ops::{Add, Mul, Sub};

use crate::{dim::*, vec::Vector};
use id::{Id, Identify};

// A column major matrix
#[repr(C)]
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Matrix<D: DynDimension, T: Copy> {
    inner: D::DynMatrix<T>,
    dim: D,
}

unsafe impl<D: Dimension, T: Copy + bytemuck::Zeroable> bytemuck::Zeroable for Matrix<D, T> {}
unsafe impl<D: Dimension, T: Copy + bytemuck::Pod> bytemuck::Pod for Matrix<D, T> {}
impl<D: Dimension, T: Copy> Copy for Matrix<D, T> {}

impl<D: DynDimension, T: Copy + Identify> Identify for Matrix<D, T> {
    fn id(&self) -> Id {
        Id::combine_it(self.inner.as_slice().iter().map(|v| v.id()))
    }
}

impl<D: Dimension, T: Copy> Matrix<D, T> {
    pub fn new(v: D::Array<Vector<D, T>>) -> Self {
        Self::from_fn(|row, col| v[col][row])
    }
    pub fn from_fn(f: impl FnMut(usize, usize) -> T) -> Self {
        Self::from_fn_and_dim(D::new(), f)
    }
}

impl<D: DynDimension, T: Copy> Matrix<D, T> {
    pub fn dim(&self) -> D {
        self.dim
    }
    pub fn from_fn_and_dim(dim: D, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let n = dim.n();
        Self {
            inner: D::DynMatrix::try_from_fn_and_len(n * n, |i| {
                let row = i % n;
                let col = i / n;
                f(row, col)
            })
            .unwrap(),
            dim,
        }
    }
    pub fn at(&self, row: usize, col: usize) -> T {
        let n = self.dim.n();
        let i = row + col * n;
        self.inner.as_slice()[i]
    }
    pub fn col(&self, col: usize) -> Vector<D, T> {
        let n = self.dim.n();
        let cols = self.inner.as_slice();
        let col = &cols[col * n..][..n];
        Vector::try_from_slice(col).unwrap()
    }
    pub fn zip<U: Copy, O: Copy>(
        &self,
        other: &Matrix<D, U>,
        mut f: impl FnMut(T, U) -> O,
    ) -> Matrix<D, O> {
        Matrix::from_fn_and_dim(self.dim, |row, col| {
            f(self.at(row, col), other.at(row, col))
        })
    }
    pub fn map<O: Copy>(&self, mut f: impl FnMut(T) -> O) -> Matrix<D, O> {
        Matrix::from_fn_and_dim(self.dim, |row, col| f(self.at(row, col)))
    }
}
impl<D: DynDimension, T: Copy + Mul<Output = T>> Matrix<D, T> {
    pub fn scaled_by(&self, by: T) -> Self {
        self.map(|v| v * by)
    }
}
impl<D: DynDimension, T: Copy> Matrix<D, T> {
    pub fn transposed(&self) -> Self {
        Self::from_fn_and_dim(self.dim, |row, col| self.at(col, row))
    }
}
impl<D: DynDimension, T: num::Zero + Copy> Matrix<D, T> {
    pub fn from_scale(scale: &Vector<D, T>) -> Self {
        Self::from_fn_and_dim(scale.dim(), |row, col| {
            if row == col {
                scale[row]
            } else {
                num::zero::<T>()
            }
        })
    }
}

impl<D: SmallerDim, T: num::Zero + num::One + Copy> Matrix<D, T> {
    pub fn from_hom_scale(scale: Vector<D::Smaller, T>) -> Self {
        let scale = scale.push_dim_small(num::one::<T>());
        Self::from_scale(&scale)
    }
}

impl<D: DynDimension, T: num::Zero + num::One + Copy> Matrix<D, T> {
    pub fn identity(dim: D) -> Self {
        Self::from_scale(&Vector::fill_with_len(num::one(), dim.n()))
    }
}

impl<D: SmallerDim, T: num::Zero + num::One + Copy> Matrix<D, T> {
    pub fn from_translation(translation_vec: Vector<D::Smaller, T>) -> Self {
        let translation_vec = translation_vec.push_dim_large(num::one::<T>());
        let dim = translation_vec.dim();
        Self::from_fn_and_dim(dim, |row, col| {
            if col == row {
                num::one::<T>()
            } else if col == 0 {
                translation_vec[row]
            } else {
                num::zero::<T>()
            }
        })
    }
}

impl<D: LargerDim, T: num::Zero + num::One + Copy> Matrix<D, T> {
    pub fn to_homogeneous(self) -> Matrix<D::Larger, T> {
        Matrix::<D::Larger, T>::from_fn_and_dim(self.dim.larger(), |row, col| {
            if col == 0 {
                if row == 0 {
                    num::one()
                } else {
                    num::zero()
                }
            } else {
                if row == 0 {
                    num::zero()
                } else {
                    self.at(row - 1, col - 1)
                }
            }
        })
    }
}

impl<D: SmallerDim, T: num::Zero + num::One + Copy> Matrix<D, T> {
    pub fn to_scaling_part(self) -> Matrix<D::Smaller, T> {
        Matrix::<D::Smaller, T>::from_fn_and_dim(self.dim.smaller(), |row, col| {
            self.at(row + 1, col + 1)
        })
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

impl<D: DynDimension, T: Copy + num::Zero + Add<Output = T> + Mul<Output = T>> Mul<&Vector<D, T>>
    for Matrix<D, T>
{
    type Output = Vector<D, T>;

    fn mul(self, rhs: &Vector<D, T>) -> Self::Output {
        let m = self.transposed();
        Self::Output::try_from_fn_and_len(self.dim.n(), |i| m.col(i).dot(&rhs)).unwrap()
    }
}

impl<D: SmallerDim, T: Copy + num::Zero + Add<Output = T> + Mul<Output = T> + num::One>
    Matrix<D, T>
{
    pub fn transform(self, rhs: &Vector<D::Smaller, T>) -> Vector<D::Smaller, T> {
        (self * &rhs.to_homogeneous_coord()).to_non_homogeneous_coord()
    }
}

impl<D: DynDimension, T: Copy + num::Zero + Add<Output = T> + Mul<Output = T>> Mul<&Matrix<D, T>>
    for Matrix<D, T>
{
    type Output = Matrix<D, T>;

    fn mul(self, rhs: &Matrix<D, T>) -> Self::Output {
        let lhs = self.transposed();
        Self::Output::from_fn_and_dim(self.dim, |row, col| lhs.col(row).dot(&rhs.col(col)))
    }
}

impl<D: Dimension, T: Copy + Add<Output = T>> Add<Matrix<D, T>> for Matrix<D, T> {
    type Output = Matrix<D, T>;

    fn add(self, rhs: Matrix<D, T>) -> Self::Output {
        self.zip(&rhs, |l, r| l + r)
    }
}

impl<D: Dimension, T: Copy + Sub<Output = T>> Sub<Matrix<D, T>> for Matrix<D, T> {
    type Output = Matrix<D, T>;

    fn sub(self, rhs: Matrix<D, T>) -> Self::Output {
        self.zip(&rhs, |l, r| l - r)
    }
}

impl<T: Copy> From<Matrix<D3, T>> for cgmath::Matrix3<T> {
    fn from(value: Matrix<D3, T>) -> Self {
        cgmath::Matrix3 {
            x: (value.col(2)).into(),
            y: (value.col(1)).into(),
            z: (value.col(0)).into(),
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
            x: (value.col(3)).into(),
            y: (value.col(2)).into(),
            z: (value.col(1)).into(),
            w: (value.col(0)).into(),
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
            let arr = np.as_array();
            let shape = arr.shape();
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
            numpy::PyArray2::from_owned_array_bound(
                py,
                numpy::ndarray::Array::from_shape_fn((4, 4), |(i, j)| self.at(i, j).clone()),
            )
            .into_py(py)
        }
    }

    impl<D: DynDimension, T: Copy> pyo3_stub_gen::PyStubType for Matrix<D, T> {
        fn type_output() -> pyo3_stub_gen::TypeInfo {
            pyo3_stub_gen::TypeInfo {
                name: format!("Matrix"),
                import: Default::default(),
            }
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
        assert_eq!(m * &v, r);
    }

    #[test]
    fn mul_mat_mat() {
        let m1 = Matrix::<D2, i32>::new([[1, 2].into(), [3, 4].into()]);
        let m2 = Matrix::<D2, i32>::new([[9, 8].into(), [0, -1].into()]);
        let r = Matrix::<D2, i32>::new([[9 + 24, 18 + 32].into(), [-3, -4].into()]);
        assert_eq!(m1 * &m2, r);

        let m1 = Matrix::<D2, i32>::identity(D2);
        let m2 = Matrix::<D2, i32>::new([[1, 2].into(), [3, 4].into()]);
        assert_eq!(m1 * &m2, m2);
        assert_eq!(m2 * &m1, m2);
        assert_eq!(m1 * &m1, m1);

        let m1 = Matrix::<D2, i32>::new([[1, 0].into(), [0, 0].into()]);
        let m2 = Matrix::<D2, i32>::new([[1, 2].into(), [3, 4].into()]);
        let r = Matrix::<D2, i32>::new([[1, 0].into(), [3, 0].into()]);
        assert_eq!(m1 * &m2, r);
    }

    #[test]
    fn add_mat_mat() {
        let m1 = Matrix::<D2, i32>::new([[1, 2].into(), [3, 4].into()]);
        let m2 = Matrix::<D2, i32>::new([[9, 8].into(), [0, -1].into()]);
        let r = Matrix::<D2, i32>::new([[1 + 9, 2 + 8].into(), [3, 3].into()]);
        assert_eq!(m1 + m2, r);
    }
}
