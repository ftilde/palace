use std::{
    mem::MaybeUninit,
    ops::{Add, Mul, Sub},
};

use crate::{
    array::ChunkInfo,
    id::{Id, Identify},
};

pub use crate::coordinate::*;
pub use crate::mat::*;
pub use crate::vec::*;

pub fn hmul<const N: usize, T: CoordinateType>(s: Vector<N, Coordinate<T>>) -> usize {
    s.into_iter().map(|v| v.raw as usize).product()
}

pub fn to_linear<const N: usize, T: CoordinateType>(
    pos: Vector<N, Coordinate<T>>,
    dim: Vector<N, Coordinate<T>>,
) -> usize {
    let mut out = pos[0].raw as usize;
    for i in 1..N {
        out = out * dim[i].raw as usize + pos[i].raw as usize;
    }
    out
}

pub fn from_linear<const N: usize, T: CoordinateType>(
    mut linear_pos: usize,
    dim: Vector<N, Coordinate<T>>,
) -> Vector<N, Coordinate<T>> {
    let mut out = Vector::<N, Coordinate<T>>::fill(0.into());
    for i in (0..N).rev() {
        let ddim = dim[i].raw as usize;
        out[i] = ((linear_pos % ddim) as u32).into();
        linear_pos /= ddim;
    }
    out
}

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

pub struct AABB<const N: usize, T> {
    min: Vector<N, T>,
    max: Vector<N, T>,
}

fn partial_ord_min<T: PartialOrd>(v1: T, v2: T) -> T {
    if v1.lt(&v2) {
        v1
    } else {
        v2
    }
}
fn partial_ord_max<T: PartialOrd>(v1: T, v2: T) -> T {
    if v1.lt(&v2) {
        v2
    } else {
        v1
    }
}

impl<const N: usize, T: Copy + PartialOrd> AABB<N, T> {
    pub fn new(p1: Vector<N, T>, p2: Vector<N, T>) -> Self {
        Self {
            min: p1.zip(p2, partial_ord_min),
            max: p1.zip(p2, partial_ord_max),
        }
    }

    pub fn from_points(mut points: impl Iterator<Item = Vector<N, T>>) -> Self {
        let first = points.next().unwrap();
        let mut s = Self {
            min: first,
            max: first,
        };
        for p in points {
            s.add_point(p);
        }
        s
    }

    pub fn add_point(&mut self, p: Vector<N, T>) {
        self.min = self.min.zip(p, partial_ord_min);
        self.max = self.max.zip(p, partial_ord_max);
    }

    pub fn lower(&self) -> Vector<N, T> {
        self.min
    }

    pub fn upper(&self) -> Vector<N, T> {
        self.max
    }

    //pub fn contains(&self, p: Vector<N, T>) -> bool {
    //    let bigger_than_min = self.min.zip(p, |v1, v2| v1.le(&v2));
    //    let smaller_than_max = p.zip(self.max, |v1, v2| v1.lt(&v2));
    //    bigger_than_min
    //        .0
    //        .iter()
    //        .chain(smaller_than_max.0.iter())
    //        .all(|v| *v)
    //}
}
impl AABB<3, f32> {
    #[must_use]
    pub fn transform(&self, t: &Matrix<4, f32>) -> Self {
        let points = (0..8).into_iter().map(|b| {
            let p = Vector::<3, f32>::from_fn(|i| {
                if (b & (1 << i)) != 0 {
                    self.min[i]
                } else {
                    self.max[i]
                }
            });
            (*t * p.to_homogeneous_coord()).drop_dim(0)
        });
        Self::from_points(points)
    }
}

pub type LocalVoxelPosition = Vector<3, LocalCoordinate>;
pub type VoxelPosition = Vector<3, GlobalCoordinate>;
pub type BrickPosition = Vector<3, ChunkCoordinate>;

pub type PixelPosition = Vector<2, GlobalCoordinate>;

fn dimension_order_stride<T: CoordinateType>(
    mem_size: Vector<3, Coordinate<T>>,
) -> (usize, usize, usize) {
    (
        mem_size.x().raw as usize * mem_size.y().raw as usize,
        mem_size.x().raw as usize,
        1,
    )
}
pub fn contiguous_shape<T: CoordinateType>(
    size: Vector<3, Coordinate<T>>,
) -> ndarray::Shape<ndarray::Ix3> {
    ndarray::ShapeBuilder::into_shape((
        size.z().raw as usize,
        size.y().raw as usize,
        size.x().raw as usize,
    ))
}
pub fn stride_shape<T: CoordinateType>(
    size: Vector<3, Coordinate<T>>,
    mem_size: Vector<3, Coordinate<T>>,
) -> ndarray::StrideShape<ndarray::Ix3> {
    use ndarray::ShapeBuilder;
    let stride = dimension_order_stride(mem_size);
    let size = (
        size.z().raw as usize,
        size.y().raw as usize,
        size.x().raw as usize,
    );
    size.strides(stride)
}

#[allow(unused)]
pub fn slice_range<T: Into<usize> + Copy>(
    begin: Vector<3, T>,
    end: Vector<3, T>,
) -> ndarray::SliceInfo<[ndarray::SliceInfoElem; 3], ndarray::Ix3, ndarray::Ix3> {
    ndarray::s![
        begin.z().into()..end.z().into(),
        begin.y().into()..end.y().into(),
        begin.x().into()..end.x().into(),
    ]
}

#[allow(unused)]
pub fn chunk<'a, T>(data: &'a [T], brick_info: &ChunkInfo<3>) -> ndarray::ArrayView3<'a, T> {
    if brick_info.is_contiguous() {
        ndarray::ArrayView3::from_shape(contiguous_shape(brick_info.logical_dimensions), data)
    } else {
        ndarray::ArrayView3::from_shape(
            stride_shape(brick_info.logical_dimensions, brick_info.mem_dimensions),
            data,
        )
    }
    .unwrap()
}

pub fn chunk_mut<'a, T>(
    data: &'a mut [T],
    brick_info: &ChunkInfo<3>,
) -> ndarray::ArrayViewMut3<'a, T> {
    if brick_info.is_contiguous() {
        ndarray::ArrayViewMut3::from_shape(contiguous_shape(brick_info.logical_dimensions), data)
    } else {
        ndarray::ArrayViewMut3::from_shape(
            stride_shape(brick_info.logical_dimensions, brick_info.mem_dimensions),
            data,
        )
    }
    .unwrap()
}

pub fn init_non_full<const N: usize, T: Clone>(
    data: &mut [std::mem::MaybeUninit<T>],
    chunk_info: &ChunkInfo<N>,
    val: T,
) {
    if !chunk_info.is_full() {
        for v in data.iter_mut() {
            v.write(val.clone());
        }
    }
}

// Unstable function copied from stdlib:
// https://doc.rust-lang.org/stable/std/mem/union.MaybeUninit.html#method.slice_assume_init_mut
pub unsafe fn slice_assume_init_mut<T>(slice: &mut [MaybeUninit<T>]) -> &mut [T] {
    // SAFETY: similar to safety notes for `slice_get_ref`, but we have a
    // mutable reference which is also guaranteed to be valid for writes.
    unsafe { &mut *(slice as *mut [MaybeUninit<T>] as *mut [T]) }
}

// Unstable function copied from stdlib:
// https://doc.rust-lang.org/stable/std/mem/union.MaybeUninit.html#method.write_slice
pub fn write_slice_uninit<'a, T>(this: &'a mut [MaybeUninit<T>], src: &[T]) -> &'a mut [T]
where
    T: Copy,
{
    // SAFETY: &[T] and &[MaybeUninit<T>] have the same layout
    let uninit_src: &[MaybeUninit<T>] = unsafe { std::mem::transmute(src) };

    this.copy_from_slice(uninit_src);

    // SAFETY: Valid elements have just been copied into `this` so it is initialized
    unsafe { slice_assume_init_mut(this) }
}

#[allow(unused)]
pub fn fill_uninit<T: Clone>(data: &mut [MaybeUninit<T>], val: T) -> &mut [T] {
    for v in data.iter_mut() {
        v.write(val.clone());
    }
    unsafe { slice_assume_init_mut(data) }
}

//pub struct Brick<'a, T> {
//    size: LocalVoxelPosition,
//    mem_size: LocalVoxelPosition,
//    data: &'a [T],
//}
//
//impl<'a, T: Copy> Brick<'a, T> {
//    pub fn new(data: &'a [T], size: LocalVoxelPosition, mem_size: LocalVoxelPosition) -> Self {
//        Self {
//            data,
//            size,
//            mem_size,
//        }
//    }
//    pub fn row(&'a self, index: Vector<2, LocalVoxelCoordinate>) -> &'a [T] {
//        let begin = to_linear(index, [self.mem_size.z(), self.mem_size.y()].into())
//            * self.mem_size.x().raw as usize;
//        let end = begin + self.mem_size.x().raw as usize;
//        &self.data[begin..end]
//    }
//    pub fn rows(&'a self) -> impl Iterator<Item = &'a [T]> + 'a {
//        itertools::iproduct! { 0..self.size.z().raw, 0..self.size.y().raw }
//            .map(|(z, y)| self.row([z.into(), y.into()].into()))
//    }
//    pub fn voxels(&'a self) -> impl Iterator<Item = T> + 'a {
//        itertools::iproduct! { 0..self.size.z().raw, 0..self.size.y().raw, 0..self.size.x().raw }
//            .map(|(z, y, x)| to_linear([z.into(), y.into(), x.into()].into(), self.mem_size))
//            .map(|i| self.data[i])
//    }
//}
//

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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn dot() {
        let v1 = Vector::<5, usize>::from_fn(|i| (i + 1) * 5);
        let v2 = Vector::<5, usize>::new([1, 0, 1, 0, 2]);
        assert_eq!(v1.dot(&v2), 5 + 15 + 50);
    }

    #[test]
    fn mul_mat_vec() {
        let v = Vector::<2, usize>::new([5, 2]);
        let m = Matrix::<2, usize>::new([[1usize, 2].into(), [3usize, 4].into()]);
        let r = Vector::<2, usize>::new([5 + 6, 10 + 8]);
        assert_eq!(m * v, r);
    }

    #[test]
    fn mul_mat_mat() {
        let m1 = Matrix::<2, i32>::new([[1, 2].into(), [3, 4].into()]);
        let m2 = Matrix::<2, i32>::new([[9, 8].into(), [0, -1].into()]);
        let r = Matrix::<2, i32>::new([[9 + 24, 18 + 32].into(), [-3, -4].into()]);
        assert_eq!(m1 * m2, r);

        let m1 = Matrix::<2, i32>::identity();
        let m2 = Matrix::<2, i32>::new([[1, 2].into(), [3, 4].into()]);
        assert_eq!(m1 * m2, m2);
        assert_eq!(m2 * m1, m2);
        assert_eq!(m1 * m1, m1);

        let m1 = Matrix::<2, i32>::new([[1, 0].into(), [0, 0].into()]);
        let m2 = Matrix::<2, i32>::new([[1, 2].into(), [3, 4].into()]);
        let r = Matrix::<2, i32>::new([[1, 0].into(), [3, 0].into()]);
        assert_eq!(m1 * m2, r);
    }

    #[test]
    fn add_mat_mat() {
        let m1 = Matrix::<2, i32>::new([[1, 2].into(), [3, 4].into()]);
        let m2 = Matrix::<2, i32>::new([[9, 8].into(), [0, -1].into()]);
        let r = Matrix::<2, i32>::new([[1 + 9, 2 + 8].into(), [3, 3].into()]);
        assert_eq!(m1 + m2, r);
    }
}
