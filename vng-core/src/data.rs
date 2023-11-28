use std::mem::MaybeUninit;

use crate::array::ChunkInfo;

pub use crate::aabb::*;
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
