use std::mem::MaybeUninit;

use crate::array::ChunkInfo;

pub use crate::aabb::*;
pub use crate::coordinate::*;
use crate::dim::*;
pub use crate::mat::*;
pub use crate::vec::*;

fn dimension_order_stride<D: DynDimension, T: CoordinateType>(
    mem_size: &Vector<D, Coordinate<T>>,
) -> D::NDArrayDimDyn {
    let nd = mem_size.len();
    let mem_size = mem_size.as_index();
    let mut out = Vector::<D, usize>::fill_with_len(1usize, nd);
    let mut rol = 1;
    for i in (0..nd).rev() {
        out[i] = rol;
        rol *= mem_size[i];
    }
    D::to_ndarray_dim_dyn(out.inner())
}
pub fn contiguous_shape<D: DynDimension, T: CoordinateType>(
    size: &Vector<D, Coordinate<T>>,
) -> ndarray::Shape<D::NDArrayDimDyn> {
    size.to_ndarray_dim().into()
}
pub fn stride_shape<D: DynDimension, T: CoordinateType>(
    size: &Vector<D, Coordinate<T>>,
    mem_size: &Vector<D, Coordinate<T>>,
) -> ndarray::StrideShape<D::NDArrayDimDyn> {
    use ndarray::ShapeBuilder;
    let stride = dimension_order_stride(mem_size);

    let size: ndarray::Shape<D::NDArrayDimDyn> = size.to_ndarray_dim().into();
    size.strides(stride)
}

#[allow(unused)]
pub fn slice_range<T: Into<usize> + Copy>(
    begin: Vector<D3, T>,
    end: Vector<D3, T>,
) -> ndarray::SliceInfo<[ndarray::SliceInfoElem; 3], ndarray::Ix3, ndarray::Ix3> {
    ndarray::s![
        begin.z().into()..end.z().into(),
        begin.y().into()..end.y().into(),
        begin.x().into()..end.x().into(),
    ]
}

#[allow(unused)]
pub fn chunk<'a, D: DynDimension, T>(
    data: &'a [T],
    brick_info: &ChunkInfo<D>,
) -> ndarray::ArrayView<'a, T, D::NDArrayDimDyn> {
    if brick_info.is_contiguous() {
        ndarray::ArrayView::from_shape(contiguous_shape(&brick_info.logical_dimensions), data)
    } else {
        ndarray::ArrayView::from_shape(
            stride_shape(&brick_info.logical_dimensions, &brick_info.mem_dimensions),
            data,
        )
    }
    .unwrap()
}

pub fn chunk_mut<'a, D: DynDimension, T>(
    data: &'a mut [T],
    brick_info: &ChunkInfo<D>,
) -> ndarray::ArrayViewMut<'a, T, D::NDArrayDimDyn> {
    if brick_info.is_contiguous() {
        ndarray::ArrayViewMut::from_shape(contiguous_shape(&brick_info.logical_dimensions), data)
    } else {
        ndarray::ArrayViewMut::from_shape(
            stride_shape(&brick_info.logical_dimensions, &brick_info.mem_dimensions),
            data,
        )
    }
    .unwrap()
}

pub fn init_non_full<D: DynDimension, T: Clone>(
    data: &mut [std::mem::MaybeUninit<T>],
    chunk_info: &ChunkInfo<D>,
    val: T,
) {
    if !chunk_info.is_full() {
        for v in data.iter_mut() {
            v.write(val.clone());
        }
    }
}

pub fn init_non_full_raw<D: Dimension>(
    data: &mut [std::mem::MaybeUninit<u8>],
    chunk_info: &ChunkInfo<D>,
    val: u8,
) {
    if !chunk_info.is_full() {
        for v in data.iter_mut() {
            v.write(val);
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

#[allow(unused)]
pub fn fill_uninit_default<T: Default>(data: &mut [MaybeUninit<T>]) -> &mut [T] {
    for v in data.iter_mut() {
        v.write(T::default());
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
