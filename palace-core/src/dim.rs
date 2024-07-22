use std::fmt::Debug;
use std::hash::Hash;

pub trait Array<T: Sized + Copy>:
    std::ops::IndexMut<usize, Output = T>
    + std::ops::Index<usize, Output = T>
    + IntoIterator<Item = T>
    + Copy
    + DynArray<T>
{
    const N: usize;
    fn from_fn(f: impl FnMut(usize) -> T) -> Self;
}

impl<const N: usize, T: Copy> Array<T> for [T; N] {
    const N: usize = N;
    fn from_fn(f: impl FnMut(usize) -> T) -> Self {
        std::array::from_fn(f)
    }
}

pub trait DynArray<T: Sized + Copy>:
    std::ops::IndexMut<usize, Output = T>
    + std::ops::Index<usize, Output = T>
    + IntoIterator<Item = T>
    + Clone
{
    fn try_from_fn_and_len(f: impl FnMut(usize) -> T, size: usize) -> Result<Self, ()>;
    fn len(&self) -> usize;
    fn as_slice(&self) -> &[T];
}

impl<const N: usize, T: Copy> DynArray<T> for [T; N] {
    fn try_from_fn_and_len(f: impl FnMut(usize) -> T, size: usize) -> Result<Self, ()> {
        if size == N {
            Ok(std::array::from_fn(f))
        } else {
            Err(())
        }
    }
    fn len(&self) -> usize {
        N
    }
    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T: Copy> DynArray<T> for Vec<T> {
    fn try_from_fn_and_len(f: impl FnMut(usize) -> T, size: usize) -> Result<Self, ()> {
        Ok((0..size).into_iter().map(f).collect())
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, bytemuck::Zeroable, Default)]
pub struct D1;
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, bytemuck::Zeroable, Default)]
pub struct D2;
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, bytemuck::Zeroable, Default)]
pub struct D3;
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, bytemuck::Zeroable, Default)]
pub struct D4;
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, bytemuck::Zeroable, Default)]
pub struct D5;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, bytemuck::Zeroable, Default)]
pub struct DDyn(usize);

pub trait Dimension:
    'static + Copy + bytemuck::Zeroable + Eq + PartialEq + Debug + Hash + Default
{
    const N: usize;
    type NDArrayDim: ndarray::Dimension;
    type Array<T: Copy>: Array<T>;
    fn to_ndarray_dim(i: Self::Array<usize>) -> Self::NDArrayDim;
}

pub trait DynDimension: 'static + Debug + Clone {
    type DynArray<T: Copy>: DynArray<T>;
    fn n_dim(&self) -> usize;
    fn try_into_dim<D: Dimension, T: Copy>(v: Self::DynArray<T>) -> Option<D::Array<T>>;

    fn into_dyn<T: Copy>(v: Self::DynArray<T>) -> Vec<T>;
}
impl<D: Dimension> DynDimension for D {
    type DynArray<T: Copy> = D::Array<T>;
    fn n_dim(&self) -> usize {
        D::N
    }
    fn try_into_dim<D2: Dimension, T: Copy>(v: Self::DynArray<T>) -> Option<D2::Array<T>> {
        if D::N == D2::N {
            Some(D2::Array::from_fn(|i| v[i]))
        } else {
            None
        }
    }

    fn into_dyn<T: Copy>(v: Self::DynArray<T>) -> Vec<T> {
        v.into_iter().collect()
    }
}

impl DynDimension for DDyn {
    type DynArray<T: Copy> = Vec<T>;
    fn n_dim(&self) -> usize {
        self.0
    }
    fn try_into_dim<D2: Dimension, T: Copy>(v: Self::DynArray<T>) -> Option<D2::Array<T>> {
        if v.len() == D2::N {
            Some(D2::Array::from_fn(|i| v[i]))
        } else {
            None
        }
    }

    fn into_dyn<T: Copy>(v: Self::DynArray<T>) -> Vec<T> {
        v
    }
}

pub trait LargerDim: Dimension {
    type Larger: SmallerDim<Smaller = Self>;
}
pub trait SmallerDim: Dimension {
    type Smaller: LargerDim<Larger = Self>;
}

impl Dimension for D1 {
    const N: usize = 1;
    type NDArrayDim = ndarray::Ix1;
    type Array<T: Copy> = [T; 1];
    fn to_ndarray_dim(i: Self::Array<usize>) -> Self::NDArrayDim {
        ndarray::Dim(i)
    }
}
impl LargerDim for D1 {
    type Larger = D2;
}

impl Dimension for D2 {
    const N: usize = 2;
    type NDArrayDim = ndarray::Ix2;
    type Array<T: Copy> = [T; 2];
    fn to_ndarray_dim(i: Self::Array<usize>) -> Self::NDArrayDim {
        ndarray::Dim(i)
    }
}
impl LargerDim for D2 {
    type Larger = D3;
}
impl SmallerDim for D2 {
    type Smaller = D1;
}

impl Dimension for D3 {
    const N: usize = 3;
    type NDArrayDim = ndarray::Ix3;
    type Array<T: Copy> = [T; 3];
    fn to_ndarray_dim(i: Self::Array<usize>) -> Self::NDArrayDim {
        ndarray::Dim(i)
    }
}
impl LargerDim for D3 {
    type Larger = D4;
}
impl SmallerDim for D3 {
    type Smaller = D2;
}

impl Dimension for D4 {
    const N: usize = 4;
    type NDArrayDim = ndarray::Ix4;
    type Array<T: Copy> = [T; 4];
    fn to_ndarray_dim(i: Self::Array<usize>) -> Self::NDArrayDim {
        ndarray::Dim(i)
    }
}
impl SmallerDim for D4 {
    type Smaller = D3;
}
impl LargerDim for D4 {
    type Larger = D5;
}

impl Dimension for D5 {
    const N: usize = 5;
    type NDArrayDim = ndarray::Ix5;
    type Array<T: Copy> = [T; 5];
    fn to_ndarray_dim(i: Self::Array<usize>) -> Self::NDArrayDim {
        ndarray::Dim(i)
    }
}
impl SmallerDim for D5 {
    type Smaller = D4;
}
