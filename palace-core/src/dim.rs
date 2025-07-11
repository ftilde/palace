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
    fn try_from_fn_and_len(size: usize, f: impl FnMut(usize) -> T) -> Result<Self, ()>;
    fn len(&self) -> usize;
    fn as_slice(&self) -> &[T];
}

impl<const N: usize, T: Copy> DynArray<T> for [T; N] {
    fn try_from_fn_and_len(size: usize, f: impl FnMut(usize) -> T) -> Result<Self, ()> {
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
    fn try_from_fn_and_len(size: usize, f: impl FnMut(usize) -> T) -> Result<Self, ()> {
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
    const N_STR: &'static str;
    type NDArrayDim: ndarray::Dimension;
    type Array<T: Copy>: Array<T>;
    type Mat<T: Copy>: Array<T>;
    fn to_ndarray_dim(i: Self::Array<usize>) -> Self::NDArrayDim;
    fn new() -> Self;
}

pub trait DynDimension: 'static + Debug + Clone + PartialEq + Copy {
    type DynArray<T: Copy>: DynArray<T>;
    type DynMatrix<T: Copy>: DynArray<T>;
    type NDArrayDimDyn: ndarray::Dimension;
    fn to_ndarray_dim_dyn(i: Self::DynArray<usize>) -> Self::NDArrayDimDyn;

    const FIXED_N: Option<usize>;

    fn try_from_n(n: usize) -> Option<Self>;
    fn n(&self) -> usize;
    fn dim_of_array<T: Copy>(v: &Self::DynArray<T>) -> Self;
    fn try_into_dim<D: Dimension, T: Copy>(v: Self::DynArray<T>) -> Option<D::Array<T>>;

    fn into_dyn<T: Copy>(v: Self::DynArray<T>) -> Vec<T>;
}
impl<D: Dimension> DynDimension for D {
    type DynArray<T: Copy> = D::Array<T>;
    type DynMatrix<T: Copy> = D::Mat<T>;
    type NDArrayDimDyn = D::NDArrayDim;

    fn to_ndarray_dim_dyn(i: Self::DynArray<usize>) -> Self::NDArrayDimDyn {
        Self::to_ndarray_dim(i)
    }

    fn n(&self) -> usize {
        D::N
    }
    fn dim_of_array<T: Copy>(_v: &Self::DynArray<T>) -> Self {
        Self::new()
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

    const FIXED_N: Option<usize> = Some(D::N);

    fn try_from_n(n: usize) -> Option<Self> {
        (n == D::N).then(|| Self::new())
    }
}

impl DynDimension for DDyn {
    type DynArray<T: Copy> = Vec<T>;
    type DynMatrix<T: Copy> = Vec<T>;
    type NDArrayDimDyn = ndarray::IxDyn;

    fn to_ndarray_dim_dyn(i: Self::DynArray<usize>) -> Self::NDArrayDimDyn {
        ndarray::Dim(i)
    }

    fn n(&self) -> usize {
        self.0
    }
    fn dim_of_array<T: Copy>(v: &Self::DynArray<T>) -> Self {
        DDyn(v.len())
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

    const FIXED_N: Option<usize> = None;

    fn try_from_n(n: usize) -> Option<Self> {
        Some(DDyn(n))
    }
}
impl SmallerDim for DDyn {
    type Smaller = DDyn;
    fn smaller(self) -> Self::Smaller {
        DDyn(self.0.checked_sub(1).unwrap())
    }
}
impl LargerDim for DDyn {
    type Larger = DDyn;
    fn larger(self) -> Self::Larger {
        DDyn(self.0.checked_add(1).unwrap())
    }
}

pub trait LargerDim: DynDimension {
    type Larger: SmallerDim<Smaller = Self>;
    fn larger(self) -> Self::Larger;
}
pub trait SmallerDim: DynDimension {
    type Smaller: LargerDim<Larger = Self>;
    fn smaller(self) -> Self::Smaller;
}

impl Dimension for D1 {
    const N: usize = 1;
    const N_STR: &'static str = "1";
    type NDArrayDim = ndarray::Ix1;
    type Array<T: Copy> = [T; 1];
    type Mat<T: Copy> = [T; 1];
    fn to_ndarray_dim(i: Self::Array<usize>) -> Self::NDArrayDim {
        ndarray::Dim(i)
    }
    fn new() -> Self {
        D1
    }
}
impl LargerDim for D1 {
    type Larger = D2;
    fn larger(self) -> Self::Larger {
        Self::Larger::new()
    }
}

impl Dimension for D2 {
    const N: usize = 2;
    const N_STR: &'static str = "2";
    type NDArrayDim = ndarray::Ix2;
    type Array<T: Copy> = [T; 2];
    type Mat<T: Copy> = [T; 4];
    fn to_ndarray_dim(i: Self::Array<usize>) -> Self::NDArrayDim {
        ndarray::Dim(i)
    }
    fn new() -> Self {
        D2
    }
}
impl LargerDim for D2 {
    type Larger = D3;
    fn larger(self) -> Self::Larger {
        Self::Larger::new()
    }
}
impl SmallerDim for D2 {
    type Smaller = D1;
    fn smaller(self) -> Self::Smaller {
        Self::Smaller::new()
    }
}

impl Dimension for D3 {
    const N: usize = 3;
    const N_STR: &'static str = "3";
    type NDArrayDim = ndarray::Ix3;
    type Array<T: Copy> = [T; 3];
    type Mat<T: Copy> = [T; 9];
    fn to_ndarray_dim(i: Self::Array<usize>) -> Self::NDArrayDim {
        ndarray::Dim(i)
    }
    fn new() -> Self {
        D3
    }
}
impl LargerDim for D3 {
    type Larger = D4;
    fn larger(self) -> Self::Larger {
        Self::Larger::new()
    }
}
impl SmallerDim for D3 {
    type Smaller = D2;
    fn smaller(self) -> Self::Smaller {
        Self::Smaller::new()
    }
}

impl Dimension for D4 {
    const N: usize = 4;
    const N_STR: &'static str = "4";
    type NDArrayDim = ndarray::Ix4;
    type Array<T: Copy> = [T; 4];
    type Mat<T: Copy> = [T; 16];
    fn to_ndarray_dim(i: Self::Array<usize>) -> Self::NDArrayDim {
        ndarray::Dim(i)
    }
    fn new() -> Self {
        D4
    }
}
impl SmallerDim for D4 {
    type Smaller = D3;
    fn smaller(self) -> Self::Smaller {
        Self::Smaller::new()
    }
}
impl LargerDim for D4 {
    type Larger = D5;
    fn larger(self) -> Self::Larger {
        Self::Larger::new()
    }
}

impl Dimension for D5 {
    const N: usize = 5;
    const N_STR: &'static str = "5";
    type NDArrayDim = ndarray::Ix5;
    type Array<T: Copy> = [T; 5];
    type Mat<T: Copy> = [T; 25];
    fn to_ndarray_dim(i: Self::Array<usize>) -> Self::NDArrayDim {
        ndarray::Dim(i)
    }
    fn new() -> Self {
        D5
    }
}
impl SmallerDim for D5 {
    type Smaller = D4;
    fn smaller(self) -> Self::Smaller {
        Self::Smaller::new()
    }
}
