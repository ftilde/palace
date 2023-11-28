pub trait Array<T: Sized + Copy>:
    std::ops::IndexMut<usize, Output = T>
    + std::ops::Index<usize, Output = T>
    + IntoIterator<Item = T>
    + Copy
{
    const N: usize;
    type Mapped<U: Copy>: Array<U>;
    fn from_fn(f: impl FnMut(usize) -> T) -> Self;
    fn map<O: Copy>(self, f: impl FnMut(T) -> O) -> Self::Mapped<O>;
}

impl<const N: usize, T: Copy> Array<T> for [T; N] {
    const N: usize = N;
    type Mapped<U: Copy> = [U; N];
    fn from_fn(f: impl FnMut(usize) -> T) -> Self {
        std::array::from_fn(f)
    }
    fn map<O: Copy>(self, f: impl FnMut(T) -> O) -> Self::Mapped<O> {
        self.map(f)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, bytemuck::Zeroable)]
pub struct D1;
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, bytemuck::Zeroable)]
pub struct D2;
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, bytemuck::Zeroable)]
pub struct D3;
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, bytemuck::Zeroable)]
pub struct D4;
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, bytemuck::Zeroable)]
pub struct D5;

pub trait Dimension:
    'static + Copy + bytemuck::Zeroable + Eq + PartialEq + std::fmt::Debug + std::hash::Hash
{
    const N: usize;
    type Array<T: Copy>: Array<T>;
}

pub trait LargerDim: Dimension {
    type Larger: Dimension;
}
pub trait SmallerDim: Dimension {
    type Smaller: Dimension;
}

impl Dimension for D1 {
    const N: usize = 1;
    type Array<T: Copy> = [T; 1];
}
impl LargerDim for D1 {
    type Larger = D2;
}

impl Dimension for D2 {
    const N: usize = 2;
    type Array<T: Copy> = [T; 2];
}
impl LargerDim for D2 {
    type Larger = D3;
}
impl SmallerDim for D2 {
    type Smaller = D1;
}

impl Dimension for D3 {
    const N: usize = 3;
    type Array<T: Copy> = [T; 3];
}
impl LargerDim for D3 {
    type Larger = D4;
}
impl SmallerDim for D3 {
    type Smaller = D2;
}

impl Dimension for D4 {
    const N: usize = 4;
    type Array<T: Copy> = [T; 4];
}
impl SmallerDim for D4 {
    type Smaller = D3;
}
impl LargerDim for D4 {
    type Larger = D5;
}

impl Dimension for D5 {
    const N: usize = 5;
    type Array<T: Copy> = [T; 5];
}
impl SmallerDim for D5 {
    type Smaller = D4;
}
