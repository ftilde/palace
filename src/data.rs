use std::ops::{Add, Div, Mul, Sub};

pub fn hmul<const N: usize, T: CoordinateType>(s: Vector<N, Coordinate<T>>) -> usize {
    s.into_iter().map(|v| v.raw as usize).product()
}

pub fn to_linear<T: CoordinateType>(
    pos: Vector<3, Coordinate<T>>,
    dim: Vector<3, Coordinate<T>>,
) -> usize {
    (pos.z().raw as usize * dim.y().raw as usize + pos.y().raw as usize) * dim.x().raw as usize
        + pos.x().raw as usize
}

pub trait CoordinateType: Copy + Clone + PartialEq + Eq {}

#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct LocalVoxelCoordinateType;
impl CoordinateType for LocalVoxelCoordinateType {}
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct GlobalVoxelCoordinateType;
impl CoordinateType for GlobalVoxelCoordinateType {}
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct BrickCoordinateType;
impl CoordinateType for BrickCoordinateType {}

#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Coordinate<T: CoordinateType> {
    pub raw: u32,
    type_: std::marker::PhantomData<T>,
}

impl<T: CoordinateType> Coordinate<T> {
    pub fn interpret_as<U: CoordinateType>(v: Coordinate<U>) -> Self {
        Coordinate {
            raw: v.raw,
            type_: Default::default(),
        }
    }
}
impl<T: CoordinateType> From<u32> for Coordinate<T> {
    fn from(value: u32) -> Self {
        Coordinate {
            raw: value,
            type_: Default::default(),
        }
    }
}
impl<T: CoordinateType> Add for Coordinate<T> {
    type Output = Coordinate<T>;

    fn add(self, rhs: Self) -> Self::Output {
        (self.raw + rhs.raw).into()
    }
}
impl<T: CoordinateType> Sub for Coordinate<T> {
    type Output = Coordinate<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        (self.raw - rhs.raw).into()
    }
}
impl<T: CoordinateType> Mul for Coordinate<T> {
    type Output = Coordinate<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        (self.raw * rhs.raw).into()
    }
}
impl<T: CoordinateType> Div for Coordinate<T> {
    type Output = Coordinate<T>;

    fn div(self, rhs: Self) -> Self::Output {
        (self.raw / rhs.raw).into()
    }
}

pub type LocalVoxelCoordinate = Coordinate<LocalVoxelCoordinateType>;
pub type GlobalVoxelCoordinate = Coordinate<GlobalVoxelCoordinateType>;
pub type BrickCoordinate = Coordinate<BrickCoordinateType>;

#[repr(C)]
#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct Vector<const N: usize, T>([T; N]);

impl<const N: usize, T> From<[T; N]> for Vector<N, T> {
    fn from(value: [T; N]) -> Self {
        Vector(value)
    }
}

impl<const N: usize, T: Copy> Vector<N, T> {
    pub fn fill(val: T) -> Self {
        Vector([val; N])
    }
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Vector<N, U> {
        Vector(std::array::from_fn(|i| f(self.0[i])))
    }
    pub fn zip<U: Copy, V>(
        self,
        other: Vector<N, U>,
        mut f: impl FnMut(T, U) -> V,
    ) -> Vector<N, V> {
        Vector(std::array::from_fn(|i| f(self.0[i], other.0[i])))
    }
}
impl<T: Copy> Vector<3, T> {
    pub fn x(&self) -> T {
        self.0[2]
    }
    pub fn y(&self) -> T {
        self.0[1]
    }
    pub fn z(&self) -> T {
        self.0[0]
    }
}
impl<const N: usize, T> IntoIterator for Vector<N, T> {
    type Item = T;

    type IntoIter = std::array::IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
impl<const N: usize, O: Copy, U: Copy, T: Copy + Add<U, Output = O>> Add<Vector<N, U>>
    for Vector<N, T>
{
    type Output = Vector<N, O>;
    fn add(self, rhs: Vector<N, U>) -> Self::Output {
        self.zip(rhs, Add::add)
    }
}
impl<const N: usize, O: Copy, U: Copy, T: Copy + Sub<U, Output = O>> Sub<Vector<N, U>>
    for Vector<N, T>
{
    type Output = Vector<N, O>;
    fn sub(self, rhs: Vector<N, U>) -> Self::Output {
        self.zip(rhs, Sub::sub)
    }
}
impl<const N: usize, O: Copy, U: Copy, T: Copy + Mul<U, Output = O>> Mul<Vector<N, U>>
    for Vector<N, T>
{
    type Output = Vector<N, O>;
    fn mul(self, rhs: Vector<N, U>) -> Self::Output {
        self.zip(rhs, Mul::mul)
    }
}
impl<const N: usize, O: Copy, U: Copy, T: Copy + Div<U, Output = O>> Div<Vector<N, U>>
    for Vector<N, T>
{
    type Output = Vector<N, O>;
    fn div(self, rhs: Vector<N, U>) -> Self::Output {
        self.zip(rhs, Div::div)
    }
}

impl Add<LocalVoxelCoordinate> for GlobalVoxelCoordinate {
    type Output = GlobalVoxelCoordinate;
    fn add(self, rhs: LocalVoxelCoordinate) -> Self::Output {
        (self.raw + rhs.raw).into()
    }
}

pub type LocalVoxelPosition = Vector<3, LocalVoxelCoordinate>;
pub type VoxelPosition = Vector<3, GlobalVoxelCoordinate>;
pub type BrickPosition = Vector<3, BrickCoordinate>;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VolumeMetaData {
    pub dimensions: VoxelPosition,
    pub brick_size: LocalVoxelPosition,
}

impl VolumeMetaData {
    pub fn num_voxels(&self) -> usize {
        hmul(self.dimensions)
    }
    pub fn dimension_in_bricks(&self) -> BrickPosition {
        self.dimensions.zip(self.brick_size, |a, b| {
            crate::util::div_round_up(a.raw, b.raw).into()
        })
    }
    pub fn brick_pos(&self, pos: VoxelPosition) -> BrickPosition {
        pos.zip(self.brick_size, |a, b| (a.raw / b.raw).into())
    }
    pub fn brick_begin(&self, pos: BrickPosition) -> VoxelPosition {
        pos.zip(self.brick_size, |a, b| (a.raw * b.raw).into())
    }
    pub fn brick_end(&self, pos: BrickPosition) -> VoxelPosition {
        let next_pos = pos + BrickPosition::fill(1.into());
        let raw_end = self.brick_begin(next_pos);
        raw_end.zip(self.dimensions, std::cmp::min)
    }
    pub fn brick_dim(&self, pos: BrickPosition) -> LocalVoxelPosition {
        (self.brick_end(pos) - self.brick_begin(pos)).map(LocalVoxelCoordinate::interpret_as)
    }

    pub fn brick_positions(&self) -> impl Iterator<Item = BrickPosition> {
        let bp = self.dimension_in_bricks();
        itertools::iproduct! { 0..bp.z().raw, 0..bp.y().raw, 0..bp.x().raw }
            .map(|(z, y, x)| [z.into(), y.into(), x.into()].into())
    }
}

pub struct Brick<'a> {
    size: LocalVoxelPosition,
    mem_size: LocalVoxelPosition,
    data: &'a [f32],
}

impl<'a> Brick<'a> {
    pub fn new(data: &'a [f32], size: LocalVoxelPosition, mem_size: LocalVoxelPosition) -> Self {
        Self {
            data,
            size,
            mem_size,
        }
    }
    pub fn voxels(&'a self) -> impl Iterator<Item = f32> + 'a {
        itertools::iproduct! { 0..self.size.z().raw, 0..self.size.y().raw, 0..self.size.x().raw }
            .map(|(z, y, x)| to_linear([z.into(), y.into(), x.into()].into(), self.mem_size))
            .map(|i| self.data[i])
    }
}
