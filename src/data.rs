use std::ops::{Add, Div, Mul, Sub};

pub fn hmul<const N: usize, T: CoordinateType>(s: Vector<N, Coordinate<T>>) -> usize {
    s.into_iter().map(|v| v.raw as usize).product()
}

//pub fn to_linear<const N: usize, T: CoordinateType>(
//    pos: Vector<N, Coordinate<T>>,
//    dim: Vector<N, Coordinate<T>>,
//) -> usize {
//    let mut out = pos.0[0].raw as usize;
//    for i in 1..N {
//        out = out * dim.0[i].raw as usize + pos.0[i].raw as usize;
//    }
//    out
//}

pub trait CoordinateType: Copy + Clone + PartialEq + Eq {}

#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct LocalVoxelCoordinateType;
impl CoordinateType for LocalVoxelCoordinateType {}
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct GlobalVoxelCoordinateType;
impl CoordinateType for GlobalVoxelCoordinateType {}
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChunkCoordinateType;
impl CoordinateType for ChunkCoordinateType {}

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
pub type ChunkCoordinate = Coordinate<ChunkCoordinateType>;

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
pub type BrickPosition = Vector<3, ChunkCoordinate>;

pub struct ChunkMemInfo<const N: usize> {
    pub mem_dimensions: Vector<N, LocalVoxelCoordinate>,
    pub logical_dimensions: Vector<N, LocalVoxelCoordinate>,
}
impl<const N: usize> ChunkMemInfo<N> {
    pub fn is_contiguous(&self) -> bool {
        for i in 1..N {
            if self.mem_dimensions.0[i] != self.logical_dimensions.0[i] {
                return false;
            }
        }
        true
    }
    pub fn mem_size(&self) -> usize {
        hmul(self.mem_dimensions)
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ArrayMetaData<const N: usize> {
    pub dimensions: Vector<N, GlobalVoxelCoordinate>,
    pub chunk_size: Vector<N, LocalVoxelCoordinate>,
}

impl<const N: usize> ArrayMetaData<N> {
    pub fn num_voxels(&self) -> usize {
        hmul(self.dimensions)
    }
    pub fn dimension_in_bricks(&self) -> Vector<N, ChunkCoordinate> {
        self.dimensions.zip(self.chunk_size, |a, b| {
            crate::util::div_round_up(a.raw, b.raw).into()
        })
    }
    pub fn chunk_pos(&self, pos: Vector<N, GlobalVoxelCoordinate>) -> Vector<N, ChunkCoordinate> {
        pos.zip(self.chunk_size, |a, b| (a.raw / b.raw).into())
    }
    pub fn chunk_begin(&self, pos: Vector<N, ChunkCoordinate>) -> Vector<N, GlobalVoxelCoordinate> {
        pos.zip(self.chunk_size, |a, b| (a.raw * b.raw).into())
    }
    pub fn chunk_end(&self, pos: Vector<N, ChunkCoordinate>) -> Vector<N, GlobalVoxelCoordinate> {
        let next_pos = pos + Vector::fill(1.into());
        let raw_end = self.chunk_begin(next_pos);
        raw_end.zip(self.dimensions, std::cmp::min)
    }
    pub fn chunk_info(&self, pos: Vector<N, ChunkCoordinate>) -> ChunkMemInfo<N> {
        ChunkMemInfo {
            mem_dimensions: self.chunk_size,
            logical_dimensions: self.logical_chunk_dim(pos),
        }
    }
    fn logical_chunk_dim(
        &self,
        pos: Vector<N, ChunkCoordinate>,
    ) -> Vector<N, LocalVoxelCoordinate> {
        (self.chunk_end(pos) - self.chunk_begin(pos)).map(LocalVoxelCoordinate::interpret_as)
    }
}

impl ArrayMetaData<3> {
    pub fn brick_positions(&self) -> impl Iterator<Item = Vector<3, ChunkCoordinate>> {
        let bp = self.dimension_in_bricks();
        itertools::iproduct! { 0..bp.z().raw, 0..bp.y().raw, 0..bp.x().raw }
            .map(|(z, y, x)| [z.into(), y.into(), x.into()].into())
    }
}

pub type VolumeMetaData = ArrayMetaData<3>;

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

pub fn slice_range<T: CoordinateType>(
    begin: Vector<3, Coordinate<T>>,
    end: Vector<3, Coordinate<T>>,
) -> ndarray::SliceInfo<[ndarray::SliceInfoElem; 3], ndarray::Ix3, ndarray::Ix3> {
    ndarray::s![
        begin.z().raw as usize..end.z().raw as usize,
        begin.y().raw as usize..end.y().raw as usize,
        begin.x().raw as usize..end.x().raw as usize,
    ]
}

pub fn chunk<'a, T>(data: &'a [T], brick_info: ChunkMemInfo<3>) -> ndarray::ArrayView3<'a, T> {
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
    brick_info: ChunkMemInfo<3>,
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
