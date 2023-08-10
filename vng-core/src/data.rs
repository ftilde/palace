use std::{
    mem::MaybeUninit,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::array::ChunkInfo;

pub fn hmul<const N: usize, T: CoordinateType>(s: Vector<N, Coordinate<T>>) -> usize {
    s.into_iter().map(|v| v.raw as usize).product()
}

pub fn to_linear<const N: usize, T: CoordinateType>(
    pos: Vector<N, Coordinate<T>>,
    dim: Vector<N, Coordinate<T>>,
) -> usize {
    let mut out = pos.0[0].raw as usize;
    for i in 1..N {
        out = out * dim.0[i].raw as usize + pos.0[i].raw as usize;
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

pub trait CoordinateType: Copy + Clone + PartialEq + Eq {}

#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct LocalCoordinateType;
impl CoordinateType for LocalCoordinateType {}
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct GlobalCoordinateType;
impl CoordinateType for GlobalCoordinateType {}
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct ChunkCoordinateType;
impl CoordinateType for ChunkCoordinateType {}

#[repr(transparent)]
#[derive(
    Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug, serde::Serialize, serde::Deserialize,
)]
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
impl<T: CoordinateType> Into<u32> for Coordinate<T> {
    fn into(self) -> u32 {
        self.raw
    }
}
impl<T: CoordinateType> TryInto<i32> for Coordinate<T> {
    type Error = <u32 as TryInto<i32>>::Error;

    fn try_into(self) -> Result<i32, Self::Error> {
        self.raw.try_into()
    }
}
impl<T: CoordinateType> TryFrom<i32> for Coordinate<T> {
    type Error = <u32 as TryInto<i32>>::Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Ok(Coordinate {
            raw: value.try_into()?,
            type_: Default::default(),
        })
    }
}
impl<T: CoordinateType> Into<usize> for Coordinate<T> {
    fn into(self) -> usize {
        self.raw as usize
    }
}
impl GlobalCoordinate {
    pub fn local(self) -> LocalCoordinate {
        self.raw.into()
    }
}
impl LocalCoordinate {
    pub fn local(self) -> GlobalCoordinate {
        self.raw.into()
    }
}

macro_rules! impl_coordinate_ops {
    ($rhs_ty:ty, $rhs_access:expr) => {
        impl<T: CoordinateType> Add<$rhs_ty> for Coordinate<T> {
            type Output = Coordinate<T>;

            fn add(self, rhs: $rhs_ty) -> Self::Output {
                (self.raw + $rhs_access(rhs)).into()
            }
        }
        impl<T: CoordinateType> Sub<$rhs_ty> for Coordinate<T> {
            type Output = Coordinate<T>;

            fn sub(self, rhs: $rhs_ty) -> Self::Output {
                (self.raw - $rhs_access(rhs)).into()
            }
        }
        impl<T: CoordinateType> Mul<$rhs_ty> for Coordinate<T> {
            type Output = Coordinate<T>;

            fn mul(self, rhs: $rhs_ty) -> Self::Output {
                (self.raw * $rhs_access(rhs)).into()
            }
        }
        impl<T: CoordinateType> Div<$rhs_ty> for Coordinate<T> {
            type Output = Coordinate<T>;

            fn div(self, rhs: $rhs_ty) -> Self::Output {
                (self.raw / $rhs_access(rhs)).into()
            }
        }
    };
}

impl_coordinate_ops!(usize, |rhs| rhs as u32);
impl_coordinate_ops!(u32, |rhs| rhs);
impl_coordinate_ops!(Self, |rhs: Self| rhs.raw);

impl Add<LocalCoordinate> for GlobalCoordinate {
    type Output = GlobalCoordinate;
    fn add(self, rhs: LocalCoordinate) -> Self::Output {
        (self.raw + rhs.raw).into()
    }
}

pub type LocalCoordinate = Coordinate<LocalCoordinateType>;
pub type GlobalCoordinate = Coordinate<GlobalCoordinateType>;
pub type ChunkCoordinate = Coordinate<ChunkCoordinateType>;

#[repr(transparent)]
#[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vector<const N: usize, T>([T; N]);

impl<const N: usize, T: serde::Serialize> serde::Serialize for Vector<N, T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq;
        let mut s = serializer.serialize_seq(Some(N))?;
        for v in &self.0 {
            s.serialize_element(&v)?;
        }
        s.end()
    }
}

struct VectorVisitor<const N: usize, T> {
    _marker: std::marker::PhantomData<T>,
}

impl<'de, const N: usize, T: serde::Deserialize<'de>> serde::de::Visitor<'de>
    for VectorVisitor<N, T>
{
    type Value = Vector<N, T>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "a Vector<{}, T>", N)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        Ok(Vector(array_init::try_array_init(
            |_| Ok(seq.next_element::<T>()?.unwrap()), /* ??? what to do here? how to construct an error? */
        )?))
    }
}

impl<'de, const N: usize, T: serde::Deserialize<'de>> serde::Deserialize<'de> for Vector<N, T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_seq(VectorVisitor {
            _marker: Default::default(),
        })
    }
}

impl<const N: usize, T, I: Copy + Into<T>> From<[I; N]> for Vector<N, T> {
    fn from(value: [I; N]) -> Self {
        Vector(std::array::from_fn(|i| value[i].into()))
    }
}

impl<const N: usize, T> Into<[T; N]> for Vector<N, T> {
    fn into(self) -> [T; N] {
        self.0
    }
}

impl<const N: usize, T: Copy> Vector<N, T> {
    pub fn new(inner: [T; N]) -> Self {
        Vector(inner)
    }
    pub fn dim() -> usize {
        N
    }
    pub fn fill(val: T) -> Self {
        Vector([val; N])
    }
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Vector<N, U> {
        Vector(std::array::from_fn(|i| f(self.0[i])))
    }
    pub fn map_element(mut self, i: usize, f: impl FnOnce(T) -> T) -> Vector<N, T> {
        self.0[i] = f(self.0[i]);
        self
    }
    pub fn fold<U>(self, mut state: U, mut f: impl FnMut(U, T) -> U) -> U {
        for v in self.0 {
            state = f(state, v);
        }
        state
    }
    pub fn zip<U: Copy, V>(
        self,
        other: Vector<N, U>,
        mut f: impl FnMut(T, U) -> V,
    ) -> Vector<N, V> {
        Vector(std::array::from_fn(|i| f(self.0[i], other.0[i])))
    }
    pub fn zip_enumerate<U: Copy, V>(
        self,
        other: Vector<N, U>,
        mut f: impl FnMut(usize, T, U) -> V,
    ) -> Vector<N, V> {
        Vector(std::array::from_fn(|i| f(i, self.0[i], other.0[i])))
    }
    pub fn into_elem<U>(self) -> Vector<N, U>
    where
        T: Into<U>,
    {
        self.map(|v| v.into())
    }
    pub fn try_into_elem<U>(self) -> Result<Vector<N, U>, T::Error>
    where
        T: TryInto<U>,
    {
        // Safety: Standard way to initialize an MaybeUninit array
        let mut out: [MaybeUninit<U>; N] = unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..N {
            out[i].write(T::try_into(self.0[i])?);
        }
        // Safety: We have just initialized all values in the loop above
        Ok(Vector(out.map(|v| unsafe { v.assume_init() })))
    }
}
impl<const N: usize, T: std::ops::Mul<Output = T> + Copy> Vector<N, T> {
    pub fn scale(self, v: T) -> Self {
        self.map(|w| w * v)
    }
}
impl<const N: usize, T> std::ops::Index<usize> for Vector<N, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl<const N: usize, T> std::ops::IndexMut<usize> for Vector<N, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
impl<const N: usize, T: CoordinateType> Vector<N, Coordinate<T>> {
    pub fn as_index(self) -> [usize; N] {
        self.map(|v| v.raw as usize).0
    }
}

impl<const N: usize> Vector<N, GlobalCoordinate> {
    pub fn local(self) -> Vector<N, LocalCoordinate> {
        self.map(LocalCoordinate::interpret_as)
    }
}
impl<const N: usize> Vector<N, LocalCoordinate> {
    pub fn global(self) -> Vector<N, GlobalCoordinate> {
        self.map(GlobalCoordinate::interpret_as)
    }
}
impl<const N: usize> Vector<N, u32> {
    pub fn global(self) -> Vector<N, GlobalCoordinate> {
        self.map(|v| v.into())
    }
}
impl<const N: usize, T: CoordinateType> Vector<N, Coordinate<T>> {
    pub fn raw(self) -> Vector<N, u32> {
        self.map(|v| v.raw)
    }
}
impl<T> std::ops::Deref for Vector<1, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0[0]
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
impl<T: Copy> Vector<2, T> {
    pub fn x(&self) -> T {
        self.0[1]
    }
    pub fn y(&self) -> T {
        self.0[0]
    }
    pub fn push_dim(self, extra: T) -> Vector<3, T> {
        [self.0[0], self.0[1], extra].into()
    }
}
impl<const N: usize, T> IntoIterator for Vector<N, T> {
    type Item = T;

    type IntoIter = std::array::IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
impl<const N: usize, T: Neg + Copy> Neg for Vector<N, T> {
    type Output = Vector<N, T::Output>;
    fn neg(self) -> Self::Output {
        self.map(|v| v.neg())
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

impl<T: Copy> From<Vector<2, T>> for cgmath::Vector2<T> {
    fn from(value: Vector<2, T>) -> Self {
        cgmath::Vector2 {
            x: value.x(),
            y: value.y(),
        }
    }
}

impl<T: Copy> From<cgmath::Vector2<T>> for Vector<2, T> {
    fn from(value: cgmath::Vector2<T>) -> Self {
        Self::from([value.y, value.x])
    }
}

impl<T: Copy> From<Vector<3, T>> for cgmath::Vector3<T> {
    fn from(value: Vector<3, T>) -> Self {
        cgmath::Vector3 {
            x: value.x(),
            y: value.y(),
            z: value.z(),
        }
    }
}
impl<T: Copy> From<Vector<3, T>> for cgmath::Point3<T> {
    fn from(value: Vector<3, T>) -> Self {
        cgmath::Point3 {
            x: value.x(),
            y: value.y(),
            z: value.z(),
        }
    }
}

impl<T: Copy> From<cgmath::Vector3<T>> for Vector<3, T> {
    fn from(value: cgmath::Vector3<T>) -> Self {
        Self::from([value.z, value.y, value.x])
    }
}
impl<T: Copy> From<cgmath::Point3<T>> for Vector<3, T> {
    fn from(value: cgmath::Point3<T>) -> Self {
        Self::from([value.z, value.y, value.x])
    }
}

impl<T: Copy> From<Vector<4, T>> for cgmath::Vector4<T> {
    fn from(value: Vector<4, T>) -> Self {
        cgmath::Vector4 {
            x: value.0[3],
            y: value.0[2],
            z: value.0[1],
            w: value.0[0],
        }
    }
}

impl<T: Copy> From<cgmath::Vector4<T>> for Vector<4, T> {
    fn from(value: cgmath::Vector4<T>) -> Self {
        Self::from([value.w, value.z, value.y, value.x])
    }
}

impl std::ops::Mul<Vector<3, f32>> for cgmath::Matrix3<f32> {
    type Output = Vector<3, f32>;

    fn mul(self, rhs: Vector<3, f32>) -> Self::Output {
        let v = cgmath::Vector3::from(rhs);
        (self * v).into()
    }
}

impl std::ops::Mul<Vector<4, f32>> for cgmath::Matrix4<f32> {
    type Output = Vector<4, f32>;

    fn mul(self, rhs: Vector<4, f32>) -> Self::Output {
        let v = cgmath::Vector4::from(rhs);
        (self * v).into()
    }
}

impl Vector<2, f32> {
    pub fn to_homogeneous_coord(self) -> Vector<3, f32> {
        Vector::from([1.0, self.y(), self.x()])
    }
}

impl Vector<3, f32> {
    pub fn to_homogeneous_coord(self) -> Vector<4, f32> {
        Vector::from([1.0, self.z(), self.y(), self.x()])
    }

    pub fn length(self) -> f32 {
        self.map(|v| v * v).fold(0.0, f32::add).sqrt()
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        let len_inv = 1.0 / len;
        self.map(|v| v * len_inv)
    }

    pub fn cross(self, other: Self) -> Self {
        let v1: cgmath::Vector3<f32> = self.into();
        let v2: cgmath::Vector3<f32> = other.into();
        v1.cross(v2).into()
    }
}

impl<T: Copy> Vector<2, T> {
    pub fn add_dim(self, dim: usize, value: T) -> Vector<3, T> {
        Vector(std::array::from_fn(|i| match i.cmp(&dim) {
            std::cmp::Ordering::Less => self.0[i],
            std::cmp::Ordering::Equal => value,
            std::cmp::Ordering::Greater => self.0[i - 1],
        }))
    }
}
impl<T: Copy> Vector<3, T> {
    pub fn drop_dim(self, dim: usize) -> Vector<2, T> {
        Vector(std::array::from_fn(|i| {
            if i < dim {
                self.0[i]
            } else {
                self.0[i + 1]
            }
        }))
    }
}
impl<T: Copy> Vector<4, T> {
    pub fn drop_dim(self, dim: usize) -> Vector<3, T> {
        Vector(std::array::from_fn(|i| {
            if i < dim {
                self.0[i]
            } else {
                self.0[i + 1]
            }
        }))
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

    pub fn contains(&self, p: Vector<N, T>) -> bool {
        let bigger_than_min = self.min.zip(p, |v1, v2| v1.le(&v2));
        let smaller_than_max = p.zip(self.max, |v1, v2| v1.lt(&v2));
        bigger_than_min
            .0
            .iter()
            .chain(smaller_than_max.0.iter())
            .all(|v| *v)
    }
}
impl AABB<3, f32> {
    pub fn transform(&self, t: &cgmath::Matrix4<f32>) -> Self {
        let points = (0..8).into_iter().map(|b| {
            let p = Vector::<3, f32>(std::array::from_fn(|i| {
                if (b & (1 << i)) != 0 {
                    self.min.0[i]
                } else {
                    self.max.0[i]
                }
            }));
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
