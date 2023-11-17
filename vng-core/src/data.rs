use std::{
    mem::MaybeUninit,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::{
    array::ChunkInfo,
    id::{Id, Identify},
};

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

#[derive(
    Copy,
    Clone,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Debug,
    state_link::StateNoPy,
    bytemuck::Zeroable,
    bytemuck::Pod,
)]
#[repr(C)]
pub struct LocalCoordinateType;
impl CoordinateType for LocalCoordinateType {}
#[derive(
    Copy,
    Clone,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Debug,
    state_link::StateNoPy,
    bytemuck::Zeroable,
    bytemuck::Pod,
)]
#[repr(C)]
pub struct GlobalCoordinateType;
impl CoordinateType for GlobalCoordinateType {}
#[derive(
    Copy,
    Clone,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Debug,
    state_link::StateNoPy,
    bytemuck::Zeroable,
    bytemuck::Pod,
)]
#[repr(C)]
pub struct ChunkCoordinateType;
impl CoordinateType for ChunkCoordinateType {}

#[repr(transparent)]
#[derive(
    Copy,
    Clone,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Debug,
    state_link::StateNoPy,
    bytemuck::Zeroable,
    bytemuck::Pod,
)]
pub struct Coordinate<T> {
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

impl<T> Identify for Coordinate<T> {
    fn id(&self) -> Id {
        self.raw.id()
    }
}

#[repr(transparent)]
#[derive(
    Copy, Clone, Hash, PartialEq, Eq, Debug, bytemuck::Pod, bytemuck::Zeroable, state_link::State,
)]
pub struct Vector<const N: usize, T>([T; N]);

impl<const N: usize, T: Identify> Identify for Vector<N, T> {
    fn id(&self) -> Id {
        (&self.0[..]).id()
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

impl<const N: usize, T> Vector<N, T> {
    pub fn from_fn(f: impl FnMut(usize) -> T) -> Self {
        Vector(std::array::from_fn(f))
    }
    pub fn new(inner: [T; N]) -> Self {
        Vector(inner)
    }
    pub fn dim() -> usize {
        N
    }
}

impl<const N: usize, T: Copy> Vector<N, T> {
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
impl<const N: usize> Vector<N, u32> {
    pub fn f32(self) -> Vector<N, f32> {
        self.map(|v| v as f32)
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
    pub fn push_dim_small(self, extra: T) -> Vector<4, T> {
        [self.0[0], self.0[1], self.0[2], extra].into()
    }
    pub fn push_dim_large(self, extra: T) -> Vector<4, T> {
        [extra, self.0[0], self.0[1], self.0[2]].into()
    }
}
impl<T: Copy> Vector<2, T> {
    pub fn x(&self) -> T {
        self.0[1]
    }
    pub fn y(&self) -> T {
        self.0[0]
    }
    pub fn push_dim_small(self, extra: T) -> Vector<3, T> {
        [self.0[0], self.0[1], extra].into()
    }
    pub fn push_dim_large(self, extra: T) -> Vector<3, T> {
        [extra, self.0[0], self.0[1]].into()
    }
}
impl<const N: usize, T> IntoIterator for Vector<N, T> {
    type Item = T;

    type IntoIter = std::array::IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
impl<const N: usize, T: Copy> Vector<N, T> {
    fn dot<O: Copy + Add<O, Output = O>, U: Copy + Mul<T, Output = O>>(
        self,
        other: &Vector<N, U>,
    ) -> O {
        let v: Vector<N, O> = other.zip(self, Mul::mul);
        let mut v = v.into_iter();
        let mut o = v.next().unwrap();
        for v in v {
            o = v + o;
        }
        o
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

impl<T: num::One + Copy> Vector<2, T> {
    pub fn to_homogeneous_coord(self) -> Vector<3, T> {
        Vector::from([num::one(), self.y(), self.x()])
    }
}

impl<T: num::One + Copy> Vector<3, T> {
    pub fn to_homogeneous_coord(self) -> Vector<4, T> {
        Vector::from([num::one(), self.z(), self.y(), self.x()])
    }
}
impl<T: Copy> Vector<4, T> {
    pub fn to_non_homogeneous_coord(self) -> Vector<3, T> {
        self.drop_dim(0)
    }
}

impl Vector<3, f32> {
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
        &self.0[col].0[row]
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
    #[must_use]
    pub fn transform(&self, t: &Matrix<4, f32>) -> Self {
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
//

#[cfg(feature = "python")]
mod py {
    use super::*;
    use pyo3::{prelude::*, types::PyList};

    // Coordinate
    impl<'source, T: CoordinateType> FromPyObject<'source> for Coordinate<T> {
        fn extract(ob: &'source PyAny) -> PyResult<Self> {
            Ok(Coordinate {
                raw: ob.extract()?,
                type_: std::marker::PhantomData,
            })
        }
    }
    impl<T: CoordinateType> IntoPy<PyObject> for Coordinate<T> {
        fn into_py(self, py: Python<'_>) -> PyObject {
            self.raw.into_py(py)
        }
    }

    // Vector
    impl<'source, const N: usize, T: FromPyObject<'source>> FromPyObject<'source> for Vector<N, T> {
        fn extract(ob: &'source PyAny) -> PyResult<Self> {
            ob.extract::<[T; N]>().map(Self)
        }
    }

    impl<const N: usize, T: IntoPy<PyObject>> IntoPy<PyObject> for Vector<N, T> {
        fn into_py(self, py: Python<'_>) -> PyObject {
            PyList::new(py, self.0.into_iter().map(|v| v.into_py(py))).into()
        }
    }

    //impl<const N: usize, T: ToPyObject> ToPyObject for Vector<N, T> {
    //    fn to_object(&self, py: Python<'_>) -> PyObject {
    //        PyList::new(py, self.0.iter().map(|v| v.to_object(py))).into()
    //    }
    //}

    impl<const N: usize, T: state_link::py::PyState> state_link::py::PyState for Vector<N, T> {
        fn build_handle(
            py: Python,
            inner: state_link::GenericNodeHandle,
            store: Py<state_link::py::Store>,
        ) -> PyObject {
            <[T; N]>::build_handle(py, inner.index(0), store)
        }
    }

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
