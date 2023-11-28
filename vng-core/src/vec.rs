use std::{
    mem::MaybeUninit,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::{
    coordinate::{ChunkCoordinate, Coordinate, CoordinateType, GlobalCoordinate, LocalCoordinate},
    id::{Id, Identify},
};

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

impl<const N: usize, T, I: Copy + Into<T>> TryFrom<Vec<I>> for Vector<N, T> {
    type Error = ();

    fn try_from(value: Vec<I>) -> Result<Self, Self::Error> {
        if value.len() != N {
            return Err(());
        }
        Ok(Self::from_fn(|i| value[i].into()))
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
    pub fn chunk(self) -> Vector<N, ChunkCoordinate> {
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
    pub fn dot<O: Copy + Add<O, Output = O>, U: Copy + Mul<T, Output = O>>(
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

#[cfg(feature = "python")]
mod py {
    use super::*;
    use pyo3::{prelude::*, types::PyList};

    // Coordinate
    impl<'source, T: CoordinateType> FromPyObject<'source> for Coordinate<T> {
        fn extract(ob: &'source PyAny) -> PyResult<Self> {
            let raw: u32 = ob.extract()?;
            Ok(Coordinate::from(raw))
        }
    }
    impl<T: CoordinateType> IntoPy<PyObject> for Coordinate<T> {
        fn into_py(self, py: Python<'_>) -> PyObject {
            self.raw.into_py(py)
        }
    }

    // Vector
    impl<'source, const N: usize, T: Copy + FromPyObject<'source>> FromPyObject<'source>
        for Vector<N, T>
    {
        fn extract(ob: &'source PyAny) -> PyResult<Self> {
            ob.extract::<[T; N]>().map(Self::from)
        }
    }

    impl<const N: usize, T: IntoPy<PyObject>> IntoPy<PyObject> for Vector<N, T> {
        fn into_py(self, py: Python<'_>) -> PyObject {
            PyList::new(py, self.into_iter().map(|v| v.into_py(py))).into()
        }
    }

    //impl<const N: usize, T: ToPyObject> ToPyObject for Vector<N, T> {
    //    fn to_object(&self, py: Python<'_>) -> PyObject {
    //        PyList::new(py, self.0.iter().map(|v| v.to_object(py))).into()
    //    }
    //}

    impl<const N: usize, T: Copy + state_link::py::PyState> state_link::py::PyState for Vector<N, T> {
        fn build_handle(
            py: Python,
            inner: state_link::GenericNodeHandle,
            store: Py<state_link::py::Store>,
        ) -> PyObject {
            <[T; N]>::build_handle(py, inner.index(0), store)
        }
    }
}
