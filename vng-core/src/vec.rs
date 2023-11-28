#![allow(unused)] //NO_PUSH_main
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{
    coordinate::{ChunkCoordinate, Coordinate, CoordinateType, GlobalCoordinate, LocalCoordinate},
    dim::*,
    id::{Id, Identify},
};

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Vector<D: Dimension, T: Copy>(D::Array<T>);

impl<D: Dimension, T: Copy> state_link::State for Vector<D, T> {
    type NodeHandle = state_link::NodeHandleSpecialized<Self>;

    fn write(
        &self,
        store: &mut state_link::Store,
        at: state_link::NodeRef,
    ) -> state_link::Result<()> {
        todo!()
    }

    fn store(&self, store: &mut state_link::Store) -> state_link::NodeRef {
        todo!()
    }

    fn load(store: &state_link::Store, location: state_link::NodeRef) -> state_link::Result<Self> {
        todo!()
    }
}

impl<D: Dimension, T: Copy + Identify> Identify for Vector<D, T> {
    fn id(&self) -> Id {
        //(&self[0..D::N]).id()
        todo!()
    }
}

unsafe impl<D: Dimension, T: Copy + bytemuck::Zeroable> bytemuck::Zeroable for Vector<D, T> {}
unsafe impl<D: Dimension, T: Copy + bytemuck::Pod> bytemuck::Pod for Vector<D, T> {}

impl<D: Dimension, T: Copy + PartialEq> PartialEq for Vector<D, T> {
    fn eq(&self, other: &Self) -> bool {
        todo!()
        //self.zip(other, |l, r| *l == r).fold(false, |l, r| r && r)
    }
}
impl<D: Dimension, T: Copy + Eq> Eq for Vector<D, T> {}

impl<D: Dimension, T: Copy + PartialOrd> PartialOrd for Vector<D, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        todo!()
    }
}
impl<D: Dimension, T: Copy + Ord> Ord for Vector<D, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        todo!()
    }
}
impl<D: Dimension, T: Copy + std::fmt::Debug> std::fmt::Debug for Vector<D, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}
impl<D: Dimension, T: Copy + std::hash::Hash> std::hash::Hash for Vector<D, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        todo!()
    }
}

macro_rules! impl_array_ops {
    ($DT:ty, $dnum:expr) => {
        impl<T: Copy, F: Copy + Into<T>> From<[F; $dnum]> for Vector<$DT, T> {
            fn from(value: [F; $dnum]) -> Self {
                Vector(std::array::from_fn(|i| value[i].into()))
            }
        }
        impl<T: Copy> Into<[T; $dnum]> for Vector<$DT, T> {
            fn into(self) -> [T; $dnum] {
                self.0
            }
        }
    };
}
impl_array_ops!(D1, 1);
impl_array_ops!(D2, 2);
impl_array_ops!(D3, 3);
impl_array_ops!(D4, 4);
impl_array_ops!(D5, 5);

impl<D: Dimension, T: Copy, I: Copy + Into<T>> TryFrom<Vec<I>> for Vector<D, T> {
    type Error = ();

    fn try_from(value: Vec<I>) -> Result<Self, Self::Error> {
        if value.len() != D::N {
            return Err(());
        }
        Ok(Self::from_fn(|i| value[i].into()))
    }
}

impl<D: Dimension, T: Copy> Vector<D, T> {
    pub fn from_fn(f: impl FnMut(usize) -> T) -> Self {
        Vector(D::Array::from_fn(f))
    }
    pub fn new(inner: D::Array<T>) -> Self {
        Vector(inner)
    }
    pub fn dim() -> usize {
        D::N
    }
}

impl<D: Dimension, T: Copy> Vector<D, T> {
    pub fn fill(val: T) -> Self {
        Self::from_fn(|_| val)
    }
    pub fn map<U: Copy>(self, mut f: impl FnMut(T) -> U) -> Vector<D, U> {
        Vector(D::Array::from_fn(|i| f(self.0[i])))
    }
    pub fn map_element(mut self, i: usize, f: impl FnOnce(T) -> T) -> Vector<D, T> {
        self.0[i] = f(self.0[i]);
        self
    }
    pub fn fold<U>(self, mut state: U, mut f: impl FnMut(U, T) -> U) -> U {
        for v in self.0 {
            state = f(state, v);
        }
        state
    }
    pub fn zip<U: Copy, V: Copy>(
        self,
        other: Vector<D, U>,
        mut f: impl FnMut(T, U) -> V,
    ) -> Vector<D, V> {
        Vector(D::Array::from_fn(|i| f(self.0[i], other.0[i])))
    }
    pub fn zip_enumerate<U: Copy, V: Copy>(
        self,
        other: Vector<D, U>,
        mut f: impl FnMut(usize, T, U) -> V,
    ) -> Vector<D, V> {
        Vector(D::Array::from_fn(|i| f(i, self.0[i], other.0[i])))
    }
    pub fn into_elem<U: Copy>(self) -> Vector<D, U>
    where
        T: Into<U>,
    {
        self.map(|v| v.into())
    }
    pub fn try_into_elem<U: Copy>(self) -> Result<Vector<D, U>, T::Error>
    where
        T: TryInto<U>,
    {
        todo!()
        //NO_PUSH_main
        //// Safety: Standard way to initialize an MaybeUninit array
        //let mut out: [MaybeUninit<U>; N] = unsafe { MaybeUninit::uninit().assume_init() };
        //for i in 0..N {
        //    out[i].write(T::try_into(self.0[i])?);
        //}
        //// Safety: We have just initialized all values in the loop above
        //Ok(Vector(out.map(|v| unsafe { v.assume_init() })))
    }
}
impl<D: Dimension, T: std::ops::Mul<Output = T> + Copy> Vector<D, T> {
    pub fn scale(self, v: T) -> Self {
        self.map(|w| w * v)
    }
}
impl<D: Dimension, T: Copy> std::ops::Index<usize> for Vector<D, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl<D: Dimension, T: Copy> std::ops::IndexMut<usize> for Vector<D, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
impl<D: Dimension, T: CoordinateType> Vector<D, Coordinate<T>> {
    pub fn as_index(self) -> D::Array<usize> {
        self.map(|v| v.raw as usize).0
    }
}

impl<D: Dimension> Vector<D, GlobalCoordinate> {
    pub fn local(self) -> Vector<D, LocalCoordinate> {
        self.map(LocalCoordinate::interpret_as)
    }
}
impl<D: Dimension> Vector<D, LocalCoordinate> {
    pub fn global(self) -> Vector<D, GlobalCoordinate> {
        self.map(GlobalCoordinate::interpret_as)
    }
}
impl<D: Dimension> Vector<D, u32> {
    pub fn global(self) -> Vector<D, GlobalCoordinate> {
        self.map(|v| v.into())
    }
    pub fn chunk(self) -> Vector<D, ChunkCoordinate> {
        self.map(|v| v.into())
    }
}
impl<D: Dimension, T: CoordinateType> Vector<D, Coordinate<T>> {
    pub fn raw(self) -> Vector<D, u32> {
        self.map(|v| v.raw)
    }
}
impl<D: Dimension> Vector<D, u32> {
    pub fn f32(self) -> Vector<D, f32> {
        self.map(|v| v as f32)
    }
}
impl<T: Copy> std::ops::Deref for Vector<D1, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0[0]
    }
}

impl<T: Copy> Vector<D3, T> {
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
impl<T: Copy> Vector<D2, T> {
    pub fn x(&self) -> T {
        self.0[1]
    }
    pub fn y(&self) -> T {
        self.0[0]
    }
}

impl<D: LargerDim, T: Copy> Vector<D, T> {
    pub fn push_dim_small(self, extra: T) -> Vector<D::Larger, T> {
        Vector::from_fn(|i| if i == D::N { extra } else { self.0[i] })
    }
    pub fn push_dim_large(self, extra: T) -> Vector<D::Larger, T> {
        Vector::from_fn(|i| if i == 0 { extra } else { self.0[i - 1] })
    }
    //pub fn add_dim(self, dim: usize, value: T) -> Vector<D3, T> {
    //    Vector(std::array::from_fn(|i| match i.cmp(&dim) {
    //        std::cmp::Ordering::Less => self.0[i],
    //        std::cmp::Ordering::Equal => value,
    //        std::cmp::Ordering::Greater => self.0[i - 1],
    //    }))
    //}
}
impl<D: LargerDim, T: num::One + Copy> Vector<D, T> {
    pub fn to_homogeneous_coord(self) -> Vector<D::Larger, T> {
        self.push_dim_large(num::one())
    }
}

impl<D: SmallerDim, T: Copy> Vector<D, T> {
    pub fn drop_dim(self, dim: usize) -> Vector<D::Smaller, T> {
        Vector::from_fn(|i| if i < dim { self.0[i] } else { self.0[i + 1] })
    }
    pub fn to_non_homogeneous_coord(self) -> Vector<D::Smaller, T> {
        self.drop_dim(0)
    }
}

impl<D: Dimension, T: Copy> IntoIterator for Vector<D, T> {
    type Item = <<D as Dimension>::Array<T> as IntoIterator>::Item;

    type IntoIter = <<D as Dimension>::Array<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<D: Dimension, T: Copy> Vector<D, T> {
    pub fn dot<O: num::Zero + Copy + Add<O, Output = O>, U: Copy + Mul<T, Output = O>>(
        &self,
        other: &Vector<D, U>,
    ) -> O {
        let mut s = num::zero();
        for i in 0..D::N {
            s = s + other.0[i] * self.0[i];
        }
        s
    }
}
impl<D: Dimension, T: Neg + Copy> Neg for Vector<D, T>
where
    T::Output: Copy,
{
    type Output = Vector<D, T::Output>;
    fn neg(self) -> Self::Output {
        self.map(|v| v.neg())
    }
}
impl<D: Dimension, O: Copy, U: Copy, T: Copy + Add<U, Output = O>> Add<Vector<D, U>>
    for Vector<D, T>
{
    type Output = Vector<D, O>;
    fn add(self, rhs: Vector<D, U>) -> Self::Output {
        self.zip(rhs, Add::add)
    }
}
impl<D: Dimension, O: Copy, U: Copy, T: Copy + Sub<U, Output = O>> Sub<Vector<D, U>>
    for Vector<D, T>
{
    type Output = Vector<D, O>;
    fn sub(self, rhs: Vector<D, U>) -> Self::Output {
        self.zip(rhs, Sub::sub)
    }
}
impl<D: Dimension, O: Copy, U: Copy, T: Copy + Mul<U, Output = O>> Mul<Vector<D, U>>
    for Vector<D, T>
{
    type Output = Vector<D, O>;
    fn mul(self, rhs: Vector<D, U>) -> Self::Output {
        self.zip(rhs, Mul::mul)
    }
}
impl<D: Dimension, O: Copy, U: Copy, T: Copy + Div<U, Output = O>> Div<Vector<D, U>>
    for Vector<D, T>
{
    type Output = Vector<D, O>;
    fn div(self, rhs: Vector<D, U>) -> Self::Output {
        self.zip(rhs, Div::div)
    }
}

impl<T: Copy> From<Vector<D2, T>> for cgmath::Vector2<T> {
    fn from(value: Vector<D2, T>) -> Self {
        cgmath::Vector2 {
            x: value.x(),
            y: value.y(),
        }
    }
}

impl<T: Copy> From<cgmath::Vector2<T>> for Vector<D2, T> {
    fn from(value: cgmath::Vector2<T>) -> Self {
        Self::from([value.y, value.x])
    }
}

impl<T: Copy> From<Vector<D3, T>> for cgmath::Vector3<T> {
    fn from(value: Vector<D3, T>) -> Self {
        cgmath::Vector3 {
            x: value.x(),
            y: value.y(),
            z: value.z(),
        }
    }
}
impl<T: Copy> From<Vector<D3, T>> for cgmath::Point3<T> {
    fn from(value: Vector<D3, T>) -> Self {
        cgmath::Point3 {
            x: value.x(),
            y: value.y(),
            z: value.z(),
        }
    }
}

impl<T: Copy> From<cgmath::Vector3<T>> for Vector<D3, T> {
    fn from(value: cgmath::Vector3<T>) -> Self {
        Self::from([value.z, value.y, value.x])
    }
}
impl<T: Copy> From<cgmath::Point3<T>> for Vector<D3, T> {
    fn from(value: cgmath::Point3<T>) -> Self {
        Self::from([value.z, value.y, value.x])
    }
}

impl<T: Copy> From<Vector<D4, T>> for cgmath::Vector4<T> {
    fn from(value: Vector<D4, T>) -> Self {
        cgmath::Vector4 {
            x: value.0[3],
            y: value.0[2],
            z: value.0[1],
            w: value.0[0],
        }
    }
}

impl<T: Copy> From<cgmath::Vector4<T>> for Vector<D4, T> {
    fn from(value: cgmath::Vector4<T>) -> Self {
        Self::from([value.w, value.z, value.y, value.x])
    }
}

impl Vector<D3, f32> {
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

#[cfg(feature = "python")]
mod py {
    use super::*;
    use pyo3::prelude::*;

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
    impl<'source, D: Dimension, T: Copy + FromPyObject<'source>> FromPyObject<'source>
        for Vector<D, T>
    {
        fn extract(ob: &'source PyAny) -> PyResult<Self> {
            //ob.extract::<D::Array<T>>().map(Self::from)
            todo!()
        }
    }

    impl<D: Dimension, T: Copy + IntoPy<PyObject>> IntoPy<PyObject> for Vector<D, T> {
        fn into_py(self, py: Python<'_>) -> PyObject {
            //PyList::new(py, self.into_iter().map(|v| v.into_py(py))).into()
            todo!()
        }
    }

    //impl<D: Dimension, T: ToPyObject> ToPyObject for Vector<D, T> {
    //    fn to_object(&self, py: Python<'_>) -> PyObject {
    //        PyList::new(py, self.0.iter().map(|v| v.to_object(py))).into()
    //    }
    //}

    impl<D: Dimension, T: Copy + state_link::py::PyState> state_link::py::PyState for Vector<D, T> {
        fn build_handle(
            py: Python,
            inner: state_link::GenericNodeHandle,
            store: Py<state_link::py::Store>,
        ) -> PyObject {
            //<D::Array<T>>::build_handle(py, inner.index(0), store)
            todo!()
        }
    }
}

pub fn hmul<D: Dimension, T: CoordinateType>(s: Vector<D, Coordinate<T>>) -> usize {
    s.into_iter().map(|v| v.raw as usize).product()
}

pub fn to_linear<D: Dimension, T: CoordinateType>(
    pos: Vector<D, Coordinate<T>>,
    dim: Vector<D, Coordinate<T>>,
) -> usize {
    let mut out = pos[0].raw as usize;
    for i in 1..D::N {
        out = out * dim[i].raw as usize + pos[i].raw as usize;
    }
    out
}

pub fn from_linear<D: Dimension, T: CoordinateType>(
    mut linear_pos: usize,
    dim: Vector<D, Coordinate<T>>,
) -> Vector<D, Coordinate<T>> {
    let mut out = Vector::<D, Coordinate<T>>::fill(0.into());
    for i in (0..D::N).rev() {
        let ddim = dim[i].raw as usize;
        out[i] = ((linear_pos % ddim) as u32).into();
        linear_pos /= ddim;
    }
    out
}

pub type LocalVoxelPosition = Vector<D3, LocalCoordinate>;
pub type VoxelPosition = Vector<D3, GlobalCoordinate>;
pub type BrickPosition = Vector<D3, ChunkCoordinate>;

pub type PixelPosition = Vector<D2, GlobalCoordinate>;
