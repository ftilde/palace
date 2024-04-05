use std::{
    mem::MaybeUninit,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::{
    coordinate::{ChunkCoordinate, Coordinate, CoordinateType, GlobalCoordinate, LocalCoordinate},
    dim::*,
};
use id::{Id, Identify};

#[repr(transparent)]
#[derive(Copy, Clone, Default)]
pub struct Vector<D: Dimension, T: Copy>(D::Array<T>);

impl<D: Dimension, T: Copy + Identify> Identify for Vector<D, T> {
    fn id(&self) -> Id {
        Id::combine_it(self.into_iter().map(|v| v.id()))
    }
}

unsafe impl<D: Dimension, T: Copy + bytemuck::Zeroable> bytemuck::Zeroable for Vector<D, T> {}
unsafe impl<D: Dimension, T: Copy + bytemuck::Pod> bytemuck::Pod for Vector<D, T> {}

impl<D: Dimension, T: Copy + PartialEq> PartialEq for Vector<D, T> {
    fn eq(&self, other: &Self) -> bool {
        Vector::zip(*self, *other, |l, r| l == r).hand()
    }
}
impl<D: Dimension, T: Copy + Eq> Eq for Vector<D, T> {}

impl<D: Dimension, T: Copy + PartialOrd> PartialOrd for Vector<D, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Vector::zip(*self, *other, |l, r| l.partial_cmp(&r)).fold(None, |l, r| match (l, r) {
            (Some(l), Some(r)) => Some(l.then(r)),
            (Some(l), None) => Some(l),
            (None, o) => o,
        })
    }
}
impl<D: Dimension, T: Copy + Ord> Ord for Vector<D, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        Vector::zip(*self, *other, |l, r| l.cmp(&r)).fold(Ordering::Equal, Ordering::then)
    }
}
impl<D: Dimension, T: Copy + std::fmt::Debug> std::fmt::Debug for Vector<D, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_list();
        for v in *self {
            f.entry(&v);
        }
        f.finish()
    }
}
impl<D: Dimension, T: Copy + std::hash::Hash> std::hash::Hash for Vector<D, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for v in *self {
            v.hash(state);
        }
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
    pub fn inner(self) -> D::Array<T> {
        self.0
    }

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
        // Safety: Standard way to initialize an MaybeUninit array
        let mut out: Vector<D, MaybeUninit<U>> = unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..D::N {
            out[i].write(T::try_into(self.0[i])?);
        }
        // Safety: We have just initialized all values in the loop above
        Ok(out.map(|v| unsafe { v.assume_init() }))
    }
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < D::N {
            Some(&self.0[index])
        } else {
            None
        }
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
impl<D: Dimension> Vector<D, u8> {
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

pub struct VecIter<D: Dimension, T: Copy> {
    vec: Vector<D, T>,
    i: usize,
}

impl<D: Dimension, T: Copy> Iterator for VecIter<D, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < D::N {
            let i = self.i;
            self.i += 1;
            Some(self.vec[i])
        } else {
            None
        }
    }
}
impl<D: Dimension, T: Copy> ExactSizeIterator for VecIter<D, T> {
    fn len(&self) -> usize {
        D::N - self.i
    }
}

impl<D: Dimension, T: Copy> IntoIterator for Vector<D, T> {
    type Item = T;

    type IntoIter = VecIter<D, T>;

    fn into_iter(self) -> Self::IntoIter {
        VecIter { vec: self, i: 0 }
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
    use pyo3::{prelude::*, types::PyList};

    impl<'source, D: Dimension, T: Copy> FromPyObject<'source> for Vector<D, T>
    where
        D::Array<T>: FromPyObject<'source>,
    {
        fn extract(ob: &'source PyAny) -> PyResult<Self> {
            ob.extract::<D::Array<T>>().map(Self)
        }
    }

    impl<D: Dimension, T: Copy + IntoPy<PyObject>> IntoPy<PyObject> for Vector<D, T> {
        fn into_py(self, py: Python<'_>) -> PyObject {
            PyList::new(py, self.into_iter().map(|v| v.into_py(py))).into()
        }
    }

    //impl<D: Dimension, T: ToPyObject> ToPyObject for Vector<D, T> {
    //    fn to_object(&self, py: Python<'_>) -> PyObject {
    //        PyList::new(py, self.0.iter().map(|v| v.to_object(py))).into()
    //    }
    //}

    impl<D: Dimension, T: Copy + state_link::py::PyState> state_link::py::PyState for Vector<D, T>
    where
        D::Array<T>: IntoPy<PyObject> + for<'f> FromPyObject<'f>,
    {
        fn build_handle(
            py: Python,
            inner: state_link::GenericNodeHandle,
            store: Py<state_link::py::Store>,
        ) -> PyObject {
            let init = state_link::py::NodeHandleArray::new::<T>(inner, D::N, store);
            PyCell::new(py, init).unwrap().to_object(py)
        }
    }
}

pub mod state_link_impl {
    use super::*;
    use state_link::*;

    pub struct VecNodeHandle<D, T> {
        inner: state_link::GenericNodeHandle,
        _d: std::marker::PhantomData<D>,
        _t: std::marker::PhantomData<T>,
    }
    impl<D: Dimension, T: Copy + State> NodeHandle for VecNodeHandle<D, T> {
        type NodeType = Vector<D, T>;

        fn pack(t: GenericNodeHandle) -> Self {
            Self {
                inner: t,
                _d: Default::default(),
                _t: Default::default(),
            }
        }

        fn unpack(&self) -> &GenericNodeHandle {
            &self.inner
        }
    }
    impl<D: Dimension, T: Copy + State> State for Vector<D, T> {
        type NodeHandle = VecNodeHandle<D, T>;

        fn store(&self, store: &mut Store) -> NodeRef {
            let refs = self.into_iter().map(|v| v.store(store)).collect();
            store.push(Node::Seq(refs))
        }

        fn load(store: &Store, location: NodeRef) -> Result<Self> {
            // If only std::array::try_from_fn were stable...
            // TODO: replace once it is https://github.com/rust-lang/rust/issues/89379
            if let ResolveResult::Seq(seq) = store.to_val(location)? {
                let results = Vector::from_fn(|i| {
                    seq.get(i)
                        .ok_or(Error::SeqTooShort)
                        .and_then(|v| T::load(store, *v))
                });
                for (i, r) in results.into_iter().enumerate() {
                    if let Err(_e) = r {
                        match results.into_iter().nth(i).unwrap() {
                            Ok(_) => std::unreachable!(),
                            Err(e) => return Err(e),
                        }
                    }
                }
                Ok(results.map(|v| match v {
                    Ok(e) => e,
                    Err(_) => std::unreachable!(),
                }))
            } else {
                Err(Error::IncorrectType)
            }
        }

        fn write(&self, store: &mut Store, at: NodeRef) -> Result<()> {
            let seq = if let ResolveResult::Seq(seq) = store.to_val(at)? {
                seq.clone() //TODO: instead of cloning we can probably also just take the old value out
            } else {
                return Err(Error::IncorrectType);
            };

            if seq.len() != D::N {
                return Err(Error::IncorrectType);
            }

            for (v, slot) in self.into_iter().zip(seq.iter()) {
                v.write(store, *slot)?;
            }

            Ok(())
        }
    }
    impl<D: Dimension, T: Copy + State> VecNodeHandle<D, T> {
        pub fn at(&self, i: usize) -> <T as State>::NodeHandle {
            <T as State>::NodeHandle::pack(self.inner.index(i))
        }
    }
}

impl<D: Dimension> Vector<D, f32> {
    pub fn hmin(&self) -> f32 {
        self.fold(f32::INFINITY, |a, b| a.min(b))
    }
    pub fn hmax(&self) -> f32 {
        self.fold(-f32::INFINITY, |a, b| a.max(b))
    }
    pub fn hadd(&self) -> f32 {
        self.fold(0.0, |a, b| a + b)
    }
    pub fn hmul(&self) -> f32 {
        self.fold(1.0, |a, b| a * b)
    }
}
impl<D: Dimension> Vector<D, bool> {
    pub fn all(&self) -> bool {
        self.fold(true, |a, b| a && b)
    }
    pub fn any(&self) -> bool {
        self.fold(false, |a, b| a || b)
    }
}
impl<D: Dimension, T: CoordinateType> Vector<D, Coordinate<T>> {
    pub fn hmul(&self) -> usize {
        self.into_iter().map(|v| v.raw as usize).product()
    }

    pub fn to_ndarray_dim(self) -> D::NDArrayDim {
        D::to_ndarray_dim(self.as_index())
    }
}
impl<D: Dimension> Vector<D, bool> {
    pub fn hand(self) -> bool {
        self.fold(true, |l, r| l && r)
    }

    pub fn hor(self) -> bool {
        self.fold(false, |l, r| l || r)
    }
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
