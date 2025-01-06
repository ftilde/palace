use std::{
    mem::MaybeUninit,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::{
    coordinate::{ChunkCoordinate, Coordinate, CoordinateType, GlobalCoordinate, LocalCoordinate},
    dim::*,
};
use id::{Id, Identify};
use num::traits::SaturatingSub;

#[repr(transparent)]
#[derive(Clone, Default)]
pub struct Vector<D: DynDimension, T: Copy>(D::DynArray<T>);

impl<D: DynDimension, T: Copy + Identify> Identify for Vector<D, T> {
    fn id(&self) -> Id {
        Id::combine_it(self.iter().map(|v| v.id()))
    }
}

unsafe impl<D: Dimension, T: Copy + bytemuck::Zeroable> bytemuck::Zeroable for Vector<D, T> {}
unsafe impl<D: Dimension, T: Copy + bytemuck::Pod> bytemuck::Pod for Vector<D, T> {}
impl<D: Dimension, T: Copy> Copy for Vector<D, T> {}

impl<D: DynDimension, T: Copy + PartialEq> PartialEq for Vector<D, T> {
    fn eq(&self, other: &Self) -> bool {
        Vector::zip(self, other, |l, r| l == r).hand()
    }
}
impl<D: DynDimension, T: Copy + Eq> Eq for Vector<D, T> {}

//impl<D: DynDimension, T: Copy + PartialOrd> PartialOrd for Vector<D, T> {
//    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//        Vector::zip(self, other, |l, r| l.partial_cmp(&r)).fold(None, |l, r| match (l, r) {
//            (Some(l), Some(r)) => Some(l.then(r)),
//            (Some(l), None) => Some(l),
//            (None, o) => o,
//        })
//    }
//}
//impl<D: DynDimension, T: Copy + Ord> Ord for Vector<D, T> {
//    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//        use std::cmp::Ordering;
//        Vector::zip(self, other, |l, r| l.cmp(&r)).fold(Ordering::Equal, Ordering::then)
//    }
//}
impl<D: DynDimension, T: Copy + std::fmt::Debug> std::fmt::Debug for Vector<D, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_list();
        for v in self.iter() {
            f.entry(&v);
        }
        f.finish()
    }
}
impl<D: DynDimension, T: Copy + std::hash::Hash> std::hash::Hash for Vector<D, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for v in self.iter() {
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

impl<D: DynDimension, T: Copy, I: Copy + Into<T>> TryFrom<Vec<I>> for Vector<D, T> {
    type Error = ();

    fn try_from(value: Vec<I>) -> Result<Self, Self::Error> {
        Self::try_from_fn_and_len(value.len(), |i| value[i].into())
    }
}

impl<D: Dimension, T: Copy> Vector<D, T> {
    pub fn from_fn(f: impl FnMut(usize) -> T) -> Self {
        Self::try_from_fn_and_len(D::N, f).unwrap()
    }

    pub fn fill(val: T) -> Self {
        Self::from_fn(|_| val)
    }
}

impl<T: Copy> Vector<DDyn, T> {
    pub fn from_fn_and_len(size: usize, f: impl FnMut(usize) -> T) -> Self {
        Self::try_from_fn_and_len(size, f).unwrap()
    }
}

impl<D: DynDimension, T: Copy> Vector<D, T> {
    pub fn into_dyn(self) -> Vector<DDyn, T> {
        Vector(D::into_dyn(self.0))
    }
    pub fn try_into_static<D2: Dimension>(self) -> Option<Vector<D2, T>> {
        D::try_into_dim::<D2, T>(self.0).map(Vector)
    }
    pub fn try_from_fn_and_len(size: usize, f: impl FnMut(usize) -> T) -> Result<Self, ()> {
        Ok(Vector(D::DynArray::try_from_fn_and_len(size, f)?))
    }
    pub fn try_from_slice(vals: &[T]) -> Result<Self, ()> {
        Vector::try_from_fn_and_len(vals.len(), |i| vals[i])
    }
    fn from_fn_and_trusted_len(size: usize, f: impl FnMut(usize) -> T) -> Self {
        Self::try_from_fn_and_len(size, f).unwrap()
    }
    pub fn fill_with_len(val: T, len: usize) -> Self {
        Self::from_fn_and_trusted_len(len, |_| val)
    }
    pub fn new(inner: D::DynArray<T>) -> Self {
        Vector(inner)
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn dim(&self) -> D {
        D::dim_of_array(&self.0)
    }
    pub fn inner(self) -> D::DynArray<T> {
        self.0
    }
    pub fn map<U: Copy>(&self, mut f: impl FnMut(T) -> U) -> Vector<D, U> {
        Vector::from_fn_and_trusted_len(self.len(), |i| f(self.0[i]))
    }
    pub fn map_element(mut self, i: usize, f: impl FnOnce(T) -> T) -> Vector<D, T> {
        self.0[i] = f(self.0[i]);
        self
    }
    pub fn fold<U>(&self, mut state: U, mut f: impl FnMut(U, T) -> U) -> U {
        for v in self.iter() {
            state = f(state, *v);
        }
        state
    }
    pub fn zip<U: Copy, V: Copy>(
        &self,
        other: &Vector<D, U>,
        mut f: impl FnMut(T, U) -> V,
    ) -> Vector<D, V> {
        assert_eq!(self.len(), other.len());
        Vector::from_fn_and_trusted_len(self.len(), |i| f(self.0[i], other.0[i]))
    }
    pub fn zip_enumerate<U: Copy, V: Copy>(
        &self,
        other: &Vector<D, U>,
        mut f: impl FnMut(usize, T, U) -> V,
    ) -> Vector<D, V> {
        assert_eq!(self.len(), other.len());
        Vector::from_fn_and_trusted_len(self.len(), |i| f(i, self.0[i], other.0[i]))
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
        for i in 0..self.len() {
            out[i].write(T::try_into(self.0[i])?);
        }
        // Safety: We have just initialized all values in the loop above
        Ok(out.map(|v| unsafe { v.assume_init() }))
    }
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len() {
            Some(&self.0[index])
        } else {
            None
        }
    }
    pub fn small_dim_element(&self) -> T {
        self.0.as_slice()[self.len() - 1]
    }
}
impl<D: DynDimension, T: std::ops::Mul<Output = T> + Copy> Vector<D, T> {
    pub fn scale(&self, v: T) -> Self {
        self.map(|w| w * v)
    }
}
impl<D: DynDimension, T: Copy> std::ops::Index<usize> for Vector<D, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl<D: DynDimension, T: Copy> std::ops::IndexMut<usize> for Vector<D, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
impl<D: DynDimension, T: CoordinateType> Vector<D, Coordinate<T>> {
    pub fn as_index(&self) -> D::DynArray<usize> {
        self.map(|v| v.raw as usize).0
    }
}

impl<D: DynDimension> Vector<D, GlobalCoordinate> {
    pub fn local(&self) -> Vector<D, LocalCoordinate> {
        self.map(LocalCoordinate::interpret_as)
    }
}
impl<D: DynDimension> Vector<D, LocalCoordinate> {
    pub fn global(&self) -> Vector<D, GlobalCoordinate> {
        self.map(GlobalCoordinate::interpret_as)
    }
}
impl<D: DynDimension> Vector<D, u32> {
    pub fn global(&self) -> Vector<D, GlobalCoordinate> {
        self.map(|v| v.into())
    }
    pub fn chunk(&self) -> Vector<D, ChunkCoordinate> {
        self.map(|v| v.into())
    }
    pub fn local(&self) -> Vector<D, LocalCoordinate> {
        self.map(|v| v.into())
    }
}
impl<D: DynDimension, T: CoordinateType> Vector<D, Coordinate<T>> {
    pub fn raw(&self) -> Vector<D, u32> {
        self.map(|v| v.raw)
    }
}
impl<D: DynDimension> Vector<D, u32> {
    pub fn f32(&self) -> Vector<D, f32> {
        self.map(|v| v as f32)
    }
}
impl<D: DynDimension> Vector<D, u8> {
    pub fn f32(&self) -> Vector<D, f32> {
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
    pub fn push_dim_small(&self, extra: T) -> Vector<D::Larger, T> {
        Vector::from_fn_and_trusted_len(self.len() + 1, |i| {
            if i == self.dim().n() {
                extra
            } else {
                self.0[i]
            }
        })
    }
    pub fn push_dim_large(&self, extra: T) -> Vector<D::Larger, T> {
        Vector::from_fn_and_trusted_len(
            self.len() + 1,
            |i| if i == 0 { extra } else { self.0[i - 1] },
        )
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
    pub fn to_homogeneous_coord(&self) -> Vector<D::Larger, T> {
        self.push_dim_large(num::one())
    }
}

impl<D: SmallerDim, T: Copy> Vector<D, T> {
    pub fn drop_dim(&self, dim: usize) -> Vector<D::Smaller, T> {
        Vector::from_fn_and_trusted_len(self.len() - 1, |i| {
            if i < dim {
                self.0[i]
            } else {
                self.0[i + 1]
            }
        })
    }
    pub fn pop_dim_small(&self) -> Vector<D::Smaller, T> {
        self.drop_dim(self.len() - 1)
    }
    pub fn pop_dim_large(&self) -> Vector<D::Smaller, T> {
        self.drop_dim(0)
    }
    pub fn to_non_homogeneous_coord(self) -> Vector<D::Smaller, T> {
        self.pop_dim_large()
    }
}

pub struct VecIter<D: DynDimension, T: Copy> {
    vec: Vector<D, T>,
    i: usize,
}

impl<D: DynDimension, T: Copy> Iterator for VecIter<D, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.vec.len() {
            let i = self.i;
            self.i += 1;
            Some(self.vec[i])
        } else {
            None
        }
    }
}
impl<D: DynDimension, T: Copy> ExactSizeIterator for VecIter<D, T> {
    fn len(&self) -> usize {
        self.vec.len() - self.i
    }
}

impl<D: DynDimension, T: Copy> IntoIterator for Vector<D, T> {
    type Item = T;

    type IntoIter = VecIter<D, T>;

    fn into_iter(self) -> Self::IntoIter {
        VecIter { vec: self, i: 0 }
    }
}

impl<D: DynDimension, T: Copy> Vector<D, T> {
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.0.as_slice().iter()
    }
}

impl<D: DynDimension, T: Copy> Vector<D, T> {
    pub fn dot<O: num::Zero + Copy + Add<O, Output = O>, U: Copy + Mul<T, Output = O>>(
        &self,
        other: &Vector<D, U>,
    ) -> O {
        assert_eq!(self.len(), other.len());
        let mut s = num::zero();
        for i in 0..self.len() {
            s = s + other.0[i] * self.0[i];
        }
        s
    }
}
impl<D: DynDimension, T: Neg + Copy> Neg for Vector<D, T>
where
    T::Output: Copy,
{
    type Output = Vector<D, T::Output>;
    fn neg(self) -> Self::Output {
        self.map(|v| v.neg())
    }
}
impl<D: DynDimension, O: Copy, U: Copy, T: Copy + Add<U, Output = O>> Add<&Vector<D, U>>
    for &Vector<D, T>
{
    type Output = Vector<D, O>;
    fn add(self, rhs: &Vector<D, U>) -> Self::Output {
        self.zip(rhs, Add::add)
    }
}
impl<D: DynDimension, O: Copy, U: Copy, T: Copy + Sub<U, Output = O>> Sub<&Vector<D, U>>
    for &Vector<D, T>
{
    type Output = Vector<D, O>;
    fn sub(self, rhs: &Vector<D, U>) -> Self::Output {
        self.zip(rhs, Sub::sub)
    }
}
impl<D: DynDimension, T: Copy + SaturatingSub> Vector<D, T> {
    pub fn saturating_sub(&self, rhs: &Vector<D, T>) -> Vector<D, T> {
        self.zip(rhs, |l, r| l.saturating_sub(&r))
    }
}

impl<D: DynDimension, O: Copy, U: Copy, T: Copy + Mul<U, Output = O>> Mul<&Vector<D, U>>
    for &Vector<D, T>
{
    type Output = Vector<D, O>;
    fn mul(self, rhs: &Vector<D, U>) -> Self::Output {
        self.zip(rhs, Mul::mul)
    }
}
impl<D: DynDimension, O: Copy, U: Copy, T: Copy + Div<U, Output = O>> Div<&Vector<D, U>>
    for &Vector<D, T>
{
    type Output = Vector<D, O>;
    fn div(self, rhs: &Vector<D, U>) -> Self::Output {
        self.zip(rhs, Div::div)
    }
}

// Add convenience versions without refs for Copy vectors
impl<D: DynDimension, O: Copy, U: Copy, T: Copy + Add<U, Output = O>> Add<Vector<D, U>>
    for Vector<D, T>
{
    type Output = Vector<D, O>;
    fn add(self, rhs: Vector<D, U>) -> Self::Output {
        self.zip(&rhs, Add::add)
    }
}
impl<D: DynDimension, O: Copy, U: Copy, T: Copy + Sub<U, Output = O>> Sub<Vector<D, U>>
    for Vector<D, T>
{
    type Output = Vector<D, O>;
    fn sub(self, rhs: Vector<D, U>) -> Self::Output {
        self.zip(&rhs, Sub::sub)
    }
}
impl<D: DynDimension, O: Copy, U: Copy, T: Copy + Mul<U, Output = O>> Mul<Vector<D, U>>
    for Vector<D, T>
{
    type Output = Vector<D, O>;
    fn mul(self, rhs: Vector<D, U>) -> Self::Output {
        self.zip(&rhs, Mul::mul)
    }
}
impl<D: DynDimension, O: Copy, U: Copy, T: Copy + Div<U, Output = O>> Div<Vector<D, U>>
    for Vector<D, T>
{
    type Output = Vector<D, O>;
    fn div(self, rhs: Vector<D, U>) -> Self::Output {
        self.zip(&rhs, Div::div)
    }
}

impl<D: DynDimension, T: Copy + Ord> Vector<D, T> {
    pub fn max(&self, rhs: &Vector<D, T>) -> Vector<D, T> {
        self.zip(rhs, |l, r| l.max(r))
    }
    pub fn min(&self, rhs: &Vector<D, T>) -> Vector<D, T> {
        self.zip(rhs, |l, r| l.min(r))
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

impl<D: DynDimension> Vector<D, f32> {
    pub fn length(&self) -> f32 {
        self.map(|v| v * v).fold(0.0, f32::add).sqrt()
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        let len_inv = 1.0 / len;
        self.map(|v| v * len_inv)
    }
}

impl Vector<D3, f32> {
    pub fn cross(self, other: Self) -> Self {
        let v1: cgmath::Vector3<f32> = self.into();
        let v2: cgmath::Vector3<f32> = other.into();
        v1.cross(v2).into()
    }
}

#[cfg(feature = "python")]
mod py {
    use super::*;
    use pyo3::{prelude::*, types::PyList, IntoPyObjectExt};

    impl<'source, D: DynDimension, T: Copy> FromPyObject<'source> for Vector<D, T>
    where
        D::DynArray<T>: FromPyObject<'source>,
    {
        fn extract_bound(ob: &Bound<'source, PyAny>) -> PyResult<Self> {
            ob.extract::<D::DynArray<T>>().map(Self)
        }
    }

    impl<'py, D: DynDimension, T: Copy + IntoPyObject<'py>> IntoPyObject<'py> for Vector<D, T> {
        type Target = PyList;

        type Output = Bound<'py, Self::Target>;

        type Error = PyErr;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            Ok(PyList::new(py, self.into_iter())?.into_pyobject(py)?)
        }
    }

    //impl<D: DynDimension, T: ToPyObject> ToPyObject for Vector<D, T> {
    //    fn to_object(&self, py: Python<'_>) -> PyObject {
    //        PyList::new(py, self.0.iter().map(|v| v.to_object(py))).into()
    //    }
    //}

    impl<D: Dimension, T: Copy + state_link::py::PyState> state_link::py::PyState for Vector<D, T>
    where
        D::Array<T>: for<'f> IntoPyObject<'f> + for<'f> FromPyObject<'f>,
    {
        fn build_handle(
            py: Python,
            inner: state_link::GenericNodeHandle,
            store: Py<state_link::py::Store>,
        ) -> PyObject {
            let init = state_link::py::NodeHandleArray::new::<T>(inner, D::N, store);
            init.into_py_any(py).unwrap()
        }
    }

    impl<D: DynDimension, T: Copy> pyo3_stub_gen::PyStubType for Vector<D, T> {
        fn type_output() -> pyo3_stub_gen::TypeInfo {
            pyo3_stub_gen::TypeInfo {
                name: format!("Vector"),
                import: Default::default(),
            }
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
    impl<D: DynDimension, T: Copy + State> VecNodeHandle<D, T> {
        pub fn at(&self, i: usize) -> <T as State>::NodeHandle {
            <T as State>::NodeHandle::pack(self.inner.index(i))
        }
    }
}

impl<D: DynDimension> Vector<D, f32> {
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
impl<D: DynDimension> Vector<D, bool> {
    pub fn all(&self) -> bool {
        self.fold(true, |a, b| a && b)
    }
    pub fn any(&self) -> bool {
        self.fold(false, |a, b| a || b)
    }
}
impl<D: DynDimension, T: CoordinateType> Vector<D, Coordinate<T>> {
    pub fn hmul(&self) -> usize {
        self.iter().map(|v| v.raw as usize).product()
    }
}
impl<D: DynDimension> Vector<D, u32> {
    pub fn hmul(&self) -> usize {
        self.iter().map(|v| *v as usize).product()
    }
}

//impl<D: Dimension, T: CoordinateType> Vector<D, Coordinate<T>> {
//    pub fn to_ndarray_dim(self) -> D::NDArrayDim {
//        D::to_ndarray_dim(self.as_index())
//    }
//}
impl<D: DynDimension, T: CoordinateType> Vector<D, Coordinate<T>> {
    pub fn to_ndarray_dim(&self) -> D::NDArrayDimDyn {
        D::to_ndarray_dim_dyn(self.as_index())
    }
}

impl<D: DynDimension> Vector<D, bool> {
    pub fn hand(self) -> bool {
        self.fold(true, |l, r| l && r)
    }

    pub fn hor(self) -> bool {
        self.fold(false, |l, r| l || r)
    }
}

pub fn to_linear<D: DynDimension, T: CoordinateType>(
    pos: &Vector<D, Coordinate<T>>,
    dim: &Vector<D, Coordinate<T>>,
) -> usize {
    assert_eq!(pos.len(), dim.len());
    let mut out = pos[0].raw as usize;
    for i in 1..pos.len() {
        out = out * dim[i].raw as usize + pos[i].raw as usize;
    }
    out
}

pub fn from_linear<D: DynDimension, T: CoordinateType>(
    linear_pos: usize,
    dim: &Vector<D, Coordinate<T>>,
) -> Vector<D, Coordinate<T>> {
    let raw = from_linear_u32(linear_pos, &dim.raw());
    Vector::from_fn_and_trusted_len(raw.len(), |i| raw[i].into())
}
//TODO: clean this up
pub fn from_linear_u32<D: DynDimension>(
    mut linear_pos: usize,
    dim: &Vector<D, u32>,
) -> Vector<D, u32> {
    let mut out = Vector::<D, u32>::from_fn_and_trusted_len(dim.len(), |_| 0u32);
    for i in (0..dim.len()).rev() {
        let ddim = dim[i] as usize;
        out[i] = (linear_pos % ddim) as u32;
        linear_pos /= ddim;
    }
    out
}

pub type LocalVoxelPosition = Vector<D3, LocalCoordinate>;
pub type VoxelPosition = Vector<D3, GlobalCoordinate>;
pub type BrickPosition = Vector<D3, ChunkCoordinate>;

pub type PixelPosition = Vector<D2, GlobalCoordinate>;
