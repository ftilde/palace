use num::traits::SaturatingSub;
use std::ops::{Add, Div, Mul, Sub};

use id::{Id, Identify};

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
impl<T: CoordinateType> TryFrom<usize> for Coordinate<T> {
    type Error = <u32 as TryInto<usize>>::Error;
    fn try_from(value: usize) -> Result<Self, Self::Error> {
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

impl<T: CoordinateType> SaturatingSub for Coordinate<T> {
    fn saturating_sub(&self, rhs: &Self) -> Self::Output {
        self.raw.saturating_sub(rhs.raw).into()
    }
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
#[cfg(feature = "python")]
mod py {
    use super::*;
    use pyo3::prelude::*;

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
}
