use std::alloc::Layout;
use std::fmt::Debug;

use id::Identify;
#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::{dim::Dimension, storage::Element, vec::Vector};

pub trait ElementType: Clone + 'static + Into<DType> + PartialEq + Debug {
    fn array_layout(&self, size: usize) -> Layout;
    fn element_layout(&self) -> Layout {
        self.array_layout(1)
    }
}

pub trait AsDynType {
    const D_TYPE: DType;
}

/// Static ---------------------------------------------------------------------

pub struct StaticElementType<T>(std::marker::PhantomData<T>);

impl<T> PartialEq for StaticElementType<T> {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl<T> Debug for StaticElementType<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StaticElementType<{}>", std::any::type_name::<T>())
    }
}

impl<T> Default for StaticElementType<T> {
    fn default() -> Self {
        StaticElementType(std::marker::PhantomData)
    }
}

impl<T> Clone for StaticElementType<T> {
    fn clone(&self) -> Self {
        Default::default()
    }
}
impl<T> Copy for StaticElementType<T> {}

impl<T: Element> ElementType for StaticElementType<T> {
    fn array_layout(&self, size: usize) -> Layout {
        Layout::array::<T>(size).unwrap()
    }
}

macro_rules! impl_conversion {
    ($ty:ty, $variant:ident) => {
        impl TryFrom<ScalarType> for StaticElementType<$ty> {
            type Error = ConversionError;

            fn try_from(value: ScalarType) -> Result<Self, Self::Error> {
                if let ScalarType::$variant = value {
                    Ok(Default::default())
                } else {
                    Err(ConversionError {
                        actual: value.into(),
                        expected: stringify!($ty),
                    })
                }
            }
        }

        impl AsDynType for $ty {
            const D_TYPE: DType = DType::scalar(ScalarType::$variant);
        }
    };
}

// TODO: Can we remove this?
impl<T: AsDynType> From<StaticElementType<T>> for DType {
    fn from(_value: StaticElementType<T>) -> Self {
        T::D_TYPE
    }
}

impl<T> TryFrom<DType> for StaticElementType<T>
where
    StaticElementType<T>: TryFrom<ScalarType, Error = ConversionError>,
{
    type Error = ConversionError;

    fn try_from(value: DType) -> Result<Self, Self::Error> {
        StaticElementType::try_from(value.scalar)
    }
}
impl<D: Dimension, T: Copy + AsDynType> AsDynType for Vector<D, T> {
    const D_TYPE: DType = DType {
        scalar: T::D_TYPE.scalar,
        size: D::N as u32 * T::D_TYPE.size,
    };
}

impl<D: Dimension, T: Copy> TryFrom<DType> for StaticElementType<Vector<D, T>>
where
    StaticElementType<T>: TryFrom<ScalarType, Error = ConversionError>,
{
    type Error = ConversionError;

    fn try_from(value: DType) -> Result<Self, Self::Error> {
        let _ = StaticElementType::<T>::try_from(value.scalar)?;
        if value.size as usize != D::N {
            return Err(ConversionError {
                actual: value.into(),
                expected: stringify!(Vector<D, T>),
            });
        }
        Ok(Default::default())
    }
}
impl<const N: usize, T: Copy + AsDynType> AsDynType for [T; N] {
    const D_TYPE: DType = DType {
        scalar: T::D_TYPE.scalar,
        size: N as u32 * T::D_TYPE.size,
    };
}

impl<const N: usize, T: Copy> TryFrom<DType> for StaticElementType<[T; N]>
where
    StaticElementType<T>: TryFrom<ScalarType, Error = ConversionError>,
{
    type Error = ConversionError;

    fn try_from(value: DType) -> Result<Self, Self::Error> {
        let _ = StaticElementType::<T>::try_from(value.scalar)?;
        if value.size as usize != N {
            return Err(ConversionError {
                actual: value.into(),
                expected: stringify!(Vector<D, T>),
            });
        }
        Ok(Default::default())
    }
}

impl_conversion!(u8, U8);
impl_conversion!(i8, I8);
impl_conversion!(u16, U16);
impl_conversion!(i16, I16);
impl_conversion!(u32, U32);
impl_conversion!(i32, I32);
impl_conversion!(f32, F32);
//impl_conversion!(Vector<D4, u8>, U8Vec4, 1);
//impl_conversion!([Vector<D4, f32>; 2], F32Vec4A2, 1);

#[derive(Debug)]
pub struct ConversionError {
    pub expected: &'static str,
    pub actual: DType,
}

impl std::error::Error for ConversionError {}
impl std::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Failed dynamic type conversion, expected {}, but got {}",
            self.expected, self.actual
        )
    }
}

#[cfg(feature = "python")]
mod py {
    use super::*;
    use pyo3::{exceptions::PyException, PyErr};

    impl From<ConversionError> for PyErr {
        fn from(e: ConversionError) -> PyErr {
            PyErr::new::<PyException, _>(format!("{}", e))
        }
    }
}

/// Dynamic --------------------------------------------------------------------

#[derive(Copy, Clone, Debug, Identify, Eq, PartialEq)]
#[cfg_attr(feature = "python", pyclass)]
pub struct DType {
    pub scalar: ScalarType,
    pub size: u32,
}

impl From<ScalarType> for DType {
    fn from(value: ScalarType) -> Self {
        Self::scalar(value)
    }
}

impl DType {
    pub const fn scalar(scalar: ScalarType) -> Self {
        Self { scalar, size: 1 }
    }
    pub fn glsl_type(&self) -> String {
        let scalar = self.scalar.glsl_type().to_owned();
        if self.size > 1 {
            format!("{}[{}]", scalar, self.size)
        } else {
            scalar
        }
    }
    pub fn glsl_ext(&self) -> Option<&'static str> {
        match self.scalar {
            ScalarType::U8 | ScalarType::I8 => Some(crate::vulkan::shader::ext::INT8_TYPES),
            ScalarType::U16 | ScalarType::I16 => Some(crate::vulkan::shader::ext::INT16_TYPES),
            ScalarType::F32 => None,
            ScalarType::U32 | ScalarType::I32 => None,
        }
    }
    pub fn vec_size(&self) -> usize {
        self.size as _
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}; {}]", self.scalar.pretty_type(), self.size)
    }
}

impl ElementType for DType {
    fn array_layout(&self, size: usize) -> Layout {
        let size = size * self.size as usize;
        // TODO: This is REALLY error prone...
        match self.scalar {
            ScalarType::U8 => Layout::array::<u8>(size).unwrap(),
            ScalarType::I8 => Layout::array::<i8>(size).unwrap(),
            ScalarType::U16 => Layout::array::<u16>(size).unwrap(),
            ScalarType::I16 => Layout::array::<i16>(size).unwrap(),
            ScalarType::U32 => Layout::array::<u32>(size).unwrap(),
            ScalarType::I32 => Layout::array::<i32>(size).unwrap(),
            ScalarType::F32 => Layout::array::<f32>(size).unwrap(),
        }
    }
}

/// Some specialized types -----------------------------------------------------
#[derive(Copy, Clone, Debug, Identify, Eq, PartialEq)]
#[cfg_attr(feature = "python", pyclass)]
pub enum ScalarType {
    U8,
    I8,
    U16,
    I16,
    F32,
    U32,
    I32,
}

impl ScalarType {
    fn glsl_type(&self) -> &'static str {
        match self {
            ScalarType::U8 => "uint8_t",
            ScalarType::I8 => "int8_t",
            ScalarType::U16 => "uint16_t",
            ScalarType::I16 => "int16_t",
            ScalarType::U32 => "uint",
            ScalarType::I32 => "int",
            ScalarType::F32 => "float",
        }
    }

    fn pretty_type(&self) -> &'static str {
        match self {
            ScalarType::U8 => "u8",
            ScalarType::I8 => "i8",
            ScalarType::U16 => "u16",
            ScalarType::I16 => "i16",
            ScalarType::U32 => "u32",
            ScalarType::I32 => "i32",
            ScalarType::F32 => "f32",
        }
    }
}
