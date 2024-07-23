use std::alloc::Layout;
use std::fmt::Debug;

use id::Identify;
#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::{dim::D4, storage::Element, vec::Vector};

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
        impl TryFrom<DType> for StaticElementType<$ty> {
            type Error = ConversionError;

            fn try_from(value: DType) -> Result<Self, Self::Error> {
                if let DType::$variant = value {
                    Ok(Default::default())
                } else {
                    Err(ConversionError {
                        actual: value.pretty_type(),
                        expected: stringify!($ty),
                    })
                }
            }
        }

        impl AsDynType for $ty {
            const D_TYPE: DType = DType::$variant;
        }
    };
}

// TODO: Can we remove this?
impl<T: AsDynType> From<StaticElementType<T>> for DType {
    fn from(_value: StaticElementType<T>) -> Self {
        T::D_TYPE
    }
}

impl_conversion!(u8, U8);
impl_conversion!(i8, I8);
impl_conversion!(u16, U16);
impl_conversion!(i16, I16);
impl_conversion!(u32, U32);
impl_conversion!(i32, I32);
impl_conversion!(f32, F32);
impl_conversion!(Vector<D4, u8>, U8Vec4);
impl_conversion!([Vector<D4, f32>; 2], F32Vec4A2);

#[derive(Debug)]
pub struct ConversionError {
    pub expected: &'static str,
    pub actual: &'static str,
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
pub enum DType {
    U8,
    I8,
    U16,
    I16,
    F32,
    U32,
    I32,
    U8Vec4,
    F32Vec4A2,
}

impl DType {
    pub fn pretty_type(&self) -> &'static str {
        match self {
            DType::U8 => "u8",
            DType::I8 => "i8",
            DType::U16 => "u16",
            DType::I16 => "i16",
            DType::U32 => "u32",
            DType::I32 => "i32",
            DType::F32 => "f32",
            DType::U8Vec4 => "u8vec4",
            DType::F32Vec4A2 => "[vec4; 2]",
        }
    }

    pub fn glsl_type(&self) -> &str {
        match self {
            DType::U8 => "uint8_t",
            DType::I8 => "int8_t",
            DType::U16 => "uint16_t",
            DType::I16 => "int16_t",
            DType::U32 => "uint",
            DType::I32 => "int",
            DType::F32 => "float",
            DType::U8Vec4 => "u8vec4",
            DType::F32Vec4A2 => "vec4[2]",
        }
    }
    pub fn glsl_ext(&self) -> Option<&'static str> {
        match self {
            DType::U8 | DType::I8 => Some(crate::vulkan::shader::ext::INT8_TYPES),
            DType::U16 | DType::I16 => Some(crate::vulkan::shader::ext::INT16_TYPES),
            DType::F32 => None,
            DType::U32 | DType::I32 => None,
            DType::U8Vec4 => Some(crate::vulkan::shader::ext::INT8_TYPES),
            DType::F32Vec4A2 => None,
        }
    }
    pub fn vec_size(&self) -> usize {
        match self {
            DType::U8
            | DType::I8
            | DType::U16
            | DType::I16
            | DType::F32
            | DType::U32
            | DType::I32 => 1,
            DType::U8Vec4 => 4,
            DType::F32Vec4A2 => 8, //TODO??
        }
    }
}

impl ElementType for DType {
    fn array_layout(&self, size: usize) -> Layout {
        // TODO: This is REALLY error prone...
        match self {
            DType::U8 => Layout::array::<u8>(size).unwrap(),
            DType::I8 => Layout::array::<i8>(size).unwrap(),
            DType::U16 => Layout::array::<u16>(size).unwrap(),
            DType::I16 => Layout::array::<i16>(size).unwrap(),
            DType::U32 => Layout::array::<u32>(size).unwrap(),
            DType::I32 => Layout::array::<i32>(size).unwrap(),
            DType::F32 => Layout::array::<f32>(size).unwrap(),
            DType::U8Vec4 => Layout::array::<Vector<D4, u8>>(size).unwrap(),
            DType::F32Vec4A2 => Layout::array::<[Vector<D4, f32>; 2]>(size).unwrap(),
        }
    }
}

/// Some specialized types -----------------------------------------------------
pub enum Scalar {
    F32,
    U32,
}
