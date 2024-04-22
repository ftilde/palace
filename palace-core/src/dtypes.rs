use std::alloc::Layout;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::{dim::D4, storage::Element, vec::Vector};

pub trait ElementType: Clone {
    fn array_layout(&self, size: usize) -> Layout;
}

/// Static ---------------------------------------------------------------------

pub struct StaticElementType<T>(std::marker::PhantomData<T>);

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

        impl From<StaticElementType<$ty>> for DType {
            fn from(_value: StaticElementType<$ty>) -> Self {
                DType::$variant
            }
        }
    };
}

impl_conversion!(f32, F32);
impl_conversion!(u32, U32);
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

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "python", pyclass)]
pub enum DType {
    F32,
    U32,
    U8Vec4,
    F32Vec4A2,
}

impl DType {
    pub fn pretty_type(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::U32 => "u32",
            DType::U8Vec4 => "u8vec4",
            DType::F32Vec4A2 => "[vec4; 2]",
        }
    }

    pub fn glsl_type(&self) -> &str {
        match self {
            DType::F32 => "float",
            DType::U32 => "uint",
            DType::U8Vec4 => "u8vec4",
            DType::F32Vec4A2 => "vec4[2]",
        }
    }
    pub fn glsl_ext(&self) -> Option<&str> {
        match self {
            DType::F32 => None,
            DType::U32 => None,
            DType::U8Vec4 => Some("GL_EXT_shader_explicit_arithmetic_types_int8"),
            DType::F32Vec4A2 => None,
        }
    }
}

impl ElementType for DType {
    fn array_layout(&self, size: usize) -> Layout {
        // TODO: This is REALLY error prone...
        match self {
            DType::F32 => Layout::array::<f32>(size).unwrap(),
            DType::U32 => Layout::array::<u32>(size).unwrap(),
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
