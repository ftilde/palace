use std::alloc::Layout;

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
                        from: value.pretty_type(),
                        to: stringify!($ty),
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

pub struct ConversionError {
    pub from: &'static str,
    pub to: &'static str,
}

/// Dynamic --------------------------------------------------------------------

#[derive(Copy, Clone)]
pub enum DType {
    F32,
    U32,
    U8Vec4,
}

impl DType {
    pub fn pretty_type(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::U32 => "u32",
            DType::U8Vec4 => "u8vec4",
        }
    }

    pub fn glsl_type(&self) -> &str {
        match self {
            DType::F32 => "float",
            DType::U32 => "uint",
            DType::U8Vec4 => "u8vec4",
        }
    }
    pub fn glsl_ext(&self) -> Option<&str> {
        match self {
            DType::F32 => None,
            DType::U32 => None,
            DType::U8Vec4 => Some("GL_EXT_shader_explicit_arithmetic_types_int8"),
        }
    }
}

impl ElementType for DType {
    fn array_layout(&self, size: usize) -> Layout {
        match self {
            DType::F32 => Layout::array::<f32>(size).unwrap(),
            DType::U32 => Layout::array::<u32>(size).unwrap(),
            DType::U8Vec4 => Layout::array::<Vector<D4, u8>>(size).unwrap(),
        }
    }
}

/// Some specialized types -----------------------------------------------------
pub enum Scalar {
    F32,
    U32,
}
