use std::alloc::Layout;
use std::fmt::Debug;

use id::Identify;
#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::{dim::Dimension, storage::Element, vec::Vector};

pub trait ElementType: Copy + Clone + 'static + Into<DType> + PartialEq + Debug {
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
                        expected: <$ty>::D_TYPE,
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
    const D_TYPE: DType = T::D_TYPE.vectorize(D::N as u32);
}

impl<D: Dimension, T: Copy + AsDynType> TryFrom<DType> for StaticElementType<Vector<D, T>>
where
    StaticElementType<T>: TryFrom<ScalarType, Error = ConversionError>,
{
    type Error = ConversionError;

    fn try_from(value: DType) -> Result<Self, Self::Error> {
        let _ = StaticElementType::<T>::try_from(value.scalar)?;
        if value.size as usize != D::N {
            return Err(ConversionError {
                actual: value.into(),
                expected: <Vector<D, T>>::D_TYPE,
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

impl<const N: usize, T: Copy + AsDynType> TryFrom<DType> for StaticElementType<[T; N]>
where
    StaticElementType<T>: TryFrom<ScalarType, Error = ConversionError>,
{
    type Error = ConversionError;

    fn try_from(value: DType) -> Result<Self, Self::Error> {
        let _ = StaticElementType::<T>::try_from(value.scalar)?;
        if value.size as usize != N {
            return Err(ConversionError {
                actual: value.into(),
                expected: <[T; N]>::D_TYPE,
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
impl_conversion!(u64, U64);
impl_conversion!(i64, I64);
impl_conversion!(f32, F32);
//impl_conversion!(Vector<D4, u8>, U8Vec4, 1);
//impl_conversion!([Vector<D4, f32>; 2], F32Vec4A2, 1);

#[derive(Debug)]
pub struct ConversionError {
    pub expected: DType,
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
#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyclass(get_all))]
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
    pub const fn vectorize(self, vec_size: u32) -> Self {
        Self {
            scalar: self.scalar,
            size: self.size * vec_size,
        }
    }
    pub fn is_scalar(&self) -> bool {
        self.size == 1
    }
    pub fn glsl_type(&self) -> String {
        let scalar = self.scalar.glsl_type().to_owned();
        if self.size > 1 {
            self.glsl_type_force_vec()
        } else {
            scalar
        }
    }
    pub fn glsl_type_force_vec(&self) -> String {
        let scalar = self.scalar.glsl_type().to_owned();
        format!("{}[{}]", scalar, self.size)
    }
    pub fn glsl_ext(&self) -> Option<&'static str> {
        match self.scalar {
            ScalarType::U8 | ScalarType::I8 => Some(crate::vulkan::shader::ext::INT8_TYPES),
            ScalarType::U16 | ScalarType::I16 => Some(crate::vulkan::shader::ext::INT16_TYPES),
            ScalarType::F32 => None,
            ScalarType::U32 | ScalarType::I32 => None,
            ScalarType::U64 | ScalarType::I64 => Some(crate::vulkan::shader::ext::INT64_TYPES),
        }
    }
    pub fn vec_size(&self) -> usize {
        self.size as _
    }
}

#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pymethods)]
#[cfg_attr(feature = "python", pymethods)]
impl DType {
    #[new]
    pub fn new(scalar: ScalarType, size: u32) -> Self {
        Self { scalar, size }
    }

    fn __str__(&self) -> String {
        format!("{}[{}]", self.scalar.pretty_type(), self.size)
    }

    fn size_in_bytes(&self) -> usize {
        self.element_layout().size()
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
            ScalarType::U64 => Layout::array::<u64>(size).unwrap(),
            ScalarType::I64 => Layout::array::<i64>(size).unwrap(),
            ScalarType::F32 => Layout::array::<f32>(size).unwrap(),
        }
    }
}

/// Some specialized types -----------------------------------------------------
#[derive(Copy, Clone, Debug, Identify, Eq, PartialEq)]
#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pyclass_enum)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
pub enum ScalarType {
    U8,
    I8,
    U16,
    I16,
    F32,
    U32,
    I32,
    U64,
    I64,
}

impl ScalarType {
    pub fn glsl_type(&self) -> &'static str {
        match self {
            ScalarType::U8 => "uint8_t",
            ScalarType::I8 => "int8_t",
            ScalarType::U16 => "uint16_t",
            ScalarType::I16 => "int16_t",
            ScalarType::U32 => "uint",
            ScalarType::I32 => "int",
            ScalarType::U64 => "uint64_t",
            ScalarType::I64 => "int64_t",
            ScalarType::F32 => "float",
        }
    }

    pub fn pretty_type(&self) -> &'static str {
        match self {
            ScalarType::U8 => "u8",
            ScalarType::I8 => "i8",
            ScalarType::U16 => "u16",
            ScalarType::I16 => "i16",
            ScalarType::U32 => "u32",
            ScalarType::I32 => "i32",
            ScalarType::U64 => "u64",
            ScalarType::I64 => "i64",
            ScalarType::F32 => "f32",
        }
    }

    pub fn try_from_pretty(s: &str) -> Result<Self, String> {
        Ok(match s {
            "u8" => Self::U8,
            "i8" => Self::I8,
            "u16" => Self::U16,
            "i16" => Self::I16,
            "u32" => Self::U32,
            "i32" => Self::I32,
            "u64" => Self::U64,
            "i64" => Self::I64,
            "f32" => Self::F32,
            _ => return Err(format!("{} is not a valar ScalarType", s).into()),
        })
    }
}

//#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pymethods)] //Broken somehow
#[cfg_attr(feature = "python", pymethods)]
impl ScalarType {
    pub fn vec(&self, size: u32) -> DType {
        DType {
            scalar: *self,
            size,
        }
    }

    pub fn is_integer(&self) -> bool {
        match self {
            ScalarType::U8
            | ScalarType::I8
            | ScalarType::U16
            | ScalarType::I16
            | ScalarType::U32
            | ScalarType::I32
            | ScalarType::U64
            | ScalarType::I64 => true,
            ScalarType::F32 => false,
        }
    }

    pub fn is_signed(&self) -> bool {
        match self {
            ScalarType::U8 | ScalarType::U16 | ScalarType::U32 | ScalarType::U64 => false,

            ScalarType::I8
            | ScalarType::I16
            | ScalarType::I32
            | ScalarType::I64
            | ScalarType::F32 => true,
        }
    }

    pub fn max_value(&self) -> f64 {
        match self {
            ScalarType::U8 => u8::max_value() as f64,
            ScalarType::I8 => i8::max_value() as f64,
            ScalarType::U16 => u16::max_value() as f64,
            ScalarType::I16 => i16::max_value() as f64,
            ScalarType::F32 => f32::INFINITY as f64,
            ScalarType::U32 => u32::max_value() as f64,
            ScalarType::I32 => i32::max_value() as f64,
            ScalarType::U64 => u64::max_value() as f64,
            ScalarType::I64 => i64::max_value() as f64,
        }
    }

    pub fn min_value(&self) -> f64 {
        match self {
            ScalarType::U8 => u8::min_value() as f64,
            ScalarType::I8 => i8::min_value() as f64,
            ScalarType::U16 => u16::min_value() as f64,
            ScalarType::I16 => i16::min_value() as f64,
            ScalarType::F32 => -f32::INFINITY as f64,
            ScalarType::U32 => u32::min_value() as f64,
            ScalarType::I32 => i32::min_value() as f64,
            ScalarType::U64 => u64::min_value() as f64,
            ScalarType::I64 => i64::min_value() as f64,
        }
    }
}
