use std::alloc::Layout;

use crate::{dim::D4, storage::ElementType, vec::Vector};

#[derive(Copy, Clone)]
pub enum DType {
    F32,
    U32,
    U8Vec4,
}

impl DType {
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

pub enum Scalar {
    F32,
    U32,
}
