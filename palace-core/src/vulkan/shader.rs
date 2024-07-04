use ahash::HashMapExt;
use ash::vk;
use spirq::ReflectConfig;
use spirv_compiler::ShaderKind;
use std::{cell::RefCell, collections::BTreeMap};

use crate::{
    data::Vector,
    dim::*,
    util::{Map, Set},
};

use super::{pipeline::DynamicDescriptorSetPool, state::VulkanState, DeviceFunctions};

pub struct ShaderDefines {
    defines: Map<String, String>,
}

impl ShaderDefines {
    pub fn new() -> Self {
        Self {
            defines: Default::default(),
        }
    }
    pub fn add(mut self, key: impl Into<String>, value: impl ToString) -> Self {
        self.defines.insert(key.into(), value.to_string());
        self
    }

    pub fn push_const_block<T: crevice::glsl::GlslStruct>(self) -> Self {
        let mut struct_def = T::glsl_definition().replace("\n", " ");
        struct_def.pop(); //Remove semicolon
        let without_leading_struct = &struct_def[7..];
        let def = format!(
            "layout(std140, push_constant) uniform {} __name",
            without_leading_struct
        );
        self.add("declare_push_consts(__name)", def)
    }
}

pub mod ext {
    pub const SCALAR_BLOCK_LAYOUT: &str = "GL_EXT_scalar_block_layout";
    pub const BUFFER_REFERENCE: &str = "GL_EXT_buffer_reference";
    pub const INT64_TYPES: &str = "GL_EXT_shader_explicit_arithmetic_types_int64";
    pub const INT8_TYPES: &str = "GL_EXT_shader_explicit_arithmetic_types_int8";
    pub const INT16_TYPES: &str = "GL_EXT_shader_explicit_arithmetic_types_int16";
}

pub struct Shader {
    pub module: vk::ShaderModule,
    pub entry_points: Vec<spirq::entry_point::EntryPoint>,
}

pub struct Config {
    version: &'static str,
    extensions: Set<&'static str>,
}

impl Config {
    pub fn new() -> Self {
        Self {
            version: "450",
            extensions: Set::default(),
        }
    }

    pub fn ext(mut self, opt_ext: Option<&'static str>) -> Self {
        if let Some(ext) = opt_ext {
            self.extensions.insert(ext);
        }
        self
    }
}

impl std::fmt::Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "#version {}", self.version)?;
        for ext in &self.extensions {
            writeln!(f, "#extension {} : require", ext)?;
        }
        Ok(())
    }
}

pub trait ShaderSource {
    fn build(self, kind: ShaderKind) -> Vec<u32>;
}

impl ShaderSource for (&str, ShaderDefines) {
    fn build(self, kind: ShaderKind) -> Vec<u32> {
        //TODO: unify all shaders with using config
        let source = self.0;
        let defines = self.1;

        use spirv_compiler::*;

        let mut compiler = CompilerBuilder::new()
            .with_source_language(SourceLanguage::GLSL)
            .generate_debug_info()
            .with_opt_level(OptimizationLevel::Performance)
            .with_target_env(TargetEnv::Vulkan, vk::API_VERSION_1_2)
            .with_include_dir(env!("GLSL_INCLUDE_DIR"));

        for (k, v) in defines.defines.into_iter() {
            compiler = compiler.with_macro(&k, Some(&v));
        }
        let mut compiler = compiler.build().unwrap();
        match compiler.compile_from_string(&source, kind) {
            Ok(r) => r,
            Err(CompilerError::Log(e)) => {
                panic!(
                    "Compilation error for shader (source {:?}):\n{}\n\nFull source:\n{}",
                    e.file, e.description, self.0,
                )
            }
            Err(CompilerError::LoadError(e)) => panic!("Load error while compiling shader: {}", e),
            Err(CompilerError::WriteError(e)) => {
                panic!("Write error while compiling shader: {}", e)
            }
        }
    }
}
impl ShaderSource for (&str, ShaderDefines, Config) {
    fn build(self, kind: ShaderKind) -> Vec<u32> {
        let source = format!("{}{}", self.2, self.0);
        (source.as_str(), self.1).build(kind)
    }
}

#[derive(Default)]
pub struct DescriptorSetLayoutBindings {
    inner: BTreeMap<u32, vk::DescriptorSetLayoutBinding>,
}

impl DescriptorSetLayoutBindings {
    fn binding_array(&self) -> Vec<vk::DescriptorSetLayoutBinding> {
        let len = self.inner.len() as u32;
        let max_binding = self.inner.last_key_value().map(|(k, _)| *k).unwrap_or(0);
        assert_eq!(
            len,
            max_binding + 1,
            "Not all descriptor binding known. There are holes in the set of bindings."
        );

        self.inner.values().cloned().collect()
    }

    fn merge(&mut self, other: Self) {
        for (k, new) in other.inner.into_iter() {
            let existing = self.inner.insert(k, new);
            if let Some(existing) = existing {
                assert_eq!(existing.descriptor_count, new.descriptor_count);
                assert_eq!(existing.descriptor_type, new.descriptor_type);
            }
        }
    }
}

#[derive(Default)]
pub struct DescriptorBindings {
    inner: Map<u32, DescriptorSetLayoutBindings>,
}

impl DescriptorBindings {
    pub fn merge(mut self, other: Self) -> Self {
        for (k, new) in other.inner.into_iter() {
            let entry = self.inner.entry(k).or_default();
            entry.merge(new);
        }
        self
    }

    pub fn create_descriptor_set_layout(
        &self,
        df: &DeviceFunctions,
        use_push_descriptor: bool,
    ) -> (
        Vec<vk::DescriptorSetLayout>,
        Vec<RefCell<DynamicDescriptorSetPool>>,
    ) {
        let mut ds_pools = Vec::new();
        let descriptor_set_layouts = self
            .inner
            .iter()
            .map(|(_, bindings)| {
                let mut type_counts: Map<vk::DescriptorType, u32> = Map::new();
                for (_, b) in &bindings.inner {
                    *type_counts.entry(b.descriptor_type).or_default() += b.descriptor_count;
                }
                let binding_array = bindings.binding_array();
                let mut dsl_info =
                    vk::DescriptorSetLayoutCreateInfo::builder().bindings(&binding_array);
                if use_push_descriptor {
                    dsl_info =
                        dsl_info.flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);
                }
                let layout =
                    unsafe { df.device.create_descriptor_set_layout(&dsl_info, None) }.unwrap();
                ds_pools.push(RefCell::new(DynamicDescriptorSetPool::new(
                    layout,
                    type_counts,
                )));
                layout
            })
            .collect::<Vec<_>>();

        (descriptor_set_layouts, ds_pools)
    }
}

pub struct ShaderBindingInfo {
    pub local_size: Option<Vector<D3, u32>>,
    pub push_const: Option<vk::PushConstantRange>,
    pub descriptor_bindings: DescriptorBindings,
}

impl ShaderSource for &str {
    fn build(self, kind: ShaderKind) -> Vec<u32> {
        (self, ShaderDefines::new()).build(kind)
    }
}

impl Shader {
    pub fn from_compiled(f: &DeviceFunctions, code: &[u32]) -> Self {
        let info = vk::ShaderModuleCreateInfo::builder().code(&code);

        let entry_points = ReflectConfig::new()
            .spv(code)
            .ref_all_rscs(true)
            .reflect()
            .unwrap();

        let module = unsafe { f.device.create_shader_module(&info, None) }.unwrap();

        Self {
            module,
            entry_points,
        }
    }
    pub fn from_source(f: &DeviceFunctions, source: impl ShaderSource, kind: ShaderKind) -> Self {
        Self::from_compiled(f, &source.build(kind))
    }

    pub fn collect_info(&self, entry_point_name: &str) -> ShaderBindingInfo {
        let mut ret = ShaderBindingInfo {
            local_size: None,
            push_const: None,
            descriptor_bindings: Default::default(),
        };
        let entry_point = self
            .entry_points
            .iter()
            .find(|e| e.name == entry_point_name)
            .expect("Shader does not have the expected entry point name");

        use spirq::prelude::*;

        let stage = match entry_point.exec_model {
            ExecutionModel::Vertex => vk::ShaderStageFlags::VERTEX,
            ExecutionModel::TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
            ExecutionModel::TessellationEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
            ExecutionModel::Geometry => vk::ShaderStageFlags::GEOMETRY,
            ExecutionModel::Fragment => vk::ShaderStageFlags::FRAGMENT,
            ExecutionModel::GLCompute => vk::ShaderStageFlags::COMPUTE,
            o => panic!("Unhandled spir-v execution model {:?}", o),
        };
        for var in &entry_point.vars {
            match var {
                Variable::Input { .. } => {}
                Variable::Output { .. } => {}
                Variable::Descriptor {
                    name: _,
                    desc_bind,
                    desc_ty,
                    ty: _,
                    nbind,
                } => {
                    assert!(*nbind > 0, "Dynamic SSBOs are currently not supported (since we are using push descriptors, see https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkDescriptorSetLayoutCreateInfo-flags-00280)");
                    let d_type = match desc_ty {
                        DescriptorType::Sampler() => todo!(),
                        DescriptorType::CombinedImageSampler() => {
                            vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                        }
                        DescriptorType::SampledImage() => todo!(),
                        DescriptorType::StorageImage(_) => todo!(),
                        DescriptorType::UniformTexelBuffer() => todo!(),
                        DescriptorType::StorageTexelBuffer(_) => todo!(),
                        DescriptorType::UniformBuffer() => vk::DescriptorType::UNIFORM_BUFFER,
                        DescriptorType::StorageBuffer(_) => {
                            if *nbind > 0 {
                                vk::DescriptorType::STORAGE_BUFFER
                            } else {
                                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                            }
                        }
                        DescriptorType::InputAttachment(_) => todo!(),
                        DescriptorType::AccelStruct() => todo!(),
                    };
                    let binding = vk::DescriptorSetLayoutBinding::builder()
                        .binding(desc_bind.bind())
                        .descriptor_type(d_type)
                        .descriptor_count(*nbind)
                        .stage_flags(stage)
                        .build();

                    let set = desc_bind.set();
                    let set_bindings = ret.descriptor_bindings.inner.entry(set).or_default();
                    set_bindings.inner.insert(desc_bind.bind(), binding);
                }
                Variable::PushConstant { name: _, ty } => {
                    let c = vk::PushConstantRange::builder()
                        .size(ty.nbyte().unwrap().try_into().unwrap())
                        .stage_flags(stage)
                        .build();
                    let prev = ret.push_const.replace(c);
                    assert!(prev.is_none(), "Should only have on push constant");
                }
                Variable::SpecConstant { .. } => panic!("Unexpected spec constant"),
            }
        }

        fn unwrap_u32(v: &ConstantValue) -> u32 {
            if let ConstantValue::U32(u) = v {
                *u
            } else {
                panic!("Not a u32");
            }
        }

        for e in &entry_point.exec_modes {
            match e.exec_mode {
                spirv::ExecutionMode::LocalSize => {
                    let x = unwrap_u32(&e.operands[0].value);
                    let y = unwrap_u32(&e.operands[1].value);
                    let z = unwrap_u32(&e.operands[2].value);
                    let prev = ret.local_size.replace([z, y, x].into());
                    assert!(prev.is_none());
                }
                _ => {}
            }
        }

        ret
    }
}

impl VulkanState for Shader {
    unsafe fn deinitialize(&mut self, context: &crate::vulkan::DeviceContext) {
        unsafe {
            context
                .functions()
                .device
                .destroy_shader_module(self.module, None)
        };
    }
}
