use ash::vk;
use spirq::{EntryPoint, ReflectConfig};
use spirv_compiler::ShaderKind;
use std::{cell::RefCell, collections::BTreeMap};

use crate::data::Vector;

use super::{pipeline::DynamicDescriptorSetPool, state::VulkanState, DeviceFunctions};

pub struct ShaderDefines {
    defines: BTreeMap<String, String>,
}

impl ShaderDefines {
    pub fn new() -> Self {
        Self {
            defines: Default::default(),
        }
    }
    pub fn add(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.defines.insert(key.into(), value.into());
        self
    }
}

pub struct Shader {
    pub module: vk::ShaderModule,
    pub entry_points: Vec<EntryPoint>,
}

pub trait ShaderSource {
    fn build(self, kind: ShaderKind) -> Vec<u32>;
}

impl ShaderSource for (&str, ShaderDefines) {
    fn build(self, kind: ShaderKind) -> Vec<u32> {
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
                    "Compilation error for shader (source {:?}):\n{}",
                    e.file, e.description
                )
            }
            Err(CompilerError::LoadError(e)) => panic!("Load error while compiling shader: {}", e),
            Err(CompilerError::WriteError(e)) => {
                panic!("Write error while compiling shader: {}", e)
            }
        }
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
    inner: BTreeMap<u32, DescriptorSetLayoutBindings>,
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
                let mut type_counts: BTreeMap<vk::DescriptorType, u32> = BTreeMap::new();
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
    pub local_size: Option<Vector<3, u32>>,
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

        let stage = match entry_point.exec_model {
            spirq::ExecutionModel::Vertex => vk::ShaderStageFlags::VERTEX,
            spirq::ExecutionModel::TessellationControl => {
                vk::ShaderStageFlags::TESSELLATION_CONTROL
            }
            spirq::ExecutionModel::TessellationEvaluation => {
                vk::ShaderStageFlags::TESSELLATION_EVALUATION
            }
            spirq::ExecutionModel::Geometry => vk::ShaderStageFlags::GEOMETRY,
            spirq::ExecutionModel::Fragment => vk::ShaderStageFlags::FRAGMENT,
            spirq::ExecutionModel::GLCompute => vk::ShaderStageFlags::COMPUTE,
            o => panic!("Unhandled spir-v execution model {:?}", o),
        };
        for var in &entry_point.vars {
            match var {
                spirq::Variable::Input { .. } => {}
                spirq::Variable::Output { .. } => {}
                spirq::Variable::Descriptor {
                    name: _,
                    desc_bind,
                    desc_ty,
                    ty: _,
                    nbind,
                } => {
                    assert!(*nbind > 0, "Dynamic SSBOs are currently not supported (since we are using push descriptors, see https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkDescriptorSetLayoutCreateInfo-flags-00280)");
                    let d_type = match desc_ty {
                        spirq::DescriptorType::Sampler() => todo!(),
                        spirq::DescriptorType::CombinedImageSampler() => todo!(),
                        spirq::DescriptorType::SampledImage() => todo!(),
                        spirq::DescriptorType::StorageImage(_) => todo!(),
                        spirq::DescriptorType::UniformTexelBuffer() => todo!(),
                        spirq::DescriptorType::StorageTexelBuffer(_) => todo!(),
                        spirq::DescriptorType::UniformBuffer() => {
                            vk::DescriptorType::UNIFORM_BUFFER
                        }
                        spirq::DescriptorType::StorageBuffer(_) => {
                            if *nbind > 0 {
                                vk::DescriptorType::STORAGE_BUFFER
                            } else {
                                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                            }
                        }
                        spirq::DescriptorType::InputAttachment(_) => todo!(),
                        spirq::DescriptorType::AccelStruct() => todo!(),
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
                spirq::Variable::PushConstant { name: _, ty } => {
                    let c = vk::PushConstantRange::builder()
                        .size(ty.nbyte().unwrap().try_into().unwrap())
                        .stage_flags(stage)
                        .build();
                    let prev = ret.push_const.replace(c);
                    assert!(prev.is_none(), "Should only have on push constant");
                }
                spirq::Variable::SpecConstant { .. } => panic!("Unexpected spec constant"),
            }
        }

        for e in &entry_point.exec_modes {
            match e.exec_mode {
                spirv::ExecutionMode::LocalSize => {
                    let x = e.operands[0].value.to_u32();
                    let y = e.operands[1].value.to_u32();
                    let z = e.operands[2].value.to_u32();
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
