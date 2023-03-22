use ash::vk;
use spirq::{EntryPoint, ReflectConfig};
use std::collections::BTreeMap;

use super::{state::VulkanState, DeviceFunctions};

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
    fn build(self) -> Vec<u32>;
}

impl ShaderSource for (&str, ShaderDefines) {
    fn build(self) -> Vec<u32> {
        let source = self.0;
        let defines = self.1;

        use spirv_compiler::*;

        let mut compiler = CompilerBuilder::new()
            .with_source_language(SourceLanguage::GLSL)
            .generate_debug_info()
            .with_opt_level(OptimizationLevel::Performance)
            .with_target_env(TargetEnv::Vulkan, vk::API_VERSION_1_2);

        for (k, v) in defines.defines.into_iter() {
            compiler = compiler.with_macro(&k, Some(&v));
        }
        let mut compiler = compiler.build().unwrap();
        let kind = ShaderKind::Compute;
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

impl ShaderSource for &str {
    fn build(self) -> Vec<u32> {
        (self, ShaderDefines::new()).build()
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
    pub fn from_source(f: &DeviceFunctions, source: impl ShaderSource) -> Self {
        Self::from_compiled(f, &source.build())
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
