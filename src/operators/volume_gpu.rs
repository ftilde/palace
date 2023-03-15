use std::collections::BTreeMap;

use ash::vk;
use crevice::std140::{AsStd140, Std140};
use futures::StreamExt;
use spirq::{EntryPoint, ReflectConfig};

use crate::{
    array::VolumeMetaData,
    data::{BrickPosition, LocalCoordinate, LocalVoxelPosition},
    id::Id,
    operator::OperatorId,
    operators::tensor::TensorOperator,
    storage::gpu::{ReadHandle, WriteHandle},
    vulkan::{DeviceContext, RessourceId, VulkanState},
};

use super::{array::ArrayOperator, scalar::ScalarOperator, volume::VolumeOperator};

//fn layout_std140<T: AsStd140>() -> Layout {
//    unsafe { Layout::from_size_align_unchecked(T::std140_size_static(), T::Output::ALIGNMENT) }
//}

struct ShaderDefines {
    defines: BTreeMap<String, String>,
}

impl ShaderDefines {
    fn new() -> Self {
        Self {
            defines: Default::default(),
        }
    }
    fn add(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.defines.insert(key.into(), value.into());
        self
    }
}

struct Shader {
    module: vk::ShaderModule,
    entry_points: Vec<EntryPoint>,
}

trait ShaderSource {
    fn build(self) -> Vec<u32>;
}

impl ShaderSource for (&str, ShaderDefines) {
    fn build(self) -> Vec<u32> {
        let source = self.0;
        let defines = self.1;

        use spirv_compiler::*;

        let mut compiler = CompilerBuilder::new()
            .with_source_language(SourceLanguage::GLSL)
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
    fn from_compiled(device: &DeviceContext, code: &[u32]) -> Self {
        let info = vk::ShaderModuleCreateInfo::builder().code(&code);

        let entry_points = ReflectConfig::new()
            .spv(code)
            .ref_all_rscs(true)
            .reflect()
            .unwrap();

        let module = unsafe { device.device.create_shader_module(&info, None) }.unwrap();

        Self {
            module,
            entry_points,
        }
    }
    fn from_source(device: &DeviceContext, source: impl ShaderSource) -> Self {
        Self::from_compiled(device, &source.build())
    }
}

impl VulkanState for Shader {
    unsafe fn deinitialize(&mut self, context: &crate::vulkan::DeviceContext) {
        unsafe { context.device.destroy_shader_module(self.module, None) };
    }
}

struct ComputePipeline {
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    local_size: usize,
    push_constant_size: Option<usize>,
}

impl ComputePipeline {
    fn new(device: &DeviceContext, shader: impl ShaderSource) -> Self {
        let mut shader = Shader::from_source(device, shader);

        let entry_point_name = "main";
        let entry_point_name_c = std::ffi::CString::new(entry_point_name).unwrap();

        let pipeline_info = vk::PipelineShaderStageCreateInfo::builder()
            .module(shader.module)
            .name(&entry_point_name_c)
            .stage(vk::ShaderStageFlags::COMPUTE);

        let entry_point = shader
            .entry_points
            .iter()
            .find(|e| e.name == entry_point_name)
            .expect("Shader does not have the expected entry point name");

        assert_eq!(entry_point.exec_model, spirq::ExecutionModel::GLCompute);

        let mut descriptor_bindings: BTreeMap<u32, Vec<_>> = BTreeMap::new();
        let mut push_constant = None;
        let mut push_constant_size = None;
        let mut local_size = None;
        let stage = vk::ShaderStageFlags::COMPUTE;

        for var in &entry_point.vars {
            match var {
                spirq::Variable::Input { .. } => panic!("Unexpected input var"),
                spirq::Variable::Output { .. } => panic!("Unexpected output var"),
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
                    let set_bindings = descriptor_bindings.entry(set).or_default();
                    set_bindings.push(binding);
                }
                spirq::Variable::PushConstant { name: _, ty } => {
                    let c = vk::PushConstantRange::builder()
                        .size(ty.nbyte().unwrap().try_into().unwrap())
                        .stage_flags(stage)
                        .build();
                    push_constant_size = Some(c.size as _);
                    let prev = push_constant.replace(c);
                    assert!(prev.is_none(), "Should only have on push constant");
                }
                spirq::Variable::SpecConstant { .. } => panic!("Unexpected spec constant"),
            }
        }

        for e in &entry_point.exec_modes {
            match e.exec_mode {
                spirv::ExecutionMode::LocalSize => {
                    let ls = e.operands.first().unwrap().value.to_u32();
                    let prev = local_size.replace(ls);
                    assert!(prev.is_none());
                }
                _ => {}
            }
        }

        let local_size = local_size
            .expect("local size should have been specified in shader")
            .try_into()
            .unwrap();

        let descriptor_set_layouts = descriptor_bindings
            .into_iter()
            .map(|(_, bindings)| {
                let dsl_info = vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&bindings)
                    .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);
                unsafe { device.device.create_descriptor_set_layout(&dsl_info, None) }.unwrap()
            })
            .collect::<Vec<_>>();

        let push_constant_ranges = push_constant
            .as_ref()
            .map(std::slice::from_ref)
            .unwrap_or(&[]);
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        let pipeline_layout = unsafe {
            device
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .unwrap();

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*pipeline_info)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            device.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[*pipeline_info],
                None,
            )
        }
        .unwrap();

        // Safety: Pipeline has been created now. Shader module is not referenced anymore.
        unsafe { shader.deinitialize(device) };

        let pipeline = pipelines[0];

        Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layouts,
            local_size,
            push_constant_size,
        }
    }

    unsafe fn bind(&self, device: &DeviceContext, cmd: vk::CommandBuffer) {
        device
            .device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
    }

    unsafe fn push_descriptor_set(
        &self,
        device: &DeviceContext,
        cmd: vk::CommandBuffer,
        bind_set: u32,
        writes: &[vk::WriteDescriptorSet],
    ) {
        device.push_descriptor_ext.cmd_push_descriptor_set(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline_layout,
            bind_set,
            writes,
        );
    }

    unsafe fn push_constant<T: AsStd140>(
        &self,
        device: &DeviceContext,
        cmd: vk::CommandBuffer,
        val: T,
    ) {
        let v = val.as_std140();
        let bytes = v.as_bytes();

        // HACK: Match up sizes:
        // - The reflect library/c compiler appears to be of the opinion that the size of the push
        // constant struct is simply the difference between the begin of the first member and the
        // end of the last, while...
        // - crevice appears to think that the size is rounded up to the alignment of the struct.
        let bytes = &bytes[..self.push_constant_size.unwrap()];
        device.device.cmd_push_constants(
            cmd,
            self.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytes,
        );
    }
    fn local_size(&self) -> usize {
        self.local_size
    }
}

impl VulkanState for ComputePipeline {
    unsafe fn deinitialize(&mut self, context: &crate::vulkan::DeviceContext) {
        unsafe {
            for descriptor_set_layout in &self.descriptor_set_layouts {
                context
                    .device
                    .destroy_descriptor_set_layout(*descriptor_set_layout, None);
            }
            context.device.destroy_pipeline(self.pipeline, None);
            context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None)
        };
    }
}

trait AsDescriptor {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo;
}

impl<'a> AsDescriptor for ReadHandle<'a> {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::builder()
            .buffer(self.buffer)
            .range(self.layout.size() as _)
            .build()
    }
}

impl<'a> AsDescriptor for WriteHandle<'a> {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::builder()
            .buffer(self.buffer)
            .range(self.size as _)
            .build()
    }
}

trait AsDescriptors {
    fn gen_buffer_info(&self) -> Vec<vk::DescriptorBufferInfo>;
}

impl<T: AsDescriptor> AsDescriptors for T {
    fn gen_buffer_info(&self) -> Vec<vk::DescriptorBufferInfo> {
        vec![self.gen_buffer_info()]
    }
}

impl<const N: usize, T: AsDescriptor> AsDescriptors for [&T; N] {
    fn gen_buffer_info(&self) -> Vec<vk::DescriptorBufferInfo> {
        self.iter().map(|i| i.gen_buffer_info()).collect()
    }
}

impl<T: AsDescriptor> AsDescriptors for &[&T] {
    fn gen_buffer_info(&self) -> Vec<vk::DescriptorBufferInfo> {
        self.iter().map(|i| i.gen_buffer_info()).collect()
    }
}

struct DescriptorConfig<const N: usize> {
    buffer_infos: [Vec<vk::DescriptorBufferInfo>; N],
}
impl<const N: usize> DescriptorConfig<N> {
    fn new(buffers: [&dyn AsDescriptors; N]) -> Self {
        let buffer_infos = std::array::from_fn(|i| buffers[i].gen_buffer_info());
        Self { buffer_infos }
    }
    fn writes(&self) -> [vk::WriteDescriptorSet; N] {
        std::array::from_fn(|i| {
            vk::WriteDescriptorSet::builder()
                .dst_binding(i as u32)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&self.buffer_infos[i])
                .build()
        })
    }
}

pub fn linear_rescale<'op>(
    input: VolumeOperator<'op>,
    scale: ScalarOperator<'op, f32>,
    offset: ScalarOperator<'op, f32>,
) -> VolumeOperator<'op> {
    #[derive(Copy, Clone, AsStd140)]
    struct PushConstants {
        chunk_pos: mint::Vector3<u32>,
        num_chunk_elems: u32,
    }
    const SHADER: &'static str = r#"
#version 450

layout (local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputScale{
    float value;
} scale;

layout(std430, binding = 1) readonly buffer InputOffset{
    float value;
} offset;

layout(std430, binding = 2) readonly buffer InputBuffer{
    float values[];
} sourceData;

layout(std430, binding = 3) buffer OutputBuffer{
    float values[];
} outputData;

layout(std140, push_constant) uniform PushConstants
{
    uvec3 chunk_pos;
    uint num_chunk_elems;
} constants;

void main()
{
    uint gID = gl_GlobalInvocationID.x;

    if(gID < constants.num_chunk_elems) {
        outputData.values[gID] = scale.value*sourceData.values[gID] + offset.value;
    }
}
"#;

    TensorOperator::with_state(
        OperatorId::new("volume_scale_gpu")
            .dependent_on(&input)
            .dependent_on(&scale)
            .dependent_on(&offset),
        input.clone(),
        (input.clone(), scale, offset),
        move |ctx, input, _| {
            async move {
                let req = input.metadata.request_scalar();
                let m = ctx.submit(req).await;
                ctx.write(m)
            }
            .into()
        },
        move |ctx, positions, (input, scale, offset), _| {
            async move {
                let device = ctx.vulkan_device();

                let (scale_gpu, offset_gpu, m) = futures::join! {
                    ctx.submit(scale.request_gpu(device.id, ())),
                    ctx.submit(offset.request_gpu(device.id, ())),
                    ctx.submit(input.metadata.request_scalar()),
                };

                let pipeline = device
                    .request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(device, SHADER)
                    });

                let mut brick_stream = ctx.submit_unordered_with_data(
                    positions
                        .iter()
                        .map(|pos| (input.bricks.request_gpu(device.id, *pos), *pos)),
                );

                while let Some((gpu_brick_in, pos)) = brick_stream.next().await {
                    let brick_info = m.chunk_info(pos);

                    let gpu_brick_out =
                        ctx.alloc_slot_gpu(device, pos, brick_info.mem_elements())?;

                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config = DescriptorConfig::new([
                            &scale_gpu,
                            &offset_gpu,
                            &gpu_brick_in,
                            &gpu_brick_out,
                        ]);
                        let descriptor_writes = descriptor_config.writes();

                        let local_size = pipeline.local_size();

                        let global_size = brick_info.mem_elements();
                        let num_wgs = crate::util::div_round_up(global_size, local_size);

                        unsafe {
                            pipeline.bind(device, cmd.raw());
                            pipeline.push_constant(
                                device,
                                cmd.raw(),
                                PushConstants {
                                    chunk_pos: pos.into_elem::<u32>().into(),
                                    num_chunk_elems: global_size.try_into().unwrap(),
                                },
                            );
                            pipeline.push_descriptor_set(device, cmd.raw(), 0, &descriptor_writes);
                            device.device.cmd_dispatch(
                                cmd.raw(),
                                num_wgs.try_into().unwrap(),
                                1,
                                1,
                            );
                        }
                    });

                    // TODO: Maybe to allow more parallel access we want to postpone this,
                    // since this involves a memory barrier
                    // Possible alternative: Only insert barriers before use/download
                    unsafe { gpu_brick_out.initialized() };
                }

                Ok(())
            }
            .into()
        },
    )
}

pub fn rechunk<'op>(
    input: VolumeOperator<'op>,
    brick_size: LocalVoxelPosition,
) -> VolumeOperator<'op> {
    #[derive(Copy, Clone, AsStd140)]
    struct PushConstants {
        mem_size_in: mint::Vector3<u32>,
        mem_size_out: mint::Vector3<u32>,
        begin_in: mint::Vector3<u32>,
        begin_out: mint::Vector3<u32>,
        region_size: mint::Vector3<u32>,
        global_size: u32,
    }
    const SHADER: &'static str = r#"
#version 450

layout (local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputBuffer{
    float values[];
} sourceData;

layout(std430, binding = 1) buffer OutputBuffer{
    float values[];
} outputData;

layout(std140, push_constant) uniform PushConstants
{
    uvec3 mem_size_in;
    uvec3 mem_size_out;
    uvec3 begin_in;
    uvec3 begin_out;
    uvec3 region_size;
    uint global_size;
} constants;

uvec3 from_linear(uint linear_pos, uvec3 size) {
    uvec3 vec_pos;
    vec_pos.x = linear_pos % size.x;
    linear_pos /= size.x;
    vec_pos.y = linear_pos % size.y;
    linear_pos /= size.y;
    vec_pos.z = linear_pos;

    return vec_pos;
}

uint to_linear(uvec3 vec_pos, uvec3 size) {
    return vec_pos.x + size.x*(vec_pos.y + size.y*vec_pos.z);
}

void main() {
    uint gID = gl_GlobalInvocationID.x;

    if(gID < constants.global_size) {
        uvec3 region_pos = from_linear(gID, constants.region_size);

        uvec3 in_pos = constants.begin_in + region_pos;
        uvec3 out_pos = constants.begin_out + region_pos;

        uint in_index = to_linear(in_pos, constants.mem_size_in);
        uint out_index = to_linear(out_pos, constants.mem_size_out);

        outputData.values[out_index] = sourceData.values[in_index];
    }
}
"#;
    TensorOperator::with_state(
        OperatorId::new("volume_rechunk_gpu")
            .dependent_on(&input)
            .dependent_on(Id::hash(&brick_size)),
        input.clone(),
        input,
        move |ctx, input, _| {
            async move {
                let req = input.metadata.request_scalar();
                let mut m = ctx.submit(req).await;
                m.chunk_size = brick_size;
                ctx.write(m)
            }
            .into()
        },
        move |ctx, positions, input, _| {
            // TODO: optimize case where input.brick_size == output.brick_size
            async move {
                let device = ctx.vulkan_device();

                let pipeline = device
                    .request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(device, SHADER)
                    });

                let m_in = ctx.submit(input.metadata.request_scalar()).await;
                let m_out = {
                    let mut m_out = m_in;
                    m_out.chunk_size = brick_size;
                    m_out
                };

                let requests = positions.into_iter().map(|pos| {
                    let out_info = m_out.chunk_info(pos);
                    let out_begin = out_info.begin();
                    let out_end = out_info.end();

                    let in_begin_brick = m_in.chunk_pos(out_begin);
                    let in_end_brick = m_in.chunk_pos(out_end.map(|v| v - 1u32));

                    let in_brick_positions = itertools::iproduct! {
                        in_begin_brick.z().raw..=in_end_brick.z().raw,
                        in_begin_brick.y().raw..=in_end_brick.y().raw,
                        in_begin_brick.x().raw..=in_end_brick.x().raw
                    }
                    .map(|(z, y, x)| BrickPosition::from([z, y, x]))
                    .collect::<Vec<_>>();
                    let intersecting_bricks = ctx.group(
                        in_brick_positions
                            .iter()
                            .map(|pos| input.bricks.request_gpu(device.id, *pos)),
                    );

                    (intersecting_bricks, (pos, in_brick_positions))
                });

                let mut stream = ctx.submit_unordered_with_data(requests);
                while let Some((intersecting_bricks, (pos, in_brick_positions))) =
                    stream.next().await
                {
                    let out_info = m_out.chunk_info(pos);

                    let gpu_brick_out = ctx
                        .alloc_slot_gpu(device, pos, out_info.mem_elements())
                        .unwrap();

                    device.with_cmd_buffer(|cmd| {
                        let out_begin = out_info.begin();
                        let out_end = out_info.end();

                        for (gpu_brick_in, in_brick_pos) in intersecting_bricks
                            .iter()
                            .zip(in_brick_positions.into_iter())
                        {
                            let in_info = m_in.chunk_info(in_brick_pos);

                            let in_begin = in_info.begin();
                            let in_end = in_info.end();

                            let overlap_begin = in_begin.zip(out_begin, |i, o| i.max(o));
                            let overlap_end = in_end.zip(out_end, |i, o| i.min(o));
                            let overlap_size =
                                (overlap_end - overlap_begin).map(LocalCoordinate::interpret_as);

                            let in_chunk_begin = in_info.in_chunk(overlap_begin);

                            let out_chunk_begin = out_info.in_chunk(overlap_begin);

                            let descriptor_config =
                                DescriptorConfig::new([gpu_brick_in, &gpu_brick_out]);
                            let descriptor_writes = descriptor_config.writes();

                            let local_size = pipeline.local_size();

                            let global_size = crate::data::hmul(overlap_size);
                            let num_wgs = crate::util::div_round_up(global_size, local_size);

                            //TODO initialization of outside regions
                            unsafe {
                                pipeline.bind(device, cmd.raw());
                                pipeline.push_constant(
                                    device,
                                    cmd.raw(),
                                    PushConstants {
                                        mem_size_in: m_in.chunk_size.into_elem::<u32>().into(),
                                        mem_size_out: m_out.chunk_size.into_elem::<u32>().into(),
                                        begin_in: in_chunk_begin.into_elem::<u32>().into(),
                                        begin_out: out_chunk_begin.into_elem::<u32>().into(),
                                        region_size: overlap_size.into_elem::<u32>().into(),
                                        global_size: global_size as _,
                                    },
                                );
                                pipeline.push_descriptor_set(
                                    device,
                                    cmd.raw(),
                                    0,
                                    &descriptor_writes,
                                );
                                device.device.cmd_dispatch(
                                    cmd.raw(),
                                    num_wgs.try_into().unwrap(),
                                    1,
                                    1,
                                );
                            }
                        }
                    });
                    unsafe { gpu_brick_out.initialized() };
                }

                Ok(())
            }
            .into()
        },
    )
}

/// A one dimensional convolution in the specified (constant) axis. Currently zero padding is the
/// only supported (and thus always applied) border handling routine.
pub fn convolution_1d<'op, const DIM: usize>(
    input: VolumeOperator<'op>,
    kernel: ArrayOperator<'op>,
) -> VolumeOperator<'op> {
    #[derive(Copy, Clone, AsStd140)]
    struct PushConstants {
        mem_dim: mint::Vector3<u32>,
        logical_dim_out: mint::Vector3<u32>,
        out_begin: mint::Vector3<u32>,
        global_dim: mint::Vector3<u32>,
        num_chunks: u32,
        first_chunk_pos: u32,
        extent: i32,
        global_size: u32,
    }
    const SHADER: &'static str = r#"
#version 450

layout (local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputBuffer{
    float values[];
} sourceData[MAX_BRICKS];

layout(std430, binding = 1) readonly buffer KernelBuffer{
    float values[];
} kernel;

layout(std430, binding = 2) buffer OutputBuffer{
    float values[];
} outputData;

layout(std140, push_constant) uniform PushConstants {
    uvec3 mem_dim;
    uvec3 logical_dim_out;
    uvec3 out_begin;
    uvec3 global_dim;
    uint num_chunks;
    uint first_chunk_pos;
    int extent;
    uint global_size;
} consts;

uvec3 from_linear(uint linear_pos, uvec3 size) {
    uvec3 vec_pos;
    vec_pos.x = linear_pos % size.x;
    linear_pos /= size.x;
    vec_pos.y = linear_pos % size.y;
    linear_pos /= size.y;
    vec_pos.z = linear_pos;

    return vec_pos;
}

uint to_linear(uvec3 vec_pos, uvec3 size) {
    return vec_pos.x + size.x*(vec_pos.y + size.y*vec_pos.z);
}

void main() {
    uint gID = gl_GlobalInvocationID.x;

    if(gID < consts.global_size) {
        uvec3 out_local = from_linear(gID, consts.mem_dim);
        float acc = 0.0;

        if(out_local.x < consts.logical_dim_out.x && out_local.y < consts.logical_dim_out.y && out_local.z < consts.logical_dim_out.z) {
            for (int i = 0; i<consts.num_chunks; ++i) {
                int chunk_pos = int(consts.first_chunk_pos) + i;
                int global_begin_pos_in = chunk_pos * int(consts.mem_dim[DIM]);

                int logical_dim_in = min(
                    global_begin_pos_in + int(consts.mem_dim[DIM]),
                    int(consts.global_dim[DIM])
                ) - global_begin_pos_in;

                int out_chunk_to_in_chunk = int(consts.out_begin[DIM]) - global_begin_pos_in;
                int out_pos_rel_to_in_pos_rel = int(out_local[DIM]) + out_chunk_to_in_chunk;

                int begin_ext = -consts.extent;
                int end_ext = consts.extent;

                int local_end = logical_dim_in;

                int l_begin = max(begin_ext + out_pos_rel_to_in_pos_rel, 0);
                int l_end = min(end_ext + out_pos_rel_to_in_pos_rel, local_end - 1);

                for (int local=l_begin; local<=l_end; ++local) {
                    int kernel_offset = local - out_pos_rel_to_in_pos_rel;
                    int kernel_buf_index = consts.extent - kernel_offset;
                    float kernel_val = kernel.values[kernel_buf_index];

                    uvec3 pos = out_local;
                    pos[DIM] = local;

                    uint local_index = to_linear(pos, consts.mem_dim);
                    float local_val = sourceData[i].values[local_index];

                    acc += kernel_val * local_val;
                }
            }
        } else {
            acc = 123.456;
        }

        outputData.values[gID] = acc;
    }
}
"#;
    TensorOperator::with_state(
        OperatorId::new("convolution_1d_gpu")
            .dependent_on(&input)
            .dependent_on(&kernel),
        input.clone(),
        (input, kernel),
        move |ctx, input, _| {
            async move {
                let req = input.metadata.request_scalar();
                let m = ctx.submit(req).await;
                ctx.write(m)
            }
            .into()
        },
        move |ctx, positions, (input, kernel), _| {
            async move {
                let device = ctx.vulkan_device();

                let (m_in, kernel_m, kernel_handle) = futures::join!(
                    ctx.submit(input.metadata.request_scalar()),
                    ctx.submit(kernel.metadata.request_scalar()),
                    ctx.submit(kernel.bricks.request_gpu(device.id, [0].into())),
                );

                let m_out = m_in;

                assert_eq!(
                    kernel_m.dimensions.raw(),
                    kernel_m.chunk_size.raw(),
                    "Only unchunked kernels are supported for now"
                );

                let kernel_size = *kernel_m.dimensions.raw();
                assert!(kernel_size % 2 == 1, "Kernel size must be odd");
                let extent = kernel_size / 2;

                let requests = positions.into_iter().map(|pos| {
                    let out_info = m_out.chunk_info(pos);
                    let out_begin = out_info.begin();
                    let out_end = out_info.end();

                    let in_begin = out_begin
                        .map_element(DIM, |v| (v.raw.saturating_sub(extent as u32)).into());
                    let in_end = out_end
                        .map_element(DIM, |v| (v + extent as u32).min(m_out.dimensions.0[DIM]));

                    let in_begin_brick = m_in.chunk_pos(in_begin);
                    let in_end_brick = m_in.chunk_pos(in_end.map(|v| v - 1u32));

                    let in_brick_positions = itertools::iproduct! {
                        in_begin_brick.z().raw..=in_end_brick.z().raw,
                        in_begin_brick.y().raw..=in_end_brick.y().raw,
                        in_begin_brick.x().raw..=in_end_brick.x().raw
                    }
                    .map(|(z, y, x)| BrickPosition::from([z, y, x]))
                    .collect::<Vec<_>>();

                    let intersecting_bricks = ctx.group(
                        in_brick_positions
                            .iter()
                            .map(|pos| input.bricks.request_gpu(device.id, *pos)),
                    );

                    (intersecting_bricks, (pos, in_brick_positions))
                });

                let max_bricks =
                    2 * crate::util::div_round_up(extent, m_in.chunk_size.0[DIM].raw) + 1;

                //TODO: This is really ugly, we should investigate to use z,y,x ordering in shaders
                //as well.
                let shader_dimension = 2 - DIM;
                let pipeline = device.request_state(
                    RessourceId::new("pipeline")
                        .of(ctx.current_op())
                        .dependent_on(Id::hash(&max_bricks))
                        .dependent_on(Id::hash(&DIM)),
                    || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .add("MAX_BRICKS", max_bricks.to_string())
                                    .add("DIM", shader_dimension.to_string()),
                            ),
                        )
                    },
                );

                let mut stream = ctx.submit_unordered_with_data(requests);
                while let Some((intersecting_bricks, (pos, in_brick_positions))) =
                    stream.next().await
                {
                    let out_info = m_out.chunk_info(pos);
                    let gpu_brick_out = ctx
                        .alloc_slot_gpu(device, pos, out_info.mem_elements())
                        .unwrap();

                    let out_begin = out_info.begin();

                    for window in in_brick_positions.windows(2) {
                        for d in 0..3 {
                            if d == DIM {
                                assert_eq!(window[0].0[d] + 1u32, window[1].0[d]);
                            } else {
                                assert_eq!(window[0].0[d], window[1].0[d]);
                            }
                        }
                    }

                    let num_chunks = in_brick_positions.len();
                    assert!(num_chunks > 0);

                    assert_eq!(num_chunks, intersecting_bricks.len());

                    let first_chunk_pos = in_brick_positions.first().unwrap().0[DIM].raw;
                    let global_size = crate::data::hmul(m_out.chunk_size);

                    let consts = PushConstants {
                        mem_dim: m_out.chunk_size.into_elem::<u32>().into(),
                        logical_dim_out: out_info.logical_dimensions.into_elem::<u32>().into(),
                        out_begin: out_begin.into_elem::<u32>().into(),
                        global_dim: m_out.dimensions.into_elem::<u32>().into(),
                        num_chunks: num_chunks as _,
                        first_chunk_pos,
                        extent: extent as i32,
                        global_size: global_size.try_into().unwrap(),
                    };

                    // TODO: This padding to max_bricks is necessary since the descriptor array in
                    // the shader has a static since. Once we use dynamic ssbos this can go away.
                    let intersecting_bricks = (0..max_bricks)
                        .map(|i| {
                            intersecting_bricks
                                .get(i as usize)
                                .unwrap_or(intersecting_bricks.get(0).unwrap())
                        })
                        .collect::<Vec<_>>();
                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config = DescriptorConfig::new([
                            &intersecting_bricks.as_slice(),
                            &kernel_handle,
                            &gpu_brick_out,
                        ]);
                        let descriptor_writes = descriptor_config.writes();

                        let local_size = pipeline.local_size();

                        let num_wgs = crate::util::div_round_up(global_size, local_size);

                        unsafe {
                            pipeline.bind(device, cmd.raw());
                            pipeline.push_descriptor_set(device, cmd.raw(), 0, &descriptor_writes);
                            pipeline.push_constant(device, cmd.raw(), consts);
                            device.device.cmd_dispatch(
                                cmd.raw(),
                                num_wgs.try_into().unwrap(),
                                1,
                                1,
                            );
                        }
                    });

                    unsafe { gpu_brick_out.initialized() };
                }

                Ok(())
            }
            .into()
        },
    )
    .into()
}

pub fn separable_convolution<'op>(
    v: VolumeOperator<'op>,
    [k0, k1, k2]: [ArrayOperator<'op>; 3],
) -> VolumeOperator<'op> {
    let v = convolution_1d::<2>(v, k2);
    let v = convolution_1d::<1>(v, k1);
    let v = convolution_1d::<0>(v, k0);
    v
}

pub fn mean<'op>(input: VolumeOperator<'op>) -> ScalarOperator<'op, f32> {
    #[derive(Copy, Clone, AsStd140)]
    struct PushConstants {
        mem_dim: mint::Vector3<u32>,
        logical_dim: mint::Vector3<u32>,
        norm_factor: f32,
        num_chunk_elems: u32,
    }
    const SHADER: &'static str = r#"
#version 450

#extension GL_KHR_shader_subgroup_arithmetic : require

layout (local_size_x = 1024) in;

layout(std430, binding = 0) readonly buffer InputBuffer{
    float values[];
} sourceData;

layout(std430, binding = 1) buffer OutputBuffer{
    uint value;
} sum;

layout(std140, push_constant) uniform PushConstants
{
    uvec3 mem_dim;
    uvec3 logical_dim;
    float norm_factor;
    uint num_chunk_elems;
} consts;

uvec3 from_linear(uint linear_pos, uvec3 size) {
    uvec3 vec_pos;
    vec_pos.x = linear_pos % size.x;
    linear_pos /= size.x;
    vec_pos.y = linear_pos % size.y;
    linear_pos /= size.y;
    vec_pos.z = linear_pos;

    return vec_pos;
}

#define atomic_add(mem, value) {\
    uint initial = 0;\
    uint new = 0;\
    do {\
        initial = mem;\
        new = floatBitsToUint(uintBitsToFloat(initial) + (value));\
        if (new == initial) {\
            break;\
        }\
    } while(atomicCompSwap(mem, initial, new) != initial);\
}

shared uint shared_sum;

void main()
{
    uint gID = gl_GlobalInvocationID.x;
    if(gl_LocalInvocationIndex == 0) {
        shared_sum = floatBitsToUint(0.0);
    }
    barrier();

    float val;
    if(gID < consts.num_chunk_elems) {
        uvec3 local = from_linear(gID, consts.mem_dim);

        if(local.x < consts.logical_dim.x && local.y < consts.logical_dim.y && local.z < consts.logical_dim.z) {
            val = sourceData.values[gID] * consts.norm_factor;
        } else {
            val = 0.0;
        }
    } else {
        val = 0.0;
    }

    float sg_sum = subgroupAdd(val);

    if(gl_SubgroupInvocationID == 0) {
        atomic_add(shared_sum, sg_sum);
    }

    barrier();

    if(gl_LocalInvocationIndex == 0) {
        atomic_add(sum.value, uintBitsToFloat(shared_sum));
    }
}
"#;

    crate::operators::scalar::scalar(
        OperatorId::new("volume_mean_gpu").dependent_on(&input),
        input,
        move |ctx, input, _| {
            async move {
                let device = ctx.vulkan_device();

                let m = ctx.submit(input.metadata.request_scalar()).await;

                let to_request = m.brick_positions().collect::<Vec<_>>();
                let batch_size = 1024;

                let pipeline = device
                    .request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(device, SHADER)
                    });

                let sum = ctx.alloc_scalar_gpu(device)?;

                let normalization_factor = 1.0 / (crate::data::hmul(m.dimensions) as f32);

                let memory_barriers = &[vk::MemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::HOST)
                    .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                    .build()];
                let barrier_info = vk::DependencyInfo::builder().memory_barriers(memory_barriers);
                device.with_cmd_buffer(|cmd| unsafe {
                    device.device.cmd_update_buffer(
                        cmd.raw(),
                        sum.buffer,
                        0,
                        bytemuck::cast_slice(&[0f32]),
                    );
                    cmd.pipeline_barrier(&barrier_info);
                });

                for chunk in to_request.chunks(batch_size) {
                    let mut stream = ctx.submit_unordered_with_data(
                        chunk
                            .iter()
                            .map(|pos| (input.bricks.request_gpu(device.id, *pos), *pos)),
                    );
                    while let Some((gpu_brick_in, pos)) = stream.next().await {
                        let brick_info = m.chunk_info(pos);

                        device.with_cmd_buffer(|cmd| {
                            let descriptor_config = DescriptorConfig::new([&gpu_brick_in, &sum]);
                            let descriptor_writes = descriptor_config.writes();

                            let local_size = pipeline.local_size();

                            let global_size = brick_info.mem_elements();
                            let num_wgs = crate::util::div_round_up(global_size, local_size);

                            unsafe {
                                pipeline.bind(device, cmd.raw());
                                pipeline.push_constant(
                                    device,
                                    cmd.raw(),
                                    PushConstants {
                                        mem_dim: brick_info
                                            .mem_dimensions
                                            .into_elem::<u32>()
                                            .into(),
                                        logical_dim: brick_info
                                            .logical_dimensions
                                            .into_elem::<u32>()
                                            .into(),
                                        norm_factor: normalization_factor,
                                        num_chunk_elems: m.num_elements().try_into().unwrap(),
                                    },
                                );
                                pipeline.push_descriptor_set(
                                    device,
                                    cmd.raw(),
                                    0,
                                    &descriptor_writes,
                                );
                                device.device.cmd_dispatch(
                                    cmd.raw(),
                                    num_wgs.try_into().unwrap(),
                                    1,
                                    1,
                                );
                            }
                        });
                    }
                    ctx.submit(device.wait_for_cmd_buffer_completion()).await;
                }
                unsafe { sum.initialized() };

                Ok(())
            }
            .into()
        },
    )
}

pub struct VoxelRasterizerGLSL {
    pub body: String,
    pub metadata: VolumeMetaData,
}

impl super::volume::VolumeOperatorState for VoxelRasterizerGLSL {
    fn operate<'a>(&'a self) -> VolumeOperator<'a> {
        #[derive(Copy, Clone, AsStd140)]
        struct PushConstants {
            offset: mint::Vector3<u32>,
            mem_dim: mint::Vector3<u32>,
            logical_dim: mint::Vector3<u32>,
            vol_dim: mint::Vector3<u32>,
            num_chunk_elems: u32,
        }

        let m = self.metadata;

        let shader = format!(
            "{}{}{}",
            r#"
#version 450

layout (local_size_x = 256) in;

layout(std430, binding = 0) buffer OutputBuffer{
    float values[];
} outputData;

layout(std140, push_constant) uniform PushConstants
{
    uvec3 offset;
    uvec3 mem_dim;
    uvec3 logical_dim;
    uvec3 vol_dim;
    uint num_chunk_elems;
} consts;

uvec3 from_linear(uint linear_pos, uvec3 size) {
    uvec3 vec_pos;
    vec_pos.x = linear_pos % size.x;
    linear_pos /= size.x;
    vec_pos.y = linear_pos % size.y;
    linear_pos /= size.y;
    vec_pos.z = linear_pos;

    return vec_pos;
}

uint to_linear(uvec3 vec_pos, uvec3 size) {
    return vec_pos.x + size.x*(vec_pos.y + size.y*vec_pos.z);
}

void main()
{
    uint gID = gl_GlobalInvocationID.x;

    if(gID < consts.num_chunk_elems) {
        uvec3 out_local = from_linear(gID, consts.mem_dim);
        float result = 0.0;
        uvec3 pos_voxel = out_local + consts.offset;
        vec3 pos_normalized = vec3(pos_voxel)/vec3(consts.vol_dim);

        if(out_local.x < consts.logical_dim.x && out_local.y < consts.logical_dim.y && out_local.z < consts.logical_dim.z) {
        "#,
            self.body,
            r#"
        } else {
            result = 123.456;
        }

        outputData.values[gID] = result;
    }
}
"#
        );

        TensorOperator::with_state(
            OperatorId::new("rasterize_gpu")
                .dependent_on(Id::hash(&shader))
                .dependent_on(Id::hash(&m)),
            (),
            shader,
            move |ctx, _, _| async move { ctx.write(m) }.into(),
            move |ctx, positions, shader, _| {
                async move {
                    let device = ctx.vulkan_device();

                    let pipeline = device
                        .request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                            ComputePipeline::new(device, shader.as_str())
                        });

                    for pos in positions {
                        let brick_info = m.chunk_info(pos);

                        let gpu_brick_out =
                            ctx.alloc_slot_gpu(device, pos, brick_info.mem_elements())?;
                        device.with_cmd_buffer(|cmd| {
                            let descriptor_config = DescriptorConfig::new([&gpu_brick_out]);
                            let descriptor_writes = descriptor_config.writes();

                            let local_size = pipeline.local_size();

                            let global_size = brick_info.mem_elements();
                            let num_wgs = crate::util::div_round_up(global_size, local_size);

                            unsafe {
                                pipeline.bind(device, cmd.raw());
                                pipeline.push_constant(
                                    device,
                                    cmd.raw(),
                                    PushConstants {
                                        num_chunk_elems: m.num_elements().try_into().unwrap(),
                                        offset: brick_info.begin.into_elem::<u32>().into(),
                                        logical_dim: brick_info
                                            .logical_dimensions
                                            .into_elem::<u32>()
                                            .into(),
                                        mem_dim: m.chunk_size.into_elem::<u32>().into(),
                                        vol_dim: m.dimensions.into_elem::<u32>().into(),
                                    },
                                );
                                pipeline.push_descriptor_set(
                                    device,
                                    cmd.raw(),
                                    0,
                                    &descriptor_writes,
                                );
                                device.device.cmd_dispatch(
                                    cmd.raw(),
                                    num_wgs.try_into().unwrap(),
                                    1,
                                    1,
                                );
                            }
                        });

                        // TODO: Maybe to allow more parallel access we want to postpone this,
                        // since this involves a memory barrier
                        // Possible alternative: Only insert barriers before use/download
                        unsafe { gpu_brick_out.initialized() };
                    }

                    Ok(())
                }
                .into()
            },
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        data::{GlobalCoordinate, Vector, VoxelPosition},
        operators::volume::VolumeOperatorState,
        test_util::*,
    };

    #[test]
    fn test_rescale_gpu() {
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let (point_vol, center) = center_point_vol(size, brick_size);

        let fill_expected = |comp: &mut ndarray::ArrayViewMut3<f32>| {
            for z in 0..size.z().raw {
                for y in 0..size.y().raw {
                    for x in 0..size.x().raw {
                        let pos = VoxelPosition::from([z, y, x]);
                        if pos != center {
                            comp[pos.as_index()] = 1.0;
                        }
                    }
                }
            }
            comp[center.as_index()] = -1.0;
        };
        let scale = (-2.0).into();
        let offset = (1.0).into();
        let input = point_vol.operate();
        let output = linear_rescale(input, scale, offset);
        compare_volume(output, size, fill_expected);
    }

    #[test]
    fn test_rechunk_gpu() {
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let input = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            crate::data::to_linear(v, size) as f32
        });

        let fill_expected = |comp: &mut ndarray::ArrayViewMut3<f32>| {
            for z in 0..size.z().raw {
                for y in 0..size.y().raw {
                    for x in 0..size.x().raw {
                        let pos = VoxelPosition::from([z, y, x]);
                        let val = crate::data::to_linear(pos, size) as f32;
                        comp[pos.as_index()] = val
                    }
                }
            }
        };
        let input = input.operate();
        for chunk_size in [[5, 1, 1], [4, 4, 1], [2, 3, 4], [1, 1, 1], [5, 5, 5]] {
            let output = rechunk(input.clone(), LocalVoxelPosition::from(chunk_size));
            compare_volume(output, size, fill_expected);
        }
    }

    #[test]
    fn test_rasterize_gpu() {
        let size = VoxelPosition::fill(5.into());

        let fill_expected = |comp: &mut ndarray::ArrayViewMut3<f32>| {
            for z in 0..size.z().raw {
                for y in 0..size.y().raw {
                    for x in 0..size.x().raw {
                        let pos = VoxelPosition::from([z, y, x]);
                        comp[pos.as_index()] = x as f32 + y as f32 + z as f32;
                    }
                }
            }
        };
        for chunk_size in [[5, 1, 1], [4, 4, 1], [2, 3, 4], [1, 1, 1], [5, 5, 5]] {
            let input = VoxelRasterizerGLSL {
                metadata: crate::array::VolumeMetaData {
                    dimensions: size,
                    chunk_size: chunk_size.into(),
                },
                body: r#"result = float(pos_voxel.x + pos_voxel.y + pos_voxel.z);"#.to_owned(),
            };
            let input = input.operate();
            let output = rechunk(input, LocalVoxelPosition::from(chunk_size));
            compare_volume(output, size, fill_expected);
        }
    }

    fn compare_convolution_1d<const DIM: usize>(
        input: &dyn VolumeOperatorState,
        size: VoxelPosition,
        kernel: &[f32],
        fill_expected: impl FnOnce(&mut ndarray::ArrayViewMut3<f32>),
    ) {
        let input = input.operate();
        let output = convolution_1d::<DIM>(input, crate::operators::array::from_static(kernel));
        compare_volume(output, size, fill_expected);
    }

    fn test_convolution_1d_generic<const DIM: usize>() {
        // Small
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let (point_vol, center) = center_point_vol(size, brick_size);
        compare_convolution_1d::<DIM>(&point_vol, size, &[1.0, -1.0, 2.0], |comp| {
            comp[center.map_element(DIM, |v| v - 1u32).as_index()] = 1.0;
            comp[center.map_element(DIM, |v| v).as_index()] = -1.0;
            comp[center.map_element(DIM, |v| v + 1u32).as_index()] = 2.0;
        });

        // Larger
        let size = VoxelPosition::fill(13.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let (point_vol, center) = center_point_vol(size, brick_size);
        let kernel_size = 7;
        let extent = kernel_size / 2;
        let mut kernel = vec![0.0; kernel_size];
        kernel[0] = -1.0;
        kernel[1] = -2.0;
        kernel[kernel_size - 1] = 1.0;
        kernel[kernel_size - 2] = 2.0;
        compare_convolution_1d::<DIM>(&point_vol, size, &kernel, |comp| {
            comp[center.map_element(DIM, |v| v - extent).as_index()] = -1.0;
            comp[center.map_element(DIM, |v| v - extent + 1u32).as_index()] = -2.0;
            comp[center.map_element(DIM, |v| v + extent).as_index()] = 1.0;
            comp[center.map_element(DIM, |v| v + extent - 1u32).as_index()] = 2.0;
        });
    }

    #[test]
    fn test_convolution_1d_x() {
        test_convolution_1d_generic::<2>();
    }
    #[test]
    fn test_convolution_1d_y() {
        test_convolution_1d_generic::<1>();
    }
    #[test]
    fn test_convolution_1d_z() {
        test_convolution_1d_generic::<0>();
    }

    #[test]
    fn test_separable_convolution() {
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let (point_vol, center) = center_point_vol(size, brick_size);

        let kernels = [&[2.0, 1.0, 2.0], &[2.0, 1.0, 2.0], &[2.0, 1.0, 2.0]];
        let kernels = std::array::from_fn(|i| crate::operators::array::from_static(kernels[i]));
        let output = separable_convolution(point_vol.operate(), kernels);
        compare_volume(output, size, |comp| {
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let offset = Vector::new([dz, dy, dx]);
                        let l1_dist = offset.map(i32::abs).fold(0, std::ops::Add::add);
                        let expected_val = 1 << l1_dist;
                        comp[(center.try_into_elem::<i32>().unwrap() + offset)
                            .try_into_elem::<GlobalCoordinate>()
                            .unwrap()
                            .as_index()] = expected_val as f32;
                    }
                }
            }
        });
    }
}
