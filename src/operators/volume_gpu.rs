use std::alloc::Layout;

use ash::vk::{self, BufferUsageFlags};
use crevice::std140::{AsStd140, Std140};
use futures::StreamExt;

use crate::{
    operator::OperatorId,
    vulkan::{DeviceContext, VulkanState},
};

use super::{scalar::ScalarOperator, volume::VolumeOperator};

#[derive(Copy, Clone, AsStd140)]
struct Config {
    offset: f32,
    scale: f32,
    num_chunk_elems: u32,
}

#[derive(Copy, Clone, AsStd140)]
struct PushConstants {
    chunk_pos: mint::Vector3<u32>,
}

const SHADER: &'static str = r#"
#version 450

layout (local_size_x = 256) in;

layout(std140, binding = 0) uniform Config{
    float offset;
    float scale;
    uint num_chunk_elems;
} config;

layout(std430, binding = 1) readonly buffer InputBuffer{
    float values[];
} sourceData;

layout(std430, binding = 2) buffer OutputBuffer{
    float values[];
} outputData;

layout(std140, push_constant) uniform PushConstants
{
	uvec3 chunk_pos;
} constants;

void main()
{
    uint gID = gl_GlobalInvocationID.x;

    if(gID < config.num_chunk_elems) {
        outputData.values[gID] = config.scale*sourceData.values[gID] + config.offset;
    }
}

"#;

fn layout_std140<T: AsStd140>() -> Layout {
    unsafe { Layout::from_size_align_unchecked(T::std140_size_static(), T::Output::ALIGNMENT) }
}

struct Shader {
    module: vk::ShaderModule,
}

impl Shader {
    fn from_compiled(device: &DeviceContext, code: &[u32]) -> Self {
        let info = vk::ShaderModuleCreateInfo::builder().code(&code);

        let module = unsafe { device.device.create_shader_module(&info, None) }.unwrap();

        Self { module }
    }
    fn from_source(device: &DeviceContext, source: &str) -> Self {
        use spirv_compiler::*;

        let mut compiler = CompilerBuilder::new()
            .with_source_language(SourceLanguage::GLSL)
            .build()
            .unwrap();
        let kind = ShaderKind::Compute;
        let compiled = compiler.compile_from_string(&source, kind).unwrap();

        Self::from_compiled(device, compiled.as_slice())
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
    descriptor_set_layout: vk::DescriptorSetLayout,
}

impl ComputePipeline {
    fn new(
        device: &DeviceContext,
        shader: &str,
        descriptor_set_bindings: &[vk::DescriptorSetLayoutBinding],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Self {
        let mut shader = Shader::from_source(device, shader);

        let dsl_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&descriptor_set_bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR);
        let descriptor_set_layout =
            unsafe { device.device.create_descriptor_set_layout(&dsl_info, None) }.unwrap();
        let dsls = &[descriptor_set_layout];

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(dsls)
            .push_constant_ranges(push_constant_ranges);

        let pipeline_layout = unsafe {
            device
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .unwrap();

        let pipeline_info = vk::PipelineShaderStageCreateInfo::builder()
            .module(shader.module)
            .name(cstr::cstr!("main"))
            .stage(vk::ShaderStageFlags::COMPUTE);

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
            descriptor_set_layout,
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
        device.device.cmd_push_constants(
            cmd,
            self.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            val.as_std140().as_bytes(),
        );
    }
}

impl VulkanState for ComputePipeline {
    unsafe fn deinitialize(&mut self, context: &crate::vulkan::DeviceContext) {
        unsafe {
            context
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            context.device.destroy_pipeline(self.pipeline, None);
            context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None)
        };
    }
}

pub fn linear_rescale<'op>(
    input: VolumeOperator<'op>,
    scale: ScalarOperator<'op, f32>,
    offset: ScalarOperator<'op, f32>,
) -> VolumeOperator<'op> {
    VolumeOperator::with_state(
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
                let (scale, offset, m) = futures::join! {
                    ctx.submit(scale.request_scalar()),
                    ctx.submit(offset.request_scalar()),
                    ctx.submit(input.metadata.request_scalar()),
                };

                let config = Config {
                    scale,
                    offset,
                    num_chunk_elems: crate::data::hmul(m.chunk_size).try_into().unwrap(),
                };

                let device = ctx.vulkan_device();

                let gpu_config = device.allocator().allocate(
                    layout_std140::<Config>(),
                    BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::UNIFORM_BUFFER,
                    gpu_allocator::MemoryLocation::GpuOnly,
                );

                let config_descriptor_info = vk::DescriptorBufferInfo::builder()
                    .buffer(gpu_config.buffer)
                    .range(gpu_config.allocation.size());

                let pipeline = device.request_state("linear_rescale_gpu_pipeline", || {
                    // TODO: It would be nice to derive these automagically from the shader.
                    // https://crates.io/crates/spirv-tools may be useful for this.
                    let bindings = [
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .build(),
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .build(),
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(2)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE)
                            .build(),
                    ];
                    let push_constants = [vk::PushConstantRange::builder()
                        .size(PushConstants::std140_size_static() as _)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .build()];

                    ComputePipeline::new(device, SHADER, &bindings, &push_constants)
                });

                let mut brick_stream = ctx.submit_unordered_with_data(
                    positions
                        .iter()
                        .map(|pos| (input.bricks.request_gpu(device.id, *pos), *pos)),
                );

                device.with_cmd_buffer(|cmd| unsafe {
                    device.device.cmd_update_buffer(
                        cmd.raw(),
                        gpu_config.buffer,
                        0,
                        config.as_std140().as_bytes(),
                    );

                    let memory_barriers = &[vk::MemoryBarrier2::builder()
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                        .build()];
                    let barrier_info =
                        vk::DependencyInfo::builder().memory_barriers(memory_barriers);
                    device
                        .device
                        .cmd_pipeline_barrier2(cmd.raw(), &barrier_info);
                });

                while let Some((gpu_brick_in, pos)) = brick_stream.next().await {
                    let brick_info = m.chunk_info(pos);

                    device.with_cmd_buffer(|cmd| {
                        let gpu_brick_out =
                            ctx.alloc_slot_gpu(cmd, pos, brick_info.mem_elements())?;

                        let db_info_in = vk::DescriptorBufferInfo::builder()
                            .buffer(gpu_brick_in.buffer)
                            .range(gpu_brick_in.layout.size() as _);

                        let db_info_out = vk::DescriptorBufferInfo::builder()
                            .buffer(gpu_brick_out.buffer)
                            .range(gpu_brick_out.size);

                        // TODO: Using build is really dangerous territory since it discards the
                        // lifetime of the builder and thus buffer_info parameters which allows us
                        // to use a temporary slice like &[*db_info_in] resulting in invalid reads.
                        let buf0 = [*config_descriptor_info];
                        let buf1 = [*db_info_in];
                        let buf2 = [*db_info_out];
                        let descriptor_writes = [
                            vk::WriteDescriptorSet::builder()
                                .dst_binding(0)
                                .dst_array_element(0)
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                .buffer_info(&buf0)
                                .build(),
                            vk::WriteDescriptorSet::builder()
                                .dst_binding(1)
                                .dst_array_element(0)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                .buffer_info(&buf1)
                                .build(),
                            vk::WriteDescriptorSet::builder()
                                .dst_binding(2)
                                .dst_array_element(0)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                .buffer_info(&buf2)
                                .build(),
                        ];

                        let local_size = 256; //TODO: somehow ensure that this is the same as specified
                                              //in shader
                        let global_size = brick_info.mem_elements();
                        let num_wgs = crate::util::div_round_up(global_size, local_size);

                        unsafe {
                            pipeline.bind(device, cmd.raw());
                            pipeline.push_constant(
                                device,
                                cmd.raw(),
                                PushConstants {
                                    chunk_pos: pos.into_elem::<u32>().into(),
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

                        // TODO: Maybe to allow more parallel access we want to postpone this,
                        // since this involves a memory barrier
                        // Possible alternative: Only insert barriers before use/download
                        unsafe { gpu_brick_out.initialized() };

                        Ok::<(), crate::Error>(())
                    })?;
                }

                ctx.submit(device.wait_for_cmd_buffer_submission()).await;
                device.allocator().deallocate(gpu_config);

                Ok(())
            }
            .into()
        },
    )
}
