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
}

const SHADER: &'static str = r#"
#version 450

layout (local_size_x = 256) in;

layout(std140, binding = 0) uniform Config{
    float offset;
    float scale;
} config;

layout(std430, binding = 1) readonly buffer InputBuffer{
    float values[];
} sourceData;

layout(std430, binding = 2) buffer OutputBuffer{
    float values[];
} outputData;


void main()
{
    //grab global ID
    uint gID = gl_GlobalInvocationID.x;
    //make sure we don't access past the buffer size
    //if(gID < matrixCount)
    //{
        // do math
        outputData.values[gID] =  config.scale*sourceData.values[gID] + config.offset;
    //}
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
    ) -> Self {
        let mut shader = Shader::from_source(device, shader);

        let dsl_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&descriptor_set_bindings);
        let descriptor_set_layout =
            unsafe { device.device.create_descriptor_set_layout(&dsl_info, None) }.unwrap();
        let dsls = &[descriptor_set_layout];

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(dsls);

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
        OperatorId::new("volume_scale")
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

                let config = Config { scale, offset };

                let device = ctx.vulkan_device();

                let mut gpu_config = device.allocator().allocate(
                    layout_std140::<Config>(),
                    BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::UNIFORM_BUFFER,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                );

                let config_descriptor_info = vk::DescriptorBufferInfo::builder()
                    .buffer(gpu_config.buffer)
                    .range(gpu_config.allocation.size());

                gpu_config
                    .allocation
                    .mapped_slice_mut()
                    .unwrap()
                    .copy_from_slice(config.as_std140().as_bytes());

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
                    ComputePipeline::new(device, SHADER, &bindings)
                });

                // ----------------------------------------------------------------------------
                // Descriptor Pool
                // ----------------------------------------------------------------------------
                let pool_sizes = [
                    vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(positions.len() as u32)
                        .build(),
                    vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(positions.len() as u32 * 2)
                        .build(),
                ];
                let max_sets = positions.len(); //??
                let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
                    .pool_sizes(&pool_sizes)
                    .max_sets(max_sets as _);

                let descriptor_pool = unsafe {
                    device
                        .device
                        .create_descriptor_pool(&descriptor_pool_info, None)
                }
                .unwrap();

                // ----------------------------------------------------------------------------
                // Descriptor Sets
                // ----------------------------------------------------------------------------
                let layouts = vec![pipeline.descriptor_set_layout; positions.len()];
                let ds_info = vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts);
                let compute_descriptor_sets =
                    unsafe { device.device.allocate_descriptor_sets(&ds_info) }.unwrap();

                let mut bufs = Vec::with_capacity(positions.len());

                let mut brick_stream = ctx.submit_unordered_with_data(
                    positions
                        .iter()
                        .enumerate()
                        .map(|(i, pos)| (input.bricks.request(*pos), (i, *pos))),
                );

                let cmd = device.begin_command_buffer();

                while let Some((brick, (i, pos))) = brick_stream.next().await {
                    let brick_info = m.chunk_info(pos);

                    let brick_layout = Layout::array::<f32>(brick_info.mem_elements()).unwrap();

                    let mut gpu_brick_in = device.allocator().allocate(
                        brick_layout,
                        BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::STORAGE_BUFFER,
                        gpu_allocator::MemoryLocation::CpuToGpu,
                    );

                    let gpu_brick_out = device.allocator().allocate(
                        brick_layout,
                        BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE_BUFFER,
                        gpu_allocator::MemoryLocation::GpuToCpu,
                    );

                    gpu_brick_in
                        .allocation
                        .mapped_slice_mut()
                        .unwrap()
                        .copy_from_slice(bytemuck::cast_slice(&*brick));

                    let db_info_in = vk::DescriptorBufferInfo::builder()
                        .buffer(gpu_brick_in.buffer)
                        .range(gpu_brick_in.allocation.size());

                    let db_info_out = vk::DescriptorBufferInfo::builder()
                        .buffer(gpu_brick_out.buffer)
                        .range(gpu_brick_out.allocation.size());

                    let descriptor_writes = [
                        vk::WriteDescriptorSet::builder()
                            .dst_set(compute_descriptor_sets[i])
                            .dst_binding(0)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&[*config_descriptor_info])
                            .build(),
                        vk::WriteDescriptorSet::builder()
                            .dst_set(compute_descriptor_sets[i])
                            .dst_binding(1)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&[*db_info_in])
                            .build(),
                        vk::WriteDescriptorSet::builder()
                            .dst_set(compute_descriptor_sets[i])
                            .dst_binding(2)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&[*db_info_out])
                            .build(),
                    ];

                    unsafe {
                        device
                            .device
                            .update_descriptor_sets(&descriptor_writes, &[])
                    };

                    let local_size = 256; //TODO: somehow ensure that this is the same as specified
                                          //in shader
                    let global_size = brick_info.mem_elements();
                    assert!(global_size % local_size == 0);
                    let num_wgs = global_size / local_size;

                    unsafe {
                        device.device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline.pipeline,
                        );
                        device.device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline.pipeline_layout,
                            0,
                            &[compute_descriptor_sets[i]],
                            &[],
                        );
                        device
                            .device
                            .cmd_dispatch(cmd, num_wgs.try_into().unwrap(), 1, 1);
                    }

                    //let copy_info = vk::BufferCopy::builder().size(brick_size_mem as _);
                    //unsafe {
                    //    device.device.cmd_copy_buffer(
                    //        cmd,
                    //        gpu_brick_in.buffer,
                    //        gpu_brick_out.buffer,
                    //        &[*copy_info],
                    //    );
                    //}

                    bufs.push((pos, gpu_brick_in, gpu_brick_out));
                }

                // TODO: This currently blocks, which we obviously don't want.
                let _fence = device.submit_command_buffer(cmd);

                for (pos, gpu_brick_in, gpu_brick_out) in bufs.into_iter() {
                    let brick_info = m.chunk_info(pos);
                    let mut output = ctx.alloc_slot(pos, brick_info.mem_elements()).unwrap();
                    //crate::data::init_non_full(&mut output, &brick_info, 0.0);
                    crate::data::write_slice_uninit(
                        &mut *output,
                        bytemuck::cast_slice(gpu_brick_out.allocation.mapped_slice().unwrap()),
                    );

                    unsafe { output.initialized() };

                    device.allocator().deallocate(gpu_brick_in);
                    device.allocator().deallocate(gpu_brick_out);
                }
                device.allocator().deallocate(gpu_config);

                unsafe { device.device.destroy_descriptor_pool(descriptor_pool, None) };

                Ok(())
            }
            .into()
        },
    )
}
