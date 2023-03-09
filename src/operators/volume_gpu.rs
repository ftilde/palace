use std::collections::BTreeMap;

use ash::vk;
use crevice::std140::{AsStd140, Std140};
use futures::StreamExt;
use spirq::{EntryPoint, ReflectConfig};

use crate::{
    data::{BrickPosition, LocalVoxelCoordinate, LocalVoxelPosition},
    id::Id,
    operator::OperatorId,
    storage::gpu::{VRamReadHandle, VRamWriteHandle},
    vulkan::{DeviceContext, VulkanState},
};

use super::{scalar::ScalarOperator, volume::VolumeOperator};

//fn layout_std140<T: AsStd140>() -> Layout {
//    unsafe { Layout::from_size_align_unchecked(T::std140_size_static(), T::Output::ALIGNMENT) }
//}

struct Shader {
    module: vk::ShaderModule,
    entry_points: Vec<EntryPoint>,
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
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    local_size: usize,
}

impl ComputePipeline {
    fn new(device: &DeviceContext, shader: &str) -> Self {
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
                    nbind: _,
                } => {
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
                            vk::DescriptorType::STORAGE_BUFFER
                        }
                        spirq::DescriptorType::InputAttachment(_) => todo!(),
                        spirq::DescriptorType::AccelStruct() => todo!(),
                    };
                    let binding = vk::DescriptorSetLayoutBinding::builder()
                        .binding(desc_bind.bind())
                        .descriptor_type(d_type)
                        .descriptor_count(1)
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

impl<'a> AsDescriptor for VRamReadHandle<'a> {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::builder()
            .buffer(self.buffer)
            .range(self.layout.size() as _)
            .build()
    }
}

impl<'a> AsDescriptor for VRamWriteHandle<'a> {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::builder()
            .buffer(self.buffer)
            .range(self.size as _)
            .build()
    }
}

struct DescriptorConfig<const N: usize> {
    buffer_infos: [vk::DescriptorBufferInfo; N],
}
impl<const N: usize> DescriptorConfig<N> {
    fn new(buffers: [&dyn AsDescriptor; N]) -> Self {
        let buffer_infos = std::array::from_fn(|i| buffers[i].gen_buffer_info());
        Self { buffer_infos }
    }
    fn writes(&self) -> [vk::WriteDescriptorSet; N] {
        std::array::from_fn(|i| {
            vk::WriteDescriptorSet::builder()
                .dst_binding(i as u32)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&self.buffer_infos[i]))
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
                let device = ctx.vulkan_device();

                let (scale_gpu, offset_gpu, m) = futures::join! {
                    ctx.submit(scale.request_gpu(device.id, ())),
                    ctx.submit(offset.request_gpu(device.id, ())),
                    ctx.submit(input.metadata.request_scalar()),
                };

                let pipeline = device.request_state("linear_rescale_gpu_pipeline", || {
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
                                    num_chunk_elems: m.num_elements().try_into().unwrap(),
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
    VolumeOperator::with_state(
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

                let pipeline = device.request_state("volume_rechunk_gpu_pipeline", || {
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
                            let overlap_size = (overlap_end - overlap_begin)
                                .map(LocalVoxelCoordinate::interpret_as);

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

#[cfg(test)]
mod test {
    use super::*;
    use crate::{data::VoxelPosition, operators::volume::VolumeOperatorState, test_util::*};

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
}
