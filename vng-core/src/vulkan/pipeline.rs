use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
};

use ash::vk;
use crevice::std140::{AsStd140, Std140};

use crate::{
    data::Vector,
    storage::gpu::{IndexHandle, ReadHandle, StateCacheHandle, WriteHandle},
    vulkan::shader::Shader,
};

use super::{
    memory::TempBuffer, shader::ShaderSource, state::VulkanState, CmdBufferEpoch, CommandBuffer,
    DeviceContext,
};

pub trait PipelineType {
    const BINDPOINT: vk::PipelineBindPoint;
}

pub struct GraphicsPipelineType;
impl PipelineType for GraphicsPipelineType {
    const BINDPOINT: vk::PipelineBindPoint = vk::PipelineBindPoint::GRAPHICS;
}

pub struct ComputePipelineType {
    local_size: Vector<3, u32>,
}
impl PipelineType for ComputePipelineType {
    const BINDPOINT: vk::PipelineBindPoint = vk::PipelineBindPoint::COMPUTE;
}

pub struct Pipeline<T> {
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    ds_pools: Vec<RefCell<DynamicDescriptorSetPool>>,
    push_constant_size: Option<usize>,
    type_: T,
}

pub type ComputePipeline = Pipeline<ComputePipelineType>;
pub type GraphicsPipeline = Pipeline<GraphicsPipelineType>;

impl ComputePipeline {
    pub fn new(
        device: &DeviceContext,
        shader: impl ShaderSource,
        use_push_descriptor: bool,
    ) -> Self {
        let df = device.functions();
        let mut shader = Shader::from_source(df, shader, spirv_compiler::ShaderKind::Compute);

        let entry_point_name = "main";
        let entry_point_name_c = std::ffi::CString::new(entry_point_name).unwrap();

        let pipeline_info = vk::PipelineShaderStageCreateInfo::builder()
            .module(shader.module)
            .name(&entry_point_name_c)
            .stage(vk::ShaderStageFlags::COMPUTE);

        let info = shader.collect_info(entry_point_name);

        let local_size = info
            .local_size
            .expect("local size should have been specified in shader")
            .try_into()
            .unwrap();

        let (descriptor_set_layouts, ds_pools) = info
            .descriptor_bindings
            .create_descriptor_set_layout(&device.functions, use_push_descriptor);

        let push_constant_size = info.push_const.map(|i| i.size as usize);
        let push_constant_ranges = info
            .push_const
            .as_ref()
            .map(std::slice::from_ref)
            .unwrap_or(&[]);
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        let pipeline_layout = unsafe {
            df.device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .unwrap();

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*pipeline_info)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            df.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[*pipeline_info], None)
        }
        .unwrap();

        // Safety: Pipeline has been created now. Shader module is not referenced anymore.
        unsafe { shader.deinitialize(device) };

        let pipeline = pipelines[0];

        Self {
            pipeline,
            pipeline_layout,
            ds_pools,
            push_constant_size,
            type_: ComputePipelineType { local_size },
        }
    }
}

impl<T: PipelineType> Pipeline<T> {
    pub unsafe fn bind<'a>(&'a self, cmd: &'a mut CommandBuffer) -> BoundPipeline<'a, T> {
        let cmd_raw = cmd.raw();
        cmd.functions()
            .cmd_bind_pipeline(cmd_raw, T::BINDPOINT, self.pipeline);
        BoundPipeline {
            pipeline: self,
            cmd,
        }
    }
}

impl<T: 'static> VulkanState for Pipeline<T> {
    unsafe fn deinitialize(&mut self, context: &crate::vulkan::DeviceContext) {
        let df = context.functions();
        unsafe {
            for pool in &mut self.ds_pools {
                pool.get_mut().deinitialize(context);
            }
            df.device.destroy_pipeline(self.pipeline, None);
            df.device
                .destroy_pipeline_layout(self.pipeline_layout, None)
        };
    }
}

impl GraphicsPipeline {
    pub fn new(
        device: &DeviceContext,
        vertex_shader: impl ShaderSource,
        fragment_shader: impl ShaderSource,
        render_pass: &vk::RenderPass,
        primitive: vk::PrimitiveTopology,
        use_push_descriptor: bool,
    ) -> Self {
        let df = device.functions();
        let mut vertex_shader =
            Shader::from_source(df, vertex_shader, spirv_compiler::ShaderKind::Vertex);
        let mut fragment_shader =
            Shader::from_source(df, fragment_shader, spirv_compiler::ShaderKind::Fragment);

        let entry_point_name = "main";
        let entry_point_name_c = std::ffi::CString::new(entry_point_name).unwrap();

        let vertex_c_info = vk::PipelineShaderStageCreateInfo::builder()
            .module(vertex_shader.module)
            .name(&entry_point_name_c)
            .stage(vk::ShaderStageFlags::VERTEX);

        let fragment_c_info = vk::PipelineShaderStageCreateInfo::builder()
            .module(fragment_shader.module)
            .name(&entry_point_name_c)
            .stage(vk::ShaderStageFlags::FRAGMENT);

        let vertex_info = vertex_shader.collect_info(entry_point_name);
        let fragment_info = fragment_shader.collect_info(entry_point_name);

        let push_const = vertex_info.push_const.or(fragment_info.push_const);
        let push_constant_size = push_const.map(|i| i.size as usize);
        let push_constant_ranges = push_const.as_ref().map(std::slice::from_ref).unwrap_or(&[]);

        let descriptor_bindings = vertex_info
            .descriptor_bindings
            .merge(fragment_info.descriptor_bindings);

        let (descriptor_set_layouts, ds_pools) = descriptor_bindings
            .create_descriptor_set_layout(&device.functions, use_push_descriptor);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        let pipeline_layout = unsafe {
            df.device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .unwrap();

        let shader_stages = [*vertex_c_info, *fragment_c_info];

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_info =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder();

        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .primitive_restart_enable(false)
            .topology(primitive);

        let viewport_state_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multi_sampling_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false);

        let color_blend_attachments = [*color_blend_attachment];
        let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&color_blend_attachments);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multi_sampling_info)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_info)
            .layout(pipeline_layout)
            .render_pass(*render_pass)
            .subpass(0);

        let pipeline_cache = vk::PipelineCache::null();
        let pipelines = unsafe {
            device
                .functions
                .create_graphics_pipelines(pipeline_cache, &[*pipeline_info], None)
        }
        .unwrap();

        let pipeline = pipelines[0];

        // Safety: Pipeline has been created now. Shader module is not referenced anymore.
        unsafe { vertex_shader.deinitialize(device) };
        unsafe { fragment_shader.deinitialize(device) };

        Self {
            pipeline,
            pipeline_layout,
            ds_pools,
            push_constant_size,
            type_: GraphicsPipelineType,
        }
    }
}

pub struct DynamicDescriptorSetPool {
    layout: vk::DescriptorSetLayout,
    type_counts: BTreeMap<vk::DescriptorType, u32>,
    used_pools: Vec<vk::DescriptorPool>,
    available: Vec<vk::DescriptorSet>,
    in_use: VecDeque<(CmdBufferEpoch, Vec<vk::DescriptorSet>)>,
}

impl DynamicDescriptorSetPool {
    pub fn new(
        layout: vk::DescriptorSetLayout,
        type_counts: BTreeMap<vk::DescriptorType, u32>,
    ) -> Self {
        Self {
            layout,
            type_counts,
            used_pools: Vec::new(),
            available: Vec::new(),
            in_use: VecDeque::new(),
        }
    }

    // Safety: Only use this set for the current epoch of the device
    unsafe fn use_set(&mut self, cmd: &CommandBuffer) -> vk::DescriptorSet {
        // Make returned available again if it is safe to do so (their use epoch is finished)
        while self
            .in_use
            .front()
            .map(|v| v.0 <= cmd.oldest_finished_epoch)
            .unwrap_or(false)
        {
            let (_, no_longer_used) = self.in_use.pop_front().unwrap();
            self.available.extend(no_longer_used);
        }

        if self.available.is_empty() {
            let num = 1 << self.used_pools.len();

            let pool_sizes = self
                .type_counts
                .iter()
                .map(|(ty, count)| {
                    vk::DescriptorPoolSize::builder()
                        .ty(*ty)
                        .descriptor_count(count * num)
                        .build()
                })
                .collect::<Vec<_>>();

            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&pool_sizes)
                .max_sets(num as _);

            let descriptor_pool = unsafe {
                cmd.functions
                    .create_descriptor_pool(&descriptor_pool_info, None)
            }
            .unwrap();

            let layouts = vec![self.layout; num as usize];

            let ds_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&layouts);
            let new_ds = unsafe { cmd.functions.allocate_descriptor_sets(&ds_info) }.unwrap();
            self.available.extend(new_ds);

            self.used_pools.push(descriptor_pool);
        }
        let ret = self.available.pop().unwrap();

        let current_epoch = cmd.id.epoch;
        if self.in_use.back().map(|(epoch, _)| *epoch) == Some(current_epoch) {
            self.in_use.back_mut().unwrap().1.push(ret);
        } else {
            self.in_use.push_back((current_epoch, vec![ret]));
        }
        ret
    }

    pub unsafe fn deinitialize(&mut self, context: &crate::vulkan::DeviceContext) {
        let df = context.functions();
        df.device.destroy_descriptor_set_layout(self.layout, None);

        for pool in &self.used_pools {
            df.device.destroy_descriptor_pool(*pool, None);
        }
    }
}

pub struct BoundPipeline<'a, T> {
    pipeline: &'a Pipeline<T>,
    cmd: &'a mut CommandBuffer, //Note: Since we take a mutable reference to the command buffer
                                //here, we ensure that this pipeline is bound for the remainder of
                                //the BoundPipeline's lifetime (barring unsafe operations).
}
impl<'a, T: PipelineType> BoundPipeline<'a, T> {
    pub fn cmd(&mut self) -> &mut CommandBuffer {
        self.cmd
    }

    pub fn push_descriptor_set<const N: usize>(
        &mut self,
        bind_set: u32,
        config: DescriptorConfig<N>,
    ) {
        let writes = config.writes_for_push();
        unsafe {
            let cmd_raw = self.cmd.raw();
            self.cmd
                .functions()
                .push_descriptor_ext
                .cmd_push_descriptor_set(
                    cmd_raw,
                    T::BINDPOINT,
                    self.pipeline.pipeline_layout,
                    bind_set,
                    &writes,
                );
        }
    }

    pub fn write_descriptor_set<const N: usize>(
        &mut self,
        bind_set: u32,
        config: DescriptorConfig<N>,
    ) {
        let ds_pool = self.pipeline.ds_pools.get(bind_set as usize).unwrap();
        let mut ds_pool = ds_pool.borrow_mut();
        // Safety: We only use `ds` with the current cmdbuffer
        let ds = unsafe { ds_pool.use_set(self.cmd) };

        let writes = config.writes_for_set(ds);
        unsafe {
            let cmd_raw = self.cmd.raw();
            self.cmd.functions().update_descriptor_sets(&writes, &[]);

            self.cmd.functions().cmd_bind_descriptor_sets(
                cmd_raw,
                T::BINDPOINT,
                self.pipeline.pipeline_layout,
                bind_set,
                &[ds],
                &[],
            );
        }
    }

    pub fn push_constant_at<V: AsStd140>(&mut self, val: V, stage: vk::ShaderStageFlags) {
        let v = val.as_std140();
        let bytes = v.as_bytes();

        // HACK: Match up sizes:
        // - The reflect library/c compiler appears to be of the opinion that the size of the push
        // constant struct is simply the difference between the begin of the first member and the
        // end of the last, while...
        // - crevice appears to think that the size is rounded up to the alignment of the struct.
        let bytes = &bytes[..self.pipeline.push_constant_size.unwrap()];
        unsafe {
            let cmd_raw = self.cmd.raw();
            self.cmd.functions().cmd_push_constants(
                cmd_raw,
                self.pipeline.pipeline_layout,
                stage,
                0,
                bytes,
            );
        }
    }
}

impl<'a> BoundPipeline<'a, ComputePipelineType> {
    pub unsafe fn dispatch(&mut self, global_size: usize) {
        self.dispatch3d([1u32, 1, global_size.try_into().unwrap()].into())
    }

    pub unsafe fn dispatch3d(&mut self, global_size: Vector<3, u32>) {
        let num_wgs = crate::util::div_round_up(global_size, self.pipeline.type_.local_size);

        let cmd_raw = self.cmd.raw();
        self.cmd
            .functions()
            .cmd_dispatch(cmd_raw, num_wgs.x(), num_wgs.y(), num_wgs.z());
    }
    pub fn push_constant<V: AsStd140>(&mut self, val: V) {
        self.push_constant_at(val, vk::ShaderStageFlags::COMPUTE);
    }
}

trait AsDescriptor {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo;
}

impl AsDescriptor for TempBuffer {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::builder()
            .buffer(self.allocation.buffer)
            .range(self.size)
            .build()
    }
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

impl<'a> AsDescriptor for IndexHandle<'a> {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::builder()
            .buffer(self.buffer)
            .range((self.num_bricks * std::mem::size_of::<vk::DeviceAddress>()) as _)
            .build()
    }
}

impl<'a> AsDescriptor for StateCacheHandle<'a> {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::builder()
            .buffer(self.buffer)
            .range(self.size as _)
            .build()
    }
}

pub trait AsDescriptors {
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

pub struct DescriptorConfig<const N: usize> {
    buffer_infos: [Vec<vk::DescriptorBufferInfo>; N],
}
impl<const N: usize> DescriptorConfig<N> {
    pub fn new(buffers: [&dyn AsDescriptors; N]) -> Self {
        let buffer_infos = std::array::from_fn(|i| buffers[i].gen_buffer_info());
        Self { buffer_infos }
    }
    pub fn writes_for_push(&self) -> [vk::WriteDescriptorSet; N] {
        self.writes_for_set(vk::DescriptorSet::null())
    }

    pub fn writes_for_set(&self, set: vk::DescriptorSet) -> [vk::WriteDescriptorSet; N] {
        std::array::from_fn(|i| {
            vk::WriteDescriptorSet::builder()
                .dst_binding(i as u32)
                .dst_array_element(0)
                .dst_set(set)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&self.buffer_infos[i])
                .build()
        })
    }
}

//fn layout_std140<T: AsStd140>() -> Layout {
//    unsafe { Layout::from_size_align_unchecked(T::std140_size_static(), T::Output::ALIGNMENT) }
//}
