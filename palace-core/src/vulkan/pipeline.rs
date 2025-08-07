use std::alloc::Layout;
use std::fmt::Write;
use std::{cell::RefCell, collections::VecDeque};

use ash::vk;
use id::Identify;

use crate::dtypes::{AsDynType, DType, ElementType};
use crate::mat::Matrix;
use crate::{
    data::Vector,
    dim::*,
    storage::gpu::{Allocation, ReadHandle, StateCacheHandle, WriteHandle},
    util::Map,
    vulkan::shader::ShaderModule,
};

use super::shader::Shader;
use super::{state::VulkanState, CmdBufferEpoch, CommandBuffer, DeviceContext};

pub trait PipelineType {
    const BINDPOINT: vk::PipelineBindPoint;
}

pub struct GraphicsPipelineType;
impl PipelineType for GraphicsPipelineType {
    const BINDPOINT: vk::PipelineBindPoint = vk::PipelineBindPoint::GRAPHICS;
}

pub struct ComputePipelineType {
    local_size: Vector<D3, u32>,
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

impl ComputePipeline {
    pub fn local_size(&self) -> &Vector<D3, u32> {
        &self.type_.local_size
    }
}

const DEFAULT_LOCAL_SIZE: u32 = 256;
const DEFAULT_LOCAL_SIZE_2D: u32 = 16;

pub enum LocalSizeConfig {
    Small,
    Large,
    Auto,
    Auto2D,
    Specific(Vector<D3, u32>),
}

pub type ComputePipeline = Pipeline<ComputePipelineType>;
pub type GraphicsPipeline = Pipeline<GraphicsPipelineType>;

pub struct ComputePipelineBuilder<'a> {
    shader: Shader<'a>,
    use_push_descriptor: bool,
    local_size: LocalSizeConfig,
}

impl<'a> ComputePipelineBuilder<'a> {
    pub fn new(shader: Shader<'a>) -> Self {
        Self {
            shader,
            use_push_descriptor: false,
            local_size: LocalSizeConfig::Auto,
        }
    }

    pub fn use_push_descriptor(mut self, use_them: bool) -> Self {
        self.use_push_descriptor = use_them;
        self
    }

    pub fn local_size(mut self, local_size: LocalSizeConfig) -> Self {
        self.local_size = local_size;
        self
    }

    pub fn build(self, device: &DeviceContext) -> Result<ComputePipeline, crate::Error> {
        let df = device.functions();

        let min_size = device.physical_device_properties_13().max_subgroup_size;
        let max_size = device
            .physical_device_properties()
            .limits
            .max_compute_work_group_size[0];

        let local_size = match self.local_size {
            LocalSizeConfig::Small => Vector::from([1, 1, min_size]),
            LocalSizeConfig::Large => Vector::from([1, 1, max_size]),
            LocalSizeConfig::Auto => {
                Vector::from([1, 1, DEFAULT_LOCAL_SIZE.clamp(min_size, max_size)])
            }
            LocalSizeConfig::Auto2D => {
                Vector::from([1, DEFAULT_LOCAL_SIZE_2D, DEFAULT_LOCAL_SIZE_2D])
            }
            LocalSizeConfig::Specific(s) => s,
        };

        let flags = match self.local_size {
            LocalSizeConfig::Auto2D => vk::PipelineShaderStageCreateFlags::default(),
            _ => vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS,
        };

        let local_size_str = format!(
            "layout (local_size_x = {}, local_size_y = {}, local_size_z = {}) in;\n",
            local_size.x(),
            local_size.y(),
            local_size.z()
        );
        let mut shader = self.shader;
        shader.program_parts.insert(0, local_size_str.into());

        let mut shader = ShaderModule::from_source(df, shader, shaderc::ShaderKind::Compute)?;

        let entry_point_name = "main";
        let entry_point_name_c = std::ffi::CString::new(entry_point_name).unwrap();

        let info = shader.collect_info(entry_point_name);

        let pipeline_info = vk::PipelineShaderStageCreateInfo::default()
            .module(shader.module)
            .name(&entry_point_name_c)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .flags(flags);

        let (descriptor_set_layouts, ds_pools) = info
            .descriptor_bindings
            .create_descriptor_set_layout(&device.functions, self.use_push_descriptor);

        let push_constant_size = info.push_const.map(|i| i.size as usize);
        let push_constant_ranges = info
            .push_const
            .as_ref()
            .map(std::slice::from_ref)
            .unwrap_or(&[]);
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        let pipeline_layout = unsafe {
            df.device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .unwrap();

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(pipeline_info)
            .layout(pipeline_layout);

        let pipelines = unsafe {
            df.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .unwrap();

        // Safety: Pipeline has been created now. Shader module is not referenced anymore.
        unsafe { shader.deinitialize(device) };

        let pipeline = pipelines[0];

        Ok(ComputePipeline {
            pipeline,
            pipeline_layout,
            ds_pools,
            push_constant_size,
            type_: ComputePipelineType { local_size },
        })
    }
}

impl<T: PipelineType> Pipeline<T> {
    pub fn has_push_constants(&self) -> bool {
        self.push_constant_size.is_some()
    }
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

pub struct GraphicsPipelineBuilder<'a> {
    vertex_shader: Shader<'a>,
    fragment_shader: Shader<'a>,
    use_push_descriptor: bool,
}
impl<'a> GraphicsPipelineBuilder<'a> {
    pub fn new(vertex_shader: Shader<'a>, fragment_shader: Shader<'a>) -> Self {
        Self {
            vertex_shader,
            fragment_shader,
            use_push_descriptor: false,
        }
    }

    pub fn use_push_descriptor(mut self, use_them: bool) -> Self {
        self.use_push_descriptor = use_them;
        self
    }

    pub fn build<
        'b,
        B: FnOnce(
            &[vk::PipelineShaderStageCreateInfo],
            vk::PipelineLayout,
            Box<dyn FnOnce(&vk::GraphicsPipelineCreateInfo) -> vk::Pipeline + 'b>,
        ) -> vk::Pipeline,
    >(
        self,
        device: &'b DeviceContext,
        build_info: B,
    ) -> Result<GraphicsPipeline, crate::Error> {
        let df = device.functions();
        let mut vertex_shader =
            ShaderModule::from_source(df, self.vertex_shader, shaderc::ShaderKind::Vertex)?;
        let mut fragment_shader =
            ShaderModule::from_source(df, self.fragment_shader, shaderc::ShaderKind::Fragment)?;

        let entry_point_name = "main";
        let entry_point_name_c = std::ffi::CString::new(entry_point_name).unwrap();

        let vertex_c_info = vk::PipelineShaderStageCreateInfo::default()
            .module(vertex_shader.module)
            .name(&entry_point_name_c)
            .stage(vk::ShaderStageFlags::VERTEX);

        let fragment_c_info = vk::PipelineShaderStageCreateInfo::default()
            .module(fragment_shader.module)
            .name(&entry_point_name_c)
            .stage(vk::ShaderStageFlags::FRAGMENT);

        let vertex_info = vertex_shader.collect_info(entry_point_name);
        let fragment_info = fragment_shader.collect_info(entry_point_name);

        let push_consts = vertex_info
            .push_const
            .into_iter()
            .chain(fragment_info.push_const.into_iter())
            .collect::<Vec<_>>();
        let push_constant_size = push_consts.first().map(|i| i.size as usize);
        if let Some(push_constant_size) = push_constant_size {
            for v in &push_consts {
                assert_eq!(v.size, push_constant_size as u32);
            }
        }
        let push_constant_ranges = &push_consts;

        let descriptor_bindings = vertex_info
            .descriptor_bindings
            .merge(fragment_info.descriptor_bindings);

        let (descriptor_set_layouts, ds_pools) = descriptor_bindings
            .create_descriptor_set_layout(&device.functions, self.use_push_descriptor);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        let pipeline_layout = unsafe {
            df.device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .unwrap();

        let shader_stages = [vertex_c_info, fragment_c_info];

        let pipeline_cache = vk::PipelineCache::null();

        let pipeline = build_info(
            &shader_stages[..],
            pipeline_layout,
            Box::new(move |info| {
                let pipelines = unsafe {
                    device
                        .functions
                        .create_graphics_pipelines(pipeline_cache, &[*info], None)
                }
                .unwrap();

                pipelines[0]
            }),
        );

        // Safety: Pipeline has been created now. Shader module is not referenced anymore.
        unsafe { vertex_shader.deinitialize(device) };
        unsafe { fragment_shader.deinitialize(device) };

        Ok(GraphicsPipeline {
            pipeline,
            pipeline_layout,
            ds_pools,
            push_constant_size,
            type_: GraphicsPipelineType,
        })
    }
}

pub struct DynamicDescriptorSetPool {
    layout: vk::DescriptorSetLayout,
    type_counts: Map<vk::DescriptorType, u32>,
    used_pools: Vec<vk::DescriptorPool>,
    available: Vec<vk::DescriptorSet>,
    in_use: VecDeque<(CmdBufferEpoch, Vec<vk::DescriptorSet>)>,
}

impl DynamicDescriptorSetPool {
    pub fn new(layout: vk::DescriptorSetLayout, type_counts: Map<vk::DescriptorType, u32>) -> Self {
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
                    vk::DescriptorPoolSize::default()
                        .ty(*ty)
                        .descriptor_count(count * num)
                })
                .collect::<Vec<_>>();

            let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(num as _);

            let descriptor_pool = unsafe {
                cmd.functions
                    .create_descriptor_pool(&descriptor_pool_info, None)
            }
            .unwrap();

            let layouts = vec![self.layout; num as usize];

            let ds_info = vk::DescriptorSetAllocateInfo::default()
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

    //Note: Since we take a mutable reference to the command buffer
    //here, we ensure that this pipeline is bound for the remainder of
    //the BoundPipeline's lifetime (barring unsafe operations).
    cmd: &'a mut CommandBuffer,
}
impl<'a, T: PipelineType> BoundPipeline<'a, T> {
    pub fn cmd(&mut self) -> &mut CommandBuffer {
        self.cmd
    }

    pub fn push_descriptor_set(&mut self, bind_set: u32, config: DescriptorConfig) {
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

    pub fn write_descriptor_set(&mut self, bind_set: u32, config: DescriptorConfig) {
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
    pub fn push_constant_bytes(&mut self, bytes: &[u8], stage: vk::ShaderStageFlags) {
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

    pub fn push_constant_at<V: bytemuck::Pod>(&mut self, val: V, stage: vk::ShaderStageFlags) {
        let bytes = bytemuck::bytes_of(&val);
        self.push_constant_bytes(bytes, stage);
    }
}

impl<'a> BoundPipeline<'a, ComputePipelineType> {
    pub unsafe fn dispatch(&mut self, ctx: &DeviceContext, global_size: usize) {
        let linear_local = self.pipeline.type_.local_size.hmul();

        let required_workgroups = global_size.div_ceil(linear_local);

        let max_workgroups = Vector::<D3, usize>::from_fn(|i| {
            ctx.physical_device_properties()
                .limits
                .max_compute_work_group_count[2 - i] as usize
        });
        let num_wgs = Vector::<_, u32>::from([
            required_workgroups.div_ceil(max_workgroups.x() * max_workgroups.y()) as u32,
            required_workgroups
                .div_ceil(max_workgroups.x())
                .min(max_workgroups.y()) as u32,
            required_workgroups.min(max_workgroups.x()) as u32,
        ]);

        if num_wgs.hmul() == 0 {
            panic!("Zero dispatch");
        }

        let cmd_raw = self.cmd.raw();
        self.cmd
            .functions()
            .cmd_dispatch(cmd_raw, num_wgs.x(), num_wgs.y(), num_wgs.z());
    }

    pub unsafe fn dispatch3d(&mut self, global_size: Vector<D3, u32>) {
        let num_wgs = crate::util::div_round_up(global_size, self.pipeline.type_.local_size);

        if num_wgs.hmul() == 0 {
            panic!("Zero dispatch");
        }

        let cmd_raw = self.cmd.raw();
        self.cmd
            .functions()
            .cmd_dispatch(cmd_raw, num_wgs.x(), num_wgs.y(), num_wgs.z());
    }

    pub unsafe fn dispatch_dyn<D: DynDimension>(
        &mut self,
        ctx: &DeviceContext,
        global_size: Vector<D, u32>,
    ) {
        let linear_size = global_size.hmul();
        self.dispatch(ctx, linear_size);
    }

    pub fn push_constant<V: bytemuck::Pod>(&mut self, val: V) {
        self.push_constant_at(val, vk::ShaderStageFlags::COMPUTE);
    }

    pub fn push_constant_dyn<F: FnOnce(&mut DynPushConstantsWriter) -> Result<(), crate::Error>>(
        &mut self,
        p: &DynPushConstants,
        write: F,
    ) {
        let mut writer = p.writer();
        write(&mut writer).unwrap();
        let bytes = writer.finish().unwrap();
        self.push_constant_bytes(&bytes, vk::ShaderStageFlags::COMPUTE);
    }
}

pub enum DescriptorInfos {
    Buffer(Vec<vk::DescriptorBufferInfo>),
    CombinedImageSampler(Vec<vk::DescriptorImageInfo>),
}

pub trait AsBufferDescriptor {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo;
}

impl AsBufferDescriptor for Allocation {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer)
            .range(self.size)
    }
}

pub struct NullBuf;

impl AsBufferDescriptor for NullBuf {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(vk::Buffer::default())
            .range(0)
    }
}

impl<'a> AsBufferDescriptor for ReadHandle<'a> {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer)
            .range(self.layout.size() as _)
    }
}

impl<'a> AsBufferDescriptor for WriteHandle<'a> {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer)
            .range(self.size as _)
    }
}

impl<'a> AsBufferDescriptor for StateCacheHandle<'a> {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer)
            .range(self.size as _)
    }
}

pub struct FixedBufferDescriptor {
    inner: vk::DescriptorBufferInfo,
}

impl AsBufferDescriptor for FixedBufferDescriptor {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo {
        self.inner
    }
}

pub trait AsBufferDescriptorExt {
    fn with_offset_and_size(&self, offset: u64, size: u64) -> FixedBufferDescriptor;
}

impl<T: AsBufferDescriptor> AsBufferDescriptorExt for T {
    fn with_offset_and_size(&self, offset: u64, size: u64) -> FixedBufferDescriptor {
        let inner = self.gen_buffer_info();
        let new_offset = inner.offset + offset;
        // Only allow reducing the range
        assert!(new_offset + size <= inner.offset + inner.range);
        FixedBufferDescriptor {
            inner: inner.offset(new_offset).range(size),
        }
    }
}

pub trait AsDescriptors {
    fn gen_buffer_info(&self) -> DescriptorInfos;
}

fn descriptor_from_iter<'a, T: 'a + AsBufferDescriptor>(
    iter: impl Iterator<Item = &'a T> + 'a,
) -> DescriptorInfos {
    DescriptorInfos::Buffer(
        iter.map(|i| i.gen_buffer_info())
            .filter(|v| v.buffer != vk::Buffer::default())
            .collect(),
    )
}

impl<T: AsBufferDescriptor> AsDescriptors for T {
    fn gen_buffer_info(&self) -> DescriptorInfos {
        descriptor_from_iter(std::iter::once(self))
    }
}

impl AsDescriptors for (vk::ImageView, vk::ImageLayout, vk::Sampler) {
    fn gen_buffer_info(&self) -> DescriptorInfos {
        DescriptorInfos::CombinedImageSampler(vec![vk::DescriptorImageInfo::default()
            .image_view(self.0)
            .image_layout(self.1)
            .sampler(self.2)])
    }
}

impl<const N: usize, T: AsBufferDescriptor> AsDescriptors for [&T; N] {
    fn gen_buffer_info(&self) -> DescriptorInfos {
        descriptor_from_iter(self.iter().cloned())
    }
}

impl<T: AsBufferDescriptor> AsDescriptors for &[&T] {
    fn gen_buffer_info(&self) -> DescriptorInfos {
        descriptor_from_iter(self.iter().cloned())
    }
}

pub struct DescriptorConfig {
    buffer_infos: Vec<DescriptorInfos>,
}
impl DescriptorConfig {
    pub fn new<const N: usize>(buffers: [&dyn AsDescriptors; N]) -> Self {
        let buffer_infos = buffers.into_iter().map(|b| b.gen_buffer_info()).collect();
        Self { buffer_infos }
    }
    pub fn from_vec(buffers: Vec<&dyn AsDescriptors>) -> Self {
        let buffer_infos = buffers.into_iter().map(|b| b.gen_buffer_info()).collect();
        Self { buffer_infos }
    }
    pub fn writes_for_push(&self) -> Vec<vk::WriteDescriptorSet> {
        self.writes_for_set(vk::DescriptorSet::null())
    }

    pub fn writes_for_set(&self, set: vk::DescriptorSet) -> Vec<vk::WriteDescriptorSet> {
        self.buffer_infos
            .iter()
            .enumerate()
            .map(|(i, v)| match v {
                DescriptorInfos::Buffer(b) => vk::WriteDescriptorSet::default()
                    .dst_binding(i as u32)
                    .dst_array_element(0)
                    .dst_set(set)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&b),
                DescriptorInfos::CombinedImageSampler(b) => vk::WriteDescriptorSet::default()
                    .dst_binding(i as u32)
                    .dst_array_element(0)
                    .dst_set(set)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&b),
            })
            .filter(|v| v.descriptor_count > 0) //May happen due to filtering above
            .collect()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Identify)]
enum MemberSize {
    Vec(usize),
    Matrix(usize),
    Scalar,
}

impl std::fmt::Display for MemberSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemberSize::Vec(n) => write!(f, "a vec of size {}", n),
            MemberSize::Matrix(n) => write!(f, "a matrix of size {n}x{n}"),
            MemberSize::Scalar => write!(f, "a scalar"),
        }
    }
}

#[derive(Identify)]
struct Member {
    size: MemberSize,
    type_: DType,
    name: &'static str,
}
impl Member {
    fn write(&self, o: &mut dyn Write) -> std::fmt::Result {
        write!(o, "{}", self.type_.glsl_type())?;
        match self.size {
            MemberSize::Scalar => {}
            MemberSize::Vec(n) => write!(o, "[{}]", n)?,
            MemberSize::Matrix(n) => write!(o, "[{n}][{n}]")?,
        }
        write!(o, " {};", self.name)
    }
}

#[derive(Identify)]
pub struct DynPushConstants {
    members: Vec<Member>,
}

impl DynPushConstants {
    pub fn new() -> Self {
        DynPushConstants {
            members: Vec::new(),
        }
    }
    pub fn scalar<T: AsDynType>(mut self, name: &'static str) -> Self {
        self.members.push(Member {
            size: MemberSize::Scalar,
            type_: T::D_TYPE,
            name,
        });
        self
    }

    pub fn vec<T: AsDynType>(mut self, size: usize, name: &'static str) -> Self {
        self.members.push(Member {
            size: MemberSize::Vec(size),
            type_: T::D_TYPE,
            name,
        });
        self
    }

    pub fn mat<T: AsDynType>(mut self, size: usize, name: &'static str) -> Self {
        self.members.push(Member {
            size: MemberSize::Matrix(size),
            type_: T::D_TYPE,
            name,
        });
        self
    }

    pub fn glsl_definition(&self) -> String {
        let mut out = String::new();
        write!(
            &mut out,
            "layout(scalar, push_constant) uniform PushConsts {{"
        )
        .unwrap();

        for member in &self.members {
            member.write(&mut out).unwrap();
        }
        write!(&mut out, "}} __name").unwrap();
        out
    }

    pub fn writer(&self) -> DynPushConstantsWriter {
        DynPushConstantsWriter {
            consts: self,
            pos: 0,
            buf: Vec::new(),
        }
    }
}

pub struct DynPushConstantsWriter<'a> {
    consts: &'a DynPushConstants,
    pos: usize,
    buf: Vec<u8>,
}

impl<'a> DynPushConstantsWriter<'a> {
    pub fn finish(self) -> Result<Vec<u8>, crate::Error> {
        if self.pos < self.consts.members.len() {
            Err(crate::Error::from(format!(
                "Only {} of {} members written",
                self.pos,
                self.consts.members.len()
            )))
        } else {
            assert_eq!(self.pos, self.consts.members.len());
            Ok(self.buf)
        }
    }

    fn checked_layout_for_value(
        &self,
        dtype: DType,
        size: MemberSize,
    ) -> Result<Layout, crate::Error> {
        let member = self.consts.members.get(self.pos).ok_or_else(|| {
            format!(
                "Tried to write member {}, but definition has only {} members",
                self.pos,
                self.consts.members.len()
            )
        })?;

        if member.size != size {
            return Err(format!("Expected {}, but got {}", member.size, size).into());
        }

        if member.type_ != dtype {
            return Err(format!("Expected type {:?}, but got {:?}", member.type_, dtype,).into());
        }

        let num_elements = match member.size {
            MemberSize::Vec(n) => n,
            MemberSize::Matrix(n) => n * n,
            MemberSize::Scalar => 1,
        };
        let layout = member.type_.array_layout(num_elements);

        Ok(layout)
    }

    pub fn scalar<V: bytemuck::Pod + AsDynType>(&mut self, v: V) -> Result<(), crate::Error> {
        let layout = self.checked_layout_for_value(V::D_TYPE, MemberSize::Scalar)?;

        while (self.buf.len() % layout.align()) != 0 {
            self.buf.push(0);
        }
        self.buf.extend_from_slice(bytemuck::bytes_of(&v));

        self.pos += 1;
        Ok(())
    }

    pub fn vec<D: DynDimension, V: bytemuck::Pod + AsDynType>(
        &mut self,
        v: &Vector<D, V>,
    ) -> Result<(), crate::Error> {
        let layout = self.checked_layout_for_value(V::D_TYPE, MemberSize::Vec(v.len()))?;

        for v in v.iter() {
            while (self.buf.len() % layout.align()) != 0 {
                self.buf.push(0);
            }
            self.buf.extend_from_slice(bytemuck::bytes_of(v));
        }

        self.pos += 1;
        Ok(())
    }

    pub fn mat<D: DynDimension, V: bytemuck::Pod + AsDynType>(
        &mut self,
        v: &Matrix<D, V>,
    ) -> Result<(), crate::Error> {
        let dim = v.dim();
        let layout = self.checked_layout_for_value(V::D_TYPE, MemberSize::Matrix(dim.n()))?;

        for c in 0..dim.n() {
            let col = v.col(c);
            for v in col.iter() {
                while (self.buf.len() % layout.align()) != 0 {
                    self.buf.push(0);
                }
                self.buf.extend_from_slice(bytemuck::bytes_of(v));
            }
        }

        self.pos += 1;
        Ok(())
    }
}

//fn layout_std140<T: AsStd140>() -> Layout {
//    unsafe { Layout::from_size_align_unchecked(T::std140_size_static(), T::Output::ALIGNMENT) }
//}
