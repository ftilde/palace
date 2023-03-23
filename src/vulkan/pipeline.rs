use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
};

use ash::vk;
use crevice::std140::{AsStd140, Std140};

use crate::{
    data::Vector,
    storage::gpu::{ReadHandle, WriteHandle},
    vulkan::shader::Shader,
};

use super::{
    shader::ShaderSource, state::VulkanState, CmdBufferEpoch, CommandBuffer, DeviceContext,
};

pub struct ComputePipeline {
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    ds_pools: Vec<RefCell<DynamicDescriptorSetPool>>,
    local_size: Vector<3, u32>,
    push_constant_size: Option<usize>,
}

impl ComputePipeline {
    pub fn new(
        device: &DeviceContext,
        shader: impl ShaderSource,
        use_push_descriptor: bool,
    ) -> Self {
        let df = device.functions();
        let mut shader = Shader::from_source(df, shader);

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
        let mut local_size: Option<Vector<3, u32>> = None;
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
                    let x = e.operands[0].value.to_u32();
                    let y = e.operands[1].value.to_u32();
                    let z = e.operands[2].value.to_u32();
                    let prev = local_size.replace([z, y, x].into());
                    assert!(prev.is_none());
                }
                _ => {}
            }
        }

        let local_size = local_size
            .expect("local size should have been specified in shader")
            .try_into()
            .unwrap();

        let mut ds_pools = Vec::new();
        let descriptor_set_layouts = descriptor_bindings
            .into_iter()
            .map(|(_, bindings)| {
                let mut type_counts: BTreeMap<vk::DescriptorType, u32> = BTreeMap::new();
                for b in &bindings {
                    *type_counts.entry(b.descriptor_type).or_default() += b.descriptor_count;
                }
                let mut dsl_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
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

        let push_constant_ranges = push_constant
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
            local_size,
            push_constant_size,
        }
    }

    pub unsafe fn bind<'a>(&'a self, cmd: &'a mut CommandBuffer) -> BoundPipeline<'a> {
        cmd.functions()
            .cmd_bind_pipeline(cmd.raw(), vk::PipelineBindPoint::COMPUTE, self.pipeline);
        BoundPipeline {
            pipeline: self,
            cmd,
        }
    }
}

impl VulkanState for ComputePipeline {
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

struct DynamicDescriptorSetPool {
    layout: vk::DescriptorSetLayout,
    type_counts: BTreeMap<vk::DescriptorType, u32>,
    used_pools: Vec<vk::DescriptorPool>,
    available: Vec<vk::DescriptorSet>,
    in_use: VecDeque<(CmdBufferEpoch, Vec<vk::DescriptorSet>)>,
}

impl DynamicDescriptorSetPool {
    fn new(
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

    unsafe fn deinitialize(&mut self, context: &crate::vulkan::DeviceContext) {
        let df = context.functions();
        df.device.destroy_descriptor_set_layout(self.layout, None);

        for pool in &self.used_pools {
            df.device.destroy_descriptor_pool(*pool, None);
        }
    }
}

pub struct BoundPipeline<'a> {
    pipeline: &'a ComputePipeline,
    cmd: &'a mut CommandBuffer, //Note: Since we take a mutable reference to the command buffer
                                //here, we ensure that this pipeline is bound for the remainder of
                                //the BoundPipeline's lifetime (barring unsafe operations).
}
impl<'a> BoundPipeline<'a> {
    pub fn push_descriptor_set<const N: usize>(&self, bind_set: u32, config: DescriptorConfig<N>) {
        let writes = config.writes_for_push();
        unsafe {
            self.cmd
                .functions()
                .push_descriptor_ext
                .cmd_push_descriptor_set(
                    self.cmd.raw(),
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline.pipeline_layout,
                    bind_set,
                    &writes,
                );
        }
    }

    pub fn write_descriptor_set<const N: usize>(&self, bind_set: u32, config: DescriptorConfig<N>) {
        let ds_pool = self.pipeline.ds_pools.get(bind_set as usize).unwrap();
        let mut ds_pool = ds_pool.borrow_mut();
        // Safety: We only use `ds` with the current cmdbuffer
        let ds = unsafe { ds_pool.use_set(self.cmd) };

        let writes = config.writes_for_set(ds);
        unsafe {
            self.cmd.functions().update_descriptor_sets(&writes, &[]);

            self.cmd.functions().cmd_bind_descriptor_sets(
                self.cmd.raw(),
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.pipeline_layout,
                bind_set,
                &[ds],
                &[],
            );
        }
    }

    pub fn push_constant<T: AsStd140>(&self, val: T) {
        let v = val.as_std140();
        let bytes = v.as_bytes();

        // HACK: Match up sizes:
        // - The reflect library/c compiler appears to be of the opinion that the size of the push
        // constant struct is simply the difference between the begin of the first member and the
        // end of the last, while...
        // - crevice appears to think that the size is rounded up to the alignment of the struct.
        let bytes = &bytes[..self.pipeline.push_constant_size.unwrap()];
        unsafe {
            self.cmd.functions().cmd_push_constants(
                self.cmd.raw(),
                self.pipeline.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
        }
    }

    pub unsafe fn dispatch(&self, global_size: usize) {
        self.dispatch3d([1u32, 1, global_size.try_into().unwrap()].into())
    }

    pub unsafe fn dispatch3d(&self, global_size: Vector<3, u32>) {
        let num_wgs = crate::util::div_round_up(global_size, self.pipeline.local_size);

        self.cmd
            .functions()
            .cmd_dispatch(self.cmd.raw(), num_wgs.x(), num_wgs.y(), num_wgs.z());
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
