use ash::vk;
use crevice::glsl::GlslStruct;
use crevice::std140::AsStd140;

use super::pipeline::{DescriptorConfig, GraphicsPipeline};
use super::shader::ShaderDefines;
use super::state::VulkanState;
use super::{
    CmdBufferEpoch, DeviceContext, DeviceId, DstBarrierInfo, SrcBarrierInfo, VulkanContext,
};
use crate::array::ChunkIndex;
use crate::data::{GlobalCoordinate, Vector};
use crate::dim::*;
use crate::operators::tensor::FrameOperator;
use crate::storage::DataVersionType;
use crate::task::OpaqueTaskContext;

type WindowSize = winit::dpi::PhysicalSize<u32>;

struct SwapChainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapChainSupportDetails {
    fn query(
        ctx: &super::VulkanContext,
        device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Self {
        let capabilities = unsafe {
            ctx.functions
                .surface_ext
                .get_physical_device_surface_capabilities(device, surface)
        }
        .unwrap();

        let formats = unsafe {
            ctx.functions
                .surface_ext
                .get_physical_device_surface_formats(device, surface)
        }
        .unwrap();

        let present_modes = unsafe {
            ctx.functions
                .surface_ext
                .get_physical_device_surface_present_modes(device, surface)
        }
        .unwrap();

        SwapChainSupportDetails {
            capabilities,
            formats,
            present_modes,
        }
    }

    fn default_config(&self) -> SwapChainConfiguration {
        let format = select_swap_surface_format(self.formats.as_slice());
        let present_mode = select_swap_present_mode(self.present_modes.as_slice());
        SwapChainConfiguration {
            format,
            present_mode,
        }
    }
}

struct SwapChainConfiguration {
    format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
}

fn select_swap_surface_format(available: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    available
        .iter()
        .find(|f| {
            f.format == vk::Format::R8G8B8A8_UNORM
                && f.color_space == vk::ColorSpaceKHR::PASS_THROUGH_EXT
        })
        .cloned()
        .unwrap_or_else(|| available[0])
}

fn select_swap_present_mode(available: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    // MAILBOX (kind of like triple buffering) causes a lot of superfluous network evaluations.
    // Instead we always use FIFO (comparable to VSYNCed updates) for now. We may want to make this
    // configurable in the future, though!
    //if available.contains(&vk::PresentModeKHR::MAILBOX) {
    //    return vk::PresentModeKHR::MAILBOX;
    //}

    assert!(available.contains(&vk::PresentModeKHR::FIFO));
    vk::PresentModeKHR::FIFO
}

fn select_swap_extent(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    window_size: WindowSize,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::max_value() {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: window_size.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: window_size.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }
}

fn create_swap_chain(
    device: &DeviceContext,
    support: &SwapChainSupportDetails,
    config: &SwapChainConfiguration,
    surface: vk::SurfaceKHR,
    extent: vk::Extent2D,
) -> vk::SwapchainKHR {
    let image_count = support.capabilities.min_image_count + 1;
    let image_count = if support.capabilities.max_image_count > 0 {
        image_count.min(support.capabilities.max_image_count)
    } else {
        image_count
    };

    let sc_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(config.format.format)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .pre_transform(support.capabilities.current_transform)
        .present_mode(config.present_mode)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .clipped(true);

    //TODO: If we want to support cards that do not have one queue for present/graphics, we
    //need CONCURRENT sharing mode here and also supply the involved queue family indices.
    let sc_info = sc_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE);

    let swap_chain = unsafe {
        device
            .functions
            .swap_chain_ext
            .create_swapchain(&sc_info, None)
    }
    .unwrap();
    swap_chain
}

fn create_render_pass(
    device: &DeviceContext,
    swap_chain_config: &SwapChainConfiguration,
) -> vk::RenderPass {
    let color_attachment = vk::AttachmentDescription::default()
        .format(swap_chain_config.format.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_attachment_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_attachment_refs = &[color_attachment_ref];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachment_refs);

    let dependency_info = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

    let attachments = &[color_attachment];
    let subpasses = &[subpass];
    let dependency_infos = &[dependency_info];
    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependency_infos);

    unsafe { device.functions.create_render_pass(&render_pass_info, None) }.unwrap()
}

fn create_framebuffers(
    device: &DeviceContext,
    image_views: &Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
) -> Vec<vk::Framebuffer> {
    image_views
        .iter()
        .map(|iv| {
            let attachments = [*iv];
            let framebuffer_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);

            unsafe { device.functions.create_framebuffer(&framebuffer_info, None) }.unwrap()
        })
        .collect::<Vec<_>>()
}

struct SwapChainData {
    inner: vk::SwapchainKHR,
    image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    extent: vk::Extent2D,
}
impl SwapChainData {
    fn new(
        device: &DeviceContext,
        support: &SwapChainSupportDetails,
        config: &SwapChainConfiguration,
        render_pass: vk::RenderPass,
        surface: vk::SurfaceKHR,
        window_size: WindowSize,
    ) -> Self {
        let extent = select_swap_extent(&support.capabilities, window_size);
        let swap_chain = create_swap_chain(device, support, config, surface, extent);
        let images = unsafe {
            device
                .functions
                .swap_chain_ext
                .get_swapchain_images(swap_chain)
        }
        .unwrap();

        let image_views = images
            .iter()
            .map(|img| {
                let info = vk::ImageViewCreateInfo::default()
                    .image(*img)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(config.format.format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );
                unsafe { device.functions.create_image_view(&info, None) }.unwrap()
            })
            .collect::<Vec<_>>();

        let framebuffers = create_framebuffers(&device, &image_views, render_pass, extent);

        Self {
            inner: swap_chain,
            image_views,
            framebuffers,
            extent,
        }
    }

    unsafe fn deinitialize(&mut self, device: &DeviceContext) {
        for fb in self.framebuffers.drain(..) {
            device.functions.destroy_framebuffer(fb, None);
        }
        for view in self.image_views.drain(..) {
            device.functions.destroy_image_view(view, None);
        }
        device
            .functions
            .swap_chain_ext
            .destroy_swapchain(self.inner, None);
    }
}

fn create_sync_objects(device: &DeviceContext) -> SyncObjects {
    let semaphore_info = vk::SemaphoreCreateInfo::default();

    SyncObjects {
        image_available_semaphore: unsafe {
            device.functions().create_semaphore(&semaphore_info, None)
        }
        .unwrap(),
        render_finished_semaphore: unsafe {
            device.functions().create_semaphore(&semaphore_info, None)
        }
        .unwrap(),
        last_use_epoch: CmdBufferEpoch::ancient(),
    }
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct SyncObjects {
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    last_use_epoch: CmdBufferEpoch,
}

impl SyncObjects {
    unsafe fn deinitialize(&mut self, device: &DeviceContext) {
        device
            .functions
            .destroy_semaphore(self.render_finished_semaphore, None);
        device
            .functions
            .destroy_semaphore(self.image_available_semaphore, None);
    }
}

pub struct Window {
    surface: vk::SurfaceKHR,
    device_id: DeviceId,
    pipeline: GraphicsPipeline,
    swap_chain: SwapChainData,
    render_pass: vk::RenderPass,
    sync_objects: [SyncObjects; MAX_FRAMES_IN_FLIGHT],
    current_frame: usize,
}

fn swap_chain_support(
    ctx: &VulkanContext,
    device: DeviceId,
    surface: vk::SurfaceKHR,
) -> Option<SwapChainSupportDetails> {
    let device = &ctx.device_contexts[device];

    let support = SwapChainSupportDetails::query(ctx, device.physical_device, surface);
    let surface_support = unsafe {
        ctx.functions
            .surface_ext
            .get_physical_device_surface_support(
                device.physical_device,
                device.queue_family_index,
                surface,
            )
            .unwrap()
    };
    if !support.formats.is_empty() && !support.present_modes.is_empty() && surface_support {
        Some(support)
    } else {
        None
    }
}

impl Window {
    // TODO: see if we can just impl drop when taking a device reference
    pub unsafe fn deinitialize(&mut self, context: &crate::vulkan::VulkanContext) {
        let device = &context.device_contexts[self.device_id];

        // Device must be idle before sync objects can be destroyed to make sure they are not in
        // use anymore
        let _ = device.functions.device_wait_idle();

        self.swap_chain.deinitialize(device);
        for s in &mut self.sync_objects {
            s.deinitialize(device);
        }
        self.pipeline.deinitialize(device);
        device.functions.destroy_render_pass(self.render_pass, None);
        context
            .functions
            .surface_ext
            .destroy_surface(self.surface, None);
    }

    pub fn new(
        ctx: &VulkanContext,
        surface: vk::SurfaceKHR,
        initial_size: WindowSize,
    ) -> Result<Self, crate::Error> {
        let (device, swap_chain_support) = ctx
            .device_contexts
            .iter()
            .enumerate()
            .find_map(|(i, device)| {
                swap_chain_support(ctx, i, surface).map(|support| (device, support))
            })
            .ok_or_else(|| "Could not find any device with present capabilities")?;

        let swap_chain_config = swap_chain_support.default_config();

        let render_pass = create_render_pass(&device, &swap_chain_config);

        let swap_chain = SwapChainData::new(
            device,
            &swap_chain_support,
            &swap_chain_config,
            render_pass,
            surface,
            initial_size,
        );

        let pipeline = GraphicsPipeline::new(
            device,
            VERTEX_SHADER,
            (
                FRAG_SHADER,
                ShaderDefines::new().push_const_block::<PushConstants>(),
            ),
            |shader_stages, pipeline_layout, build_pipeline| {
                let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
                let dynamic_info =
                    vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

                let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

                let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::default()
                    .primitive_restart_enable(false)
                    .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

                let viewport_state_info = vk::PipelineViewportStateCreateInfo::default()
                    .viewport_count(1)
                    .scissor_count(1);

                let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::default()
                    .depth_clamp_enable(false)
                    .rasterizer_discard_enable(false)
                    .polygon_mode(vk::PolygonMode::FILL)
                    .line_width(1.0)
                    .cull_mode(vk::CullModeFlags::BACK)
                    .front_face(vk::FrontFace::CLOCKWISE)
                    .depth_bias_enable(false);

                let multi_sampling_info = vk::PipelineMultisampleStateCreateInfo::default()
                    .sample_shading_enable(false)
                    .rasterization_samples(vk::SampleCountFlags::TYPE_1);

                let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(
                        vk::ColorComponentFlags::R
                            | vk::ColorComponentFlags::G
                            | vk::ColorComponentFlags::B
                            | vk::ColorComponentFlags::A,
                    )
                    .blend_enable(false);

                let color_blend_attachments = [color_blend_attachment];
                let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
                    .logic_op_enable(false)
                    .attachments(&color_blend_attachments);

                let info = vk::GraphicsPipelineCreateInfo::default()
                    .stages(shader_stages)
                    .vertex_input_state(&vertex_input_info)
                    .input_assembly_state(&input_assembly_info)
                    .viewport_state(&viewport_state_info)
                    .rasterization_state(&rasterizer_info)
                    .multisample_state(&multi_sampling_info)
                    .color_blend_state(&color_blending)
                    .dynamic_state(&dynamic_info)
                    .layout(pipeline_layout)
                    .render_pass(render_pass)
                    .subpass(0);
                build_pipeline(&info)
            },
            true,
        )?;

        let sync_objects = std::array::from_fn(|_| create_sync_objects(device));

        Ok(Self {
            surface,
            device_id: device.id,
            swap_chain,
            pipeline,
            render_pass,
            sync_objects,
            current_frame: 0,
        })
    }
    pub fn size(&self) -> Vector<D2, GlobalCoordinate> {
        [self.swap_chain.extent.height, self.swap_chain.extent.width].into()
    }
    pub fn resize(&mut self, size: WindowSize, ctx: &VulkanContext) {
        let device = &ctx.device_contexts[self.device_id];

        let swap_chain_support = swap_chain_support(ctx, self.device_id, self.surface).unwrap();
        let swap_chain_config = swap_chain_support.default_config();

        unsafe {
            device.functions.device_wait_idle().unwrap();
            self.swap_chain.deinitialize(device);

            self.swap_chain = SwapChainData::new(
                device,
                &swap_chain_support,
                &swap_chain_config,
                self.render_pass,
                self.surface,
                size,
            );
        }
    }

    pub async fn render<'cref, 'inv: 'cref, 'op: 'inv>(
        &mut self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        input: &'inv FrameOperator,
    ) -> Result<DataVersionType, crate::Error> {
        let m = input.metadata;

        if m.dimensions != m.chunk_size.global() {
            return Err("Image must consist of a single chunk".into());
        }

        let device = &ctx.device_contexts[self.device_id];

        let img = ctx
            .submit(input.chunks.request_gpu(
                device.id,
                ChunkIndex(0),
                DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                },
            ))
            .await;

        let version = img.version;

        let descriptor_config = DescriptorConfig::new([&img]);

        // The rendering part:

        let f = self.current_frame;
        let sync_object = &mut self.sync_objects[f];

        ctx.submit(device.wait_for_cmd_buffer_completion(sync_object.last_use_epoch))
            .await;

        let image_index = match unsafe {
            device.functions.swap_chain_ext.acquire_next_image(
                self.swap_chain.inner,
                u64::max_value(),
                sync_object.image_available_semaphore,
                vk::Fence::null(),
            )
        } {
            Ok(i) => i.0,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(version),
            Err(e) => panic!("{}", e),
        };

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.swap_chain.framebuffers[image_index as usize])
            .render_area(
                vk::Rect2D::default()
                    .offset(vk::Offset2D::default().x(0).y(0))
                    .extent(self.swap_chain.extent),
            )
            .clear_values(&clear_values);

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.swap_chain.extent.width as f32)
            .height(self.swap_chain.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::default()
            .offset(vk::Offset2D::default().x(0).y(0))
            .extent(self.swap_chain.extent);

        let push_constants = PushConstants {
            size: m.dimensions.try_into_elem().unwrap().into(),
        };

        device.with_cmd_buffer(|cmd| unsafe {
            sync_object.last_use_epoch = cmd.id().epoch;

            let mut pipeline = self.pipeline.bind(cmd);

            device.functions().cmd_begin_render_pass(
                pipeline.cmd().raw(),
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
            device
                .functions()
                .cmd_set_viewport(pipeline.cmd().raw(), 0, &[viewport]);
            device
                .functions()
                .cmd_set_scissor(pipeline.cmd().raw(), 0, &[scissor]);

            pipeline.push_descriptor_set(0, descriptor_config);

            pipeline.push_constant_at(push_constants, vk::ShaderStageFlags::FRAGMENT);

            device.functions().cmd_draw(cmd.raw(), 3, 1, 0, 0);

            device.functions().cmd_end_render_pass(cmd.raw());

            cmd.wait_semaphore(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(sync_object.image_available_semaphore)
                    .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
            );
            cmd.signal_semaphore(
                vk::SemaphoreSubmitInfo::default().semaphore(sync_object.render_finished_semaphore),
            );
        });

        // Avoid Present-after-Write hazard (swapchain after image layout transition). Seems like
        // this works? I cannot find much information about this specific hazard
        ctx.submit(device.barrier(
            SrcBarrierInfo {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_WRITE,
            },
            DstBarrierInfo {
                stage: vk::PipelineStageFlags2::TRANSFER,
                access: vk::AccessFlags2::TRANSFER_READ,
            },
        ))
        .await;

        ctx.submit(device.wait_for_current_cmd_buffer_submission())
            .await;

        let swap_chains = &[self.swap_chain.inner];
        let image_indices = &[image_index];
        let wait_sem = &[sync_object.render_finished_semaphore];
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(swap_chains)
            .image_indices(image_indices)
            .wait_semaphores(wait_sem);

        match unsafe {
            device
                .functions
                .swap_chain_ext
                .queue_present(*device.queues.first().unwrap(), &present_info)
        } {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(version),
            Err(e) => panic!("{}", e),
        }

        self.current_frame = (self.current_frame + 1) % self.sync_objects.len();

        Ok(version)
    }
}
#[derive(Copy, Clone, AsStd140, GlslStruct)]
struct PushConstants {
    size: cgmath::Vector2<u32>,
}

const VERTEX_SHADER: &str = "
#version 450

vec2 positions[3] = vec2[](
    vec2(-1.0,-3.0),
    vec2( 3.0, 1.0),
    vec2(-1.0, 1.0)
);

vec2 texture_positions[3] = vec2[](
    vec2(0.0, -1.0),
    vec2(2.0,  1.0),
    vec2(0.0,  1.0)
);

layout(location = 0) out vec2 texture_pos;

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    texture_pos = texture_positions[gl_VertexIndex];
}
";

const FRAG_SHADER: &str = "
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_scalar_block_layout : require

#include <color.glsl>

layout(scalar, binding = 0) readonly buffer InputBuffer{
    u8vec4 values[];
} sourceData;

declare_push_consts(constants);

layout(location = 0) out vec4 out_color;
layout(location = 0) in vec2 texture_pos;

void main() {
    vec2 norm_pos = texture_pos;
    uvec2 buffer_pos = uvec2(norm_pos * vec2(constants.size));
    uint buffer_index = buffer_pos.x + buffer_pos.y * constants.size.x;

    out_color = to_uniform(sourceData.values[buffer_index]);
}
";
