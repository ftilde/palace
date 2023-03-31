use std::cell::RefCell;

#[cfg(target_family = "unix")]
use ash::extensions::khr::{WaylandSurface, XlibSurface};

#[cfg(target_family = "windows")]
use ash::extensions::khr::Win32Surface;

use ash::vk;
use crevice::std140::AsStd140;
use winit::event_loop::EventLoopWindowTarget;
use winit::window::WindowBuilder;

use crate::data::{BrickPosition, GlobalCoordinate, Vector};
use crate::operators::volume::VolumeOperator;
use crate::task::OpaqueTaskContext;
use crate::vulkan::shader::Shader;

use super::pipeline::{DescriptorConfig, DynamicDescriptorSetPool};
use super::shader::ShaderSource;
use super::state::VulkanState;
use super::{CmdBufferEpoch, DeviceContext, DeviceId, VulkanContext};

type WindowSize = winit::dpi::PhysicalSize<u32>;

#[cfg(target_family = "unix")]
fn create_surface_wayland(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> Option<vk::SurfaceKHR> {
    use winit::platform::wayland::WindowExtWayland;
    let loader = WaylandSurface::new(entry, instance);

    let create_info = vk::WaylandSurfaceCreateInfoKHR::builder()
        .display(window.wayland_display()?)
        .surface(window.wayland_surface()?)
        .build();

    Some(unsafe { loader.create_wayland_surface(&create_info, None) }.unwrap())
}

#[cfg(target_family = "unix")]
fn create_surface_x11(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> Option<vk::SurfaceKHR> {
    use winit::platform::x11::WindowExtX11;

    let x11_display = window.xlib_display()?;
    let x11_window = window.xlib_window()?;
    let create_info = vk::XlibSurfaceCreateInfoKHR::builder()
        .window(x11_window as vk::Window)
        .dpy(x11_display as *mut vk::Display);

    let xlib_surface_loader = XlibSurface::new(entry, instance);
    Some(unsafe {
        xlib_surface_loader
            .create_xlib_surface(&create_info, None)
            .unwrap()
    })
}

#[cfg(target_family = "unix")]
fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> vk::SurfaceKHR {
    if let Some(s) = create_surface_wayland(entry, instance, window) {
        return s;
    } else {
        create_surface_x11(entry, instance, window).unwrap()
    }
}

#[cfg(target_family = "windows")]
fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> vk::SurfaceKHR {
    use std::os::raw::c_void;
    use winit::platform::windows::WindowExtWindows;

    let hinstance = window.hinstance();
    let hwnd = window.hwnd();
    let win32_create_info = vk::Win32SurfaceCreateInfoKHR::builder()
        .hinstance(hinstance as *const c_void)
        .hwnd(hwnd as *const c_void);
		
    let win32_surface_loader = Win32Surface::new(entry, instance);
	unsafe{ win32_surface_loader.create_win32_surface(&win32_create_info, None) }.unwrap()
}

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
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .cloned()
        .unwrap_or_else(|| available[0])
}

fn select_swap_present_mode(available: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    if available.contains(&vk::PresentModeKHR::MAILBOX) {
        return vk::PresentModeKHR::MAILBOX;
    }
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

    let sc_info = vk::SwapchainCreateInfoKHR::builder()
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
    let color_attachment = vk::AttachmentDescription::builder()
        .format(swap_chain_config.format.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_attachment_refs = &[*color_attachment_ref];
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachment_refs);

    let dependency_info = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

    let attachments = &[*color_attachment];
    let subpasses = &[*subpass];
    let dependency_infos = &[*dependency_info];
    let render_pass_info = vk::RenderPassCreateInfo::builder()
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
            let framebuffer_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);

            unsafe { device.functions.create_framebuffer(&framebuffer_info, None) }.unwrap()
        })
        .collect::<Vec<_>>()
}

struct GraphicsPipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    ds_pools: Vec<RefCell<DynamicDescriptorSetPool>>,
    push_constant_size: Option<usize>,
}

impl VulkanState for GraphicsPipeline {
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
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

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
        }
    }
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
                let info = vk::ImageViewCreateInfo::builder()
                    .image(*img)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(config.format.format)
                    .components(vk::ComponentMapping::builder().build())
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
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
    let semaphore_info = vk::SemaphoreCreateInfo::builder();

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
    winit_win: winit::window::Window,
    surface: vk::SurfaceKHR,
    device_id: DeviceId,
    pipeline: GraphicsPipeline,
    swap_chain_support: SwapChainSupportDetails,
    swap_chain_config: SwapChainConfiguration,
    swap_chain: SwapChainData,
    render_pass: vk::RenderPass,
    sync_objects: [SyncObjects; MAX_FRAMES_IN_FLIGHT],
    current_frame: usize,
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

    pub fn new<T>(
        ctx: &VulkanContext,
        target: &EventLoopWindowTarget<T>,
    ) -> Result<Self, crate::Error> {
        let winit_win = WindowBuilder::new().build(&target).unwrap();
        let surface = create_surface(&ctx.entry, &ctx.instance, &winit_win);
        let (device, swap_chain_support) = ctx
            .device_contexts
            .iter()
            .find_map(|device| {
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
                if !support.formats.is_empty()
                    && !support.present_modes.is_empty()
                    && surface_support
                {
                    Some((device, support))
                } else {
                    None
                }
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
            winit_win.inner_size(),
        );

        let pipeline =
            GraphicsPipeline::new(device, VERTEX_SHADER, FRAG_SHADER, &render_pass, true);

        let sync_objects = std::array::from_fn(|_| create_sync_objects(device));

        Ok(Self {
            winit_win,
            surface,
            swap_chain_config,
            device_id: device.id,
            swap_chain_support,
            swap_chain,
            pipeline,
            render_pass,
            sync_objects,
            current_frame: 0,
        })
    }
    pub fn size(&self) -> Vector<2, GlobalCoordinate> {
        [self.swap_chain.extent.height, self.swap_chain.extent.width].into()
    }
    pub fn inner(&self) -> &winit::window::Window {
        &self.winit_win
    }
    pub fn resize(&mut self, size: WindowSize, ctx: &VulkanContext) {
        let device = &ctx.device_contexts[self.device_id];

        unsafe {
            device.functions.device_wait_idle().unwrap();
            self.swap_chain.deinitialize(device);

            self.swap_chain = SwapChainData::new(
                device,
                &self.swap_chain_support,
                &self.swap_chain_config,
                self.render_pass,
                self.surface,
                size,
            );
        }
    }

    pub async fn render<'cref, 'inv: 'cref, 'op: 'inv>(
        &mut self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        input: &'inv VolumeOperator<'op>,
    ) -> Result<(), crate::Error> {
        let m = ctx.submit(input.metadata.request_scalar()).await;

        if m.dimensions != m.chunk_size.global() {
            return Err("Image must consist of a single chunk".into());
        }

        if m.dimensions.x().raw != 4 {
            return Err("Image must have exactly four channels".into());
        }

        let device = &ctx.device_contexts[self.device_id];

        let img = ctx
            .submit(input.bricks.request_gpu(
                device.id,
                BrickPosition::fill(0.into()),
                super::DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                },
            ))
            .await;

        let descriptor_config = DescriptorConfig::new([&img]);
        let writes = descriptor_config.writes_for_push();

        // The rendering part:

        let f = self.current_frame;
        self.current_frame = (self.current_frame + 1) % self.sync_objects.len();
        let sync_object = &mut self.sync_objects[f];

        ctx.submit(device.wait_for_cmd_buffer_completion(sync_object.last_use_epoch))
            .await;

        let image_index = unsafe {
            device
                .functions
                .swap_chain_ext
                .acquire_next_image(
                    self.swap_chain.inner,
                    u64::max_value(),
                    sync_object.image_available_semaphore,
                    vk::Fence::null(),
                )
                .unwrap()
                .0
        };

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(self.swap_chain.framebuffers[image_index as usize])
            .render_area(
                vk::Rect2D::builder()
                    .offset(vk::Offset2D::builder().x(0).y(0).build())
                    .extent(self.swap_chain.extent)
                    .build(),
            )
            .clear_values(&clear_values);

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(self.swap_chain.extent.width as f32)
            .height(self.swap_chain.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D::builder().x(0).y(0).build())
            .extent(self.swap_chain.extent);

        let push_constans = PushConstants {
            size: m.dimensions.drop_dim(2).try_into_elem().unwrap().into(),
        };

        device.with_cmd_buffer(|cmd| unsafe {
            sync_object.last_use_epoch = cmd.id().epoch;

            cmd.functions().cmd_bind_pipeline(
                cmd.raw(),
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );

            device.functions().cmd_begin_render_pass(
                cmd.raw(),
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
            device
                .functions()
                .cmd_set_viewport(cmd.raw(), 0, &[*viewport]);
            device
                .functions()
                .cmd_set_scissor(cmd.raw(), 0, &[*scissor]);

            device
                .functions()
                .push_descriptor_ext
                .cmd_push_descriptor_set(
                    cmd.raw(),
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.pipeline_layout,
                    0,
                    &writes,
                );

            {
                let v = push_constans.as_std140();
                let bytes = v.as_bytes();

                let bytes = &bytes[..self.pipeline.push_constant_size.unwrap()];
                device.functions().cmd_push_constants(
                    cmd.raw(),
                    self.pipeline.pipeline_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytes,
                );
            }

            device.functions().cmd_draw(cmd.raw(), 3, 1, 0, 0);

            device.functions().cmd_end_render_pass(cmd.raw());

            cmd.wait_semaphore(sync_object.image_available_semaphore);
            cmd.signal_semaphore(sync_object.render_finished_semaphore);
        });

        ctx.submit(device.wait_for_current_cmd_buffer_submission())
            .await;

        let swap_chains = &[self.swap_chain.inner];
        let image_indices = &[image_index];
        let wait_sem = &[sync_object.render_finished_semaphore];
        let present_info = vk::PresentInfoKHR::builder()
            .swapchains(swap_chains)
            .image_indices(image_indices)
            .wait_semaphores(wait_sem);

        unsafe {
            device
                .functions
                .swap_chain_ext
                .queue_present(*device.queues.first().unwrap(), &present_info)
        }
        .unwrap();

        Ok(())
    }
}
#[derive(Copy, Clone, AsStd140)]
struct PushConstants {
    size: mint::Vector2<u32>,
}

const VERTEX_SHADER: &str = "
#version 450

vec2 positions[3] = vec2[](
    vec2(-1.0, -3.0),
    vec2(3.0, 1.0),
    vec2(-1.0, 1.0)
);

vec2 colors[3] = vec2[](
    vec2(0.0, 2.0),
    vec2(2.0, 0.0),
    vec2(0.0, 0.0)
);

layout(location = 0) out vec2 fragColor;

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
}
";

const FRAG_SHADER: &str = "
#version 450

layout(std430, binding = 0) readonly buffer InputBuffer{
    float values[];
} sourceData;

layout(std140, push_constant) uniform PushConstants
{
    uvec2 size;
} constants;

layout(location = 0) out vec4 outColor;
layout(location = 0) in vec2 fragColor;

void main() {
    vec2 norm_pos = fragColor;
    uvec2 buffer_pos = uvec2(norm_pos * vec2(constants.size));
    uint buffer_index = buffer_pos.x + buffer_pos.y * constants.size.x;

    for(int c = 0; c < 4; ++c) {
        outColor[c] = sourceData.values[4*buffer_index + c];
    }
}
";
