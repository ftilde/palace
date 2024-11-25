use std::{
    cell::{Cell, RefCell},
    ffi::c_void,
    rc::Rc,
};

use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};
use derive_more::Deref;
use egui::{Rect, TextureId, TexturesDelta};

pub use egui;
use winit::keyboard::PhysicalKey;

use crate::{
    array::ChunkIndex,
    data::Vector,
    dim::*,
    event::{EventChain, EventStream},
    op_descriptor,
    operator::OperatorDescriptor,
    operators::tensor::TensorOperator,
    runtime::RunTime,
    storage::gpu::ImageAllocation,
    task::OpaqueTaskContext,
    util::Map,
    vulkan::{
        memory::TempRessource,
        pipeline::{DescriptorConfig, GraphicsPipelineBuilder},
        shader::Shader,
        state::{ResourceId, VulkanState},
        DeviceContext, DeviceId, DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::tensor::FrameOperator;

// TODO: We really need to clean this up for this to work properly with python. (Also we need to
// generate some kind of accessor for all the egui functionality anyway)
#[derive(Clone, Deref)]
pub struct GuiState(Rc<RefCell<GuiStateInner>>);

impl GuiState {
    pub fn on_device(device_id: DeviceId) -> Self {
        Self(Rc::new(RefCell::new(GuiStateInner {
            egui_ctx: Default::default(),
            version: Default::default(),
            latest_size: Cell::new(Vector::fill(0)),
            textures_delta: Default::default(),
            textures: Default::default(),
            scale_factor: 1.0,
            device: device_id,
        })))
    }
    pub fn new(ctx: &OpaqueTaskContext<'_, '_>) -> Self {
        Self::on_device(ctx.preferred_device().id)
    }

    pub unsafe fn deinit(&mut self, rt: &RunTime) {
        let device = self.0.borrow().device;
        unsafe { self.deinitialize(&rt.vulkan.device_contexts()[device]) };
    }
    pub fn destroy(mut self, rt: &RunTime) {
        // Safety we are actually taken ownership of the state
        unsafe { self.deinit(rt) };
    }
}

pub struct GuiStateInner {
    egui_ctx: egui::Context,
    version: u64,
    latest_size: Cell<Vector<D2, u32>>,
    textures_delta: TexturesDelta,
    textures: Map<TextureId, (ImageAllocation, vk::ImageView)>,
    scale_factor: f32,
    device: DeviceId,
}

impl GuiStateInner {
    async fn update(&mut self, ctx: &OpaqueTaskContext<'_, '_>, size: Vector<D2, u32>) {
        let device = ctx.device_ctx(self.device);

        self.latest_size.set(size);

        let textures_delta = &mut self.textures_delta;
        let textures = &mut self.textures;
        for (id, delta) in textures_delta.set.drain(..) {
            // Extract pixel data from egui
            let data: Vec<u8> = match &delta.image {
                egui::ImageData::Color(image) => {
                    assert_eq!(
                        image.width() * image.height(),
                        image.pixels.len(),
                        "Mismatch between texture size and texel count"
                    );
                    image
                        .pixels
                        .iter()
                        .flat_map(|color| color.to_array())
                        .collect()
                }
                egui::ImageData::Font(image) => image
                    .srgba_pixels(None)
                    .flat_map(|color| color.to_array())
                    .collect(),
            };

            let extent = vk::Extent3D {
                width: delta.image.width() as u32,
                height: delta.image.height() as u32,
                depth: 1,
            };
            let create_info = vk::ImageCreateInfo::default()
                .array_layers(1)
                .extent(extent)
                .flags(vk::ImageCreateFlags::empty())
                .format(vk::Format::R8G8B8A8_UNORM)
                .image_type(vk::ImageType::TYPE_2D)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .mip_levels(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                );

            let buffer_layout = std::alloc::Layout::array::<u8>(data.len()).unwrap();
            // TODO: Provide and use some general staging buffer infrastructure
            let staging_buffer = TempRessource::new(
                device,
                ctx.submit(device.storage.request_allocate_raw(
                    device,
                    buffer_layout,
                    vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                ))
                .await,
            );

            // TODO: at least make sure to abstract this away
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    staging_buffer.mapped_ptr().unwrap().as_ptr().cast(),
                    buffer_layout.size(),
                )
            }

            let img = ctx
                .submit(device.storage.request_allocate_image(device, create_info))
                .await;
            let region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(delta.image.width() as u32)
                .buffer_image_height(delta.image.height() as u32)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_array_layer(0)
                        .layer_count(1)
                        .mip_level(0),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: delta.image.width() as u32,
                    height: delta.image.height() as u32,
                    depth: 1,
                });

            transistion_image_layout_with_barrier(
                device,
                img.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            device.with_cmd_buffer(|cmd| unsafe {
                device.functions().cmd_copy_buffer_to_image(
                    cmd.raw(),
                    staging_buffer.buffer,
                    img.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            });

            transistion_image_layout_with_barrier(
                device,
                img.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );

            if let Some(_pos) = delta.pos {
                panic!("Texture update not yet implemented");
            } else {
                let create_info = vk::ImageViewCreateInfo::default()
                    .components(vk::ComponentMapping::default())
                    .flags(vk::ImageViewCreateFlags::empty())
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .image(img.image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_array_layer(0)
                            .base_mip_level(0)
                            .layer_count(1)
                            .level_count(1),
                    )
                    .view_type(vk::ImageViewType::TYPE_2D);
                let img_view = unsafe {
                    device
                        .functions()
                        .create_image_view(&create_info, None)
                        .unwrap()
                };

                textures.insert(id, (img, img_view));
            }
        }

        for id in textures_delta.free.drain(..) {
            let (img, img_view) = textures.remove(&id).unwrap();
            let _ = TempRessource::new(device, img);
            let _ = TempRessource::new(device, img_view);
        }
    }
    fn pointer_pos(&self, p: Vector<D2, i32>) -> egui::Pos2 {
        let pos = p.map(|v| v as f32 / self.scale_factor);
        (pos.x(), pos.y()).into()
    }
}

impl VulkanState for GuiState {
    unsafe fn deinitialize(&mut self, context: &DeviceContext) {
        let mut inner = self.0.borrow_mut();
        for (mut img, mut img_view) in std::mem::take(&mut inner.textures).into_values() {
            img.deinitialize(context);
            img_view.deinitialize(context);
        }
    }
}

pub struct GuiRenderState {
    inner: GuiState,
    clipped_primitives: Vec<egui::ClippedPrimitive>,
}

// Taken from egui (License MIT/Apatche)
fn translate_virtual_key_code(key: winit::keyboard::KeyCode) -> Option<egui::Key> {
    use egui::Key;
    use winit::keyboard::KeyCode;

    Some(match key {
        KeyCode::ArrowDown => Key::ArrowDown,
        KeyCode::ArrowLeft => Key::ArrowLeft,
        KeyCode::ArrowRight => Key::ArrowRight,
        KeyCode::ArrowUp => Key::ArrowUp,

        KeyCode::Escape => Key::Escape,
        KeyCode::Tab => Key::Tab,
        KeyCode::Backspace => Key::Backspace,
        KeyCode::Enter => Key::Enter,
        KeyCode::Space => Key::Space,

        KeyCode::Insert => Key::Insert,
        KeyCode::Delete => Key::Delete,
        KeyCode::Home => Key::Home,
        KeyCode::End => Key::End,
        KeyCode::PageUp => Key::PageUp,
        KeyCode::PageDown => Key::PageDown,

        KeyCode::Minus => Key::Minus,
        // Using Mac the key with the Plus sign on it is reported as the Equals key
        // (with both English and Swedish keyboard).
        KeyCode::Equal => Key::PlusEquals,

        KeyCode::Digit0 | KeyCode::Numpad0 => Key::Num0,
        KeyCode::Digit1 | KeyCode::Numpad1 => Key::Num1,
        KeyCode::Digit2 | KeyCode::Numpad2 => Key::Num2,
        KeyCode::Digit3 | KeyCode::Numpad3 => Key::Num3,
        KeyCode::Digit4 | KeyCode::Numpad4 => Key::Num4,
        KeyCode::Digit5 | KeyCode::Numpad5 => Key::Num5,
        KeyCode::Digit6 | KeyCode::Numpad6 => Key::Num6,
        KeyCode::Digit7 | KeyCode::Numpad7 => Key::Num7,
        KeyCode::Digit8 | KeyCode::Numpad8 => Key::Num8,
        KeyCode::Digit9 | KeyCode::Numpad9 => Key::Num9,

        KeyCode::KeyA => Key::A,
        KeyCode::KeyB => Key::B,
        KeyCode::KeyC => Key::C,
        KeyCode::KeyD => Key::D,
        KeyCode::KeyE => Key::E,
        KeyCode::KeyF => Key::F,
        KeyCode::KeyG => Key::G,
        KeyCode::KeyH => Key::H,
        KeyCode::KeyI => Key::I,
        KeyCode::KeyJ => Key::J,
        KeyCode::KeyK => Key::K,
        KeyCode::KeyL => Key::L,
        KeyCode::KeyM => Key::M,
        KeyCode::KeyN => Key::N,
        KeyCode::KeyO => Key::O,
        KeyCode::KeyP => Key::P,
        KeyCode::KeyQ => Key::Q,
        KeyCode::KeyR => Key::R,
        KeyCode::KeyS => Key::S,
        KeyCode::KeyT => Key::T,
        KeyCode::KeyU => Key::U,
        KeyCode::KeyV => Key::V,
        KeyCode::KeyW => Key::W,
        KeyCode::KeyX => Key::X,
        KeyCode::KeyY => Key::Y,
        KeyCode::KeyZ => Key::Z,

        KeyCode::F1 => Key::F1,
        KeyCode::F2 => Key::F2,
        KeyCode::F3 => Key::F3,
        KeyCode::F4 => Key::F4,
        KeyCode::F5 => Key::F5,
        KeyCode::F6 => Key::F6,
        KeyCode::F7 => Key::F7,
        KeyCode::F8 => Key::F8,
        KeyCode::F9 => Key::F9,
        KeyCode::F10 => Key::F10,
        KeyCode::F11 => Key::F11,
        KeyCode::F12 => Key::F12,
        KeyCode::F13 => Key::F13,
        KeyCode::F14 => Key::F14,
        KeyCode::F15 => Key::F15,
        KeyCode::F16 => Key::F16,
        KeyCode::F17 => Key::F17,
        KeyCode::F18 => Key::F18,
        KeyCode::F19 => Key::F19,
        KeyCode::F20 => Key::F20,

        _ => {
            return None;
        }
    })
}

impl GuiState {
    pub fn setup(
        &self,
        events: &mut EventStream,
        run_ui: impl FnOnce(&egui::Context),
    ) -> GuiRenderState {
        let mut this = self.borrow_mut();

        let mut latest_events = Vec::new();
        this.scale_factor = events.latest_state().scale_factor;

        events.act(|e| match e.change {
            winit::event::WindowEvent::CursorMoved { .. } => {
                let pos = this.pointer_pos(e.state.mouse_state.as_ref().unwrap().pos);
                latest_events.push(egui::Event::PointerMoved(pos));
                if this.egui_ctx.is_using_pointer() {
                    EventChain::Consumed
                } else {
                    e.into()
                }
            }
            winit::event::WindowEvent::MouseInput { state, button, .. }
                if this.egui_ctx.is_pointer_over_area() =>
            {
                if let Some(m_state) = e.state.mouse_state {
                    let pos = this.pointer_pos(m_state.pos);
                    use egui::PointerButton;
                    use winit::event::MouseButton;
                    latest_events.push(egui::Event::PointerButton {
                        pos,
                        button: match button {
                            MouseButton::Left => PointerButton::Primary,
                            MouseButton::Right => PointerButton::Secondary,
                            MouseButton::Middle => PointerButton::Middle,
                            MouseButton::Other(_) => PointerButton::Extra1,
                            MouseButton::Forward => PointerButton::Extra2,
                            MouseButton::Back => PointerButton::Extra2,
                        },
                        pressed: matches!(state, winit::event::ElementState::Pressed),
                        modifiers: egui::Modifiers::NONE, //TODO
                    });
                    EventChain::Consumed
                } else {
                    e.into()
                }
            }
            winit::event::WindowEvent::KeyboardInput { ref event, .. }
                if this.egui_ctx.wants_keyboard_input() =>
            {
                if let Some(text) = &event.text {
                    latest_events.push(egui::Event::Text(text.to_string()));
                    EventChain::Consumed
                } else if let PhysicalKey::Code(key_orig) = event.physical_key {
                    let key = translate_virtual_key_code(key_orig);
                    if let Some(key) = key {
                        let modifiers = egui::Modifiers {
                            alt: false,
                            ctrl: e.state.ctrl_pressed(),
                            shift: e.state.shift_pressed(),
                            mac_cmd: false,
                            command: false,
                        };
                        latest_events.push(egui::Event::Key {
                            key,
                            pressed: matches!(event.state, winit::event::ElementState::Pressed),
                            repeat: false,
                            modifiers,
                        });
                        EventChain::Consumed
                    } else {
                        e.into()
                    }
                } else {
                    e.into()
                }
            }
            _ => e.into(),
        });
        this.version = this.version.wrapping_add(1);

        let modifiers = egui::Modifiers {
            alt: false,
            ctrl: events.latest_state().ctrl_pressed(),
            shift: events.latest_state().shift_pressed(),
            mac_cmd: false,
            command: false,
        };

        let size2d = this.latest_size.get().map(|v| v as f32 / this.scale_factor);
        let raw_input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(
                Default::default(),
                egui::Vec2 {
                    x: size2d.x(),
                    y: size2d.y(),
                },
            )),
            pixels_per_point: Some(this.scale_factor),
            max_texture_side: None,
            time: None,               //TODO: specify
            predicted_dt: 1.0 / 60.0, //TODO: specify
            modifiers,
            events: latest_events,
            hovered_files: Vec::new(),
            dropped_files: Vec::new(),
            focused: true,
        };

        let full_output = this.egui_ctx.run(raw_input, run_ui);
        let clipped_primitives = this.egui_ctx.tessellate(full_output.shapes);

        this.textures_delta = full_output.textures_delta;

        GuiRenderState {
            inner: self.clone(),
            clipped_primitives,
        }
    }
}

fn transistion_image_layout_with_barrier(
    device: &DeviceContext,
    image: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    //TODO: Figure out finer grained synchronization here. We do not actually require ALL_COMMANDS
    //and MEMORY_READ/MEMORY_WRITE for src/dst. I _think_ this is not problematic though (at least
    //for now) since the barrier is associated with this specific image.
    let barriers = [vk::ImageMemoryBarrier2::default()
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .level_count(1)
                .layer_count(1)
                .aspect_mask(vk::ImageAspectFlags::COLOR),
        )
        .old_layout(from)
        .new_layout(to)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
        .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
    let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);

    device.with_cmd_buffer(|cmd| unsafe {
        device
            .functions()
            .cmd_pipeline_barrier2(cmd.raw(), &dep_info);
    });
}

impl GuiRenderState {
    pub fn render(self, input: FrameOperator) -> FrameOperator {
        #[derive(Copy, Clone, AsStd140, GlslStruct)]
        struct PushConstants {
            frame_size: cgmath::Vector2<u32>,
        }

        const VERTEX_SHADER: &str = "
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_uv;

declare_push_consts(consts);

vec3 srgb_to_linear(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    vec3 lower = srgb / vec3(12.92);
    vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    return mix(higher, lower, cutoff);
}

void main() {
    vec2 out_pos = 2.0 * pos / consts.frame_size - vec2(1.0);
    gl_Position = vec4(out_pos, 0.0, 1.0);
    out_color = color;
    out_color.xyz = srgb_to_linear(out_color.xyz);
    out_uv = uv;
}
";

        const FRAG_SHADER: &str = "
#include <util.glsl>

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec4 out_color;

layout(binding = 0, set = 0) uniform sampler2D img_tex;

void main() {
    vec4 col = color * texture(img_tex, uv);
    out_color = vec4(col.rgb, col.a * 0.8);
    //out_color = col;
}
";

        let version = self.inner.0.borrow().version;
        TensorOperator::unbatched(
            op_descriptor!()
                .dependent_on(&input)
                .dependent_on_data(&version)
                .ephemeral(),
            Default::default(),
            {
                let m = input.metadata;
                assert_eq!(m.dimension_in_chunks(), Vector::fill(1.into()));
                m
            },
            (input.clone(), self),
            move |ctx, pos, _, (input, state)| {
                async move {
                    let device = ctx.preferred_device();

                    let m_out = input.metadata;
                    let out_info = m_out.chunk_info(pos);

                    let format = vk::Format::R8G8B8A8_UNORM;

                    let render_pass =
                        device.request_state(ResourceId::new().of(ctx.current_op()), || {
                            let color_attachment = vk::AttachmentDescription::default()
                                .format(format)
                                .samples(vk::SampleCountFlags::TYPE_1)
                                .load_op(vk::AttachmentLoadOp::LOAD)
                                .store_op(vk::AttachmentStoreOp::STORE)
                                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                                .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

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
                                .src_stage_mask(vk::PipelineStageFlags::ALL_COMMANDS)
                                .src_access_mask(vk::AccessFlags::empty())
                                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

                            let subpasses = &[subpass];
                            let dependency_infos = &[dependency_info];
                            let attachments = &[color_attachment];
                            let render_pass_info = vk::RenderPassCreateInfo::default()
                                .attachments(attachments)
                                .subpasses(subpasses)
                                .dependencies(dependency_infos);

                            Ok(unsafe {
                                device
                                    .functions()
                                    .create_render_pass(&render_pass_info, None)
                            }
                            .unwrap())
                        })?;
                    let pipeline =
                        device.request_state(ResourceId::new().of(ctx.current_op()), || {
                            GraphicsPipelineBuilder::new(
                                Shader::new(VERTEX_SHADER).push_const_block::<PushConstants>(),
                                Shader::new(FRAG_SHADER)
                                    .define("BRICK_MEM_SIZE", out_info.mem_elements()),
                            )
                            .use_push_descriptor(true)
                            .build(
                                device,
                                |shader_stages, pipeline_layout, build_pipeline| {
                                    let dynamic_states =
                                        [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
                                    let dynamic_info =
                                        vk::PipelineDynamicStateCreateInfo::default()
                                            .dynamic_states(&dynamic_states);

                                    let vertex_bindings =
                                        [vk::VertexInputBindingDescription::default()
                                            .binding(0)
                                            .input_rate(vk::VertexInputRate::VERTEX)
                                            .stride(
                                                4 * std::mem::size_of::<f32>() as u32
                                                    + 4 * std::mem::size_of::<u8>() as u32,
                                            )];

                                    let vertex_attributes = [
                                        // position
                                        vk::VertexInputAttributeDescription::default()
                                            .binding(0)
                                            .offset(0)
                                            .location(0)
                                            .format(vk::Format::R32G32_SFLOAT),
                                        // uv
                                        vk::VertexInputAttributeDescription::default()
                                            .binding(0)
                                            .offset(8)
                                            .location(1)
                                            .format(vk::Format::R32G32_SFLOAT),
                                        // color
                                        vk::VertexInputAttributeDescription::default()
                                            .binding(0)
                                            .offset(16)
                                            .location(2)
                                            .format(vk::Format::R8G8B8A8_UNORM),
                                    ];

                                    let vertex_input_info =
                                        vk::PipelineVertexInputStateCreateInfo::default()
                                            .vertex_attribute_descriptions(&vertex_attributes)
                                            .vertex_binding_descriptions(&vertex_bindings);

                                    let input_assembly_info =
                                        vk::PipelineInputAssemblyStateCreateInfo::default()
                                            .primitive_restart_enable(false)
                                            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

                                    let viewport_state_info =
                                        vk::PipelineViewportStateCreateInfo::default()
                                            .viewport_count(1)
                                            .scissor_count(1);

                                    let rasterizer_info =
                                        vk::PipelineRasterizationStateCreateInfo::default()
                                            .depth_clamp_enable(false)
                                            .rasterizer_discard_enable(false)
                                            .polygon_mode(vk::PolygonMode::FILL)
                                            .line_width(1.0)
                                            .cull_mode(vk::CullModeFlags::NONE)
                                            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                                            .depth_bias_enable(false);

                                    let multi_sampling_info =
                                        vk::PipelineMultisampleStateCreateInfo::default()
                                            .sample_shading_enable(false)
                                            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

                                    let color_blend_attachment =
                                        vk::PipelineColorBlendAttachmentState::default()
                                            .color_write_mask(
                                                vk::ColorComponentFlags::R
                                                    | vk::ColorComponentFlags::G
                                                    | vk::ColorComponentFlags::B
                                                    | vk::ColorComponentFlags::A,
                                            )
                                            .src_color_blend_factor(vk::BlendFactor::ONE)
                                            .dst_color_blend_factor(
                                                vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                                            )
                                            .src_alpha_blend_factor(vk::BlendFactor::ONE)
                                            .dst_color_blend_factor(
                                                vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                                            )
                                            .color_blend_op(vk::BlendOp::ADD)
                                            .alpha_blend_op(vk::BlendOp::MAX)
                                            .blend_enable(true);

                                    let color_blend_attachments = [color_blend_attachment];
                                    let color_blending =
                                        vk::PipelineColorBlendStateCreateInfo::default()
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
                                        .render_pass(*render_pass)
                                        .subpass(0);
                                    build_pipeline(&info)
                                },
                            )
                        })?;
                    let sampler =
                        device.request_state(ResourceId::new().of(ctx.current_op()), || {
                            let create_info = vk::SamplerCreateInfo::default()
                                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                                .anisotropy_enable(false)
                                .min_filter(vk::Filter::LINEAR)
                                .mag_filter(vk::Filter::LINEAR)
                                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                                .min_lod(0.0)
                                .max_lod(vk::LOD_CLAMP_NONE);
                            Ok(unsafe {
                                device
                                    .functions()
                                    .create_sampler(&create_info, None)
                                    .unwrap()
                            })
                        })?;

                    let out_dim = out_info.logical_dimensions;
                    let width = out_dim.x().into();
                    let height = out_dim.y().into();
                    let extent = vk::Extent2D::default().width(width).height(height);

                    let img_info = vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .extent(extent.into())
                        .mip_levels(1)
                        .array_layers(1)
                        .format(format)
                        .tiling(vk::ImageTiling::LINEAR)
                        .initial_layout(vk::ImageLayout::PREINITIALIZED)
                        .usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::TRANSFER_SRC
                                | vk::ImageUsageFlags::TRANSFER_DST,
                        )
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE);

                    let output_texture = TempRessource::new(
                        device,
                        ctx.submit(device.storage.request_allocate_image(device, img_info))
                            .await,
                    );

                    let info = vk::ImageViewCreateInfo::default()
                        .image(output_texture.image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .components(vk::ComponentMapping::default())
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        );

                    let img_view = TempRessource::new(
                        device,
                        unsafe { device.functions().create_image_view(&info, None) }.unwrap(),
                    );

                    let attachments = [*img_view];

                    let framebuffer_info = vk::FramebufferCreateInfo::default()
                        .render_pass(*render_pass)
                        .attachments(&attachments)
                        .width(width)
                        .height(height)
                        .layers(1);

                    let framebuffer = TempRessource::new(
                        device,
                        unsafe {
                            device
                                .functions()
                                .create_framebuffer(&framebuffer_info, None)
                        }
                        .unwrap(),
                    );

                    assert_eq!(pos, ChunkIndex(0));
                    let gpu_brick_in = ctx
                        .submit(input.chunks.request_gpu(
                            device.id,
                            pos,
                            DstBarrierInfo {
                                stage: vk::PipelineStageFlags2::TRANSFER,
                                access: vk::AccessFlags2::TRANSFER_READ,
                            },
                        ))
                        .await;

                    let copy_info = vk::BufferImageCopy::default()
                        .image_extent(extent.into())
                        .buffer_row_length(width)
                        .buffer_image_height(height)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .mip_level(0)
                                .layer_count(1)
                                .aspect_mask(vk::ImageAspectFlags::COLOR),
                        );

                    transistion_image_layout_with_barrier(
                        device,
                        output_texture.image,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    );

                    device.with_cmd_buffer(|cmd| unsafe {
                        device.functions().cmd_copy_buffer_to_image(
                            cmd.raw(),
                            gpu_brick_in.buffer,
                            output_texture.image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[copy_info],
                        );
                    });

                    transistion_image_layout_with_barrier(
                        device,
                        output_texture.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    );

                    // Actual rendering
                    let render_pass_info = vk::RenderPassBeginInfo::default()
                        .render_pass(*render_pass)
                        .framebuffer(*framebuffer)
                        .render_area(
                            vk::Rect2D::default()
                                .offset(vk::Offset2D::default().x(0).y(0))
                                .extent(extent),
                        )
                        .clear_values(&[]);

                    let mut state_i = state.inner.borrow_mut();
                    let size2d = m_out.dimensions.raw();
                    let vp_size = size2d.map(|v| v as f32 * state_i.scale_factor);
                    let viewport = vk::Viewport::default()
                        .x(0.0)
                        .y(0.0)
                        .width(vp_size.x())
                        .height(vp_size.y())
                        .min_depth(0.0)
                        .max_depth(1.0);

                    let push_constants = PushConstants {
                        frame_size: out_info.mem_dimensions.try_into_elem().unwrap().into(),
                    };

                    state_i.update(&ctx, size2d).await;

                    let textures = &state_i.textures;
                    for primitive in state.clipped_primitives.iter() {
                        let mesh = match &primitive.primitive {
                            egui::epaint::Primitive::Mesh(m) => m,
                            egui::epaint::Primitive::Callback(_) => {
                                panic!("egui callback not supported")
                            }
                        };

                        let mut clip_rect = primitive.clip_rect;
                        *clip_rect.left_mut() *= state_i.scale_factor;
                        *clip_rect.right_mut() *= state_i.scale_factor;
                        *clip_rect.top_mut() *= state_i.scale_factor;
                        *clip_rect.bottom_mut() *= state_i.scale_factor;
                        let clip_rect_extent = vk::Extent2D::default()
                            .width(clip_rect.width().round() as u32)
                            .height(clip_rect.height().round() as u32);
                        let scissor = vk::Rect2D::default()
                            .offset(
                                vk::Offset2D::default()
                                    .x(clip_rect.left().round() as i32)
                                    .y(clip_rect.top().round() as i32),
                            )
                            .extent(clip_rect_extent);

                        let indices = &mesh.indices;
                        let vertices = &mesh.vertices;

                        let vertex_buf_layout =
                            std::alloc::Layout::array::<egui::epaint::Vertex>(vertices.len())
                                .unwrap();
                        let index_buf_layout =
                            std::alloc::Layout::array::<u32>(indices.len()).unwrap();

                        let vertex_flags = vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::VERTEX_BUFFER;
                        let index_flags =
                            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER;
                        let buf_type = gpu_allocator::MemoryLocation::CpuToGpu;

                        let (vertex_alloc, index_alloc) = futures::join!(
                            ctx.submit(device.storage.request_allocate_raw(
                                device,
                                vertex_buf_layout,
                                vertex_flags,
                                buf_type,
                            ),),
                            ctx.submit(device.storage.request_allocate_raw(
                                device,
                                index_buf_layout,
                                index_flags,
                                buf_type,
                            ),)
                        );
                        let vertex_buffer = TempRessource::new(device, vertex_alloc);
                        let index_buffer = TempRessource::new(device, index_alloc);

                        let (_, img_view) = textures.get(&mesh.texture_id).unwrap();

                        let descriptor_config = DescriptorConfig::new([&(
                            *img_view,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            *sampler,
                        )]);

                        for (out, in_, layout) in [
                            (
                                &vertex_buffer,
                                vertices.as_ptr() as *const c_void,
                                vertex_buf_layout,
                            ),
                            (
                                &index_buffer,
                                indices.as_ptr() as *const c_void,
                                index_buf_layout,
                            ),
                        ] {
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    in_,
                                    out.mapped_ptr().unwrap().as_ptr(),
                                    layout.size(),
                                )
                            }
                        }

                        device.with_cmd_buffer(|cmd| unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            device.functions().cmd_bind_vertex_buffers(
                                pipeline.cmd().raw(),
                                0,
                                &[vertex_buffer.buffer],
                                &[0],
                            );
                            device.functions().cmd_bind_index_buffer(
                                pipeline.cmd().raw(),
                                index_buffer.buffer,
                                0,
                                vk::IndexType::UINT32,
                            );

                            device.functions().cmd_begin_render_pass(
                                pipeline.cmd().raw(),
                                &render_pass_info,
                                vk::SubpassContents::INLINE,
                            );
                            device.functions().cmd_set_viewport(
                                pipeline.cmd().raw(),
                                0,
                                &[viewport],
                            );
                            device
                                .functions()
                                .cmd_set_scissor(pipeline.cmd().raw(), 0, &[scissor]);

                            pipeline.push_descriptor_set(0, descriptor_config);
                            pipeline.push_constant_at(push_constants, vk::ShaderStageFlags::VERTEX);

                            device.functions().cmd_draw_indexed(
                                cmd.raw(),
                                indices.len() as _,
                                1,
                                0,
                                0,
                                0,
                            );

                            device.functions().cmd_end_render_pass(cmd.raw());
                        });
                    }

                    transistion_image_layout_with_barrier(
                        device,
                        output_texture.image,
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    );

                    let copy_info = vk::BufferImageCopy::default()
                        .image_extent(extent.into())
                        .buffer_row_length(width)
                        .buffer_image_height(height)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .mip_level(0)
                                .layer_count(1)
                                .aspect_mask(vk::ImageAspectFlags::COLOR),
                        );

                    let gpu_brick_out = ctx
                        .submit(ctx.alloc_slot_gpu(device, pos, out_info.mem_elements()))
                        .await;

                    device.with_cmd_buffer(|cmd| unsafe {
                        device.functions().cmd_copy_image_to_buffer(
                            cmd.raw(),
                            output_texture.image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            gpu_brick_out.buffer,
                            &[copy_info],
                        );
                    });

                    unsafe {
                        gpu_brick_out.initialized(
                            *ctx,
                            SrcBarrierInfo {
                                stage: vk::PipelineStageFlags2::TRANSFER,
                                access: vk::AccessFlags2::TRANSFER_WRITE,
                            },
                        )
                    };

                    Ok(())
                }
                .into()
            },
        )
    }
}
