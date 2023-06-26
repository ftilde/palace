use std::ffi::c_void;

use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};
use egui::Rect;

use crate::{
    data::Vector,
    operator::OperatorId,
    operators::tensor::TensorOperator,
    vulkan::{
        memory::TempRessource, pipeline::GraphicsPipeline, shader::ShaderDefines,
        state::RessourceId, DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::volume::VolumeOperator;

pub fn gui<'a>(input: VolumeOperator<'a>) -> VolumeOperator<'a> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        frame_size: cgmath::Vector2<u32>,
    }

    const VERTEX_SHADER: &str = "
#version 450

layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;

layout(location = 0) out vec4 out_color;

declare_push_consts(consts);

void main() {
    vec2 out_pos = 2.0 * pos / consts.frame_size - vec2(1.0);
    gl_Position = vec4(out_pos, 0.0, 1.0);
    out_color = color;
}
";

    const FRAG_SHADER: &str = "
#version 450

#include <util.glsl>

layout(location = 0) in vec4 color;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(color.rgb, color.a*0.8); //NO_PUSH_main
}
";
    TensorOperator::unbatched(
        OperatorId::new("entry_exit_points").dependent_on(&input),
        input.clone(),
        input,
        move |ctx, input, _| {
            async move {
                let m = ctx.submit(input.metadata.request_scalar()).await;
                assert_eq!(m.dimension_in_bricks(), Vector::fill(1.into()));
                ctx.write(m)
            }
            .into()
        },
        move |ctx, pos, input, _| {
            async move {
                let device = ctx.vulkan_device();

                let m_out = ctx.submit(input.metadata.request_scalar()).await;
                let out_info = m_out.chunk_info(pos);

                let format = vk::Format::R32G32B32A32_SFLOAT;

                let render_pass = device.request_state(
                    RessourceId::new("renderpass").of(ctx.current_op()),
                    || {
                        let color_attachment = vk::AttachmentDescription::builder()
                            .format(format)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .initial_layout(vk::ImageLayout::GENERAL)
                            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

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

                        let subpasses = &[*subpass];
                        let dependency_infos = &[*dependency_info];
                        let attachments = &[*color_attachment];
                        let render_pass_info = vk::RenderPassCreateInfo::builder()
                            .attachments(attachments)
                            .subpasses(subpasses)
                            .dependencies(dependency_infos);

                        unsafe {
                            device
                                .functions()
                                .create_render_pass(&render_pass_info, None)
                        }
                        .unwrap()
                    },
                );
                let pipeline =
                    device.request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        GraphicsPipeline::new(
                            device,
                            (
                                VERTEX_SHADER,
                                ShaderDefines::new().push_const_block::<PushConstants>(),
                            ),
                            (
                                FRAG_SHADER,
                                ShaderDefines::new().add("BRICK_MEM_SIZE", out_info.mem_elements()),
                            ),
                            |shader_stages, pipeline_layout, build_pipeline| {
                                let dynamic_states =
                                    [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
                                let dynamic_info = vk::PipelineDynamicStateCreateInfo::builder()
                                    .dynamic_states(&dynamic_states);

                                let vertex_bindings =
                                    [vk::VertexInputBindingDescription::builder()
                                        .binding(0)
                                        .input_rate(vk::VertexInputRate::VERTEX)
                                        .stride(
                                            4 * std::mem::size_of::<f32>() as u32
                                                + 4 * std::mem::size_of::<u8>() as u32,
                                        )
                                        .build()];

                                let vertex_attributes = [
                                    // position
                                    vk::VertexInputAttributeDescription::builder()
                                        .binding(0)
                                        .offset(0)
                                        .location(0)
                                        .format(vk::Format::R32G32_SFLOAT)
                                        .build(),
                                    // uv
                                    vk::VertexInputAttributeDescription::builder()
                                        .binding(0)
                                        .offset(8)
                                        .location(1)
                                        .format(vk::Format::R32G32_SFLOAT)
                                        .build(),
                                    // color
                                    vk::VertexInputAttributeDescription::builder()
                                        .binding(0)
                                        .offset(16)
                                        .location(2)
                                        .format(vk::Format::R8G8B8A8_UNORM)
                                        .build(),
                                ];

                                let vertex_input_info =
                                    vk::PipelineVertexInputStateCreateInfo::builder()
                                        .vertex_attribute_descriptions(&vertex_attributes)
                                        .vertex_binding_descriptions(&vertex_bindings);

                                let input_assembly_info =
                                    vk::PipelineInputAssemblyStateCreateInfo::builder()
                                        .primitive_restart_enable(false)
                                        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

                                let viewport_state_info =
                                    vk::PipelineViewportStateCreateInfo::builder()
                                        .viewport_count(1)
                                        .scissor_count(1);

                                let rasterizer_info =
                                    vk::PipelineRasterizationStateCreateInfo::builder()
                                        .depth_clamp_enable(false)
                                        .rasterizer_discard_enable(false)
                                        .polygon_mode(vk::PolygonMode::FILL)
                                        .line_width(1.0)
                                        .cull_mode(vk::CullModeFlags::NONE)
                                        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                                        .depth_bias_enable(false);

                                let multi_sampling_info =
                                    vk::PipelineMultisampleStateCreateInfo::builder()
                                        .sample_shading_enable(false)
                                        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

                                let color_blend_attachment =
                                    vk::PipelineColorBlendAttachmentState::builder()
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
                                        .blend_enable(true);

                                let color_blend_attachments = [*color_blend_attachment];
                                let color_blending =
                                    vk::PipelineColorBlendStateCreateInfo::builder()
                                        .attachments(&color_blend_attachments);

                                let info = vk::GraphicsPipelineCreateInfo::builder()
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
                            true,
                        )
                    });

                let out_dim = out_info.logical_dimensions.drop_dim(2);
                let width = out_dim.x().into();
                let height = out_dim.y().into();
                let extent = vk::Extent2D::builder().width(width).height(height).build();

                let img_info = vk::ImageCreateInfo::builder()
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
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build();

                let output_texture =
                    TempRessource::new(device, device.storage.allocate_image(device, img_info));

                let info = vk::ImageViewCreateInfo::builder()
                    .image(output_texture.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
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

                let img_view = TempRessource::new(
                    device,
                    unsafe { device.functions().create_image_view(&info, None) }.unwrap(),
                );

                let attachments = [*img_view];

                let framebuffer_info = vk::FramebufferCreateInfo::builder()
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

                assert_eq!(pos, Vector::fill(0.into()));
                let inplace_result = ctx
                    .submit(input.bricks.request_inplace_gpu(
                        device.id,
                        pos,
                        ctx.current_op(),
                        DstBarrierInfo {
                            stage: vk::PipelineStageFlags2::FRAGMENT_SHADER
                                | vk::PipelineStageFlags2::TRANSFER,
                            access: vk::AccessFlags2::SHADER_WRITE
                                | vk::AccessFlags2::TRANSFER_WRITE,
                        },
                    ))
                    .await
                    .unwrap();

                let (gpu_brick_in_buffer, gpu_brick_out) = match &inplace_result {
                    crate::storage::gpu::InplaceResult::Inplace(rw, _) => (rw.buffer, rw),
                    crate::storage::gpu::InplaceResult::New(r, w) => (r.buffer, w),
                };

                let copy_info = vk::BufferImageCopy::builder()
                    .image_extent(extent.into())
                    .buffer_row_length(width)
                    .buffer_image_height(height)
                    .image_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .mip_level(0)
                            .layer_count(1)
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .build(),
                    )
                    .build();

                let barriers = [vk::ImageMemoryBarrier2::builder()
                    .image(output_texture.image)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .level_count(1)
                            .layer_count(1)
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .build(),
                    )
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .build()];
                let dep_info = vk::DependencyInfo::builder()
                    .image_memory_barriers(&barriers)
                    .build();
                device.with_cmd_buffer(|cmd| unsafe {
                    device
                        .functions()
                        .cmd_pipeline_barrier2(cmd.raw(), &dep_info);
                });

                device.with_cmd_buffer(|cmd| unsafe {
                    device.functions().cmd_copy_buffer_to_image(
                        cmd.raw(),
                        gpu_brick_in_buffer,
                        output_texture.image,
                        vk::ImageLayout::GENERAL,
                        &[copy_info],
                    );
                });

                ctx.submit(device.barrier(
                    SrcBarrierInfo {
                        stage: vk::PipelineStageFlags2::TRANSFER,
                        access: vk::AccessFlags2::TRANSFER_WRITE,
                    },
                    DstBarrierInfo {
                        stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        access: vk::AccessFlags2::SHADER_WRITE,
                    },
                ))
                .await;

                // Actual rendering
                let render_pass_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(*render_pass)
                    .framebuffer(*framebuffer)
                    .render_area(
                        vk::Rect2D::builder()
                            .offset(vk::Offset2D::builder().x(0).y(0).build())
                            .extent(extent)
                            .build(),
                    )
                    .clear_values(&[]);

                let size2d = m_out.dimensions.drop_dim(2).raw();
                let viewport = vk::Viewport::builder()
                    .x(0.0)
                    .y(0.0)
                    .width(size2d.x() as f32)
                    .height(size2d.y() as f32)
                    .min_depth(0.0)
                    .max_depth(1.0);

                let push_constants = PushConstants {
                    frame_size: out_info
                        .mem_dimensions
                        .drop_dim(2)
                        .try_into_elem()
                        .unwrap()
                        .into(),
                };

                let raw_input = egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(
                        Default::default(),
                        egui::Vec2 {
                            x: size2d.x() as f32,
                            y: size2d.y() as f32,
                        },
                    )),
                    pixels_per_point: Some(1.0),
                    max_texture_side: None,
                    time: None,               //TODO: specify
                    predicted_dt: 1.0 / 60.0, //TODO: specify
                    modifiers: egui::Modifiers::NONE,
                    events: Vec::new(),
                    hovered_files: Vec::new(),
                    dropped_files: Vec::new(),
                    focused: true,
                };

                let egui_ctx = egui::Context::default();

                let mut counter = 0;
                let full_output = egui_ctx.run(raw_input, |ctx| {
                    //egui::Window::new("Some title").show(&ctx, |ui| {
                    egui::CentralPanel::default().show(&ctx, |ui| {
                        ui.horizontal(|ui| {
                            if ui.button("-").clicked() {
                                counter -= 1;
                            }
                            ui.label(counter.to_string());
                            if ui.button("+").clicked() {
                                counter += 1;
                            }
                        });
                    });
                });
                let clipped_primitives = egui_ctx.tessellate(full_output.shapes);

                for primitive in clipped_primitives {
                    let mesh = match primitive.primitive {
                        egui::epaint::Primitive::Mesh(m) => m,
                        egui::epaint::Primitive::Callback(_) => {
                            panic!("egui callback not supported")
                        }
                    };

                    let clip_rect = primitive.clip_rect;
                    let clip_rect_extent = vk::Extent2D::builder()
                        .width(clip_rect.width().round() as u32)
                        .height(clip_rect.height().round() as u32)
                        .build();
                    let scissor = vk::Rect2D::builder()
                        .offset(
                            vk::Offset2D::builder()
                                .x(clip_rect.left().round() as i32)
                                .y(clip_rect.top().round() as i32)
                                .build(),
                        )
                        .extent(clip_rect_extent);

                    let indices = mesh.indices;
                    let vertices = mesh.vertices;

                    let vertex_buf_layout =
                        std::alloc::Layout::array::<egui::epaint::Vertex>(vertices.len()).unwrap();
                    let index_buf_layout = std::alloc::Layout::array::<u32>(indices.len()).unwrap();

                    let vertex_flags =
                        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER;
                    let index_flags =
                        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER;
                    let buf_type = gpu_allocator::MemoryLocation::CpuToGpu;

                    let vertex_buffer = TempRessource::new(
                        device,
                        device
                            .storage
                            .allocate(device, vertex_buf_layout, vertex_flags, buf_type),
                    );
                    let index_buffer = TempRessource::new(
                        device,
                        device
                            .storage
                            .allocate(device, index_buf_layout, index_flags, buf_type),
                    );

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
                        device
                            .functions()
                            .cmd_set_viewport(pipeline.cmd().raw(), 0, &[*viewport]);
                        device
                            .functions()
                            .cmd_set_scissor(pipeline.cmd().raw(), 0, &[*scissor]);

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

                let copy_info = vk::BufferImageCopy::builder()
                    .image_extent(extent.into())
                    .buffer_row_length(width)
                    .buffer_image_height(height)
                    .image_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .mip_level(0)
                            .layer_count(1)
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .build(),
                    )
                    .build();
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
                    inplace_result.initialized(
                        *ctx,
                        SrcBarrierInfo {
                            stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                            access: vk::AccessFlags2::SHADER_WRITE,
                        },
                    )
                };

                Ok(())
            }
            .into()
        },
    )
}
