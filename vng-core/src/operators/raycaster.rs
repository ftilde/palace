use ash::vk;

use crate::{
    array::{ImageMetaData, VolumeMetaData},
    operator::OperatorId,
    operators::tensor::TensorOperator,
    vulkan::{
        memory::TempRessource,
        pipeline::{DescriptorConfig, GraphicsPipeline},
        state::{RessourceId, VulkanState},
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::{scalar::ScalarOperator, volume::VolumeOperator};

//TODO NO_PUSH_main move
impl VulkanState for vk::RenderPass {
    unsafe fn deinitialize(&mut self, context: &crate::vulkan::DeviceContext) {
        unsafe { context.functions().destroy_render_pass(*self, None) };
    }
}

pub fn render_entry_exit<'a>(
    input_metadata: ScalarOperator<'a, VolumeMetaData>,
    result_metadata: ScalarOperator<'a, ImageMetaData>,
    projection_mat: ScalarOperator<'a, cgmath::Matrix4<f32>>,
) -> VolumeOperator<'a> {
    //#[derive(Copy, Clone, AsStd140, GlslStruct)]
    //struct PushConstants {
    //    size: cgmath::Vector2<u32>,
    //}

    const VERTEX_SHADER: &str = "
#version 450

layout(std430, binding = 0) buffer Transform {
    mat4 value;
} projection;

vec3 positions[14] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(1.0, 0.0, 1.0),
    vec3(0.0, 0.0, 0.0),
    vec3(0.0, 0.0, 1.0),
    vec3(0.0, 1.0, 1.0),
    vec3(1.0, 0.0, 1.0),
    vec3(1.0, 1.0, 1.0),
    vec3(1.0, 0.0, 0.0),
    vec3(1.0, 1.0, 0.0),
    vec3(0.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 1.0, 1.0),
    vec3(1.0, 1.0, 0.0),
    vec3(1.0, 1.0, 1.0)
);

layout(location = 0) out vec3 norm_pos;

void main() {
    gl_Position = projection.value * vec4(positions[gl_VertexIndex], 1.0);
    norm_pos = positions[gl_VertexIndex];
}
";

    const FRAG_SHADER: &str = "
#version 450

layout(location = 0) out vec4 out_color_front;
layout(location = 1) out vec4 out_color_back;

layout(location = 0) in vec3 norm_pos;

void main() {
    if(gl_FrontFacing) {
        out_color_front = vec4(norm_pos, 1.0);
        out_color_back = vec4(0.0);
    } else {
        out_color_back = vec4(norm_pos, 1.0);
        out_color_front = vec4(0.0);
    }
}
";
    const N_CHANNELS: u32 = 4;
    fn full_info(r: ImageMetaData) -> VolumeMetaData {
        VolumeMetaData {
            dimensions: [r.dimensions.y(), r.dimensions.x(), N_CHANNELS.into()].into(),
            chunk_size: [r.chunk_size.y(), r.chunk_size.x(), N_CHANNELS.into()].into(),
        }
    }

    TensorOperator::unbatched(
        OperatorId::new("render_entry_exit")
            .dependent_on(&input_metadata)
            .dependent_on(&result_metadata)
            .dependent_on(&projection_mat),
        result_metadata.clone(),
        (input_metadata, result_metadata, projection_mat),
        move |ctx, result_metadata, _| {
            async move {
                let r = ctx.submit(result_metadata.request_scalar()).await;
                let m = full_info(r);
                ctx.write(m)
            }
            .into()
        },
        move |ctx, pos, (_m_in, result_metadata, projection_mat), _| {
            async move {
                let device = ctx.vulkan_device();

                let dst_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };

                let format = vk::Format::R32G32B32A32_SFLOAT;
                let render_pass = device.request_state(
                    RessourceId::new("renderpass").of(ctx.current_op()),
                    || {
                        let color_attachment = vk::AttachmentDescription::builder()
                            .format(format)
                            .samples(vk::SampleCountFlags::TYPE_1)
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE)
                            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                            .initial_layout(vk::ImageLayout::UNDEFINED)
                            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);

                        let color_attachment_front_ref = vk::AttachmentReference::builder()
                            .attachment(0)
                            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

                        let color_attachment_back_ref = vk::AttachmentReference::builder()
                            .attachment(1)
                            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

                        let color_attachment_refs =
                            &[*color_attachment_front_ref, *color_attachment_back_ref];
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

                        let attachments = &[*color_attachment, *color_attachment];
                        let subpasses = &[*subpass];
                        let dependency_infos = &[*dependency_info];
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
                            VERTEX_SHADER,
                            //(
                            FRAG_SHADER,
                            //ShaderDefines::new().push_const_block::<PushConstants>(),
                            //),
                            |shader_stages, pipeline_layout, build_pipeline| {
                                let dynamic_states =
                                    [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
                                let dynamic_info = vk::PipelineDynamicStateCreateInfo::builder()
                                    .dynamic_states(&dynamic_states);

                                let vertex_input_info =
                                    vk::PipelineVertexInputStateCreateInfo::builder();

                                let input_assembly_info =
                                    vk::PipelineInputAssemblyStateCreateInfo::builder()
                                        .primitive_restart_enable(false)
                                        .topology(vk::PrimitiveTopology::TRIANGLE_STRIP);

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
                                        .front_face(vk::FrontFace::CLOCKWISE)
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
                                        .dst_color_blend_factor(vk::BlendFactor::ONE)
                                        .color_blend_op(vk::BlendOp::ADD)
                                        .src_alpha_blend_factor(vk::BlendFactor::ONE)
                                        .dst_alpha_blend_factor(vk::BlendFactor::ONE)
                                        .alpha_blend_op(vk::BlendOp::ADD)
                                        .blend_enable(true);

                                let color_blend_attachments =
                                    [*color_blend_attachment, *color_blend_attachment];
                                let color_blending =
                                    vk::PipelineColorBlendStateCreateInfo::builder()
                                        .logic_op_enable(false)
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

                let (m2d, transform_gpu) = futures::join! {
                    ctx.submit(result_metadata.request_scalar()),
                    ctx.submit(projection_mat.request_scalar_gpu(device.id, dst_info)),
                };
                let m_out = full_info(m2d);
                let out_info = m_out.chunk_info(pos);

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
                    .initial_layout(vk::ImageLayout::UNDEFINED) //TODO ???
                    .usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                    )
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build();

                // TODO: Make renderpass a vulkanstate

                let img_front =
                    TempRessource::new(device, device.storage.allocate_image(device, img_info));
                let img_back =
                    TempRessource::new(device, device.storage.allocate_image(device, img_info));

                let info = vk::ImageViewCreateInfo::builder()
                    .image(img_front.image)
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
                let img_view_front = TempRessource::new(
                    device,
                    unsafe { device.functions().create_image_view(&info, None) }.unwrap(),
                );
                let info = vk::ImageViewCreateInfo::builder()
                    .image(img_back.image)
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
                let img_view_back = TempRessource::new(
                    device,
                    unsafe { device.functions().create_image_view(&info, None) }.unwrap(),
                );

                let attachments = [*img_view_front, *img_view_back];
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

                // Actual rendering
                let clear_value = vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                };
                let clear_values = [clear_value, clear_value];

                let render_pass_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(*render_pass)
                    .framebuffer(*framebuffer)
                    .render_area(
                        vk::Rect2D::builder()
                            .offset(vk::Offset2D::builder().x(0).y(0).build())
                            .extent(extent)
                            .build(),
                    )
                    .clear_values(&clear_values);

                let viewport = vk::Viewport::builder()
                    .x(0.0)
                    .y(0.0)
                    .width(width as _)
                    .height(height as _)
                    .min_depth(0.0)
                    .max_depth(1.0);

                let scissor = vk::Rect2D::builder()
                    .offset(vk::Offset2D::builder().x(0).y(0).build())
                    .extent(extent);

                //let push_constants = PushConstants {
                //    size: m_out.dimensions.drop_dim(2).try_into_elem().unwrap().into(),
                //};
                let descriptor_config = DescriptorConfig::new([&transform_gpu]);

                device.with_cmd_buffer(|cmd| unsafe {
                    let mut pipeline = pipeline.bind(cmd);

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

                    pipeline.push_descriptor_set(0, descriptor_config);

                    //pipeline.push_constant_at(push_constants, vk::ShaderStageFlags::FRAGMENT);

                    device.functions().cmd_draw(cmd.raw(), 14, 1, 0, 0);

                    device.functions().cmd_end_render_pass(cmd.raw());
                });

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

                let gpu_brick_out = ctx
                    .alloc_slot_gpu(device, pos, out_info.mem_elements())
                    .unwrap();

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
                        img_front.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL, /*???*/
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
