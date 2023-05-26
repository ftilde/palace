use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};

use crate::{
    array::{ImageMetaData, VolumeMetaData},
    operator::OperatorId,
    operators::tensor::TensorOperator,
    vulkan::{
        pipeline::{DescriptorConfig, GraphicsPipeline},
        shader::ShaderDefines,
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
    vec3(1.0, 0.0, 0.0), // Front-top-left
    vec3(0.0, 0.0, 0.0), // Front-top-right
    vec3(1.0, 1.0, 0.0), // Front-bottom-left
    vec3(0.0, 1.0, 0.0), // Front-bottom-right
    vec3(0.0, 1.0, 1.0), // Back-bottom-right
    vec3(0.0, 0.0, 0.0), // Front-top-right
    vec3(0.0, 0.0, 1.0), // Back-top-right
    vec3(1.0, 0.0, 0.0), // Front-top-left
    vec3(1.0, 0.0, 1.0), // Back-top-left
    vec3(1.0, 1.0, 0.0), // Front-bottom-left
    vec3(1.0, 1.0, 1.0), // Back-bottom-left
    vec3(0.0, 1.0, 1.0), // Back-bottom-right
    vec3(1.0, 0.0, 1.0), // Back-top-left
    vec3(0.0, 0.0, 1.0)  // Back-top-right
);

layout(location = 0) out vec3 norm_pos;

void main() {
    gl_Position = projection.value * vec4(positions[gl_VertexIndex], 1.0);
    norm_pos = positions[gl_VertexIndex];
}
";

    const FRAG_SHADER: &str = "
#version 450

layout(location = 0) out vec4 out_color;
layout(location = 0) in vec3 norm_pos;

void main() {
    out_color = vec4(norm_pos, 1.0);
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
                            &render_pass,
                            vk::PrimitiveTopology::TRIANGLE_STRIP,
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

                let img = device.tmp_buffers.request_image(device, img_info);

                let info = vk::ImageViewCreateInfo::builder()
                    .image(img.allocation.image)
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
                let img_view =
                    unsafe { device.functions().create_image_view(&info, None) }.unwrap();

                let attachments = [img_view];
                let framebuffer_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(*render_pass)
                    .attachments(&attachments)
                    .width(width)
                    .height(height)
                    .layers(1);

                let framebuffer = unsafe {
                    device
                        .functions()
                        .create_framebuffer(&framebuffer_info, None)
                }
                .unwrap();

                // Actual rendering
                let clear_values = [vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                }];

                let render_pass_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(*render_pass)
                    .framebuffer(framebuffer)
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

                    device.functions().cmd_draw(cmd.raw(), 12, 1, 0, 0);

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
                        img.allocation.image,
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

                // TODO NO_PUSH_main: remove
                ctx.submit(device.wait_for_current_cmd_buffer_submission())
                    .await;

                // TODO NO_PUSH_main: Not sure if this flies
                unsafe { device.functions().destroy_framebuffer(framebuffer, None) };

                unsafe { device.functions().destroy_image_view(img_view, None) };

                unsafe { device.tmp_buffers.return_image(device, img) };

                Ok(())
            }
            .into()
        },
    )
}
