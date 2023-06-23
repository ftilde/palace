use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};

use crate::{
    data::Vector,
    operator::OperatorId,
    operators::tensor::TensorOperator,
    vulkan::{
        memory::TempRessource,
        pipeline::{DescriptorConfig, GraphicsPipeline},
        shader::ShaderDefines,
        state::RessourceId,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::volume::VolumeOperator;

pub fn gui<'a>(input: VolumeOperator<'a>) -> VolumeOperator<'a> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        out_mem_dim: cgmath::Vector2<u32>,
    }

    const VERTEX_SHADER: &str = "
#version 450

vec2 positions[3] = vec2[](
    vec2(-1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(0.0, 0.0)
);

layout(location = 0) out vec2 norm_pos;

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    norm_pos = positions[gl_VertexIndex];
}
";

    const FRAG_SHADER: &str = "
#version 450

#include <util.glsl>

layout(std430, binding = 0) buffer OutputBuffer{
    float values[BRICK_MEM_SIZE];
} outputData;

layout(location = 0) in vec2 norm_pos;

declare_push_consts(consts);

void main() {
    uint n_channels = 4;
    uvec2 pos = uvec2(gl_FragCoord.xy);
    uint linear_pos = n_channels * to_linear2(pos, consts.out_mem_dim);

    vec4 color = vec4(norm_pos, 0.0, 1.0);
    for(int c=0; c<4; ++c) {
        outputData.values[linear_pos + c] = color[c];
    }
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

                let render_pass = device.request_state(
                    RessourceId::new("renderpass").of(ctx.current_op()),
                    || {
                        let subpass = vk::SubpassDescription::builder()
                            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                            .color_attachments(&[]);

                        let dependency_info = vk::SubpassDependency::builder()
                            .src_subpass(vk::SUBPASS_EXTERNAL)
                            .dst_subpass(0)
                            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                            .src_access_mask(vk::AccessFlags::empty())
                            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

                        let subpasses = &[*subpass];
                        let dependency_infos = &[*dependency_info];
                        let render_pass_info = vk::RenderPassCreateInfo::builder()
                            .attachments(&[])
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
                            (
                                FRAG_SHADER,
                                ShaderDefines::new()
                                    .push_const_block::<PushConstants>()
                                    .add("BRICK_MEM_SIZE", out_info.mem_elements()),
                            ),
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

                let out_dim = out_info.logical_dimensions.drop_dim(2);
                let width = out_dim.x().into();
                let height = out_dim.y().into();
                let extent = vk::Extent2D::builder().width(width).height(height).build();

                let framebuffer_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(*render_pass)
                    .attachments(&[])
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

                let gpu_brick_out = match &inplace_result {
                    crate::storage::gpu::InplaceResult::Inplace(rw, _) => rw,
                    crate::storage::gpu::InplaceResult::New(r, w) => {
                        let copy_info = vk::BufferCopy::builder().size(r.layout.size() as u64);
                        device.with_cmd_buffer(|cmd| unsafe {
                            device.functions().cmd_copy_buffer(
                                cmd.raw(),
                                r.buffer,
                                w.buffer,
                                &[*copy_info],
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

                        w
                    }
                };

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

                // Setup the viewport such that only the content of the current tile is rendered
                let offset = out_info.begin().drop_dim(2).map(|v| v.raw as f32);
                let full_size = m_out.dimensions.drop_dim(2).map(|v| v.raw as f32);
                let tile_size = out_info
                    .logical_dimensions
                    .drop_dim(2)
                    .map(|v| v.raw as f32);
                let scale_factor = full_size / tile_size;
                let size = tile_size * scale_factor;
                let viewport = vk::Viewport::builder()
                    .x(-offset.x())
                    .y(-offset.y())
                    .width(size.x())
                    .height(size.y())
                    .min_depth(0.0)
                    .max_depth(1.0);

                let scissor = vk::Rect2D::builder()
                    .offset(vk::Offset2D::builder().x(0).y(0).build())
                    .extent(extent);

                let push_constants = PushConstants {
                    out_mem_dim: out_info
                        .mem_dimensions
                        .drop_dim(2)
                        .try_into_elem()
                        .unwrap()
                        .into(),
                };
                let descriptor_config = DescriptorConfig::new([gpu_brick_out]);

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

                    pipeline.push_constant_at(push_constants, vk::ShaderStageFlags::FRAGMENT);

                    device.functions().cmd_draw(cmd.raw(), 3, 1, 0, 0);

                    device.functions().cmd_end_render_pass(cmd.raw());
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
