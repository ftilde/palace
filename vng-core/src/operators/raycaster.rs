use std::alloc::Layout;

use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};

use crate::{
    array::{ImageMetaData, VolumeMetaData},
    data::{from_linear, hmul, Vector},
    operator::{OpaqueOperator, OperatorId},
    operators::tensor::TensorOperator,
    storage::{gpu::Allocation, DataVersionType},
    task::OpaqueTaskContext,
    vulkan::{
        memory::TempRessource,
        pipeline::{ComputePipeline, DescriptorConfig, GraphicsPipeline},
        shader::ShaderDefines,
        state::{RessourceId, VulkanState},
        CommandBuffer, DeviceContext, DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::{scalar::ScalarOperator, volume::VolumeOperator};

pub fn entry_exit_points<'a>(
    input_metadata: ScalarOperator<'a, VolumeMetaData>,
    result_metadata: ScalarOperator<'a, ImageMetaData>,
    projection_mat: ScalarOperator<'a, cgmath::Matrix4<f32>>,
) -> VolumeOperator<'a> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        size: cgmath::Vector2<u32>,
    }

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

#include <util.glsl>

layout(std430, binding = 1) buffer OutputBuffer{
    float values[BRICK_MEM_SIZE];
} outputData;

layout(location = 0) in vec3 norm_pos;

declare_push_consts(consts);

void main() {
    uint n_channels = 8;
    uvec2 pos = uvec2(gl_FragCoord.xy);
    uint linear_pos = n_channels * to_linear2(pos, consts.size);

    vec4 color = vec4(norm_pos, 1.0);
    if(gl_FrontFacing) {
        for(int c=0; c<4; ++c) {
            outputData.values[linear_pos + c] = color[c];
        }
    } else {
        for(int c=0; c<4; ++c) {
            outputData.values[linear_pos + c + 4] = color[c];
        }
    }
}
";
    const N_CHANNELS: u32 = 8;
    fn full_info(r: ImageMetaData) -> VolumeMetaData {
        VolumeMetaData {
            dimensions: [r.dimensions.y(), r.dimensions.x(), N_CHANNELS.into()].into(),
            chunk_size: [r.chunk_size.y(), r.chunk_size.x(), N_CHANNELS.into()].into(),
        }
    }

    TensorOperator::unbatched(
        OperatorId::new("entry_exit_points")
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
                    stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };

                let (m2d, transform_gpu) = futures::join! {
                    ctx.submit(result_metadata.request_scalar()),
                    ctx.submit(projection_mat.request_scalar_gpu(device.id, dst_info)),
                };
                let m_out = full_info(m2d);
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

                // Actual rendering
                let gpu_brick_out = ctx
                    .alloc_slot_gpu(device, pos, out_info.mem_elements())
                    .unwrap();

                device.with_cmd_buffer(|cmd| {
                    unsafe {
                        device.functions().cmd_fill_buffer(
                            cmd.raw(),
                            gpu_brick_out.buffer,
                            0,
                            gpu_brick_out.size,
                            0x0,
                        )
                    };
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

                let push_constants = PushConstants {
                    size: m_out.dimensions.drop_dim(2).try_into_elem().unwrap().into(),
                };
                let descriptor_config = DescriptorConfig::new([&transform_gpu, &gpu_brick_out]);

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

                    device.functions().cmd_draw(cmd.raw(), 14, 1, 0, 0);

                    device.functions().cmd_end_render_pass(cmd.raw());
                });

                unsafe {
                    gpu_brick_out.initialized(
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

pub struct BrickRequestTable {
    num_elements: usize,
    allocation: Allocation,
    layout: Layout,
}

impl BrickRequestTable {
    // Note: a barrier is needed after initialization to make values visible
    pub fn new(num_elements: usize, device: &DeviceContext) -> Self {
        let request_table_buffer_layout = Layout::array::<u64>(num_elements).unwrap();
        let flags = vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::STORAGE_BUFFER;
        let buf_type = gpu_allocator::MemoryLocation::GpuOnly;
        let allocation =
            device
                .storage
                .allocate(device, request_table_buffer_layout, flags, buf_type);

        let ret = BrickRequestTable {
            num_elements,
            allocation,
            layout: request_table_buffer_layout,
        };

        device.with_cmd_buffer(|cmd| ret.clear(cmd));

        ret
    }

    // Note: a barrier is needed after clearing to make values visible
    pub fn clear(&self, cmd: &mut CommandBuffer) {
        unsafe {
            cmd.functions().cmd_fill_buffer(
                cmd.raw(),
                self.allocation.buffer,
                0,
                vk::WHOLE_SIZE,
                0xffffffff,
            )
        };
    }

    pub fn buffer(&self) -> &Allocation {
        &self.allocation
    }

    // Note: any changes to the buffer have to be made visible to the cpu side via a barrier first
    pub async fn download_requested<'cref, 'inv>(
        &self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        device: &'cref DeviceContext,
    ) -> Vec<u64> {
        let mut request_table_cpu = vec![0u64; self.num_elements];
        let request_table_cpu_bytes = bytemuck::cast_slice_mut(request_table_cpu.as_mut_slice());
        unsafe {
            crate::vulkan::memory::copy_to_cpu(
                ctx,
                device,
                self.allocation.buffer,
                self.layout,
                request_table_cpu_bytes.as_mut_ptr(),
            )
            .await
        };

        let to_request_linear = request_table_cpu
            .into_iter()
            .filter(|v| *v != u64::max_value())
            .collect::<Vec<u64>>();

        to_request_linear
    }
}

impl VulkanState for BrickRequestTable {
    unsafe fn deinitialize(&mut self, context: &DeviceContext) {
        self.allocation.deinitialize(context);
    }
}

pub fn raycast<'a>(
    input: VolumeOperator<'a>,
    entry_exit_points: VolumeOperator<'a>,
) -> VolumeOperator<'a> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        out_mem_dim: cgmath::Vector2<u32>,
        dimensions: cgmath::Vector3<u32>,
        chunk_size: cgmath::Vector3<u32>,
    }
    const SHADER: &'static str = r#"
#version 450

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require

#include <util.glsl>
#include <hash.glsl>
#include <sample.glsl>

layout (local_size_x = 32, local_size_y = 32) in;

layout(std430, binding = 0) buffer OutputBuffer{
    float values[];
} output_data;

layout(std430, binding = 1) buffer EntryExitPoints{
    float values[];
} entry_exit_points;

layout(std430, binding = 2) buffer RefBuffer {
    BrickType values[NUM_BRICKS];
} bricks;

layout(std430, binding = 3) buffer QueryTable {
    uint64_t values[REQUEST_TABLE_SIZE];
} request_table;

struct State {
    uint iteration;
    float intensity;
};

layout(std430, binding = 4) buffer StateBuffer {
    State values[];
} state_cache;

declare_push_consts(consts);

void main()
{
    uvec2 out_pos = gl_GlobalInvocationID.xy;
    uint gID = out_pos.x + out_pos.y * consts.out_mem_dim.x;
    if(out_pos.x < consts.out_mem_dim.x && out_pos.y < consts.out_mem_dim.y) {
        vec4 entry_point;
        vec4 exit_point;

        for(int c=0; c<4; ++c) {
            entry_point[c] = entry_exit_points.values[8*gID+c];
            exit_point[c] = entry_exit_points.values[8*gID+c+4];
        }

        VolumeMetaData m_in;
        m_in.dimensions = consts.dimensions;
        m_in.chunk_size = consts.chunk_size;


        vec4 color;
        if(entry_point.a > 0.0 && exit_point.a > 0.0) {

            State state = state_cache.values[gID];

            uint sample_points = 100;
            for(; state.iteration < sample_points; ++state.iteration) {
                float a = float(state.iteration)/float(sample_points-1);
                vec3 p = a * exit_point.xyz + (1-a) * entry_point.xyz;

                uvec3 pos_voxel = uvec3(round(p * vec3(m_in.dimensions - uvec3(1))));

                bool found;
                uint sample_brick_pos_linear;
                float sampled_intensity;
                try_sample(pos_voxel, m_in, bricks.values, found, sample_brick_pos_linear, sampled_intensity);
                if(found) {
                    state.intensity = max(state.intensity, sampled_intensity);
                } else {
                    uint64_t sbp = uint64_t(sample_brick_pos_linear);
                    try_insert_into_hash_table(request_table.values, REQUEST_TABLE_SIZE, sample_brick_pos_linear);
                    break;
                }
            }
            state_cache.values[gID] = state;

            color = vec4(vec3(state.intensity), 1.0);
        } else {
            color = vec4(0.0);
        }

        for(int c=0; c<4; ++c) {
            output_data.values[4*gID+c] = color[c];
        }
    }
}
"#;
    const N_CHANNELS: u32 = 4;
    fn full_info(r: VolumeMetaData) -> VolumeMetaData {
        assert_eq!(r.dimensions.x().raw, 2 * N_CHANNELS);
        VolumeMetaData {
            dimensions: [r.dimensions.z(), r.dimensions.y(), N_CHANNELS.into()].into(),
            chunk_size: [r.chunk_size.z(), r.chunk_size.y(), N_CHANNELS.into()].into(),
        }
    }

    TensorOperator::unbatched(
        OperatorId::new("raycast")
            .dependent_on(&input)
            .dependent_on(&entry_exit_points),
        entry_exit_points.clone(),
        (input, entry_exit_points),
        move |ctx, entry_exit_points, _| {
            async move {
                let r = ctx
                    .submit(entry_exit_points.metadata.request_scalar())
                    .await;
                let m = full_info(r);
                ctx.write(m)
            }
            .into()
        },
        move |ctx, pos, (input, entry_exit_points), _| {
            async move {
                let device = ctx.vulkan_device();

                let dst_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };

                let (m_in, im, eep) = futures::join! {
                    ctx.submit(input.metadata.request_scalar()),
                    ctx.submit(entry_exit_points.metadata.request_scalar()),
                    ctx.submit(entry_exit_points.bricks.request_gpu(device.id, Vector::fill(0.into()), dst_info)),
                };
                let m_out = full_info(im);
                let out_info = m_out.chunk_info(pos);

                let request_table_size = 256;
                let request_batch_size = 32;

                let num_bricks = hmul(m_in.dimension_in_bricks());

                let brick_index = device
                    .storage
                    .get_index(*ctx, device, input.bricks.id(), num_bricks, dst_info)
                    .await;

                let pipeline =
                    device.request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new().push_const_block::<PushConstants>()
                                    .add("BRICK_MEM_SIZE", hmul(m_out.chunk_size))
                                    .add("NUM_BRICKS", num_bricks)
                                    .add("REQUEST_TABLE_SIZE", request_table_size),
                            ),
                            false,
                        )
                    });

                let state_initialized = ctx
                    .access_state_cache(
                        device,
                        pos,
                        "initialized",
                        Layout::array::<(u32, f32)>(hmul(im.chunk_size)).unwrap(),
                    )
                    .unwrap();
                let state_initialized = state_initialized.init(|v| {
                    device.with_cmd_buffer(|cmd| unsafe {
                        device.functions().cmd_fill_buffer(
                            cmd.raw(),
                            v.buffer,
                            0,
                            vk::WHOLE_SIZE,
                            0,
                        );
                    });
                });

                let request_table = TempRessource::new(device, BrickRequestTable::new(request_table_size, device));

                let dim_in_bricks = m_in.dimension_in_bricks();

                let chunk_size = out_info.mem_dimensions.drop_dim(2).raw();
                let consts = PushConstants {
                    out_mem_dim: chunk_size.into(),
                    dimensions: m_in.dimensions.raw().into(),
                    chunk_size: m_in.chunk_size.raw().into(),
                };

                // Actual rendering
                let gpu_brick_out = ctx
                    .alloc_slot_gpu(device, pos, out_info.mem_elements())
                    .unwrap();
                let mut it = 1;
                let timed_out = 'outer: loop {

                    // Make writes to the request table visible (including initialization)
                    ctx.submit(device.barrier(
                        SrcBarrierInfo {
                            stage: vk::PipelineStageFlags2::TRANSFER,
                            access: vk::AccessFlags2::TRANSFER_WRITE,
                        },
                        DstBarrierInfo {
                            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                        },
                    ))
                    .await;


                    // Now first try a render pass to collect bricks to load (or just to finish the
                    // rendering
                    let global_size = [1, chunk_size.y(), chunk_size.x()].into();

                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config = DescriptorConfig::new([&gpu_brick_out, &eep,
                            &brick_index, request_table.buffer(), &state_initialized]);

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            pipeline.push_constant(consts);
                            pipeline.write_descriptor_set(0, descriptor_config);
                            pipeline.dispatch3d(global_size);
                        }
                    });

                    // Make requests visible
                    ctx.submit(device.barrier(SrcBarrierInfo {
                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::SHADER_WRITE,
                    }, DstBarrierInfo {
                        stage: vk::PipelineStageFlags2::TRANSFER,
                        access: vk::AccessFlags2::TRANSFER_READ,
                    })).await;

                    let mut to_request_linear = request_table.download_requested(*ctx, device).await;

                    if to_request_linear.is_empty() {
                        break false;
                    }

                    // Fulfill requests
                    to_request_linear.sort_unstable();

                    for batch in to_request_linear.chunks(request_batch_size) {
                        let to_request = batch.iter().map(|v| {
                            assert!(*v < num_bricks as _);
                            input.bricks.request_gpu(
                                device.id,
                                from_linear(*v as usize, dim_in_bricks),
                                DstBarrierInfo {
                                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                    access: vk::AccessFlags2::SHADER_READ,
                                },
                            )
                        });
                        let requested_bricks = ctx.submit(ctx.group(to_request)).await;

                        for (brick, brick_linear_pos) in requested_bricks
                            .into_iter()
                            .zip(batch.into_iter())
                        {
                            brick_index.insert(*brick_linear_pos, brick);
                        }

                        if ctx.past_deadline() {
                            break 'outer true;
                        }
                    }

                    // Clear request table for the next iteration
                    device.with_cmd_buffer(|cmd| request_table.clear(cmd));

                    it += 1;
                };

                let src_info = SrcBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_WRITE,
                };
                if timed_out {
                    unsafe {
                        println!("Raycaster: Time out result after {} it", it);
                        gpu_brick_out.initialized_version(*ctx, src_info, DataVersionType::Preview)
                    };
                } else {
                    unsafe { gpu_brick_out.initialized(*ctx, src_info) };
                }

                Ok(())
            }
            .into()
        },
    )
}
