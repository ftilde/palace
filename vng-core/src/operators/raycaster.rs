use std::alloc::Layout;

use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};

use crate::{
    array::{ImageMetaData, VolumeEmbeddingData, VolumeMetaData},
    chunk_utils::ChunkRequestTable,
    data::{from_linear, hmul, GlobalCoordinate, Matrix, Vector},
    operator::{OpaqueOperator, OperatorId},
    operators::tensor::TensorOperator,
    storage::DataVersionType,
    vulkan::{
        memory::TempRessource,
        pipeline::{ComputePipeline, DescriptorConfig, GraphicsPipeline},
        shader::ShaderDefines,
        state::RessourceId,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

#[cfg(feature = "python")]
use pyo3::prelude::*;

use super::{
    scalar::ScalarOperator,
    volume::{EmbeddedVolumeOperator, VolumeOperator},
};

#[cfg_attr(feature = "python", pyclass)]
#[derive(state_link::State, Clone)]
pub struct TrackballState {
    #[pyo3(get, set)]
    pub eye: Vector<3, f32>,
    #[pyo3(get, set)]
    pub center: Vector<3, f32>,
    #[pyo3(get, set)]
    pub up: Vector<3, f32>,
}

#[cfg_attr(feature = "python", pymethods)]
impl TrackballState {
    #[new]
    pub fn new(eye: Vector<3, f32>, center: Vector<3, f32>, up: Vector<3, f32>) -> Self {
        Self { eye, center, up }
    }

    fn store(&self, py: pyo3::Python, store: Py<::state_link::py::Store>) -> pyo3::PyObject {
        self.store_py(py, store)
    }

    pub fn pan_around(&mut self, delta: Vector<2, i32>) {
        let look = self.center - self.eye;
        let look_len = look.length();
        let left = self.up.cross(look).normalized();
        let move_factor = 0.005;
        let delta = delta.map(|v| v as f32 * move_factor);

        let new_look = (look.normalized() + self.up.scale(delta.y()) + left.scale(-delta.x()))
            .normalized()
            .scale(look_len);

        self.eye = self.center - new_look;
        let left = self.up.cross(new_look);
        self.up = new_look.cross(left).normalized();
    }
    pub fn move_inout(&mut self, delta: f32) {
        let look = self.center - self.eye;
        let new_look = look.scale(1.0 - delta * 0.1);
        self.eye = self.center - new_look;
    }
    pub fn view_mat(&self) -> Matrix<4, f32> {
        cgmath::Matrix4::look_at_rh(self.eye.into(), self.center.into(), self.up.into()).into()
    }
}

#[derive(Clone, state_link::State)]
#[cfg_attr(feature = "python", pyclass)]
pub struct CameraState {
    #[pyo3(get, set)]
    pub fov: f32,
    #[pyo3(get, set)]
    pub trackball: TrackballState,
}

#[cfg_attr(feature = "python", pymethods)]
impl CameraState {
    #[new]
    pub fn new(trackball: TrackballState, fov: f32) -> Self {
        Self { trackball, fov }
    }

    fn store(&self, py: pyo3::Python, store: Py<::state_link::py::Store>) -> pyo3::PyObject {
        self.store_py(py, store)
    }

    pub fn projection_mat(&self, size: Vector<2, GlobalCoordinate>) -> Matrix<4, f32> {
        let perspective: Matrix<4, f32> = cgmath::perspective(
            cgmath::Deg(self.fov),
            size.x().raw as f32 / size.y().raw as f32,
            0.001,
            100.0,
        )
        .into();
        let matrix = perspective * self.trackball.view_mat();
        matrix
    }
}

pub fn entry_exit_points(
    input_metadata: ScalarOperator<VolumeMetaData>,
    embedding_data: ScalarOperator<VolumeEmbeddingData>,
    result_metadata: ScalarOperator<ImageMetaData>,
    projection_mat: ScalarOperator<Matrix<4, f32>>,
) -> VolumeOperator {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        transform: cgmath::Matrix4<f32>,
        out_mem_dim: cgmath::Vector2<u32>,
    }

    const VERTEX_SHADER: &str = "
#version 450
#extension GL_EXT_scalar_block_layout : require

#include <mat.glsl>

declare_push_consts(consts);

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
    gl_Position = consts.transform * vec4(positions[gl_VertexIndex], 1.0);
    norm_pos = positions[gl_VertexIndex];
}
";

    const FRAG_SHADER: &str = "
#version 450

#include <util.glsl>

layout(std430, binding = 0) buffer OutputBuffer{
    float values[BRICK_MEM_SIZE];
} outputData;

layout(location = 0) in vec3 norm_pos;

declare_push_consts(consts);

void main() {
    uint n_channels = 8;
    uvec2 pos = uvec2(gl_FragCoord.xy);
    uint linear_pos = n_channels * to_linear2(pos, consts.out_mem_dim);

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
        (
            input_metadata,
            embedding_data,
            result_metadata,
            projection_mat,
        ),
        move |ctx, result_metadata| {
            async move {
                let r = ctx.submit(result_metadata.request_scalar()).await;
                let m = full_info(r);
                ctx.write(m)
            }
            .into()
        },
        move |ctx, pos, (m_in, embedding_data, result_metadata, projection_mat)| {
            async move {
                //TODO: Use spacing information of _m_in (or similar) here
                let device = ctx.vulkan_device();

                let (m_in, embedding_data, m2d, transform) = futures::join! {
                    ctx.submit(m_in.request_scalar()),
                    ctx.submit(embedding_data.request_scalar()),
                    ctx.submit(result_metadata.request_scalar()),
                    ctx.submit(projection_mat.request_scalar()),
                };
                let m_out = full_info(m2d);
                let out_info = m_out.chunk_info(pos);

                let norm_to_world = Matrix::from_scale(
                    m_in.dimensions.map(|v| v.raw as f32) * embedding_data.spacing,
                )
                .to_homogeneuous()
                    * Matrix::from_translation(Vector::fill(-0.5));
                let transform = transform * norm_to_world;

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
                            (
                                VERTEX_SHADER,
                                ShaderDefines::new().push_const_block::<PushConstants>(),
                            ),
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
                        access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
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
                    transform: transform.into(),
                };
                let descriptor_config = DescriptorConfig::new([&gpu_brick_out]);

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

                    pipeline.push_constant_at(
                        push_constants,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    );

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

pub fn raycast(input: EmbeddedVolumeOperator, entry_exit_points: VolumeOperator) -> VolumeOperator {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        out_mem_dim: cgmath::Vector2<u32>,
        dimensions: cgmath::Vector3<u32>,
        chunk_size: cgmath::Vector3<u32>,
        spacing: cgmath::Vector3<f32>,
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
    uint values[REQUEST_TABLE_SIZE];
} request_table;

struct State {
    float t;
    float intensity;
};

layout(std430, binding = 4) buffer StateBuffer {
    State values[];
} state_cache;

declare_push_consts(consts);

vec4 map_to_color(float v) {
    v = clamp(v, 0.0, 1.0);
    return vec4(v, v, v, 1.0);
}

struct EEPoint {
    vec3 entry;
    vec3 exit;
};

bool sample_ee(uvec2 pos, out EEPoint eep) {

    if(pos.x >= consts.out_mem_dim.x || pos.y >= consts.out_mem_dim.y) {
        return false;
    }

    uint gID = pos.x + pos.y * consts.out_mem_dim.x;

    vec4 entry;
    vec4 exit;
    for(int c=0; c<4; ++c) {
        entry[c] = entry_exit_points.values[8*gID+c];
        exit[c] = entry_exit_points.values[8*gID+c+4];
    }
    eep.entry = entry.xyz * consts.spacing;
    eep.exit = exit.xyz * consts.spacing;

    return entry.a > 0.0 && exit.a > 0.0;
}

void main()
{
    uvec2 out_pos = gl_GlobalInvocationID.xy;
    uint gID = out_pos.x + out_pos.y * consts.out_mem_dim.x;

    EEPoint eep;
    bool valid = sample_ee(out_pos, eep);

    if(!(out_pos.x >= consts.out_mem_dim.x || out_pos.y >= consts.out_mem_dim.y)) {

        vec4 color;
        if(valid) {
            VolumeMetaData m_in;
            m_in.dimensions = consts.dimensions;
            m_in.chunk_size = consts.chunk_size;

            EEPoint eep_x;
            if(!sample_ee(out_pos + uvec2(1, 0), eep_x)) {
                if(!sample_ee(out_pos - uvec2(1, 0), eep_x)) {
                    eep_x.entry = vec3(0.0);
                    eep_x.exit = vec3(1.0);
                }
            }
            EEPoint eep_y;
            if(!sample_ee(out_pos + uvec2(0, 1), eep_y)) {
                if(!sample_ee(out_pos - uvec2(0, 1), eep_y)) {
                    eep_y.entry = vec3(0.0);
                    eep_y.exit = vec3(1.0);
                }
            }
            vec3 neigh_x = eep_x.entry;
            vec3 neigh_y = eep_y.entry;
            vec3 center = eep.entry;
            vec3 front = eep.exit - eep.entry;

            vec3 rough_dir_x = neigh_x - center;
            vec3 rough_dir_y = neigh_y - center;
            vec3 dir_x = normalize(cross(rough_dir_y, front));
            vec3 dir_y = normalize(cross(rough_dir_x, front));

            State state = state_cache.values[gID];

            uint sample_points = 100;

            vec3 start = eep.entry;
            vec3 end = eep.exit;
            float t_end = distance(start, end);
            vec3 dir = normalize(end - start);

            float start_pixel_dist = abs(dot(dir_x, eep_x.entry - eep.entry));
            float end_pixel_dist = abs(dot(dir_x, eep_x.exit - eep.exit));

            while(state.t <= t_end) {
                vec3 p = start + state.t*dir;

                float alpha = state.t/t_end;
                float pixel_dist = start_pixel_dist * (1.0-alpha) + end_pixel_dist;

                vec3 pos_voxel = round(p/consts.spacing * vec3(m_in.dimensions) - vec3(0.5));

                int res;
                uint sample_brick_pos_linear;
                float sampled_intensity;
                try_sample(pos_voxel, m_in, bricks.values, res, sample_brick_pos_linear, sampled_intensity);
                if(res == SAMPLE_RES_FOUND) {
                    state.intensity = max(state.intensity, sampled_intensity);
                } else if(res == SAMPLE_RES_NOT_PRESENT) {
                    try_insert_into_hash_table(request_table.values, REQUEST_TABLE_SIZE, sample_brick_pos_linear);
                    break;
                } else /*res == SAMPLE_RES_OUTSIDE*/ {
                    // Should only happen at the border of the volume due to rounding errors
                }

                state.t += pixel_dist;
            }
            state_cache.values[gID] = state;

            color = map_to_color(state.intensity);
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
        move |ctx, entry_exit_points| {
            async move {
                let r = ctx
                    .submit(entry_exit_points.metadata.request_scalar())
                    .await;
                let m = full_info(r);
                ctx.write(m)
            }
            .into()
        },
        move |ctx, pos, (input, entry_exit_points)| {
            async move {
                let device = ctx.vulkan_device();

                let dst_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };

                let (m_in, emd, im, eep) = futures::join! {
                    ctx.submit(input.metadata.request_scalar()),
                    ctx.submit(input.embedding_data.request_scalar()),
                    ctx.submit(entry_exit_points.metadata.request_scalar()),
                    ctx.submit(entry_exit_points.chunks.request_gpu(device.id, pos, dst_info)),
                };
                let m_out = full_info(im);
                let out_info = m_out.chunk_info(pos);

                let request_table_size = 256;
                let request_batch_size = 32;

                let num_bricks = hmul(m_in.dimension_in_chunks());

                let brick_index = device
                    .storage
                    .get_index(*ctx, device, input.chunks.id(), num_bricks, dst_info)
                    .await;

                let pipeline =
                    device.request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .push_const_block::<PushConstants>()
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

                let request_table =
                    TempRessource::new(device, ChunkRequestTable::new(request_table_size, device));

                let dim_in_bricks = m_in.dimension_in_chunks();

                let chunk_size = out_info.mem_dimensions.drop_dim(2).raw();
                let consts = PushConstants {
                    out_mem_dim: chunk_size.into(),
                    dimensions: m_in.dimensions.raw().into(),
                    chunk_size: m_in.chunk_size.raw().into(),
                    spacing: emd.spacing.into(),
                };

                // Actual rendering
                let gpu_brick_out = ctx
                    .alloc_slot_gpu(device, pos, out_info.mem_elements())
                    .unwrap();
                let global_size = [1, chunk_size.y(), chunk_size.x()].into();

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
                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config = DescriptorConfig::new([
                            &gpu_brick_out,
                            &eep,
                            &brick_index,
                            request_table.buffer(),
                            &state_initialized,
                        ]);

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            pipeline.push_constant(consts);
                            pipeline.write_descriptor_set(0, descriptor_config);
                            pipeline.dispatch3d(global_size);
                        }
                    });

                    // Make requests visible
                    ctx.submit(device.barrier(
                        SrcBarrierInfo {
                            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            access: vk::AccessFlags2::SHADER_WRITE,
                        },
                        DstBarrierInfo {
                            stage: vk::PipelineStageFlags2::TRANSFER,
                            access: vk::AccessFlags2::TRANSFER_READ,
                        },
                    ))
                    .await;

                    let mut to_request_linear =
                        request_table.download_requested(*ctx, device).await;

                    if to_request_linear.is_empty() {
                        break false;
                    }

                    // Fulfill requests
                    to_request_linear.sort_unstable();

                    for batch in to_request_linear.chunks(request_batch_size) {
                        let to_request = batch.iter().map(|v| {
                            assert!(*v < num_bricks as _);
                            input.chunks.request_gpu(
                                device.id,
                                from_linear(*v as usize, dim_in_bricks),
                                DstBarrierInfo {
                                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                    access: vk::AccessFlags2::SHADER_READ,
                                },
                            )
                        });
                        let requested_bricks = ctx.submit(ctx.group(to_request)).await;

                        for (brick, brick_linear_pos) in
                            requested_bricks.into_iter().zip(batch.into_iter())
                        {
                            brick_index.insert(*brick_linear_pos as u64, brick);
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
