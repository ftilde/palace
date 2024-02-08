use std::alloc::Layout;

use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};

use crate::{
    array::{ImageMetaData, VolumeEmbeddingData, VolumeMetaData},
    chunk_utils::ChunkRequestTable,
    data::{GlobalCoordinate, Matrix, Vector},
    dim::*,
    id::{Id, Identify},
    operator::{OpaqueOperator, OperatorDescriptor},
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
    tensor::{FrameOperator, ImageOperator},
    volume::LODVolumeOperator,
};

#[cfg_attr(feature = "python", pyclass)]
#[derive(state_link::State, Clone)]
pub struct TrackballState {
    #[pyo3(get, set)]
    pub eye: Vector<D3, f32>, //Relative to center
    #[pyo3(get, set)]
    pub center: Vector<D3, f32>,
    #[pyo3(get, set)]
    pub up: Vector<D3, f32>,
}

#[cfg_attr(feature = "python", pymethods)]
impl TrackballState {
    #[new]
    pub fn new(eye: Vector<D3, f32>, center: Vector<D3, f32>, up: Vector<D3, f32>) -> Self {
        Self { eye, center, up }
    }

    fn store(&self, py: pyo3::Python, store: Py<::state_link::py::Store>) -> pyo3::PyObject {
        self.store_py(py, store)
    }

    pub fn pan_around(&mut self, delta: Vector<D2, i32>) {
        let look = -self.eye;
        let look_len = look.length();
        let left = self.up.cross(look).normalized();
        let move_factor = 0.005;
        let delta = delta.map(|v| v as f32 * move_factor);

        let new_look = (look.normalized() + self.up.scale(delta.y()) + left.scale(-delta.x()))
            .normalized()
            .scale(look_len);

        self.eye = -new_look;
        let left = self.up.cross(new_look);
        self.up = new_look.cross(left).normalized();
    }
    pub fn move_inout(&mut self, delta: f32) {
        let look = -self.eye;
        let new_look = look.scale(1.0 - delta * 0.1);
        self.eye = -new_look;
    }
    pub fn view_mat(&self) -> Matrix<D4, f32> {
        cgmath::Matrix4::look_at_rh(
            (self.eye + self.center).into(),
            self.center.into(),
            self.up.into(),
        )
        .into()
    }
}

#[derive(Clone, state_link::State)]
#[cfg_attr(feature = "python", pyclass)]
pub struct CameraState {
    #[pyo3(get, set)]
    pub fov: f32,
    #[pyo3(get, set)]
    pub trackball: TrackballState,
    #[pyo3(get, set)]
    pub near_plane: f32,
    #[pyo3(get, set)]
    pub far_plane: f32,
}

#[cfg_attr(feature = "python", pymethods)]
impl CameraState {
    #[new]
    pub fn new(trackball: TrackballState, fov: f32, near_plane: f32, far_plane: f32) -> Self {
        Self {
            trackball,
            fov,
            near_plane,
            far_plane,
        }
    }

    #[staticmethod]
    pub fn for_volume(
        input_metadata: VolumeMetaData,
        embedding_data: VolumeEmbeddingData,
        fov: f32,
    ) -> Self {
        let real_size = embedding_data.spacing * input_metadata.dimensions.raw().f32();
        let dist = real_size.length() * 1.5;

        let eye = [dist, 0.0, 0.0].into();
        let center = real_size.scale(0.5);
        let up = [0.0, 1.0, 0.0].into();

        let near_plane = embedding_data.spacing.hmin();
        let far_plane = 20.0 * dist;

        let trackball = TrackballState { eye, center, up };
        Self {
            trackball,
            fov,
            near_plane,
            far_plane,
        }
    }

    fn store(&self, py: pyo3::Python, store: Py<::state_link::py::Store>) -> pyo3::PyObject {
        self.store_py(py, store)
    }

    pub fn projection_mat(&self, size: Vector<D2, GlobalCoordinate>) -> Matrix<D4, f32> {
        let perspective: Matrix<D4, f32> = cgmath::perspective(
            cgmath::Deg(self.fov),
            size.x().raw as f32 / size.y().raw as f32,
            self.near_plane,
            self.far_plane,
        )
        .into();
        let matrix = perspective * self.trackball.view_mat();
        matrix
    }
}

pub fn entry_exit_points(
    input_metadata: VolumeMetaData,
    embedding_data: VolumeEmbeddingData,
    result_metadata: ImageMetaData,
    projection_mat: Matrix<D4, f32>,
) -> ImageOperator<[Vector<D4, f32>; 2]> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        transform: cgmath::Matrix4<f32>,
        out_mem_dim: cgmath::Vector2<u32>,
    }

    const VERTEX_SHADER: &str = "
#version 450

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
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include <util.glsl>

layout(scalar, binding = 0) buffer OutputBuffer{
    vec4[2] values[BRICK_MEM_SIZE];
} outputData;

layout(location = 0) in vec3 norm_pos;

declare_push_consts(consts);

void main() {
    uvec2 pos = uvec2(gl_FragCoord.xy);
    uint linear_pos = to_linear2(pos, consts.out_mem_dim);

    vec4 color = vec4(norm_pos, 1.0);
    if(gl_FrontFacing) {
        outputData.values[linear_pos][0] = color;
    } else {
        outputData.values[linear_pos][1] = color;
    }
}
";

    TensorOperator::unbatched(
        OperatorDescriptor::new("entry_exit_points")
            .dependent_on_data(&input_metadata)
            .dependent_on_data(&result_metadata)
            .dependent_on_data(&projection_mat)
            .unstable(),
        result_metadata,
        (
            input_metadata,
            embedding_data,
            result_metadata,
            projection_mat,
        ),
        move |ctx, pos, _, (m_in, embedding_data, m_out, transform)| {
            async move {
                let device = ctx.vulkan_device();

                let out_info = m_out.chunk_info(pos);

                let norm_to_world = Matrix::from_scale(
                    m_in.dimensions.map(|v| v.raw as f32) * embedding_data.spacing,
                )
                .to_homogeneous();
                let transform = *transform * norm_to_world;

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

                let out_dim = out_info.logical_dimensions;
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
                    .submit(ctx.alloc_slot_gpu(device, pos, out_info.mem_elements()))
                    .await;

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
                let offset = out_info.begin().map(|v| v.raw as f32);
                let full_size = m_out.dimensions.map(|v| v.raw as f32);
                let tile_size = out_info.logical_dimensions.map(|v| v.raw as f32);
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
                    out_mem_dim: out_info.mem_dimensions.try_into_elem().unwrap().into(),
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

#[cfg_attr(feature = "python", pyclass)]
#[derive(state_link::State, Clone, Copy)]
pub struct RaycasterConfig {
    #[pyo3(get, set)]
    pub lod_coarseness: f32,
    #[pyo3(get, set)]
    pub oversampling_factor: f32,
}

#[cfg_attr(feature = "python", pymethods)]
impl RaycasterConfig {
    #[new]
    pub fn new() -> Self {
        Default::default()
    }

    fn store(&self, py: pyo3::Python, store: Py<::state_link::py::Store>) -> pyo3::PyObject {
        self.store_py(py, store)
    }
}

impl Identify for RaycasterConfig {
    fn id(&self) -> Id {
        Id::combine(&[self.lod_coarseness.id(), self.oversampling_factor.id()])
    }
}

impl Default for RaycasterConfig {
    fn default() -> Self {
        RaycasterConfig {
            lod_coarseness: 1.0,
            oversampling_factor: 1.0,
        }
    }
}

pub fn raycast(
    input: LODVolumeOperator<f32>,
    entry_exit_points: ImageOperator<[Vector<D4, f32>; 2]>,
    config: RaycasterConfig,
) -> FrameOperator {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        out_mem_dim: cgmath::Vector2<u32>,
        lod_coarseness: f32,
        oversampling_factor: f32,
    }
    const SHADER: &'static str = r#"
#version 450

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include <util.glsl>
#include <hash.glsl>
#include <sample.glsl>
#include <vec.glsl>
#include <color.glsl>


layout(buffer_reference, std430) buffer IndexType {
    BrickType values[];
};

layout(buffer_reference, std430) buffer QueryTableType {
    uint values[REQUEST_TABLE_SIZE];
};

struct LOD {
    IndexType index;
    QueryTableType queryTable;
    UVec3 dimensions;
    UVec3 chunk_size;
    Vec3 spacing;
    uint _padding;
};

layout (local_size_x = 32, local_size_y = 32) in;

layout(std430, binding = 0) buffer OutputBuffer{
    u8vec4 values[];
} output_data;

layout(scalar, binding = 1) buffer EntryExitPoints{
    vec4[2] values[];
} entry_exit_points;

layout(scalar, binding = 2) buffer LodBuffer {
    LOD levels[NUM_LEVELS];
} vol;

struct State {
    float t;
    float intensity;
};

layout(std430, binding = 3) buffer StateBuffer {
    State values[];
} state_cache;

declare_push_consts(consts);

struct EEPoint {
    vec3 entry;
    vec3 exit;
};

vec3 norm_to_voxel(vec3 pos, LOD l) {
    return pos * vec3(to_glsl_uvec3(l.dimensions)) - vec3(0.5);
}

vec3 voxel_to_world(vec3 pos, LOD l) {
    return pos * to_glsl_vec3(l.spacing);
}

vec3 world_to_voxel(vec3 pos, LOD l) {
    return pos / to_glsl_vec3(l.spacing);
}

vec3 norm_to_world(vec3 pos, LOD l) {
    return voxel_to_world(norm_to_voxel(pos, l), l);
}

bool sample_ee(uvec2 pos, out EEPoint eep, LOD l) {

    if(pos.x >= consts.out_mem_dim.x || pos.y >= consts.out_mem_dim.y) {
        return false;
    }

    uint gID = pos.x + pos.y * consts.out_mem_dim.x;

    vec4 entry = entry_exit_points.values[gID][0];
    vec4 exit = entry_exit_points.values[gID][1];

    eep.entry = norm_to_world(entry.xyz, l);
    eep.exit = norm_to_world(exit.xyz, l);

    return entry.a > 0.0 && exit.a > 0.0;
}

#define T_DONE -1.0

void main()
{
    uvec2 out_pos = gl_GlobalInvocationID.xy;
    uint gID = out_pos.x + out_pos.y * consts.out_mem_dim.x;

    EEPoint eep;

    if(!(out_pos.x >= consts.out_mem_dim.x || out_pos.y >= consts.out_mem_dim.y)) {

        LOD root_level = vol.levels[0];
        bool valid = sample_ee(out_pos, eep, root_level);

        u8vec4 color;
        if(valid) {
            State state = state_cache.values[gID];

            if(state.t != T_DONE) {

                EEPoint eep_x;
                if(!sample_ee(out_pos + uvec2(1, 0), eep_x, root_level)) {
                    if(!sample_ee(out_pos - uvec2(1, 0), eep_x, root_level)) {
                        eep_x.entry = vec3(0.0);
                        eep_x.exit = vec3(1.0);
                    }
                }
                EEPoint eep_y;
                if(!sample_ee(out_pos + uvec2(0, 1), eep_y, root_level)) {
                    if(!sample_ee(out_pos - uvec2(0, 1), eep_y, root_level)) {
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

                vec3 start = eep.entry;
                vec3 end = eep.exit;
                float t_end = distance(start, end);
                vec3 dir = normalize(end - start);

                float start_pixel_dist = abs(dot(dir_x, eep_x.entry - eep.entry));
                float end_pixel_dist = abs(dot(dir_x, eep_x.exit - eep.exit));


                float lod_coarseness = consts.lod_coarseness;
                float oversampling_factor = consts.oversampling_factor;

                uint level_num = 0;
                while(state.t <= t_end) {
                    float alpha = state.t/t_end;
                    float pixel_dist = start_pixel_dist * (1.0-alpha) + end_pixel_dist * alpha;

                    while(level_num < NUM_LEVELS - 1) {
                        uint next = level_num+1;
                        vec3 next_spacing = to_glsl_vec3(vol.levels[next].spacing);
                        float left_spacing_dist = length(abs(dir_x) * next_spacing);
                        if(left_spacing_dist >= pixel_dist * lod_coarseness) {
                            break;
                        }
                        level_num = next;
                    }
                    LOD level = vol.levels[level_num];

                    TensorMetaData(3) m_in;
                    m_in.dimensions = level.dimensions.vals;
                    m_in.chunk_size = level.chunk_size.vals;

                    vec3 p = start + state.t*dir;

                    vec3 pos_voxel_g = round(world_to_voxel(p, level));
                    float[3] pos_voxel = from_glsl(pos_voxel_g);

                    int res;
                    uint sample_brick_pos_linear;
                    float sampled_intensity;
                    try_sample(3, pos_voxel, m_in, level.index.values, res, sample_brick_pos_linear, sampled_intensity);
                    if(res == SAMPLE_RES_FOUND) {
                        state.intensity = max(state.intensity, sampled_intensity);
                    } else if(res == SAMPLE_RES_NOT_PRESENT) {
                        try_insert_into_hash_table(level.queryTable.values, REQUEST_TABLE_SIZE, sample_brick_pos_linear);
                        break;
                    } else /*res == SAMPLE_RES_OUTSIDE*/ {
                        // Should only happen at the border of the volume due to rounding errors
                    }

                    float step = length(abs(dir) * to_glsl_vec3(level.spacing)) / oversampling_factor;

                    state.t += step;
                }
                if(state.t > t_end) {
                    state.t = T_DONE;
                    //if(level_num > 0) {
                    //    state.intensity = 0.0;
                    //}
                }

                state_cache.values[gID] = state;
            }

            color = intensity_to_grey(state.intensity);
        } else {
            color = u8vec4(0);
        }

        output_data.values[gID] = color;
    }
}
"#;
    #[repr(C)]
    #[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
    struct LOD {
        index: u64,
        query_table: u64,
        dim: Vector<D3, u32>,
        chunk_dim: Vector<D3, u32>,
        spacing: Vector<D3, f32>,
        _padding: u32,
    }

    TensorOperator::unbatched(
        OperatorDescriptor::new("raycast")
            .dependent_on(&input)
            .dependent_on(&entry_exit_points)
            .dependent_on_data(&config),
        entry_exit_points.metadata,
        (input, entry_exit_points.clone()),
        move |ctx, pos, _, (input, entry_exit_points)| {
            async move {
                let device = ctx.vulkan_device();

                let request_table_size = 256;

                let pipeline =
                    device.request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .push_const_block::<PushConstants>()
                                    .add("NUM_LEVELS", input.levels.len())
                                    .add("REQUEST_TABLE_SIZE", request_table_size),
                            ),
                            false,
                        )
                    });

                let dst_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };

                let m_out = entry_exit_points.metadata;
                let eep = ctx
                    .submit(
                        entry_exit_points
                            .chunks
                            .request_gpu(device.id, pos, dst_info),
                    )
                    .await;
                let out_info = m_out.chunk_info(pos);

                let chunk_size = out_info.mem_dimensions.raw();
                let consts = PushConstants {
                    out_mem_dim: chunk_size.into(),
                    oversampling_factor: config.oversampling_factor,
                    lod_coarseness: config.lod_coarseness,
                };

                let mut lods = Vec::new();
                let mut lod_data = Vec::new();
                for level in &input.levels {
                    let m_in = level.metadata;
                    let emd = level.embedding_data;
                    let num_bricks = m_in.dimension_in_chunks().hmul();
                    //let dim_in_bricks = m_in.dimension_in_chunks();

                    let brick_index = device
                        .storage
                        .get_index(
                            *ctx,
                            device,
                            level.chunks.descriptor(),
                            num_bricks,
                            dst_info,
                        )
                        .await;

                    let info =
                        ash::vk::BufferDeviceAddressInfo::builder().buffer(brick_index.buffer);
                    let index_addr = unsafe { device.functions().get_buffer_device_address(&info) };

                    let request_table = TempRessource::new(
                        device,
                        ctx.submit(ChunkRequestTable::new(request_table_size, device))
                            .await,
                    );

                    let info = ash::vk::BufferDeviceAddressInfo::builder()
                        .buffer(request_table.buffer().buffer);
                    let req_table_addr =
                        unsafe { device.functions().get_buffer_device_address(&info) };

                    lod_data.push((brick_index, request_table, m_in));

                    lods.push(LOD {
                        index: index_addr,
                        query_table: req_table_addr,
                        dim: m_in.dimensions.raw().into(),
                        chunk_dim: m_in.chunk_size.raw().into(),
                        spacing: emd.spacing.into(),
                        _padding: 0,
                    });
                }
                let layout = Layout::array::<LOD>(lods.len()).unwrap();
                let flags = ash::vk::BufferUsageFlags::STORAGE_BUFFER
                    | ash::vk::BufferUsageFlags::TRANSFER_DST;

                let location = crate::storage::gpu::MemoryLocation::GpuOnly;
                let lod_data_gpu = TempRessource::new(
                    device,
                    ctx.submit(
                        device
                            .storage
                            .request_allocate_raw(device, layout, flags, location),
                    )
                    .await,
                );

                let in_bytes: &[u8] = bytemuck::cast_slice(lods.as_slice());
                let in_ptr = in_bytes.as_ptr().cast();
                unsafe {
                    crate::vulkan::memory::copy_to_gpu(
                        *ctx,
                        device,
                        in_ptr,
                        layout,
                        lod_data_gpu.buffer,
                    )
                    .await
                };

                let state_initialized = ctx
                    .submit(ctx.access_state_cache(
                        device,
                        pos,
                        "initialized",
                        Layout::array::<(u32, f32)>(m_out.chunk_size.hmul()).unwrap(),
                    ))
                    .await;
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

                let gpu_brick_out = ctx
                    .submit(ctx.alloc_slot_gpu(device, pos, out_info.mem_elements()))
                    .await;
                let global_size = [1, chunk_size.y(), chunk_size.x()].into();

                // Actual rendering
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
                            &*lod_data_gpu,
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

                    let mut requested_anything = false;
                    for (level, data) in (input.levels.iter().zip(lod_data.iter())).rev() {
                        let mut to_request_linear = data.1.download_requested(*ctx, device).await;

                        if to_request_linear.is_empty() {
                            continue;
                        }
                        requested_anything = true;

                        if let Err(crate::chunk_utils::Timeout) =
                            crate::chunk_utils::request_to_index_with_timeout(
                                &*ctx,
                                device,
                                &mut to_request_linear,
                                level,
                                &data.0,
                            )
                            .await
                        {
                            break 'outer true;
                        }

                        // Clear request table for the next iteration
                        device.with_cmd_buffer(|cmd| data.1.clear(cmd));
                    }

                    if !requested_anything {
                        break 'outer false;
                    }
                };

                let src_info = SrcBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_WRITE,
                };
                if timed_out {
                    unsafe {
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
