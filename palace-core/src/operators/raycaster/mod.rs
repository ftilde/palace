use std::alloc::Layout;

use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};

use crate::{
    array::{
        ImageMetaData, PyTensorEmbeddingData, PyTensorMetaData, VolumeEmbeddingData, VolumeMetaData,
    },
    chunk_utils::ChunkRequestTable2,
    data::{GlobalCoordinate, Matrix, Vector},
    dim::*,
    dtypes::StaticElementType,
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
use id::Identify;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use super::{
    array::ArrayOperator,
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
impl CameraState {
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
    #[pyo3(name = "for_volume")]
    pub fn for_volume_py(
        input_metadata: PyTensorMetaData,
        embedding_data: PyTensorEmbeddingData,
        fov: f32,
    ) -> PyResult<Self> {
        Ok(Self::for_volume(
            input_metadata.try_into_dim()?,
            embedding_data.try_into_dim()?,
            fov,
        ))
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
        let matrix = perspective * &self.trackball.view_mat();
        matrix
    }
}

pub fn entry_exit_points(
    input_metadata: VolumeMetaData,
    embedding_data: VolumeEmbeddingData,
    result_metadata: ImageMetaData,
    projection_mat: Matrix<D4, f32>,
) -> ImageOperator<StaticElementType<[f32; 8]>> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        transform: cgmath::Matrix4<f32>,
        out_mem_dim: cgmath::Vector2<u32>,
    }

    TensorOperator::unbatched(
        OperatorDescriptor::new("entry_exit_points")
            .dependent_on_data(&input_metadata)
            .dependent_on_data(&result_metadata)
            .dependent_on_data(&projection_mat)
            .unstable(),
        Default::default(),
        result_metadata,
        (
            input_metadata,
            embedding_data,
            result_metadata,
            projection_mat,
        ),
        move |ctx, pos, _, (m_in, embedding_data, m_out, transform)| {
            async move {
                let device = ctx.preferred_device();

                let out_info = m_out.chunk_info(pos);

                let norm_to_world = Matrix::from_scale(
                    &(&m_in.dimensions.map(|v| v.raw as f32) * &embedding_data.spacing),
                )
                .to_homogeneous();
                let transform = *transform * &norm_to_world;

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

                        Ok(unsafe {
                            device
                                .functions()
                                .create_render_pass(&render_pass_info, None)
                        }
                        .unwrap())
                    },
                )?;
                let pipeline = device.request_state(
                    RessourceId::new("pipeline").of(ctx.current_op()),
                    || {
                        GraphicsPipeline::new(
                            device,
                            (
                                include_str!("entryexitpoints.vert"),
                                ShaderDefines::new().push_const_block::<PushConstants>(),
                            ),
                            (
                                include_str!("entryexitpoints.frag"),
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
                    },
                )?;

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
#[derive(state_link::State, Clone, Copy, Debug, PartialEq, Eq, id::Identify)]
pub enum CompositingMode {
    MOP,
    DVR,
}

impl CompositingMode {
    fn define_name(&self) -> &'static str {
        match self {
            CompositingMode::MOP => "COMPOSITING_MOP",
            CompositingMode::DVR => "COMPOSITING_DVR",
        }
    }
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(state_link::State, Clone, Copy, Debug, PartialEq, Eq, id::Identify)]
pub enum Shading {
    None,
    Phong,
}

impl Shading {
    fn define_name(&self) -> &'static str {
        match self {
            Shading::None => "SHADING_NONE",
            Shading::Phong => "SHADING_PHONG",
        }
    }
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(state_link::State, Clone, Copy, Debug, Identify)]
pub struct RaycasterConfig {
    #[pyo3(get, set)]
    pub lod_coarseness: f32,
    #[pyo3(get, set)]
    pub preview_lod_coarseness_modifier: f32,
    #[pyo3(get, set)]
    pub oversampling_factor: f32,
    #[pyo3(get, set)]
    pub compositing_mode: CompositingMode,
    #[pyo3(get, set)]
    pub shading: Shading,
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

impl Default for RaycasterConfig {
    fn default() -> Self {
        RaycasterConfig {
            lod_coarseness: 1.0,
            preview_lod_coarseness_modifier: 4.0,
            oversampling_factor: 1.0,
            compositing_mode: CompositingMode::MOP,
            shading: Shading::None,
        }
    }
}

pub type TransFuncTableOperator = ArrayOperator<StaticElementType<Vector<D4, u8>>>;

#[derive(Clone, Identify)]
pub struct TransFuncOperator {
    pub table: TransFuncTableOperator,
    pub min: f32,
    pub max: f32,
}

impl TransFuncOperator {
    pub fn data(&self) -> TransFuncData {
        TransFuncData {
            len: self.table.metadata.dimensions[0].raw,
            min: self.min,
            max: self.max,
        }
    }
    pub fn gen(min: f32, max: f32, len: usize, g: impl FnMut(usize) -> Vector<D4, u8>) -> Self {
        let vals = (0..len).map(g).collect::<Vec<_>>();
        let table = super::array::from_rc(std::rc::Rc::from(vals));
        Self { table, min, max }
    }
    pub fn gen_normalized(
        min: f32,
        max: f32,
        len: usize,
        mut g: impl FnMut(f32) -> Vector<D4, u8>,
    ) -> Self {
        let vals = (0..len)
            .map(|i| g(i as f32 / len as f32))
            .collect::<Vec<_>>();
        let table = super::array::from_rc(std::rc::Rc::from(vals));
        Self { table, min, max }
    }
    pub fn grey_ramp(min: f32, max: f32) -> Self {
        Self::gen(min, max, 256, |i| Vector::fill(i as u8))
    }
    pub fn red_ramp(min: f32, max: f32) -> Self {
        Self::gen(min, max, 256, |i| Vector::from([i as u8, 0, 0, i as u8]))
    }
    pub fn normalized(mut self) -> Self {
        self.min = 0.0;
        self.max = 1.0;
        self
    }
}

#[derive(Copy, Clone, AsStd140, GlslStruct)]
pub struct TransFuncData {
    pub len: u32,
    pub min: f32,
    pub max: f32,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
enum RaycastingState {
    Empty = 0,
    RenderingPreview = 1,
    PreviewDone = 2,
    RenderingFull = 3,
    Done = 4,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::AnyBitPattern)]
struct RawRaycastingState(u8);

impl From<RaycastingState> for RawRaycastingState {
    fn from(value: RaycastingState) -> Self {
        RawRaycastingState(value as u8)
    }
}
impl RawRaycastingState {
    fn unpack(&self) -> RaycastingState {
        (*self).into()
    }
}

impl From<RawRaycastingState> for RaycastingState {
    fn from(value: RawRaycastingState) -> Self {
        match value.0 {
            0 => RaycastingState::Empty,
            1 => RaycastingState::RenderingPreview,
            2 => RaycastingState::PreviewDone,
            3 => RaycastingState::RenderingFull,
            4 => RaycastingState::Done,
            _ => panic!("Invalid state"),
        }
    }
}

pub fn raycast(
    input: LODVolumeOperator<StaticElementType<f32>>,
    entry_exit_points: ImageOperator<StaticElementType<[f32; 8]>>,
    tf: TransFuncOperator,
    config: RaycasterConfig,
) -> FrameOperator {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        out_mem_dim: cgmath::Vector2<u32>,
        lod_coarseness: f32,
        oversampling_factor: f32,
        tf_min: f32,
        tf_max: f32,
        tf_len: u32,
        reset_state: u32,
    }

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
            .dependent_on_data(&tf)
            .dependent_on_data(&config),
        Default::default(),
        entry_exit_points.metadata,
        (input, entry_exit_points.clone(), tf),
        move |ctx, pos, _, (input, entry_exit_points, tf)| {
            async move {
                let device = ctx.preferred_device();

                let progress_state = ctx
                    .submit(ctx.access_state_cache::<RawRaycastingState>(pos, "progress_state", 1))
                    .await;
                let mut progress_state = unsafe {
                    progress_state.init(|r| {
                        crate::data::fill_uninit(r, RaycastingState::Empty.into());
                    })
                };
                let [ref mut progress_state] = &mut *progress_state else {
                    panic!("Invalid size");
                };

                let lod_coarseness = match progress_state.unpack() {
                    RaycastingState::Empty | RaycastingState::RenderingPreview => {
                        config.lod_coarseness * config.preview_lod_coarseness_modifier
                    }
                    _ => config.lod_coarseness,
                };

                let mut reset_state = matches!(
                    progress_state.unpack(),
                    RaycastingState::Empty | RaycastingState::PreviewDone
                );
                let m_out = entry_exit_points.metadata;

                let in_preview = progress_state.unpack() < RaycastingState::RenderingFull;

                let state_ray = ctx
                    .submit(ctx.access_state_cache_gpu(
                        device,
                        pos,
                        "state_ray",
                        Layout::array::<f32>(m_out.chunk_size.hmul()).unwrap(),
                    ))
                    .await;
                let state_ray = state_ray.init(|v| {
                    device.with_cmd_buffer(|cmd| unsafe {
                        device.functions().cmd_fill_buffer(
                            cmd.raw(),
                            v.buffer,
                            0,
                            vk::WHOLE_SIZE,
                            0,
                        );
                        // Reset state if we lose cache
                        *progress_state = RaycastingState::Empty.into();
                    });
                });

                let state_img = ctx
                    .submit(ctx.access_state_cache_gpu(
                        device,
                        pos,
                        "state_img",
                        Layout::array::<Vector<D4, u8>>(m_out.chunk_size.hmul()).unwrap(),
                    ))
                    .await;
                let state_img = state_img.init(|v| {
                    device.with_cmd_buffer(|cmd| unsafe {
                        device.functions().cmd_fill_buffer(
                            cmd.raw(),
                            v.buffer,
                            0,
                            vk::WHOLE_SIZE,
                            0,
                        );
                        // Reset state if we lose cache
                        *progress_state = RaycastingState::Empty.into();
                    });
                });
                let out_info = m_out.chunk_info(pos);

                let request_table_size = 256;
                let raw_request_tables = ctx
                    .submit(ctx.group((0..input.levels.len()).into_iter().map(|i| {
                        ctx.access_state_cache_gpu(
                            device,
                            pos,
                            &format!("lod_table{}", i),
                            Layout::array::<Vector<D4, u8>>(request_table_size).unwrap(),
                        )
                    })))
                    .await;
                let request_tables = raw_request_tables
                    .into_iter()
                    .map(|raw| ChunkRequestTable2::new(device, raw))
                    .collect::<Vec<_>>();

                let reuse_res = ctx.alloc_try_reuse_gpu(device, pos, out_info.mem_elements());
                let gpu_brick_out = ctx.submit(reuse_res.request).await;

                if reuse_res.new
                    || ctx.past_deadline(in_preview).is_none()
                    || progress_state.unpack() < RaycastingState::RenderingFull
                {
                    let pipeline = device.request_state(
                        RessourceId::new("pipeline")
                            .of(ctx.current_op())
                            .dependent_on(&input.levels.len())
                            .dependent_on(&config.compositing_mode)
                            .dependent_on(&config.shading),
                        || {
                            ComputePipeline::new(
                                device,
                                (
                                    include_str!("raycaster.glsl"),
                                    ShaderDefines::new()
                                        .push_const_block::<PushConstants>()
                                        .add("NUM_LEVELS", input.levels.len())
                                        .add("REQUEST_TABLE_SIZE", request_table_size)
                                        .add(config.compositing_mode.define_name(), 1)
                                        .add(config.shading.define_name(), 1),
                                ),
                                false,
                            )
                        },
                    )?;

                    let request_batch_size = ctx
                        .submit(ctx.access_state_cache(
                            pos,
                            "request_batch_size",
                            input.levels.len(),
                        ))
                        .await;
                    let mut request_batch_size = unsafe {
                        request_batch_size.init(|r| {
                            crate::data::fill_uninit(r, 1usize);
                        })
                    };

                    let dst_info = DstBarrierInfo {
                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::SHADER_READ,
                    };

                    let eep = ctx
                        .submit(
                            entry_exit_points
                                .chunks
                                .request_gpu(device.id, pos, dst_info),
                        )
                        .await;

                    assert_eq!(tf.table.metadata.dimension_in_chunks()[0].raw, 1);
                    let tf_data_gpu = ctx
                        .submit(tf.table.chunks.request_scalar_gpu(device.id, dst_info))
                        .await;

                    let chunk_size = out_info.mem_dimensions.raw();
                    let tf_data = tf.data();

                    let mut lods = Vec::new();
                    let mut lod_data = Vec::new();
                    for (level, request_table) in
                        input.levels.iter().zip(request_tables.into_iter())
                    {
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
                        let index_addr =
                            unsafe { device.functions().get_buffer_device_address(&info) };

                        let info = ash::vk::BufferDeviceAddressInfo::builder()
                            .buffer(request_table.buffer());
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
                    let global_size = [1, chunk_size.y(), chunk_size.x()].into();

                    // Actual rendering
                    let timed_out = 'outer: loop {
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

                        let mut done = true;
                        let mut timeout = false;
                        for ((level, request_batch_size), data) in (input
                            .levels
                            .iter()
                            .zip(request_batch_size.iter_mut())
                            .zip(lod_data.iter_mut()))
                        .rev()
                        {
                            if !data.1.newly_initialized && !reset_state {
                                let mut to_request_linear =
                                    data.1.download_requested(*ctx, device).await;

                                if to_request_linear.is_empty() {
                                    continue;
                                }

                                done = false;

                                if let Err(crate::chunk_utils::Timeout) =
                                    crate::chunk_utils::request_to_index_with_timeout(
                                        &*ctx,
                                        device,
                                        &mut to_request_linear,
                                        level,
                                        &data.0,
                                        request_batch_size,
                                        in_preview,
                                    )
                                    .await
                                {
                                    timeout = true;
                                }

                                // Clear request table for the next iteration
                                device.with_cmd_buffer(|cmd| data.1.clear(cmd));
                            } else {
                                data.1.newly_initialized = false;
                                done = false;
                            }
                        }

                        // Make writes to the request table visible (including initialization)
                        ctx.submit(device.barrier(
                            SrcBarrierInfo {
                                stage: vk::PipelineStageFlags2::TRANSFER,
                                access: vk::AccessFlags2::TRANSFER_WRITE,
                            },
                            DstBarrierInfo {
                                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                access: vk::AccessFlags2::SHADER_READ
                                    | vk::AccessFlags2::SHADER_WRITE,
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
                                &state_ray,
                                &state_img,
                                &tf_data_gpu,
                            ]);

                            let consts = PushConstants {
                                tf_min: tf_data.min,
                                tf_max: tf_data.max,
                                tf_len: tf_data.len,
                                out_mem_dim: chunk_size.into(),
                                oversampling_factor: config.oversampling_factor,
                                lod_coarseness,
                                reset_state: reset_state as u32,
                            };
                            reset_state = false;

                            unsafe {
                                let mut pipeline = pipeline.bind(cmd);

                                pipeline.push_constant(consts);
                                pipeline.write_descriptor_set(0, descriptor_config);
                                pipeline.dispatch3d(global_size);
                            }
                        });

                        if done {
                            break 'outer false;
                        }

                        if timeout {
                            break 'outer true;
                        }
                    };

                    let new_state = match progress_state.unpack() {
                        RaycastingState::Empty | RaycastingState::RenderingPreview => {
                            if timed_out {
                                RaycastingState::RenderingPreview
                            } else {
                                RaycastingState::PreviewDone
                            }
                        }
                        RaycastingState::PreviewDone | RaycastingState::RenderingFull => {
                            if timed_out {
                                RaycastingState::RenderingFull
                            } else {
                                RaycastingState::Done
                            }
                        }
                        RaycastingState::Done => RaycastingState::Done,
                    };
                    //println!("{:?} -> {:?}", progress_state.unpack(), new_state);
                    *progress_state = new_state.into();

                    //println!(
                    //    "Request table size ================ {:?}",
                    //    &*request_batch_size
                    //);
                }

                let src_info = SrcBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_WRITE,
                };

                if matches!(progress_state.unpack(), RaycastingState::Done) {
                    unsafe { gpu_brick_out.initialized(*ctx, src_info) };
                } else {
                    unsafe {
                        gpu_brick_out.initialized_version(*ctx, src_info, DataVersionType::Preview)
                    };
                }

                Ok(())
            }
            .into()
        },
    )
}
