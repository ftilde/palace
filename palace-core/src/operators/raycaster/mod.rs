use std::alloc::Layout;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use crevice::glsl::GlslStruct;

use crate::{
    array::{
        ImageMetaData, PyTensorEmbeddingData, PyTensorMetaData, VolumeEmbeddingData, VolumeMetaData,
    },
    chunk_utils::{FeedbackTableElement, RequestTable, RequestTableResult, UseTable},
    data::{GlobalCoordinate, Matrix, Vector},
    dim::*,
    dtypes::{DType, ElementType, StaticElementType},
    op_descriptor,
    operator::{DataParam, OperatorDescriptor, OperatorNetworkNode},
    operators::tensor::TensorOperator,
    runtime::FrameNumber,
    storage::{
        gpu::{buffer_address, BufferAddress},
        DataVersionType,
    },
    transfunc::TransFuncOperator,
    vulkan::{
        memory::TempRessource,
        pipeline::{
            ComputePipelineBuilder, DescriptorConfig, GraphicsPipelineBuilder, LocalSizeConfig,
        },
        shader::Shader,
        DstBarrierInfo, SrcBarrierInfo,
    },
    Error,
};
use id::Identify;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3_stub_gen::derive::*;

use super::tensor::{FrameOperator, ImageOperator, LODVolumeOperator};

#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(feature = "python", gen_stub_pyclass)]
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
#[cfg_attr(feature = "python", gen_stub_pymethods)]
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
#[cfg_attr(feature = "python", gen_stub_pyclass)]
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
#[cfg_attr(feature = "python", gen_stub_pymethods)]
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
    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable, GlslStruct)]
    struct PushConstantsFirstEEP {
        norm_to_projection: Matrix<D4, f32>,
        out_mem_dim: Vector<D2, u32>,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable, GlslStruct)]
    struct PushConstantsInVolumeFix {
        projection_to_norm: Matrix<D4, f32>,
        out_mem_dim: Vector<D2, u32>,
    }

    TensorOperator::unbatched(
        op_descriptor!().unstable(),
        Default::default(),
        result_metadata,
        (
            DataParam(input_metadata),
            DataParam(embedding_data),
            DataParam(result_metadata),
            DataParam(projection_mat),
        ),
        move |ctx, pos, loc, (m_in, embedding_data, m_out, transform)| {
            async move {
                let device = ctx.preferred_device(loc);

                let out_info = m_out.chunk_info(pos);

                let norm_to_world = Matrix::from_scale(
                    &(&m_in.dimensions.map(|v| v.raw as f32) * &embedding_data.spacing),
                )
                .to_homogeneous();
                let norm_to_projection = **transform * &norm_to_world;
                let projection_to_norm = norm_to_projection.invert().unwrap();

                let (render_pass, pipeline_eep) = device.request_state((), |device, ()| {
                    let subpass = vk::SubpassDescription::default()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .color_attachments(&[]);

                    let dependency_info = vk::SubpassDependency::default()
                        .src_subpass(vk::SUBPASS_EXTERNAL)
                        .dst_subpass(0)
                        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

                    let subpasses = &[subpass];
                    let dependency_infos = &[dependency_info];
                    let render_pass_info = vk::RenderPassCreateInfo::default()
                        .attachments(&[])
                        .subpasses(subpasses)
                        .dependencies(dependency_infos);

                    let render_pass = unsafe {
                        device
                            .functions()
                            .create_render_pass(&render_pass_info, None)
                    }?;

                    let pipeline = GraphicsPipelineBuilder::new(
                        Shader::new(include_str!("entryexitpoints.vert"))
                            .push_const_block::<PushConstantsFirstEEP>(),
                        Shader::new(include_str!("entryexitpoints.frag"))
                            .push_const_block::<PushConstantsFirstEEP>(),
                    )
                    .use_push_descriptor(true)
                    .build(
                        device,
                        |shader_stages, pipeline_layout, build_pipeline| {
                            let dynamic_states =
                                [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
                            let dynamic_info = vk::PipelineDynamicStateCreateInfo::default()
                                .dynamic_states(&dynamic_states);

                            let vertex_input_info =
                                vk::PipelineVertexInputStateCreateInfo::default();

                            let input_assembly_info =
                                vk::PipelineInputAssemblyStateCreateInfo::default()
                                    .primitive_restart_enable(false)
                                    .topology(vk::PrimitiveTopology::TRIANGLE_STRIP);

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
                                    .front_face(vk::FrontFace::CLOCKWISE)
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
                                    .dst_color_blend_factor(vk::BlendFactor::ONE)
                                    .color_blend_op(vk::BlendOp::ADD)
                                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                                    .dst_alpha_blend_factor(vk::BlendFactor::ONE)
                                    .alpha_blend_op(vk::BlendOp::ADD)
                                    .blend_enable(true);

                            let color_blend_attachments =
                                [color_blend_attachment, color_blend_attachment];
                            let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
                                .logic_op_enable(false)
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
                                .render_pass(render_pass)
                                .subpass(0);
                            build_pipeline(&info)
                        },
                    )?;

                    Ok((render_pass, pipeline))
                })?;

                let pipeline_in_volume_fix = device.request_state((), |device, ()| {
                    ComputePipelineBuilder::new(
                        Shader::new(include_str!("entrypoints_inside.glsl"))
                            .push_const_block::<PushConstantsInVolumeFix>(),
                    )
                    .local_size(LocalSizeConfig::Auto2D)
                    .build(device)
                })?;

                let out_dim = out_info.logical_dimensions;
                let width = out_dim.x().into();
                let height = out_dim.y().into();
                let extent = vk::Extent2D::default().width(width).height(height);

                let framebuffer_info = vk::FramebufferCreateInfo::default()
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
                    .submit(ctx.alloc_slot_gpu(device, pos, &out_info.mem_dimensions))
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

                let render_pass_info = vk::RenderPassBeginInfo::default()
                    .render_pass(*render_pass)
                    .framebuffer(*framebuffer)
                    .render_area(
                        vk::Rect2D::default()
                            .offset(vk::Offset2D::default().x(0).y(0))
                            .extent(extent),
                    )
                    .clear_values(&[]);

                // Setup the viewport such that only the content of the current tile is rendered
                let offset = out_info.begin().map(|v| v.raw as f32);
                let full_size = m_out.dimensions.map(|v| v.raw as f32);
                let tile_size = out_info.logical_dimensions.map(|v| v.raw as f32);
                let scale_factor = full_size / tile_size;
                let size = tile_size * scale_factor;
                let viewport = vk::Viewport::default()
                    .x(-offset.x())
                    .y(-offset.y())
                    .width(size.x())
                    .height(size.y())
                    .min_depth(0.0)
                    .max_depth(1.0);

                let scissor = vk::Rect2D::default()
                    .offset(vk::Offset2D::default().x(0).y(0))
                    .extent(extent);

                let push_constants = PushConstantsFirstEEP {
                    out_mem_dim: out_info.mem_dimensions.try_into_elem().unwrap(),
                    norm_to_projection,
                };
                let descriptor_config = DescriptorConfig::new([&gpu_brick_out]);

                device.with_cmd_buffer(|cmd| unsafe {
                    let mut pipeline = pipeline_eep.bind(cmd);

                    device.functions().cmd_begin_render_pass(
                        pipeline.cmd().raw(),
                        &render_pass_info,
                        vk::SubpassContents::INLINE,
                    );
                    device
                        .functions()
                        .cmd_set_viewport(pipeline.cmd().raw(), 0, &[viewport]);
                    device
                        .functions()
                        .cmd_set_scissor(pipeline.cmd().raw(), 0, &[scissor]);

                    pipeline.push_descriptor_set(0, descriptor_config);

                    pipeline.push_constant_at(
                        push_constants,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    );

                    device.functions().cmd_draw(cmd.raw(), 14, 1, 0, 0);

                    device.functions().cmd_end_render_pass(cmd.raw());
                });

                // Fix entry points in case the camera is inside the volume:

                ctx.submit(device.barrier(
                    SrcBarrierInfo {
                        stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        access: vk::AccessFlags2::SHADER_WRITE,
                    },
                    DstBarrierInfo {
                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                    },
                ))
                .await;

                let global_size = out_info.mem_dimensions.raw().push_dim_large(1);

                device.with_cmd_buffer(|cmd| {
                    let descriptor_config = DescriptorConfig::new([&gpu_brick_out]);

                    let consts = PushConstantsInVolumeFix {
                        projection_to_norm,
                        out_mem_dim: out_info.mem_dimensions.try_into_elem().unwrap(),
                    };

                    unsafe {
                        let mut pipeline = pipeline_in_volume_fix.bind(cmd);

                        pipeline.push_constant(consts);
                        pipeline.write_descriptor_set(0, descriptor_config);
                        pipeline.dispatch3d(global_size);
                    }
                });

                unsafe {
                    gpu_brick_out.initialized(
                        *ctx,
                        SrcBarrierInfo {
                            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
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

#[cfg_attr(feature = "python", gen_stub_pyclass_enum)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
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

#[cfg_attr(feature = "python", gen_stub_pyclass_enum)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
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

#[cfg_attr(feature = "python", gen_stub_pyclass)]
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

#[repr(C)]
#[derive(Copy, Clone)]
struct RawSharedRaycastingStateInner {
    frame: FrameNumber,
    preview_done: bool,
    next_preview_done: bool,
}

impl Default for RawSharedRaycastingStateInner {
    fn default() -> Self {
        Self {
            frame: FrameNumber::first(),
            preview_done: false,
            next_preview_done: false,
        }
    }
}

#[repr(C)]
#[derive(Default)]
struct RawSharedRaycastingState {
    inner: std::cell::Cell<RawSharedRaycastingStateInner>,
}

impl RawSharedRaycastingState {
    fn preview_all_done(&self, current_frame: FrameNumber) -> bool {
        let mut current = self.inner.get();
        if current.frame != current_frame {
            current.frame = current_frame;
            current.preview_done = current.next_preview_done;
            current.next_preview_done = true;
            self.inner.set(current);
        }
        current.preview_done
    }
    fn mark_preview_not_done(&self) {
        let mut current = self.inner.get();
        current.next_preview_done = false;
        self.inner.set(current);
    }
}

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

pub fn raycast<E: ElementType>(
    input: LODVolumeOperator<E>,
    entry_exit_points: ImageOperator<StaticElementType<[f32; 8]>>,
    tf: TransFuncOperator,
    config: RaycasterConfig,
) -> Result<FrameOperator, Error> {
    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable, GlslStruct)]
    struct PushConstants {
        request_table: BufferAddress,
        out_mem_dim: Vector<D2, u32>,
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
        page_table_root: BufferAddress,
        use_table: BufferAddress,
        dim: Vector<D3, u32>,
        chunk_dim: Vector<D3, u32>,
        spacing: Vector<D3, f32>,
        _padding: u32,
    }

    let dtype: DType = input.dtype().into();

    if dtype.size != 1 {
        return Err(format!("Tensor element must be one-dimensional: {:?}", dtype).into());
    }

    Ok(TensorOperator::unbatched(
        op_descriptor!(),
        Default::default(),
        entry_exit_points.metadata,
        (
            input,
            entry_exit_points.clone(),
            DataParam(tf),
            DataParam(config),
            DataParam(dtype),
        ),
        |ctx, pos, loc, (input, entry_exit_points, tf, config, dtype)| {
            async move {
                let device = ctx.preferred_device(loc);

                let global_progress_state = ctx
                    .submit(ctx.access_state_cache_shared::<RawSharedRaycastingState>(
                        "global_progress_state",
                        1,
                    ))
                    .await;
                let [ref global_progress_state] = &*global_progress_state else {
                    panic!("Invalid size");
                };

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

                let request_table_size = 2048;
                let use_table_size = 2048;
                let raw_request_table = ctx
                    .submit(ctx.access_state_cache_gpu(
                        device,
                        pos,
                        "request_table",
                        Layout::array::<FeedbackTableElement>(request_table_size).unwrap(),
                    ))
                    .await;
                let mut request_table = RequestTable::new(device, raw_request_table);
                let request_table_addr = request_table.buffer_address();

                let raw_use_tables = ctx
                    .submit(ctx.group((0..input.levels.len()).into_iter().map(|i| {
                        ctx.access_state_cache_gpu(
                            device,
                            pos,
                            &format!("use_table{}", i),
                            Layout::array::<FeedbackTableElement>(use_table_size).unwrap(),
                        )
                    })))
                    .await;
                let use_tables = raw_use_tables
                    .into_iter()
                    .map(|raw| UseTable::new(device, raw))
                    .collect::<Vec<_>>();

                let pipeline = device.request_state(
                    (
                        input.levels.len(),
                        request_table_size,
                        use_table_size,
                        config,
                        dtype,
                    ),
                    |device, (num_levels, request_table_size, use_table_size, config, dtype)| {
                        ComputePipelineBuilder::new(
                            Shader::new(include_str!("raycaster.glsl"))
                                .push_const_block::<PushConstants>()
                                .define("NUM_LEVELS", num_levels)
                                .define("USE_TABLE_SIZE", use_table_size)
                                .define("REQUEST_TABLE_SIZE", request_table_size)
                                .define(config.compositing_mode.define_name(), 1)
                                .define(config.shading.define_name(), 1)
                                .define("INPUT_DTYPE", dtype.glsl_type())
                                .ext(dtype.glsl_ext()),
                        )
                        .local_size(LocalSizeConfig::Auto2D)
                        .build(device)
                    },
                )?;

                let reuse_res = ctx.alloc_try_reuse_gpu(device, pos, out_info.mem_elements());
                let gpu_brick_out = ctx.submit(reuse_res.request).await;
                if reuse_res.new {
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
                }

                let wait_for_others = !global_progress_state.preview_all_done(ctx.current_frame)
                    && progress_state.unpack() >= RaycastingState::PreviewDone;

                let should_progress = ctx.past_deadline(in_preview).is_none() && !wait_for_others;

                if reuse_res.new || should_progress {
                    let request_batch_size = ctx
                        .submit(ctx.access_state_cache(pos, "request_batch_size", 1))
                        .await;
                    let mut request_batch_size = unsafe {
                        request_batch_size.init(|r| {
                            crate::data::fill_uninit(r, 1usize);
                        })
                    };
                    let request_batch_size = &mut request_batch_size[0];

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
                    for (level, use_table) in input.levels.iter().zip(use_tables.into_iter()) {
                        let m_in = level.metadata;
                        let emd = level.embedding_data;
                        //let dim_in_bricks = m_in.dimension_in_chunks();

                        let brick_index = device
                            .storage
                            .get_page_table(*ctx, device, level.chunks.descriptor(), dst_info)
                            .await;

                        let page_table_root = buffer_address(device, brick_index.buffer);
                        let use_table_addr = use_table.buffer_address();

                        lod_data.push((brick_index, use_table, m_in));

                        lods.push(LOD {
                            page_table_root,
                            use_table: use_table_addr,
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
                                access: vk::AccessFlags2::SHADER_WRITE
                                    | vk::AccessFlags2::SHADER_READ,
                            },
                            DstBarrierInfo {
                                stage: vk::PipelineStageFlags2::TRANSFER
                                    | vk::PipelineStageFlags2::COMPUTE_SHADER,
                                access: vk::AccessFlags2::TRANSFER_READ
                                    | vk::AccessFlags2::SHADER_WRITE,
                            },
                        ))
                        .await;

                        let tensors_and_pts = input
                            .levels
                            .iter()
                            .map(|t| &t.inner)
                            .zip(lod_data.iter().map(|d| &d.0))
                            .collect::<Vec<_>>();

                        let request_result = request_table
                            .download_and_insert(
                                *ctx,
                                device,
                                tensors_and_pts,
                                request_batch_size,
                                in_preview,
                                reset_state,
                            )
                            .await;

                        for data in lod_data.iter_mut().rev() {
                            data.1.download_and_note_use(*ctx, device, &data.0).await;
                        }

                        // Make writes to the request table visible (including initialization)
                        ctx.submit(device.barrier(
                            SrcBarrierInfo {
                                stage: vk::PipelineStageFlags2::TRANSFER
                                    | vk::PipelineStageFlags2::COMPUTE_SHADER,
                                access: vk::AccessFlags2::TRANSFER_WRITE
                                    | vk::AccessFlags2::SHADER_WRITE,
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
                                request_table: request_table_addr,
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

                        match request_result {
                            RequestTableResult::Done => break 'outer false,
                            RequestTableResult::Timeout => break 'outer true,
                            RequestTableResult::Continue => {}
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

                if progress_state.unpack() < RaycastingState::PreviewDone {
                    global_progress_state.mark_preview_not_done();
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
    ))
}
