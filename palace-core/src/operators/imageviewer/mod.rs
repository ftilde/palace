use std::alloc::Layout;

use ash::vk;
use id::Identify;

use super::{
    sliceviewer::RenderConfig2D,
    tensor::{FrameOperator, LODImageOperator},
};

use crate::{
    array::{ImageEmbeddingData, ImageMetaData, PyTensorEmbeddingData, PyTensorMetaData},
    chunk_utils::{FeedbackTableElement, RequestTable, RequestTableResult, UseTable},
    coordinate::GlobalCoordinate,
    data::Vector,
    dim::*,
    dtypes::StaticElementType,
    mat::Matrix,
    op_descriptor,
    operator::{DataParam, OpaqueOperator, OperatorDescriptor},
    operators::tensor::TensorOperator,
    storage::DataVersionType,
    vulkan::{
        pipeline::{ComputePipelineBuilder, DescriptorConfig, DynPushConstants, LocalSizeConfig},
        shader::Shader,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3_stub_gen::derive::*;

#[derive(Clone, state_link::State, Identify)]
#[cfg_attr(feature = "python", gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyclass)]
pub struct ImageViewerState {
    #[pyo3(get, set)]
    pub offset: Vector<D2, f32>,
    #[pyo3(get, set)]
    pub zoom_level: f32,
}

#[cfg_attr(feature = "python", pymethods)]
impl ImageViewerState {
    #[new]
    pub fn new() -> Self {
        Self {
            offset: Vector::fill(0.0),
            zoom_level: 1.0,
        }
    }

    fn store(&self, py: pyo3::Python, store: Py<::state_link::py::Store>) -> pyo3::PyObject {
        self.store_py(py, store)
    }

    pub fn drag(&mut self, delta: Vector<D2, f32>) {
        self.offset = self.offset + delta;
    }

    pub fn zoom(&mut self, delta: f32, on: Vector<D2, f32>) {
        let zoom_change = (-delta * 0.05).exp(); //TODO: not entirely happy about the magic
                                                 //constant here...
        self.zoom_level *= zoom_change;

        self.offset = (self.offset - on) / Vector::fill(zoom_change) + on;
    }

    #[pyo3(name = "projection_mat")]
    pub fn projection_mat_py(
        &self,
        input_metadata: PyTensorMetaData,
        embedding_data: PyTensorEmbeddingData,
        output_size: Vector<D2, GlobalCoordinate>,
    ) -> PyResult<Matrix<D3, f32>> {
        Ok(projection_mat(
            input_metadata.try_into_dim()?,
            embedding_data.try_into_dim()?,
            output_size,
            self.offset,
            self.zoom_level,
        ))
    }
}

impl ImageViewerState {
    pub fn projection_mat(
        &self,
        input_metadata: ImageMetaData,
        embedding_data: ImageEmbeddingData,
        output_size: Vector<D2, GlobalCoordinate>,
    ) -> Matrix<D3, f32> {
        projection_mat(
            input_metadata,
            embedding_data,
            output_size,
            self.offset,
            self.zoom_level,
        )
    }
}

fn projection_mat(
    input_data: ImageMetaData,
    embedding_data: ImageEmbeddingData,
    output_size: Vector<D2, GlobalCoordinate>,
    offset: Vector<D2, f32>,
    zoom_level: f32,
) -> Matrix<D3, f32> {
    let input_dim_pixel = input_data.dimensions.map(|v| v.raw as f32);
    let input_dim = input_dim_pixel * embedding_data.spacing;
    let img_dim = output_size.map(|v| v.raw as f32);

    let h_size = input_dim[1];
    let v_size = input_dim[0];

    let aspect_ratio_img = img_dim.x() / img_dim.y();
    let aspect_ratio_vol = h_size / v_size;
    let scaling_factor = if aspect_ratio_img > aspect_ratio_vol {
        v_size / img_dim.y()
    } else {
        h_size / img_dim.x()
    };

    let offset_x = (img_dim.x() - (h_size / scaling_factor)).max(0.0) * 0.5;
    let offset_y = (img_dim.y() - (v_size / scaling_factor)).max(0.0) * 0.5;

    let to_pixel_center = Matrix::from_translation(Vector::<D2, f32>::fill(0.5));
    let pixel_transform = Matrix::from_translation(Vector::from([-offset_y, -offset_x]))
        * &Matrix::from_scale(&Vector::<D2, _>::fill(zoom_level)).to_homogeneous()
        * &Matrix::from_translation(offset.map(|v| -v))
        * &to_pixel_center;

    let translation = Vector::<D2, f32>::fill(-0.5); //For "centered" input positions
    let scale = Matrix::from_scale(&Vector::<D2, _>::fill(scaling_factor)).to_homogeneous();
    let slice_select = Matrix::from_translation(translation);
    let rw_to_voxel = Matrix::from_scale(&embedding_data.spacing.map(|v| 1.0 / v)).to_homogeneous();
    let mat = slice_select * &rw_to_voxel * &scale * &pixel_transform;

    mat
}

pub fn view_image(
    input: LODImageOperator<StaticElementType<Vector<D4, u8>>>,
    result_metadata: ImageMetaData,
    view_state: ImageViewerState,
    config: RenderConfig2D,
) -> FrameOperator {
    let push_constants = DynPushConstants::new()
        .scalar::<u64>("page_table_root")
        .scalar::<u64>("use_table")
        .mat::<f32>(3, "transform")
        .vec::<u32>(2, "input_dim")
        .vec::<u32>(2, "chunk_dim")
        .vec::<u32>(2, "out_begin")
        .vec::<u32>(2, "out_mem_dim");

    TensorOperator::unbatched(
        op_descriptor!().unstable(),
        Default::default(),
        result_metadata,
        (
            input,
            DataParam(result_metadata),
            DataParam(view_state),
            DataParam(config),
            DataParam(push_constants),
        ),
        move |ctx, pos, loc, (input, result_metadata, view_state, config, push_constants)| {
            async move {
                let device = ctx.preferred_device(loc);

                let dst_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };

                let m_out = result_metadata;
                let emd_0 = input.levels[0].embedding_data;
                let out_to_pixel_in = projection_mat(
                    input.fine_metadata(),
                    input.fine_embedding_data(),
                    m_out.dimensions,
                    view_state.offset,
                    view_state.zoom_level,
                );

                let transform_rw = emd_0.voxel_to_physical() * &out_to_pixel_in;

                let (level_num, level) = crate::operators::sliceviewer::select_level(
                    &input,
                    transform_rw,
                    &[[0.0, 1.0].into(), [1.0, 0.0].into()],
                    **config,
                );

                let m_in = level.metadata;
                let emd_l = level.embedding_data;

                let transform =
                    emd_l.physical_to_voxel() * &emd_0.voxel_to_physical() * &out_to_pixel_in;

                let out_info = m_out.chunk_info(pos);

                let page_table = device
                    .storage
                    .get_page_table(*ctx, device, level.chunks.operator_descriptor(), dst_info)
                    .await;
                let page_table_addr = page_table.root();

                let request_table_size = 256;
                let use_table_size = 2048;

                let pipeline = device.request_state(
                    (
                        m_in.chunk_size.hmul(),
                        request_table_size,
                        use_table_size,
                        push_constants,
                    ),
                    |device, (mem_size, request_table_size, use_table_size, push_constants)| {
                        ComputePipelineBuilder::new(
                            Shader::new(include_str!("imageviewer.glsl"))
                                .push_const_block_dyn(push_constants)
                                .define("BRICK_MEM_SIZE", mem_size)
                                .define("REQUEST_TABLE_SIZE", request_table_size)
                                .define("USE_TABLE_SIZE", use_table_size),
                        )
                        .local_size(LocalSizeConfig::Auto2D)
                        .build(device)
                    },
                )?;

                let request_batch_size = ctx
                    .submit(ctx.access_state_cache(pos, "request_batch_size", input.levels.len()))
                    .await;
                let mut request_batch_size = unsafe {
                    request_batch_size.init(|r| {
                        crate::data::fill_uninit(r, 1usize);
                    })
                };
                let request_batch_size = &mut request_batch_size[level_num];

                let state_initialized = ctx
                    .submit(ctx.access_state_cache_gpu(
                        device,
                        pos,
                        "initialized",
                        Layout::array::<u32>(m_out.chunk_size.hmul()).unwrap(),
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
                let state_values = ctx
                    .submit(ctx.access_state_cache_gpu(
                        device,
                        pos,
                        "values",
                        Layout::array::<f32>(m_out.chunk_size.hmul()).unwrap(),
                    ))
                    .await
                    .unpack();

                let raw_request_table = ctx
                    .submit(ctx.access_state_cache_gpu(
                        device,
                        pos,
                        &format!("request_table"),
                        Layout::array::<FeedbackTableElement>(request_table_size).unwrap(),
                    ))
                    .await;
                let mut request_table = RequestTable::new(device, raw_request_table);

                let raw_use_table = ctx
                    .submit(ctx.access_state_cache_gpu(
                        device,
                        pos,
                        &format!("use_table"),
                        Layout::array::<FeedbackTableElement>(use_table_size).unwrap(),
                    ))
                    .await;
                let mut use_table = UseTable::new(device, raw_use_table);
                let use_table_addr = use_table.buffer_address();

                let gpu_brick_out = ctx
                    .submit(ctx.alloc_slot_gpu(device, pos, &out_info.mem_dimensions))
                    .await;

                let chunk_size = m_out.chunk_size.raw();
                let global_size = [1, chunk_size.y(), chunk_size.x()].into();

                let timed_out = loop {
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

                    let (request_result, ()) = futures::join!(
                        request_table.download_and_insert(
                            *ctx,
                            device,
                            vec![(level, &page_table)],
                            request_batch_size,
                            true,
                            false,
                        ),
                        use_table.download_and_note_use(*ctx, device, &page_table)
                    );

                    // Make writes to the request table, use table and page table visible
                    // (including initialization)
                    ctx.submit(device.barrier(
                        SrcBarrierInfo {
                            stage: vk::PipelineStageFlags2::COMPUTE_SHADER
                                | vk::PipelineStageFlags2::TRANSFER,
                            access: vk::AccessFlags2::SHADER_WRITE
                                | vk::AccessFlags2::TRANSFER_WRITE,
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
                            &request_table,
                            &state_initialized,
                            &state_values,
                        ]);

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            pipeline.push_constant_dyn(&push_constants, |consts| {
                                consts.scalar(page_table_addr.0)?;
                                consts.scalar(use_table_addr.0)?;
                                consts.mat(&transform)?;
                                consts.vec(&m_in.dimensions.raw().into())?;
                                consts.vec(&m_in.chunk_size.raw().into())?;
                                consts.vec(&out_info.begin.raw().into())?;
                                consts.vec(&out_info.mem_dimensions.raw().into())?;
                                Ok(())
                            });

                            pipeline.write_descriptor_set(0, descriptor_config);
                            pipeline.dispatch3d(global_size);
                        }
                    });

                    match request_result {
                        RequestTableResult::Done => break false,
                        RequestTableResult::Timeout => break true,
                        RequestTableResult::Continue => {}
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
