use std::alloc::Layout;

use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};
use id::Identify;

use super::tensor::{FrameOperator, LODImageOperator};

use crate::{
    array::{ImageEmbeddingData, ImageMetaData},
    chunk_utils::ChunkRequestTable,
    coordinate::GlobalCoordinate,
    data::Vector,
    dim::*,
    dtypes::StaticElementType,
    mat::Matrix,
    op_descriptor,
    operator::{OpaqueOperator, OperatorDescriptor},
    operators::tensor::TensorOperator,
    storage::DataVersionType,
    vulkan::{
        memory::TempRessource,
        pipeline::{ComputePipelineBuilder, DescriptorConfig, LocalSizeConfig},
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
) -> FrameOperator {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        transform: cgmath::Matrix3<f32>,
        input_dim: cgmath::Vector2<u32>,
        chunk_dim: cgmath::Vector2<u32>,
        out_begin: cgmath::Vector2<u32>,
        out_mem_dim: cgmath::Vector2<u32>,
    }

    TensorOperator::unbatched(
        op_descriptor!()
            .dependent_on(&input)
            .dependent_on_data(&result_metadata)
            .dependent_on_data(&view_state)
            .unstable(),
        Default::default(),
        result_metadata,
        (input, result_metadata, view_state),
        move |ctx, pos, _, (input, result_metadata, view_state)| {
            async move {
                let device = ctx.preferred_device();

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
                );

                let m_in = level.metadata;
                let emd_l = level.embedding_data;

                let transform =
                    emd_l.physical_to_voxel() * &emd_0.voxel_to_physical() * &out_to_pixel_in;

                let out_info = m_out.chunk_info(pos);

                let num_bricks = m_in.dimension_in_chunks().hmul();

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

                let request_table_size = 256;

                let pipeline = device.request_state(
                    (m_in.chunk_size.hmul(), num_bricks, request_table_size),
                    |device, (mem_size, num_bricks, request_table_size)| {
                        ComputePipelineBuilder::new(
                            Shader::new(include_str!("imageviewer.glsl"))
                                .push_const_block::<PushConstants>()
                                .define("BRICK_MEM_SIZE", mem_size)
                                .define("NUM_BRICKS", num_bricks)
                                .define("REQUEST_TABLE_SIZE", request_table_size),
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

                let request_table = TempRessource::new(
                    device,
                    ctx.submit(ChunkRequestTable::new(request_table_size, device))
                        .await,
                );

                let consts = PushConstants {
                    input_dim: m_in.dimensions.raw().into(),
                    chunk_dim: m_in.chunk_size.raw().into(),
                    out_begin: out_info.begin.raw().into(),
                    out_mem_dim: out_info.mem_dimensions.raw().into(),
                    transform: transform.into(),
                };

                let gpu_brick_out = ctx
                    .submit(ctx.alloc_slot_gpu(device, pos, out_info.mem_elements()))
                    .await;

                let chunk_size = m_out.chunk_size.raw();
                let global_size = [1, chunk_size.y(), chunk_size.x()].into();

                let timed_out = loop {
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
                            &brick_index,
                            request_table.buffer(),
                            &state_initialized,
                            &state_values,
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

                    if let Err(crate::chunk_utils::Timeout) =
                        crate::chunk_utils::request_to_index_with_timeout(
                            &*ctx,
                            device,
                            &mut to_request_linear,
                            level,
                            &brick_index,
                            request_batch_size,
                            true,
                        )
                        .await
                    {
                        break true;
                    }

                    // Clear request table for the next iteration
                    device.with_cmd_buffer(|cmd| request_table.clear(cmd));
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
