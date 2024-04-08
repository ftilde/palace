use std::alloc::Layout;

use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};

use super::{
    tensor::{EmbeddedTensorOperator, FrameOperator, LODTensorOperator},
    volume::LODVolumeOperator,
};

use crate::{
    array::{ImageMetaData, VolumeEmbeddingData, VolumeMetaData},
    chunk_utils::ChunkRequestTable,
    data::{GlobalCoordinate, Matrix, Vector},
    dim::*,
    operator::{OpaqueOperator, OperatorDescriptor},
    operators::tensor::TensorOperator,
    storage::DataVersionType,
    vulkan::{
        memory::TempRessource,
        pipeline::{ComputePipeline, DescriptorConfig},
        shader::ShaderDefines,
        state::RessourceId,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[derive(Clone, state_link::State)]
#[cfg_attr(feature = "python", pyclass)]
pub struct SliceviewState {
    #[pyo3(get, set)]
    pub depth: f32,
    #[pyo3(get, set)]
    pub offset: Vector<D2, f32>,
    #[pyo3(get, set)]
    pub zoom_level: f32,
    pub dim: u32,
    pub dim_spacing: f32,
    pub dim_end: f32,
}

#[cfg_attr(feature = "python", pymethods)]
impl SliceviewState {
    #[staticmethod]
    pub fn for_volume(
        input_metadata: VolumeMetaData,
        embedding_data: VolumeEmbeddingData,
        dim: u32,
    ) -> Self {
        let real_size = embedding_data.spacing * input_metadata.dimensions.raw().f32();
        let dim_spacing = embedding_data.spacing[dim as usize];
        let dim_end = real_size[dim as usize];
        Self {
            dim_spacing,
            dim_end,
            dim,
            offset: Vector::fill(0.0),
            zoom_level: 1.0,
            depth: dim_end * 0.5,
        }
    }

    fn store(&self, py: pyo3::Python, store: Py<::state_link::py::Store>) -> pyo3::PyObject {
        self.store_py(py, store)
    }

    pub fn drag(&mut self, delta: Vector<D2, f32>) {
        self.offset = self.offset + delta;
    }

    pub fn scroll(&mut self, delta: i32) {
        self.depth = (self.depth + delta as f32 * self.dim_spacing).clamp(0.0, self.dim_end);
    }

    pub fn zoom(&mut self, delta: f32, on: Vector<D2, f32>) {
        let zoom_change = (-delta * 0.05).exp(); //TODO: not entirely happy about the magic
                                                 //constant here...
        self.zoom_level *= zoom_change;

        self.offset = (self.offset - on) / Vector::fill(zoom_change) + on;
    }
    pub fn projection_mat(
        &self,
        input_data: VolumeMetaData,
        embedding_data: VolumeEmbeddingData,
        output_size: Vector<D2, GlobalCoordinate>,
    ) -> Matrix<D4, f32> {
        slice_projection_mat(
            self.dim as _,
            input_data,
            embedding_data,
            output_size,
            ((self.depth / self.dim_spacing).round() as u32).into(),
            self.offset,
            self.zoom_level,
        )
    }
}

pub fn slice_projection_mat_z_scaled_fit(
    input_data: VolumeMetaData,
    output_data: ImageMetaData,
    selected_slice: GlobalCoordinate,
) -> Matrix<D4, f32> {
    let to_pixel_center =
        Matrix::from_translation(Vector::<D2, f32>::fill(0.5).push_dim_large(0.0));
    let vol_dim = input_data.dimensions.map(|v| v.raw as f32);
    let img_dim = output_data.dimensions.map(|v| v.raw as f32);
    let scale = Matrix::from_scale(
        vol_dim
            .drop_dim(0)
            .zip(img_dim, |v, i| v / i)
            .push_dim_large(1.0),
    )
    .to_homogeneous();
    let slice_select =
        Matrix::from_translation(Vector::from([selected_slice.raw as f32, 0.0, 0.0]));
    let to_voxel_center = Matrix::from_translation(Vector::<D3, f32>::fill(-0.5));
    let out = to_voxel_center * slice_select * scale * to_pixel_center;
    out
}

pub fn slice_projection_mat(
    dim: usize,
    input_data: VolumeMetaData,
    embedding_data: VolumeEmbeddingData,
    output_size: Vector<D2, GlobalCoordinate>,
    selected_slice: GlobalCoordinate,
    offset: Vector<D2, f32>,
    zoom_level: f32,
) -> Matrix<D4, f32> {
    assert!(dim < 3);

    let vol_dim_voxel = input_data.dimensions.map(|v| v.raw as f32);
    let vol_dim = vol_dim_voxel * embedding_data.spacing;
    let img_dim = output_size.map(|v| v.raw as f32);

    let (h_dim, v_dim) = match dim {
        0 => (2, 1),
        1 => (2, 0),
        2 => (1, 0),
        _ => unreachable!(),
    };

    let h_size = vol_dim[h_dim];
    let v_size = vol_dim[v_dim];

    let aspect_ratio_img = img_dim.x() / img_dim.y();
    let aspect_ratio_vol = h_size / v_size;
    let scaling_factor = if aspect_ratio_img > aspect_ratio_vol {
        v_size / img_dim.y()
    } else {
        h_size / img_dim.x()
    };

    let offset_x = (img_dim.x() - (h_size / scaling_factor)).max(0.0) * 0.5;
    let offset_y = (img_dim.y() - (v_size / scaling_factor)).max(0.0) * 0.5;

    let zero = Vector::<D3, f32>::fill(0.0);
    let col0 = zero.map_element(h_dim, |_| 1.0);
    let col1 = zero.map_element(v_dim, |_| 1.0);
    let permute = Matrix::<D3, _>::new([zero, col1, col0]).to_homogeneous();

    let to_pixel_center =
        Matrix::from_translation(Vector::<D2, f32>::fill(0.5).push_dim_large(0.0));
    let pixel_transform = permute
        * Matrix::from_translation(Vector::from([0.0, -offset_y, -offset_x]))
        * Matrix::from_scale(Vector::<D3, _>::fill(zoom_level)).to_homogeneous()
        * Matrix::from_translation(offset.map(|v| -v).push_dim_large(0.0))
        * to_pixel_center;

    let mut translation = Vector::<D3, f32>::fill(-0.5); //For "centered" voxel positions
    translation[dim] += selected_slice.raw as f32;
    let scale = Matrix::from_scale(Vector::<D3, _>::fill(scaling_factor)).to_homogeneous();
    let slice_select = Matrix::from_translation(translation);
    let rw_to_voxel = Matrix::from_scale(embedding_data.spacing.map(|v| 1.0 / v)).to_homogeneous();
    let mat = slice_select * rw_to_voxel * scale * pixel_transform;

    mat
}

pub fn slice_projection_mat_centered_rotate(
    input_data: VolumeMetaData,
    embedding_data: VolumeEmbeddingData,
    output_data: ImageMetaData,
    rotation: f32,
) -> Matrix<D4, f32> {
    let vol_dim = input_data.dimensions.map(|v| v.raw as f32) * embedding_data.spacing;
    let img_dim = output_data.dimensions.map(|v| v.raw as f32);

    let min_dim_img = img_dim.x().min(img_dim.y());
    let min_dim_vol = vol_dim.fold(f32::INFINITY, |a, b| a.min(b));

    let to_pixel_center =
        Matrix::from_translation(Vector::<D2, f32>::fill(0.5).push_dim_large(0.0));
    let central_normalized = Matrix::from_scale(Vector::<D3, _>::fill(2.0 / min_dim_img))
        .to_homogeneous()
        * Matrix::from_translation(img_dim.map(|v| -v * 0.5).push_dim_large(0.0))
        * to_pixel_center;

    let rotation = Matrix::from_angle_y(rotation);
    let norm_to_rw = Matrix::from_translation(vol_dim.map(|v| v * 0.5))
        * Matrix::<D3, _>::from_scale(Vector::fill(min_dim_vol * 0.5)).to_homogeneous();
    let rw_to_voxel = Matrix::from_scale(embedding_data.spacing.map(|v| 1.0 / v)).to_homogeneous();
    let to_voxel_center = Matrix::from_translation(Vector::<D3, f32>::fill(-0.5));
    let out = to_voxel_center * rw_to_voxel * norm_to_rw * rotation * central_normalized;
    out
}

pub fn select_level<'a, D: SmallerDim, T>(
    lod: &'a LODTensorOperator<D::Smaller, T>,
    transform_rw: Matrix<D, f32>,
    neighbor_dirs: &[Vector<D::Smaller, f32>],
) -> &'a EmbeddedTensorOperator<D::Smaller, T> {
    let neighbor_dirs = neighbor_dirs
        .iter()
        .map(|p| (transform_rw * p.push_dim_large(0.0)).to_non_homogeneous_coord())
        .collect::<Vec<_>>();

    let mut selected_level = lod.levels.len() - 1;

    let coarse_lod_factor = 1.0; //TODO: make configurable
    'outer: for (i, level) in lod.levels.iter().enumerate() {
        let emd = level.embedding_data;

        for dir in &neighbor_dirs {
            let abs_dir = dir.map(|v| v.abs()).normalized();
            let dir_spacing_dist = (abs_dir * emd.spacing).length();
            let pixel_dist = dir.length();
            if dir_spacing_dist >= pixel_dist * coarse_lod_factor {
                selected_level = i;
                break 'outer;
            }
        }
    }

    &lod.levels[selected_level]
}

pub fn render_slice(
    input: LODVolumeOperator<f32>,
    result_metadata: ImageMetaData,
    projection_mat: Matrix<D4, f32>,
) -> FrameOperator {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        transform: cgmath::Matrix4<f32>,
        vol_dim: cgmath::Vector3<u32>,
        chunk_dim: cgmath::Vector3<u32>,
        out_begin: cgmath::Vector2<u32>,
        out_mem_dim: cgmath::Vector2<u32>,
    }

    const SHADER: &'static str = r#"
#version 450

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : require

#define ChunkValue float

#include <util.glsl>
#include <color.glsl>
#include <hash.glsl>
#include <sample.glsl>

layout (local_size_x = 32, local_size_y = 32) in;

layout(scalar, binding = 0) buffer OutputBuffer{
    u8vec4 values[];
} output_data;

layout(std430, binding = 1) buffer RefBuffer {
    Chunk values[NUM_BRICKS];
} bricks;

layout(std430, binding = 2) buffer QueryTable {
    uint values[REQUEST_TABLE_SIZE];
} request_table;

layout(std430, binding = 3) buffer StateBuffer {
    uint values[];
} state;

layout(std430, binding = 4) buffer ValueBuffer{
    float values[];
} brick_values;

declare_push_consts(consts);

#define UNINIT 0
#define INIT_VAL 1
#define INIT_EMPTY 2

void main()
{
    uvec2 out_pos = gl_GlobalInvocationID.xy;
    uint gID = out_pos.x + out_pos.y * consts.out_mem_dim.x;
    if(out_pos.x < consts.out_mem_dim.x && out_pos.y < consts.out_mem_dim.y) {
        uint s = state.values[gID];

        u8vec4 val;
        if(s == INIT_VAL) {
            float v = brick_values.values[gID];
            val = intensity_to_grey(v);
        } else if(s == INIT_EMPTY) {
            val = u8vec4(0, 0, 255, 255);
        } else {
            vec3 pos = vec3(vec2(out_pos + consts.out_begin), 0);
            //vec3 sample_pos_f = mulh_mat4(transform.value, pos);
            vec3 sample_pos_f = (consts.transform * vec4(pos, 1)).xyz;

            TensorMetaData(3) m_in;
            m_in.dimensions = from_glsl(consts.vol_dim);
            m_in.chunk_size = from_glsl(consts.chunk_dim);

            // Round to nearest neighbor
            // Floor+0.5 is chosen instead of round to ensure compatibility with f32::round() (see
            // test_sliceviewer below)
            vec3 sample_pos_g = floor(sample_pos_f + vec3(0.5));
            float[3] sample_pos = from_glsl(sample_pos_g);

            ivec3 vol_dim = ivec3(consts.vol_dim);

            int res;
            uint sample_brick_pos_linear;
            float sampled_intensity;
            try_sample(3, sample_pos, m_in, bricks.values, res, sample_brick_pos_linear, sampled_intensity);

            if(res == SAMPLE_RES_FOUND) {
                val = intensity_to_grey(sampled_intensity);

                state.values[gID] = INIT_VAL;
                brick_values.values[gID] = sampled_intensity;
            } else if(res == SAMPLE_RES_NOT_PRESENT) {
                try_insert_into_hash_table(request_table.values, REQUEST_TABLE_SIZE, sample_brick_pos_linear);
                val = u8vec4(255, 0, 0, 255);
            } else /* SAMPLE_RES_OUTSIDE */ {
                val = u8vec4(0, 0, 255, 255);

                state.values[gID] = INIT_EMPTY;
            }
        }

        output_data.values[gID] = val;
    }
}
"#;

    TensorOperator::unbatched(
        OperatorDescriptor::new("sliceviewer")
            .dependent_on(&input)
            .dependent_on_data(&result_metadata)
            .dependent_on_data(&projection_mat)
            .unstable(),
        result_metadata,
        (input, result_metadata, projection_mat),
        move |ctx, pos, _, (input, result_metadata, projection_mat)| {
            async move {
                let device = ctx.preferred_device();

                let dst_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };

                let m_out = result_metadata;
                let emd_0 = input.levels[0].embedding_data;
                let pixel_to_voxel = projection_mat;

                let transform_rw = emd_0.voxel_to_physical() * *pixel_to_voxel;
                let level = select_level(
                    &input,
                    transform_rw,
                    &[[0.0, 0.0, 1.0].into(), [0.0, 1.0, 0.0].into()],
                );

                let m_in = level.metadata;
                let emd_l = level.embedding_data;

                let transform =
                    emd_l.physical_to_voxel() * emd_0.voxel_to_physical() * *pixel_to_voxel;

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

                let pipeline =
                    device.request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .push_const_block::<PushConstants>()
                                    .add("BRICK_MEM_SIZE", m_in.chunk_size.hmul())
                                    .add("NUM_BRICKS", num_bricks)
                                    .add("REQUEST_TABLE_SIZE", request_table_size),
                            ),
                            false,
                        )
                    });

                let state_initialized = ctx
                    .submit(ctx.access_state_cache(
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
                    .submit(ctx.access_state_cache(
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
                    vol_dim: m_in.dimensions.raw().into(),
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

#[cfg(test)]
mod test {
    use crate::{
        array::ImageMetaData,
        data::{GlobalCoordinate, Vector},
        dim::*,
        test_util::compare_tensor_fn,
    };

    fn test_sliceviewer_configuration(
        img_size: Vector<D2, GlobalCoordinate>,
        vol_size: Vector<D3, GlobalCoordinate>,
    ) {
        let img_size_c = Vector::<D2, GlobalCoordinate>::from([img_size.y(), img_size.x()]);
        for z in 0..vol_size.z().raw {
            let fill_expected = |comp: &mut ndarray::ArrayViewMut2<Vector<D4, u8>>| {
                for y in 0..img_size_c[0].raw {
                    for x in 0..img_size_c[1].raw {
                        let pos = Vector::<D2, GlobalCoordinate>::from([y, x]);
                        let voxel_y = (y as f32 + 0.5) / img_size.y().raw as f32
                            * vol_size.y().raw as f32
                            - 0.5;
                        let voxel_x = (x as f32 + 0.5) / img_size.x().raw as f32
                            * vol_size.x().raw as f32
                            - 0.5;

                        let mut out = Vector::fill(
                            (((voxel_x.round() + voxel_y.round() + z as f32) / 32.0) * 255.0) as u8,
                        );
                        out[3] = 255;
                        comp[pos.as_index()] = out;
                    }
                }
            };

            let input = crate::operators::procedural::rasterize(
                crate::array::VolumeMetaData {
                    dimensions: vol_size,
                    chunk_size: (vol_size / Vector::fill(2u32)).local(),
                },
                r#"float run(float[3] pos_normalized, uint[3] pos_voxel) { return float(pos_voxel[0] + pos_voxel[1] + pos_voxel[2])/32.0; }"#,
            );

            let img_meta = ImageMetaData {
                dimensions: img_size,
                chunk_size: (img_size / Vector::fill(3u32)).local(),
            };
            let slice_proj = super::slice_projection_mat_z_scaled_fit(
                input.metadata.clone(),
                img_meta.into(),
                z.into(),
            );
            let slice = super::render_slice(
                input
                    .embedded(crate::array::TensorEmbeddingData {
                        spacing: Vector::fill(1.0),
                    })
                    .single_level_lod(),
                img_meta.into(),
                slice_proj,
            );
            compare_tensor_fn(slice, fill_expected);
        }
    }
    #[test]
    fn test_sliceviewer() {
        for img_size in [[5, 3], [6, 6], [10, 20]] {
            for vol_size in [[5, 3, 2], [6, 6, 6], [2, 10, 20]] {
                test_sliceviewer_configuration(img_size.into(), vol_size.into())
            }
        }
    }
}
