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
    pub selected: u32,
    #[pyo3(get, set)]
    pub offset: Vector<2, f32>,
    #[pyo3(get, set)]
    pub zoom_level: f32,
}

impl SliceviewState {
    pub fn projection_mat(
        &self,
        dim: usize,
        input_data: ScalarOperator<VolumeMetaData>,
        embedding_data: ScalarOperator<VolumeEmbeddingData>,
        output_size: Vector<2, GlobalCoordinate>,
    ) -> ScalarOperator<Matrix<4, f32>> {
        use crate::operators::scalar::constant;
        slice_projection_mat(
            dim,
            input_data,
            embedding_data,
            output_size,
            constant(self.selected.into()),
            constant(self.offset),
            constant(self.zoom_level),
        )
    }
}

#[cfg_attr(feature = "python", pymethods)]
impl SliceviewState {
    #[new]
    pub fn new(selected: u32, offset: Vector<2, f32>, zoom_level: f32) -> Self {
        Self {
            selected,
            offset,
            zoom_level,
        }
    }

    fn store(&self, py: pyo3::Python, store: Py<::state_link::py::Store>) -> pyo3::PyObject {
        self.store_py(py, store)
    }

    pub fn drag(&mut self, delta: Vector<2, f32>) {
        self.offset = self.offset + delta;
    }

    pub fn scroll(&mut self, delta: i32) {
        self.selected = (self.selected as i32 + delta).max(0) as u32;
    }

    pub fn zoom(&mut self, delta: f32, on: Vector<2, f32>) {
        let zoom_change = (-delta * 0.05).exp(); //TODO: not entirely happy about the magic
                                                 //constant here...
        self.zoom_level *= zoom_change;

        self.offset = (self.offset - on) / Vector::fill(zoom_change) + on;
    }
}

use super::{
    scalar::ScalarOperator,
    volume::{LODVolumeOperator, VolumeOperator},
};

pub fn slice_projection_mat_z_scaled_fit(
    input_data: ScalarOperator<VolumeMetaData>,
    output_data: ScalarOperator<ImageMetaData>,
    selected_slice: ScalarOperator<GlobalCoordinate>,
) -> ScalarOperator<Matrix<4, f32>> {
    crate::operators::scalar::scalar(
        OperatorId::new("slice_projection_mat_z")
            .dependent_on(&input_data)
            .dependent_on(&output_data)
            .dependent_on(&selected_slice),
        (input_data, output_data, selected_slice),
        move |ctx, (input_data, output_data, selected_slice)| {
            async move {
                let (input_data, output_data, selected_slice) = futures::join! {
                    ctx.submit(input_data.request_scalar()),
                    ctx.submit(output_data.request_scalar()),
                    ctx.submit(selected_slice.request_scalar()),
                };

                let to_pixel_center =
                    Matrix::from_translation(Vector::<2, f32>::fill(0.5).push_dim_large(0.0));
                let vol_dim = input_data.dimensions.map(|v| v.raw as f32);
                let img_dim = output_data.dimensions.map(|v| v.raw as f32);
                let scale = Matrix::from_scale(
                    vol_dim
                        .drop_dim(0)
                        .zip(img_dim, |v, i| v / i)
                        .push_dim_large(1.0),
                )
                .to_homogeneuous();
                let slice_select =
                    Matrix::from_translation(Vector::from([selected_slice.raw as f32, 0.0, 0.0]));
                let to_voxel_center = Matrix::from_translation(Vector::<3, f32>::fill(-0.5));
                let out = to_voxel_center * slice_select * scale * to_pixel_center;
                ctx.write(out)
            }
            .into()
        },
    )
}

pub fn slice_projection_mat(
    dim: usize,
    input_data: ScalarOperator<VolumeMetaData>,
    embedding_data: ScalarOperator<VolumeEmbeddingData>,
    output_size: Vector<2, GlobalCoordinate>,
    selected_slice: ScalarOperator<GlobalCoordinate>,
    offset: ScalarOperator<Vector<2, f32>>,
    zoom_level: ScalarOperator<f32>,
) -> ScalarOperator<Matrix<4, f32>> {
    assert!(dim < 3);
    crate::operators::scalar::scalar(
        OperatorId::new("slice_projection_mat")
            .dependent_on(&dim)
            .dependent_on(&input_data)
            .dependent_on(&output_size)
            .dependent_on(&selected_slice)
            .dependent_on(&offset)
            .dependent_on(&zoom_level),
        (input_data, embedding_data, output_size, selected_slice, offset, zoom_level),
        move |ctx, (input_data, embedding_data, output_size, selected_slice, offset, zoom_level)| {
            async move {
                let (input_data, embedding_data, selected_slice, offset, zoom_level) = futures::join! {
                    ctx.submit(input_data.request_scalar()),
                    ctx.submit(embedding_data.request_scalar()),
                    ctx.submit(selected_slice.request_scalar()),
                    ctx.submit(offset.request_scalar()),
                    ctx.submit(zoom_level.request_scalar()),
                };

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

                let zero = Vector::<3, f32>::fill(0.0);
                let col0 = zero.map_element(h_dim, |_| 1.0);
                let col1 = zero.map_element(v_dim, |_| 1.0);
                let permute = Matrix::new([zero, col1, col0]).to_homogeneuous();

                let to_pixel_center = Matrix::from_translation(Vector::<2, f32>::fill(0.5).push_dim_large(0.0));
                let pixel_transform = permute
                    * Matrix::from_translation(Vector::from([0.0, -offset_y, -offset_x]))
                    * Matrix::from_scale(Vector::fill(zoom_level)).to_homogeneuous()
                    * Matrix::from_translation(offset.map(|v| -v).push_dim_large(0.0))
                    * to_pixel_center;

                let mut translation = Vector::<3, f32>::fill(-0.5); //For "centered" voxel positions
                translation[dim] += selected_slice.raw as f32;
                let scale = Matrix::from_scale(Vector::fill(scaling_factor)).to_homogeneuous();
                let slice_select = Matrix::from_translation(translation);
                let rw_to_voxel = Matrix::from_scale(embedding_data.spacing.map(|v| 1.0/v)).to_homogeneuous();
                let mat = slice_select * rw_to_voxel * scale * pixel_transform;

                ctx.write(mat)
            }
            .into()
        },
    )
}

pub fn slice_projection_mat_centered_rotate(
    input_data: ScalarOperator<VolumeMetaData>,
    embedding_data: ScalarOperator<VolumeEmbeddingData>,
    output_data: ScalarOperator<ImageMetaData>,
    rotation: ScalarOperator<f32>,
) -> ScalarOperator<Matrix<4, f32>> {
    crate::operators::scalar::scalar(
        OperatorId::new("slice_projection_mat_centered_rotate")
            .dependent_on(&input_data)
            .dependent_on(&output_data)
            .dependent_on(&rotation),
        (input_data, embedding_data, output_data, rotation),
        move |ctx, (input_data, embedding_data, output_data, rotation)| {
            async move {
                let (input_data, embedding_data, output_data, rotation) = futures::join! {
                    ctx.submit(input_data.request_scalar()),
                    ctx.submit(embedding_data.request_scalar()),
                    ctx.submit(output_data.request_scalar()),
                    ctx.submit(rotation.request_scalar()),
                };

                let vol_dim = input_data.dimensions.map(|v| v.raw as f32) * embedding_data.spacing;
                let img_dim = output_data.dimensions.map(|v| v.raw as f32);

                let min_dim_img = img_dim.x().min(img_dim.y());
                let min_dim_vol = vol_dim.fold(f32::INFINITY, |a, b| a.min(b));

                let to_pixel_center =
                    Matrix::from_translation(Vector::<2, f32>::fill(0.5).push_dim_large(0.0));
                let central_normalized = Matrix::from_scale(Vector::fill(2.0 / min_dim_img))
                    .to_homogeneuous()
                    * Matrix::from_translation(img_dim.map(|v| -v * 0.5).push_dim_large(0.0))
                    * to_pixel_center;

                let rotation = Matrix::from_angle_y(rotation);
                let norm_to_rw = Matrix::from_translation(vol_dim.map(|v| v * 0.5))
                    * Matrix::from_scale(Vector::fill(min_dim_vol * 0.5)).to_homogeneuous();
                let rw_to_voxel =
                    Matrix::from_scale(embedding_data.spacing.map(|v| 1.0 / v)).to_homogeneuous();
                let to_voxel_center = Matrix::from_translation(Vector::<3, f32>::fill(-0.5));
                let out =
                    to_voxel_center * rw_to_voxel * norm_to_rw * rotation * central_normalized;
                ctx.write(out)
            }
            .into()
        },
    )
}

pub fn render_slice(
    input: LODVolumeOperator<f32>,
    result_metadata: ScalarOperator<ImageMetaData>,
    projection_mat: ScalarOperator<Matrix<4, f32>>,
) -> VolumeOperator<f32> {
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
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <hash.glsl>
#include <mat.glsl>
#include <sample.glsl>

layout (local_size_x = 32, local_size_y = 32) in;

layout(std430, binding = 0) buffer OutputBuffer{
    float values[];
} output_data;

layout(std430, binding = 1) buffer RefBuffer {
    BrickType values[NUM_BRICKS];
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

vec4 map_to_color(float v) {
    v = clamp(v, 0.0, 1.0);
    return vec4(v, v, v, 1.0);
}

void main()
{
    uvec2 out_pos = gl_GlobalInvocationID.xy;
    uint gID = out_pos.x + out_pos.y * consts.out_mem_dim.x;
    if(out_pos.x < consts.out_mem_dim.x && out_pos.y < consts.out_mem_dim.y) {
        uint s = state.values[gID];

        vec4 val;
        if(s == INIT_VAL) {
            float v = brick_values.values[gID];
            val = map_to_color(v);
        } else if(s == INIT_EMPTY) {
            val = vec4(0.0, 0.0, 1.0, 0.0);
        } else {
            vec3 pos = vec3(vec2(out_pos + consts.out_begin), 0);
            //vec3 sample_pos_f = mulh_mat4(transform.value, pos);
            vec3 sample_pos_f = (consts.transform * vec4(pos, 1)).xyz;

            VolumeMetaData m_in;
            m_in.dimensions = consts.vol_dim;
            m_in.chunk_size = consts.chunk_dim;

            // Round to nearest neighbor
            // Floor+0.5 is chosen instead of round to ensure compatibility with f32::round() (see
            // test_sliceviewer below)
            vec3 sample_pos = floor(sample_pos_f + vec3(0.5));

            ivec3 vol_dim = ivec3(consts.vol_dim);

            int res;
            uint sample_brick_pos_linear;
            float sampled_intensity;
            try_sample(sample_pos, m_in, bricks.values, res, sample_brick_pos_linear, sampled_intensity);

            if(res == SAMPLE_RES_FOUND) {
                val = map_to_color(sampled_intensity);

                state.values[gID] = INIT_VAL;
                brick_values.values[gID] = sampled_intensity;
            } else if(res == SAMPLE_RES_NOT_PRESENT) {
                try_insert_into_hash_table(request_table.values, REQUEST_TABLE_SIZE, sample_brick_pos_linear);
                val = vec4(1.0, 0.0, 0.0, 1.0);
            } else /* SAMPLE_RES_OUTSIDE */ {
                val = vec4(0.0, 0.0, 1.0, 0.0);

                state.values[gID] = INIT_EMPTY;
            }
        }

        for(int c=0; c<4; ++c) {
            output_data.values[4*gID+c] = val[c];
        }
    }
}
"#;

    const N_CHANNELS: u32 = 4;
    fn full_info(r: ImageMetaData) -> VolumeMetaData {
        VolumeMetaData {
            dimensions: r.dimensions.push_dim_small(N_CHANNELS.into()),
            chunk_size: r.chunk_size.push_dim_small(N_CHANNELS.into()),
        }
    }

    TensorOperator::unbatched(
        OperatorId::new("sliceviewer")
            .dependent_on(&input)
            .dependent_on(&result_metadata)
            .dependent_on(&projection_mat),
        result_metadata.clone(),
        (input, result_metadata, projection_mat),
        move |ctx, result_metadata| {
            async move {
                let r = ctx.submit(result_metadata.request_scalar()).await;
                let m = full_info(r);
                ctx.write(m)
            }
            .into()
        },
        move |ctx, pos, (input, result_metadata, projection_mat)| {
            async move {
                let device = ctx.vulkan_device();

                let dst_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };

                let (m2d, emd_0, pixel_to_voxel) = futures::join! {
                    ctx.submit(result_metadata.request_scalar()),
                    ctx.submit(input.levels[0].embedding_data.request_scalar()),
                    ctx.submit(projection_mat.request_scalar()),
                };

                let center: Vector<2, f32> = [0.0, 0.0].into();
                let neighbors: [Vector<2, f32>; 2] = [[0.0, 1.0].into(), [1.0, 0.0].into()];

                let transform_rw = emd_0.voxel_to_physical() * pixel_to_voxel;
                let projected_center = transform_rw.transform(center.push_dim_large(0.0));
                let projected_neighbors =
                    neighbors.map(|v| transform_rw.transform(v.push_dim_large(0.0)));

                let neighbor_dirs = projected_neighbors.map(|v| projected_center - v);
                let mut selected_level = input.levels.len() - 1;

                let coarse_lod_factor = 1.0; //TODO: make configurable
                'outer: for (i, level) in input.levels.iter().enumerate() {
                    let emd = ctx.submit(level.embedding_data.request_scalar()).await;

                    for dir in neighbor_dirs {
                        let abs_dir = dir.map(|v| v.abs()).normalized();
                        let dir_spacing_dist = (abs_dir * emd.spacing).length();
                        let pixel_dist = dir.length();
                        if dir_spacing_dist >= pixel_dist * coarse_lod_factor {
                            selected_level = i;
                            break 'outer;
                        }
                    }
                }

                let level = &input.levels[selected_level];

                let (m_in, emd_l) = futures::join! {
                    ctx.submit(level.metadata.request_scalar()),
                    ctx.submit(level.embedding_data.request_scalar()),
                };

                let transform =
                    emd_l.physical_to_voxel() * emd_0.voxel_to_physical() * pixel_to_voxel;

                let m_out = full_info(m2d);
                let out_info = m_out.chunk_info(pos);

                let num_bricks = hmul(m_in.dimension_in_chunks());

                let brick_index = device
                    .storage
                    .get_index(*ctx, device, level.chunks.id(), num_bricks, dst_info)
                    .await;

                let request_table_size = 256;
                let request_batch_size = 32;

                let pipeline =
                    device.request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .push_const_block::<PushConstants>()
                                    .add("BRICK_MEM_SIZE", hmul(m_in.chunk_size))
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
                        Layout::array::<u32>(hmul(m2d.chunk_size)).unwrap(),
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
                let state_values = ctx
                    .access_state_cache(
                        device,
                        pos,
                        "values",
                        Layout::array::<f32>(hmul(m2d.chunk_size)).unwrap(),
                    )
                    .unwrap()
                    .unpack();

                let request_table =
                    TempRessource::new(device, ChunkRequestTable::new(request_table_size, device));

                let dim_in_bricks = m_in.dimension_in_chunks();
                let consts = PushConstants {
                    vol_dim: m_in.dimensions.raw().into(),
                    chunk_dim: m_in.chunk_size.raw().into(),
                    out_begin: out_info.begin.drop_dim(2).raw().into(),
                    out_mem_dim: out_info.mem_dimensions.drop_dim(2).raw().into(),
                    transform: transform.into(),
                };

                let gpu_brick_out = ctx
                    .alloc_slot_gpu(device, pos, out_info.mem_elements())
                    .unwrap();
                let chunk_size = m2d.chunk_size.raw();
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

                    // Fulfill requests
                    to_request_linear.sort_unstable();

                    for batch in to_request_linear.chunks(request_batch_size) {
                        let to_request = batch.iter().map(|v| {
                            assert!(*v < num_bricks as _);
                            level.chunks.request_gpu(
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
                        println!("Sliceviewer: Time out result after {} it", it);
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
        data::{GlobalCoordinate, Vector, VoxelPosition},
        operators::{volume::VolumeOperatorState, volume_gpu::VoxelRasterizerGLSL},
        test_util::compare_volume,
    };

    fn test_sliceviewer_configuration(
        img_size: Vector<2, GlobalCoordinate>,
        vol_size: Vector<3, GlobalCoordinate>,
    ) {
        let num_channels = 4;
        let img_size_c = VoxelPosition::from([img_size.y(), img_size.x(), num_channels.into()]);
        for z in 0..vol_size.z().raw {
            let fill_expected = |comp: &mut ndarray::ArrayViewMut3<f32>| {
                for y in 0..img_size_c[0].raw {
                    for x in 0..img_size_c[1].raw {
                        for c in 0..img_size_c[2].raw {
                            let pos = VoxelPosition::from([y, x, c]);
                            let voxel_y = (y as f32 + 0.5) / img_size.y().raw as f32
                                * vol_size.y().raw as f32
                                - 0.5;
                            let voxel_x = (x as f32 + 0.5) / img_size.x().raw as f32
                                * vol_size.x().raw as f32
                                - 0.5;
                            let val = if c == num_channels - 1 {
                                1.0
                            } else {
                                // Note: We are dividing by 32 here to avoid running into clamping
                                // between [0, 1]. We may be able to avoid this once we have proper
                                // color mapping.
                                (voxel_x.round() + voxel_y.round() + z as f32) / 32.0
                            };
                            comp[pos.as_index()] = val;
                        }
                    }
                }
            };

            let input = VoxelRasterizerGLSL {
                metadata: crate::array::VolumeMetaData {
                    dimensions: vol_size,
                    chunk_size: (vol_size / Vector::fill(2u32)).local(),
                },
                body: r#"result = float(pos_voxel.x + pos_voxel.y + pos_voxel.z)/32.0;"#.to_owned(),
            };

            let img_meta = ImageMetaData {
                dimensions: img_size,
                chunk_size: (img_size / Vector::fill(3u32)).local(),
            };
            let input = input.operate();
            let slice_proj = super::slice_projection_mat_z_scaled_fit(
                input.metadata.clone(),
                img_meta.into(),
                crate::operators::scalar::constant(z.into()),
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
            compare_volume(slice, fill_expected);
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
