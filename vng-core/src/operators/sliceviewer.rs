use ash::vk;
use crevice::std140::AsStd140;
use futures::StreamExt;

use crate::{
    array::{ImageMetaData, VolumeMetaData},
    data::{BrickPosition, GlobalCoordinate, Vector},
    operator::OperatorId,
    operators::tensor::TensorOperator,
    task::RequestStream,
    vulkan::{
        pipeline::{ComputePipeline, DescriptorConfig},
        state::RessourceId,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::{scalar::ScalarOperator, volume::VolumeOperator};

pub fn slice_projection_mat_z<'a>(
    input_data: ScalarOperator<'a, VolumeMetaData>,
    output_data: ScalarOperator<'a, ImageMetaData>,
    selected_slice: ScalarOperator<'a, GlobalCoordinate>,
) -> ScalarOperator<'a, mint::ColumnMatrix3<f32>> {
    crate::operators::scalar::scalar(
        OperatorId::new("slice_projection_mat_z")
            .dependent_on(&input_data)
            .dependent_on(&output_data)
            .dependent_on(&selected_slice),
        (input_data, output_data, selected_slice),
        move |ctx, (input_data, output_data, selected_slice), _| {
            async move {
                let (input_data, output_data, selected_slice) = futures::join! {
                    ctx.submit(input_data.request_scalar()),
                    ctx.submit(output_data.request_scalar()),
                    ctx.submit(selected_slice.request_scalar()),
                };

                let vol_dim = input_data.dimensions.map(|v| v.raw as f32);
                let img_dim = output_data.dimensions.map(|v| v.raw as f32);

                let aspect_ratio_img = img_dim.x() / img_dim.y();
                let aspect_ratio_vol = vol_dim.x() / vol_dim.y();
                let scaling_factor = if aspect_ratio_img > aspect_ratio_vol {
                    vol_dim.y() / img_dim.y()
                } else {
                    vol_dim.x() / img_dim.x()
                };

                let offset_x = (img_dim.x() - (vol_dim.x() / scaling_factor)).max(0.0) * 0.5;
                let offset_y = (img_dim.y() - (vol_dim.y() / scaling_factor)).max(0.0) * 0.5;

                let offset_pixel = cgmath::Matrix3::from_translation(cgmath::Vector2 {
                    x: -offset_x,
                    y: -offset_y,
                });

                let scale = cgmath::Matrix3::from_scale(scaling_factor);
                let mut mat = scale * offset_pixel;
                mat.z.z = selected_slice.raw as f32 + 0.5; //For +0.5 see below

                let out = mat.into();
                ctx.write(out)
            }
            .into()
        },
    )
}

pub fn render_slice<'a>(
    input: VolumeOperator<'a>,
    result_metadata: ScalarOperator<'a, ImageMetaData>,
    projection_mat: ScalarOperator<'a, mint::ColumnMatrix3<f32>>,
) -> VolumeOperator<'a> {
    #[derive(Copy, Clone, AsStd140)]
    struct PushConstants {
        transform: mint::ColumnMatrix3<f32>,
        vol_dim: mint::Vector3<u32>,
        chunk_dim: mint::Vector3<u32>,
        brick_region_size: mint::Vector3<u32>,
        llb_brick: mint::Vector3<u32>,
        out_begin: mint::Vector2<u32>,
        out_mem_dim: mint::Vector2<u32>,
    }
    const SHADER: &'static str = r#"
#version 450

#extension GL_EXT_buffer_reference : require

layout (local_size_x = 32, local_size_y = 32) in;

layout(buffer_reference, std430) buffer BrickType {
    float values[];
};


layout(std430, binding = 0) buffer OutputBuffer{
    float values[];
} outputData;

layout(std430, binding = 1) buffer RefBuffer {
    BrickType values[];
} bricks;

layout(std140, push_constant) uniform PushConstants
{
    mat3 transform;
    uvec3 vol_dim;
    uvec3 chunk_dim;
    uvec3 brick_region_size;
    uvec3 llb_brick;
    uvec2 out_begin;
    uvec2 out_mem_dim;
} consts;

uvec2 from_linear(uint linear_pos, uvec2 size) {
    uvec2 vec_pos;
    vec_pos.x = linear_pos % size.x;
    linear_pos /= size.x;
    vec_pos.y = linear_pos % size.y;

    return vec_pos;
}

uint to_linear(uvec3 vec_pos, uvec3 size) {
    return vec_pos.x + size.x*(vec_pos.y + size.y*vec_pos.z);
}

void main()
{
    uvec2 out_pos = gl_GlobalInvocationID.xy;
    uint gID = out_pos.x + out_pos.y * consts.out_mem_dim.x;
    if(out_pos.x < consts.out_mem_dim.x && out_pos.y < consts.out_mem_dim.y) {
        vec4 val;

        //TODO: Maybe revisit this +0.5 -0.5 business.
        vec3 pos = vec3(vec2(out_pos + consts.out_begin) + vec2(0.5), 1);
        vec3 sample_pos_f = consts.transform * pos - vec3(0.5);
        ivec3 sample_pos = ivec3(floor(sample_pos_f + vec3(0.5)));
        ivec3 vol_dim = ivec3(consts.vol_dim);

        if(all(lessThanEqual(ivec3(0), sample_pos)) && all(lessThan(sample_pos, vol_dim))) {
            uvec3 sample_brick = sample_pos / consts.chunk_dim;
            uvec3 urb_brick = consts.brick_region_size + consts.llb_brick;
            if(all(lessThanEqual(consts.llb_brick, sample_brick)) && all(lessThan(sample_brick, urb_brick))) {
                uvec3 sample_brick_region = sample_brick - consts.llb_brick;
                uint sample_brick_pos_linear = to_linear(sample_brick_region, consts.brick_region_size);

                uvec3 brick_begin = sample_brick * consts.chunk_dim;
                uvec3 local = sample_pos - brick_begin;
                uint local_index = to_linear(local, consts.chunk_dim);
                float v = bricks.values[sample_brick_pos_linear].values[local_index];
                val = vec4(v, v, v, 1.0);
            } else {
                val = vec4(0.0, 0.0, 0.0, 0.0);
            }
        } else {
            val = vec4(0.0, 0.0, 0.0, 0.0);
        }

        for(int c=0; c<4; ++c) {
            outputData.values[4*gID+c] = val[c];
        }
    }
}
"#;

    const N_CHANNELS: u32 = 4;
    fn full_info(r: ImageMetaData) -> VolumeMetaData {
        VolumeMetaData {
            dimensions: [r.dimensions.y(), r.dimensions.x(), N_CHANNELS.into()].into(),
            chunk_size: [r.chunk_size.y(), r.chunk_size.x(), N_CHANNELS.into()].into(),
        }
    }

    TensorOperator::with_state(
        OperatorId::new("sliceviewer")
            .dependent_on(&input)
            .dependent_on(&result_metadata)
            .dependent_on(&projection_mat),
        result_metadata.clone(),
        (input, result_metadata, projection_mat),
        move |ctx, result_metadata, _| {
            async move {
                let r = ctx.submit(result_metadata.request_scalar()).await;
                let m = full_info(r);
                ctx.write(m)
            }
            .into()
        },
        move |ctx, positions, (input, result_metadata, projection_mat), _| {
            async move {
                let device = ctx.vulkan_device();

                let m_in = ctx.submit(input.metadata.request_scalar()).await;

                let (m2d, transform) = futures::join! {
                    ctx.submit(result_metadata.request_scalar()),
                    ctx.submit(projection_mat.request_scalar()),
                };
                let m = full_info(m2d);

                let mut max_bricks = 0;
                let requests = positions
                    .into_iter()
                    .map(|pos| {
                        let out_info = m.chunk_info(pos);
                        let out_begin_pixel = out_info.begin().raw().map(|v| v as f32).drop_dim(2);
                        let out_end_pixel = out_info.end().raw().map(|v| v as f32).drop_dim(2);

                        let out_begin_voxel = transform * out_begin_pixel.to_homogeneous_coord();
                        let out_end_voxel = transform * out_end_pixel.to_homogeneous_coord();
                        let llb =
                            out_begin_voxel.zip(out_end_voxel, |a, b| a.min(b).floor() as u32);
                        let urb =
                            out_begin_voxel.zip(out_end_voxel, |a, b| a.max(b).floor() as u32);

                        let max_brick_pos = m_in.dimension_in_bricks() - Vector::fill(1u32);
                        let llb_brick = m_in
                            .chunk_pos(llb.global())
                            .zip(max_brick_pos, |a, b| a.min(b));
                        // Clamp to valid range:
                        let urb_brick = m_in
                            .chunk_pos(urb.global())
                            .zip(max_brick_pos, |a, b| a.min(b));

                        let brick_region_size = urb_brick + Vector::fill(1u32) - llb_brick;

                        max_bricks = max_bricks.max(crate::data::hmul(brick_region_size));

                        let low = llb_brick.raw();
                        let high = urb_brick.raw();

                        let in_brick_positions = itertools::iproduct! {
                            low.z()..=high.z(),
                            low.y()..=high.y(),
                            low.x()..=high.x()
                        }
                        .map(|(z, y, x)| BrickPosition::from([z, y, x]))
                        .collect::<Vec<_>>();
                        let intersecting_bricks = ctx.group(in_brick_positions.iter().map(|pos| {
                            input.bricks.request_gpu(
                                device.id,
                                *pos,
                                DstBarrierInfo {
                                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                    access: vk::AccessFlags2::SHADER_READ,
                                },
                            )
                        }));

                        (intersecting_bricks, (pos, llb_brick, brick_region_size))
                    })
                    .collect::<Vec<_>>();

                let pipeline = device
                    .request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(device, SHADER, false)
                    });

                let mut stream = ctx
                    .submit_unordered_with_data(requests.into_iter())
                    .then_req_with_data(
                        *ctx,
                        |(intersecting_bricks, (pos, llb_brick, brick_region_size))| {
                            let out_info = m.chunk_info(pos);
                            let gpu_brick_out = ctx
                                .alloc_slot_gpu(device, pos, out_info.mem_elements())
                                .unwrap();

                            let brick_index_layout =
                                std::alloc::Layout::array::<u64>(max_bricks).unwrap();
                            let brick_index =
                                device.tmp_buffers.request(device, brick_index_layout);

                            let addrs = intersecting_bricks
                                .iter()
                                .map(|brick| {
                                    let info = ash::vk::BufferDeviceAddressInfo::builder()
                                        .buffer(brick.buffer);
                                    unsafe { device.functions().get_buffer_device_address(&info) }
                                })
                                .collect::<Vec<_>>();

                            device.with_cmd_buffer(|cmd| {
                                unsafe {
                                    device.functions().cmd_update_buffer(
                                        cmd.raw(),
                                        brick_index.allocation.buffer,
                                        0,
                                        bytemuck::cast_slice(&addrs),
                                    )
                                };
                            });
                            let barrier_req = device.barrier(
                                SrcBarrierInfo {
                                    stage: vk::PipelineStageFlags2::TRANSFER,
                                    access: vk::AccessFlags2::TRANSFER_WRITE,
                                },
                                DstBarrierInfo {
                                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                    access: vk::AccessFlags2::SHADER_READ,
                                },
                            );

                            let consts = PushConstants {
                                transform,
                                vol_dim: m_in.dimensions.raw().into(),
                                chunk_dim: m_in.chunk_size.raw().into(),
                                brick_region_size: brick_region_size.raw().into(),
                                llb_brick: llb_brick.raw().into(),
                                out_begin: out_info.begin.drop_dim(2).raw().into(),
                                out_mem_dim: out_info.mem_dimensions.drop_dim(2).raw().into(),
                            };

                            (barrier_req, (consts, brick_index, gpu_brick_out))
                        },
                    );

                while let Some(((), (consts, brick_index, gpu_brick_out))) = stream.next().await {
                    let chunk_size = m2d.chunk_size.raw();
                    let global_size = [1, chunk_size.y(), chunk_size.x()].into();

                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config =
                            DescriptorConfig::new([&gpu_brick_out, &brick_index]);

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            pipeline.push_constant(consts);
                            pipeline.write_descriptor_set(0, descriptor_config);
                            pipeline.dispatch3d(global_size);
                        }
                    });

                    // Safety: The buffer was requested above from the same device
                    unsafe { device.tmp_buffers.return_buf(device, brick_index) };

                    unsafe {
                        gpu_brick_out.initialized(SrcBarrierInfo {
                            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            access: vk::AccessFlags2::SHADER_WRITE,
                        })
                    };
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
                for y in 0..img_size_c.0[0].raw {
                    for x in 0..img_size_c.0[1].raw {
                        for c in 0..img_size_c.0[2].raw {
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
                                voxel_x.round() + voxel_y.round() + z as f32
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
                body: r#"result = float(pos_voxel.x + pos_voxel.y + pos_voxel.z);"#.to_owned(),
            };

            let img_meta = ImageMetaData {
                dimensions: img_size,
                chunk_size: (img_size / Vector::fill(3u32)).local(),
            };
            let input = input.operate();
            let slice_proj = super::slice_projection_mat_z(
                input.metadata.clone(),
                crate::operators::scalar::constant_hash(img_meta),
                crate::operators::scalar::constant_hash(z.into()),
            );
            let slice = super::render_slice(
                input,
                crate::operators::scalar::constant_hash(img_meta),
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
