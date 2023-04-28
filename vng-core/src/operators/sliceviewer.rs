use std::alloc::Layout;

use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};

use crate::{
    array::{ImageMetaData, VolumeMetaData},
    data::{from_linear, hmul, GlobalCoordinate, Vector},
    operator::OperatorId,
    operators::tensor::TensorOperator,
    storage::DataVersionType,
    vulkan::{
        pipeline::{ComputePipeline, DescriptorConfig},
        shader::ShaderDefines,
        state::RessourceId,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::{scalar::ScalarOperator, volume::VolumeOperator};

pub fn slice_projection_mat_z_scaled_fit<'a>(
    input_data: ScalarOperator<'a, VolumeMetaData>,
    output_data: ScalarOperator<'a, ImageMetaData>,
    selected_slice: ScalarOperator<'a, GlobalCoordinate>,
) -> ScalarOperator<'a, cgmath::Matrix4<f32>> {
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
                let out = cgmath::Matrix4::from_translation(cgmath::Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: selected_slice.raw as f32 + 0.5,
                }) * cgmath::Matrix4::from_nonuniform_scale(
                    vol_dim.x() / img_dim.x(),
                    vol_dim.y() / img_dim.y(),
                    1.0,
                );
                ctx.write(out)
            }
            .into()
        },
    )
}

pub fn slice_projection_mat_z<'a>(
    input_data: ScalarOperator<'a, VolumeMetaData>,
    output_data: ScalarOperator<'a, ImageMetaData>,
    selected_slice: ScalarOperator<'a, GlobalCoordinate>,
    offset: ScalarOperator<'a, Vector<2, f32>>,
    zoom_level: ScalarOperator<'a, f32>,
) -> ScalarOperator<'a, cgmath::Matrix4<f32>> {
    crate::operators::scalar::scalar(
        OperatorId::new("slice_projection_mat_z")
            .dependent_on(&input_data)
            .dependent_on(&output_data)
            .dependent_on(&selected_slice)
            .dependent_on(&offset)
            .dependent_on(&zoom_level),
        (input_data, output_data, selected_slice, offset, zoom_level),
        move |ctx, (input_data, output_data, selected_slice, offset, zoom_level), _| {
            async move {
                let (input_data, output_data, selected_slice, offset, zoom_level) = futures::join! {
                    ctx.submit(input_data.request_scalar()),
                    ctx.submit(output_data.request_scalar()),
                    ctx.submit(selected_slice.request_scalar()),
                    ctx.submit(offset.request_scalar()),
                    ctx.submit(zoom_level.request_scalar()),
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

                let pixel_transform = cgmath::Matrix4::from_translation(cgmath::Vector3 {
                    x: -offset_x,
                    y: -offset_y,
                    z: 0.0,
                }) * cgmath::Matrix4::from_scale(zoom_level)
                    * cgmath::Matrix4::from_translation(cgmath::Vector3 {
                        x: -offset.x(),
                        y: -offset.y(),
                        z: 0.0,
                    });

                let scale = cgmath::Matrix4::from_scale(scaling_factor);
                let slice_select = cgmath::Matrix4::from_translation(cgmath::Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: selected_slice.raw as f32 + 0.5, //For +0.5 see below
                });
                let mat = slice_select * scale * pixel_transform;

                let out = mat.into();
                ctx.write(out)
            }
            .into()
        },
    )
}

pub fn slice_projection_mat_centered_rotate<'a>(
    input_data: ScalarOperator<'a, VolumeMetaData>,
    output_data: ScalarOperator<'a, ImageMetaData>,
    rotation: ScalarOperator<'a, f32>,
) -> ScalarOperator<'a, cgmath::Matrix4<f32>> {
    crate::operators::scalar::scalar(
        OperatorId::new("slice_projection_mat_centered_rotate")
            .dependent_on(&input_data)
            .dependent_on(&output_data)
            .dependent_on(&rotation),
        (input_data, output_data, rotation),
        move |ctx, (input_data, output_data, rotation), _| {
            async move {
                let (input_data, output_data, rotation) = futures::join! {
                    ctx.submit(input_data.request_scalar()),
                    ctx.submit(output_data.request_scalar()),
                    ctx.submit(rotation.request_scalar()),
                };

                let vol_dim = input_data.dimensions.map(|v| v.raw as f32);
                let img_dim = output_data.dimensions.map(|v| v.raw as f32);

                let min_dim_img = img_dim.x().min(img_dim.y());
                let min_dim_vol = vol_dim.fold(f32::INFINITY, |a, b| a.min(b));

                let central_normalized = cgmath::Matrix4::from_scale(2.0 / min_dim_img)
                    * cgmath::Matrix4::from_translation(cgmath::Vector3 {
                        x: -img_dim.x() * 0.5,
                        y: -img_dim.y() * 0.5,
                        z: 0.0,
                    });

                let rotation = cgmath::Matrix4::from_angle_y(cgmath::Rad(rotation));
                let norm_to_vol =
                    cgmath::Matrix4::from_translation((vol_dim * Vector::fill(0.5)).into())
                        * cgmath::Matrix4::from_scale(min_dim_vol * 0.5);
                let out = norm_to_vol * rotation * central_normalized;
                ctx.write(out)
            }
            .into()
        },
    )
}

pub fn render_slice<'a>(
    input: VolumeOperator<'a>,
    result_metadata: ScalarOperator<'a, ImageMetaData>,
    projection_mat: ScalarOperator<'a, cgmath::Matrix4<f32>>,
) -> VolumeOperator<'a> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        vol_dim: cgmath::Vector3<u32>,
        chunk_dim: cgmath::Vector3<u32>,
        dim_in_bricks: cgmath::Vector3<u32>,
        out_begin: cgmath::Vector2<u32>,
        out_mem_dim: cgmath::Vector2<u32>,
    }

    const SHADER: &'static str = r#"
#version 450

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require

#include <util.glsl>
#include <hash.glsl>

layout (local_size_x = 32, local_size_y = 32) in;

layout(buffer_reference, std430) buffer BrickType {
    float values[BRICK_MEM_SIZE];
};


layout(std430, binding = 0) buffer OutputBuffer{
    float values[];
} outputData;

layout(std430, binding = 1) buffer Transform {
    mat4 value;
} transform;

layout(std430, binding = 2) buffer RefBuffer {
    BrickType values[NUM_BRICKS];
} bricks;

layout(std430, binding = 3) buffer QueryTable {
    uint64_t values[REQUEST_TABLE_SIZE];
} request_table;

declare_push_consts(consts);

void main()
{
    uvec2 out_pos = gl_GlobalInvocationID.xy;
    uint gID = out_pos.x + out_pos.y * consts.out_mem_dim.x;
    if(out_pos.x < consts.out_mem_dim.x && out_pos.y < consts.out_mem_dim.y) {
        vec4 val;

        //TODO: Maybe revisit this +0.5 -0.5 business.
        vec4 pos = vec4(vec2(out_pos + consts.out_begin) + vec2(0.5), 0, 1);
        vec3 sample_pos_f = (transform.value * pos).xyz - vec3(0.5);
        ivec3 sample_pos = ivec3(floor(sample_pos_f + vec3(0.5)));
        ivec3 vol_dim = ivec3(consts.vol_dim);

        if(all(lessThanEqual(ivec3(0), sample_pos)) && all(lessThan(sample_pos, vol_dim))) {
            uvec3 sample_brick = sample_pos / consts.chunk_dim;

            uint sample_brick_pos_linear = to_linear3(sample_brick, consts.dim_in_bricks);

            BrickType brick = bricks.values[sample_brick_pos_linear];
            if(uint64_t(brick) == 0) {
                uint64_t sbp = uint64_t(sample_brick_pos_linear);
                try_insert_into_hash_table(request_table.values, REQUEST_TABLE_SIZE, sample_brick_pos_linear);
                val = vec4(1.0, 0.0, 0.0, 1.0);
            } else {
                uvec3 brick_begin = sample_brick * consts.chunk_dim;
                uvec3 local = sample_pos - brick_begin;
                uint local_index = to_linear3(local, consts.chunk_dim);
                float v = brick.values[local_index];
                val = vec4(v, v, v, 1.0);
            }
        } else {
            val = vec4(0.0, 0.0, 1.0, 0.0);
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

    TensorOperator::unbatched(
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
        move |ctx, pos, (input, result_metadata, projection_mat), _| {
            async move {
                let device = ctx.vulkan_device();

                let m_in = ctx.submit(input.metadata.request_scalar()).await;

                let dst_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };

                let (m2d, transform_gpu) = futures::join! {
                    ctx.submit(result_metadata.request_scalar()),
                    ctx.submit(projection_mat.request_scalar_gpu(device.id, dst_info)),
                };
                let m_out = full_info(m2d);
                let out_info = m_out.chunk_info(pos);

                let num_bricks = hmul(m_in.dimension_in_bricks()); //TODO: rename

                let load_factor = 3.0;
                let request_table_size = (((num_bricks as f32).powf(2.0 / 3.0) * load_factor)
                    .round() as usize)
                    .next_power_of_two();

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

                let brick_index_layout = Layout::array::<u64>(num_bricks).unwrap();

                // TODO: Manage brick indices globally to avoid duplicate work.
                let brick_index = device.tmp_buffers.request(device, brick_index_layout);

                let request_table_buffer_layout = Layout::array::<u64>(request_table_size).unwrap();
                let request_table_buffer = device
                    .tmp_buffers
                    .request(device, request_table_buffer_layout);

                device.with_cmd_buffer(|cmd| unsafe {
                    device.functions().cmd_fill_buffer(
                        cmd.raw(),
                        request_table_buffer.allocation.buffer,
                        0,
                        vk::WHOLE_SIZE,
                        0xffffffff,
                    );
                    device.functions().cmd_fill_buffer(
                        cmd.raw(),
                        brick_index.allocation.buffer,
                        0,
                        vk::WHOLE_SIZE,
                        0,
                    );
                });

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

                let dim_in_bricks = m_in.dimension_in_bricks();
                let consts = PushConstants {
                    vol_dim: m_in.dimensions.raw().into(),
                    chunk_dim: m_in.chunk_size.raw().into(),
                    dim_in_bricks: dim_in_bricks.raw().into(),
                    out_begin: out_info.begin.drop_dim(2).raw().into(),
                    out_mem_dim: out_info.mem_dimensions.drop_dim(2).raw().into(),
                };

                let gpu_brick_out = ctx
                    .alloc_slot_gpu(device, pos, out_info.mem_elements())
                    .unwrap();
                let chunk_size = m2d.chunk_size.raw();
                let global_size = [1, chunk_size.y(), chunk_size.x()].into();

                // TODO: Free bricks once it is safe to do so. Ties in to a global handling of the
                // brick index.
                // Once we do that, we have to maintain a temporary buffer that stores which output
                // pixels have been written already, though!
                let mut collected_bricks = Vec::new();

                let mut it = 0; //NO_PUSH_main
                let timed_out = loop {
                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config = DescriptorConfig::new([
                            &gpu_brick_out,
                            &transform_gpu,
                            &brick_index,
                            &request_table_buffer,
                        ]);

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            pipeline.push_constant(consts);
                            pipeline.write_descriptor_set(0, descriptor_config);
                            pipeline.dispatch3d(global_size);
                        }
                    });

                    let src_info = SrcBarrierInfo {
                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::SHADER_WRITE,
                    };
                    let dst_info = DstBarrierInfo {
                        stage: vk::PipelineStageFlags2::TRANSFER,
                        access: vk::AccessFlags2::TRANSFER_READ,
                    };
                    ctx.submit(device.barrier(src_info, dst_info)).await;

                    let mut request_table_cpu = vec![0u64; request_table_size];
                    let request_table_cpu_bytes =
                        bytemuck::cast_slice_mut(request_table_cpu.as_mut_slice());
                    unsafe {
                        crate::vulkan::memory::copy_to_cpu(
                            *ctx,
                            device,
                            request_table_buffer.allocation.buffer,
                            request_table_buffer_layout,
                            request_table_cpu_bytes.as_mut_ptr(),
                        )
                        .await
                    };

                    let to_request_linear = request_table_cpu
                        .into_iter()
                        .filter(|v| *v != u64::max_value())
                        .collect::<Vec<u64>>();
                    if to_request_linear.is_empty() {
                        break false;
                    }
                    if ctx.past_deadline() {
                        break true;
                    }

                    let to_request = to_request_linear.iter().map(|v| {
                        input.bricks.request_gpu(
                            device.id,
                            from_linear(*v as usize, dim_in_bricks),
                            dst_info,
                        )
                    });
                    let requested_bricks = ctx.submit(ctx.group(to_request)).await;

                    device.with_cmd_buffer(|cmd| unsafe {
                        for (brick, brick_linear_pos) in
                            requested_bricks.iter().zip(to_request_linear.into_iter())
                        {
                            let info =
                                ash::vk::BufferDeviceAddressInfo::builder().buffer(brick.buffer);
                            let addr = device.functions().get_buffer_device_address(&info);
                            device.functions().cmd_update_buffer(
                                cmd.raw(),
                                brick_index.allocation.buffer,
                                brick_linear_pos * std::mem::size_of::<u64>() as u64,
                                bytemuck::bytes_of(&addr),
                            );
                        }
                        device.functions().cmd_fill_buffer(
                            cmd.raw(),
                            request_table_buffer.allocation.buffer,
                            0,
                            vk::WHOLE_SIZE,
                            0xffffffff,
                        );
                    });

                    // Make sure the bricks are not freed as long as we reference them from the
                    // index buffer
                    collected_bricks.extend(requested_bricks);

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
                    it += 1;
                };

                // Safety: The buffer was requested above from the same device
                unsafe { device.tmp_buffers.return_buf(device, brick_index) };

                // Safety: The buffer was requested above from the same device
                unsafe { device.tmp_buffers.return_buf(device, request_table_buffer) };

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
            let slice_proj = super::slice_projection_mat_z_scaled_fit(
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
