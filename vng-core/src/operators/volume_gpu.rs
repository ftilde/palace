use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};
use futures::StreamExt;

use crate::{
    array::VolumeMetaData,
    data::{BrickPosition, LocalCoordinate, Vector},
    id::Id,
    operator::OperatorId,
    operators::tensor::TensorOperator,
    storage::gpu,
    vulkan::{
        pipeline::{AsDescriptors, ComputePipeline, DescriptorConfig},
        shader::ShaderDefines,
        state::RessourceId,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::{
    array::ArrayOperator,
    scalar::ScalarOperator,
    volume::{ChunkSize, VolumeOperator},
};

pub fn linear_rescale<'op>(
    input: VolumeOperator<'op>,
    scale: ScalarOperator<'op, f32>,
    offset: ScalarOperator<'op, f32>,
) -> VolumeOperator<'op> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        chunk_pos: cgmath::Vector3<u32>,
        num_chunk_elems: u32,
    }
    const SHADER: &'static str = r#"
#version 450

#include <util.glsl>

layout (local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputScale{
    float value;
} scale;

layout(std430, binding = 1) readonly buffer InputOffset{
    float value;
} offset;

// Note: We cannot use `restrict` here and below since we bind the same buffer to sourceData and
// outputData in the inplace update case.
layout(std430, binding = 2) readonly buffer InputBuffer{
    float values[];
} sourceData;

layout(std430, binding = 3) buffer OutputBuffer{
    float values[];
} outputData;

declare_push_consts(constants);

void main()
{
    uint gID = gl_GlobalInvocationID.x;

    if(gID < constants.num_chunk_elems) {
        outputData.values[gID] = scale.value*sourceData.values[gID] + offset.value;
    }
}
"#;

    TensorOperator::with_state(
        OperatorId::new("volume_scale_gpu")
            .dependent_on(&input)
            .dependent_on(&scale)
            .dependent_on(&offset),
        input.clone(),
        (input.clone(), scale, offset),
        move |ctx, input, _| {
            async move {
                let req = input.metadata.request_scalar();
                let m = ctx.submit(req).await;
                ctx.write(m)
            }
            .into()
        },
        move |ctx, positions, (input, scale, offset), _| {
            async move {
                let device = ctx.vulkan_device();

                let access_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                };
                let (scale_gpu, offset_gpu, m) = futures::join! {
                    ctx.submit(scale.request_gpu(device.id, (), access_info)),
                    ctx.submit(offset.request_gpu(device.id, (), access_info)),
                    ctx.submit(input.metadata.request_scalar()),
                };

                let pipeline =
                    device.request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new().push_const_block::<PushConstants>(),
                            ),
                            true,
                        )
                    });

                let mut brick_stream =
                    ctx.submit_unordered_with_data(positions.iter().map(|pos| {
                        (
                            input.bricks.request_inplace_gpu(
                                device.id,
                                *pos,
                                ctx.current_op(),
                                access_info,
                            ),
                            *pos,
                        )
                    }));

                while let Some((inplace, pos)) = brick_stream.next().await {
                    let brick_info = m.chunk_info(pos);
                    let inplace = inplace?;

                    let (gpu_brick_in, gpu_brick_out): (&dyn AsDescriptors, &dyn AsDescriptors) =
                        match &inplace {
                            gpu::InplaceResult::Inplace(rw) => (rw, rw),
                            gpu::InplaceResult::New(r, w) => (r, w),
                        };

                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config = DescriptorConfig::new([
                            &scale_gpu,
                            &offset_gpu,
                            gpu_brick_in,
                            gpu_brick_out,
                        ]);

                        let global_size = brick_info.mem_elements();

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            pipeline.push_constant(PushConstants {
                                chunk_pos: pos.into_elem::<u32>().into(),
                                num_chunk_elems: global_size.try_into().unwrap(),
                            });
                            pipeline.push_descriptor_set(0, descriptor_config);
                            pipeline.dispatch(global_size);
                        }
                    });

                    unsafe {
                        inplace.initialized(SrcBarrierInfo {
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

pub fn rechunk<'op>(
    input: VolumeOperator<'op>,
    brick_size: Vector<3, ChunkSize>,
) -> VolumeOperator<'op> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        mem_size_in: cgmath::Vector3<u32>,
        mem_size_out: cgmath::Vector3<u32>,
        begin_in: cgmath::Vector3<u32>,
        begin_out: cgmath::Vector3<u32>,
        region_size: cgmath::Vector3<u32>,
        global_size: u32,
    }
    const SHADER: &'static str = r#"
#version 450

#include <util.glsl>

layout (local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputBuffer{
    float values[];
} sourceData;

layout(std430, binding = 1) buffer OutputBuffer{
    float values[];
} outputData;

declare_push_consts(constants);

void main() {
    uint gID = gl_GlobalInvocationID.x;

    if(gID < constants.global_size) {
        uvec3 region_pos = from_linear3(gID, constants.region_size);

        uvec3 in_pos = constants.begin_in + region_pos;
        uvec3 out_pos = constants.begin_out + region_pos;

        uint in_index = to_linear3(in_pos, constants.mem_size_in);
        uint out_index = to_linear3(out_pos, constants.mem_size_out);

        outputData.values[out_index] = sourceData.values[in_index];
    }
}
"#;
    TensorOperator::with_state(
        OperatorId::new("volume_rechunk_gpu")
            .dependent_on(&input)
            .dependent_on(Id::hash(&brick_size)),
        input.clone(),
        input,
        move |ctx, input, _| {
            async move {
                let req = input.metadata.request_scalar();
                let mut m = ctx.submit(req).await;
                m.chunk_size = brick_size.zip(m.dimensions, |v, d| v.apply(d));
                ctx.write(m)
            }
            .into()
        },
        move |ctx, positions, input, _| {
            // TODO: optimize case where input.brick_size == output.brick_size
            async move {
                let device = ctx.vulkan_device();

                let pipeline =
                    device.request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new().push_const_block::<PushConstants>(),
                            ),
                            true,
                        )
                    });

                let m_in = ctx.submit(input.metadata.request_scalar()).await;
                let m_out = {
                    let mut m_out = m_in;
                    m_out.chunk_size = brick_size.zip(m_in.dimensions, |v, d| v.apply(d));
                    m_out
                };

                let requests = positions.into_iter().map(|pos| {
                    let out_info = m_out.chunk_info(pos);
                    let out_begin = out_info.begin();
                    let out_end = out_info.end();

                    let in_begin_brick = m_in.chunk_pos(out_begin);
                    let in_end_brick = m_in.chunk_pos(out_end.map(|v| v - 1u32));

                    let in_brick_positions = itertools::iproduct! {
                        in_begin_brick.z().raw..=in_end_brick.z().raw,
                        in_begin_brick.y().raw..=in_end_brick.y().raw,
                        in_begin_brick.x().raw..=in_end_brick.x().raw
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

                    (intersecting_bricks, (pos, in_brick_positions))
                });

                let mut stream = ctx.submit_unordered_with_data(requests);
                while let Some((intersecting_bricks, (pos, in_brick_positions))) =
                    stream.next().await
                {
                    let out_info = m_out.chunk_info(pos);

                    let gpu_brick_out = ctx
                        .alloc_slot_gpu(device, pos, out_info.mem_elements())
                        .unwrap();

                    device.with_cmd_buffer(|cmd| {
                        let out_begin = out_info.begin();
                        let out_end = out_info.end();

                        for (gpu_brick_in, in_brick_pos) in intersecting_bricks
                            .iter()
                            .zip(in_brick_positions.into_iter())
                        {
                            let in_info = m_in.chunk_info(in_brick_pos);

                            let in_begin = in_info.begin();
                            let in_end = in_info.end();

                            let overlap_begin = in_begin.zip(out_begin, |i, o| i.max(o));
                            let overlap_end = in_end.zip(out_end, |i, o| i.min(o));
                            let overlap_size =
                                (overlap_end - overlap_begin).map(LocalCoordinate::interpret_as);

                            let in_chunk_begin = in_info.in_chunk(overlap_begin);

                            let out_chunk_begin = out_info.in_chunk(overlap_begin);

                            let descriptor_config =
                                DescriptorConfig::new([gpu_brick_in, &gpu_brick_out]);

                            let global_size = crate::data::hmul(overlap_size);

                            //TODO initialization of outside regions
                            unsafe {
                                let mut pipeline = pipeline.bind(cmd);

                                pipeline.push_constant(PushConstants {
                                    mem_size_in: m_in.chunk_size.into_elem::<u32>().into(),
                                    mem_size_out: m_out.chunk_size.into_elem::<u32>().into(),
                                    begin_in: in_chunk_begin.into_elem::<u32>().into(),
                                    begin_out: out_chunk_begin.into_elem::<u32>().into(),
                                    region_size: overlap_size.into_elem::<u32>().into(),
                                    global_size: global_size as _,
                                });
                                pipeline.push_descriptor_set(0, descriptor_config);
                                pipeline.dispatch(global_size);
                            }
                        }
                    });
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

/// A one dimensional convolution in the specified (constant) axis. Currently zero padding is the
/// only supported (and thus always applied) border handling routine.
pub fn convolution_1d<'op, const DIM: usize>(
    input: VolumeOperator<'op>,
    kernel: ArrayOperator<'op>,
) -> VolumeOperator<'op> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        mem_dim: cgmath::Vector3<u32>,
        logical_dim_out: cgmath::Vector3<u32>,
        out_begin: cgmath::Vector3<u32>,
        global_dim: cgmath::Vector3<u32>,
        num_chunks: u32,
        first_chunk_pos: u32,
        extent: i32,
        global_size: u32,
    }
    const SHADER: &'static str = r#"
#version 450

#include <util.glsl>

layout (local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputBuffer{
    float values[];
} sourceData[MAX_BRICKS];

layout(std430, binding = 1) readonly buffer KernelBuffer{
    float values[];
} kernel;

layout(std430, binding = 2) buffer OutputBuffer{
    float values[];
} outputData;

declare_push_consts(consts);

void main() {
    uint gID = gl_GlobalInvocationID.x;

    if(gID < consts.global_size) {
        uvec3 out_local = from_linear3(gID, consts.mem_dim);
        float acc = 0.0;

        if(all(lessThan(out_local, consts.logical_dim_out))) {
            for (int i = 0; i<consts.num_chunks; ++i) {
                int chunk_pos = int(consts.first_chunk_pos) + i;
                int global_begin_pos_in = chunk_pos * int(consts.mem_dim[DIM]);

                int logical_dim_in = min(
                    global_begin_pos_in + int(consts.mem_dim[DIM]),
                    int(consts.global_dim[DIM])
                ) - global_begin_pos_in;

                int out_chunk_to_in_chunk = int(consts.out_begin[DIM]) - global_begin_pos_in;
                int out_pos_rel_to_in_pos_rel = int(out_local[DIM]) + out_chunk_to_in_chunk;

                int begin_ext = -consts.extent;
                int end_ext = consts.extent;

                int local_end = logical_dim_in;

                int l_begin = max(begin_ext + out_pos_rel_to_in_pos_rel, 0);
                int l_end = min(end_ext + out_pos_rel_to_in_pos_rel, local_end - 1);

                for (int local=l_begin; local<=l_end; ++local) {
                    int kernel_offset = local - out_pos_rel_to_in_pos_rel;
                    int kernel_buf_index = consts.extent - kernel_offset;
                    float kernel_val = kernel.values[kernel_buf_index];

                    uvec3 pos = out_local;
                    pos[DIM] = local;

                    uint local_index = to_linear3(pos, consts.mem_dim);
                    float local_val = sourceData[i].values[local_index];

                    acc += kernel_val * local_val;
                }
            }
        } else {
            acc = NaN;
        }

        outputData.values[gID] = acc;
    }
}
"#;
    TensorOperator::with_state(
        OperatorId::new("convolution_1d_gpu")
            .dependent_on(&input)
            .dependent_on(&kernel),
        input.clone(),
        (input, kernel),
        move |ctx, input, _| {
            async move {
                let req = input.metadata.request_scalar();
                let m = ctx.submit(req).await;
                ctx.write(m)
            }
            .into()
        },
        move |ctx, positions, (input, kernel), _| {
            async move {
                let device = ctx.vulkan_device();

                let (m_in, kernel_m, kernel_handle) = futures::join!(
                    ctx.submit(input.metadata.request_scalar()),
                    ctx.submit(kernel.metadata.request_scalar()),
                    ctx.submit(kernel.bricks.request_gpu(
                        device.id,
                        [0].into(),
                        DstBarrierInfo {
                            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            access: vk::AccessFlags2::SHADER_READ,
                        }
                    )),
                );

                let m_out = m_in;

                assert_eq!(
                    kernel_m.dimensions.raw(),
                    kernel_m.chunk_size.raw(),
                    "Only unchunked kernels are supported for now"
                );

                let kernel_size = *kernel_m.dimensions.raw();
                assert!(kernel_size % 2 == 1, "Kernel size must be odd");
                let extent = kernel_size / 2;

                let requests = positions.into_iter().map(|pos| {
                    let out_info = m_out.chunk_info(pos);
                    let out_begin = out_info.begin();
                    let out_end = out_info.end();

                    let in_begin = out_begin
                        .map_element(DIM, |v| (v.raw.saturating_sub(extent as u32)).into());
                    let in_end = out_end
                        .map_element(DIM, |v| (v + extent as u32).min(m_out.dimensions[DIM]));

                    let in_begin_brick = m_in.chunk_pos(in_begin);
                    let in_end_brick = m_in.chunk_pos(in_end.map(|v| v - 1u32));

                    let in_brick_positions = itertools::iproduct! {
                        in_begin_brick.z().raw..=in_end_brick.z().raw,
                        in_begin_brick.y().raw..=in_end_brick.y().raw,
                        in_begin_brick.x().raw..=in_end_brick.x().raw
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

                    (intersecting_bricks, (pos, in_brick_positions))
                });

                let max_bricks =
                    2 * crate::util::div_round_up(extent, m_in.chunk_size[DIM].raw) + 1;

                //TODO: This is really ugly, we should investigate to use z,y,x ordering in shaders
                //as well.
                let shader_dimension = 2 - DIM;
                let pipeline = device.request_state(
                    RessourceId::new("pipeline")
                        .of(ctx.current_op())
                        .dependent_on(Id::hash(&max_bricks))
                        .dependent_on(Id::hash(&DIM)),
                    || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .add("MAX_BRICKS", max_bricks.to_string())
                                    .add("DIM", shader_dimension.to_string())
                                    .push_const_block::<PushConstants>(),
                            ),
                            true,
                        )
                    },
                );

                let mut stream = ctx.submit_unordered_with_data(requests);
                while let Some((intersecting_bricks, (pos, in_brick_positions))) =
                    stream.next().await
                {
                    let out_info = m_out.chunk_info(pos);
                    let gpu_brick_out = ctx
                        .alloc_slot_gpu(device, pos, out_info.mem_elements())
                        .unwrap();

                    let out_begin = out_info.begin();

                    for window in in_brick_positions.windows(2) {
                        for d in 0..3 {
                            if d == DIM {
                                assert_eq!(window[0][d] + 1u32, window[1][d]);
                            } else {
                                assert_eq!(window[0][d], window[1][d]);
                            }
                        }
                    }

                    let num_chunks = in_brick_positions.len();
                    assert!(num_chunks > 0);

                    assert_eq!(num_chunks, intersecting_bricks.len());

                    let first_chunk_pos = in_brick_positions.first().unwrap()[DIM].raw;
                    let global_size = crate::data::hmul(m_out.chunk_size);

                    let consts = PushConstants {
                        mem_dim: m_out.chunk_size.into_elem::<u32>().into(),
                        logical_dim_out: out_info.logical_dimensions.into_elem::<u32>().into(),
                        out_begin: out_begin.into_elem::<u32>().into(),
                        global_dim: m_out.dimensions.into_elem::<u32>().into(),
                        num_chunks: num_chunks as _,
                        first_chunk_pos,
                        extent: extent as i32,
                        global_size: global_size.try_into().unwrap(),
                    };

                    // TODO: This padding to max_bricks is necessary since the descriptor array in
                    // the shader has a static since. Once we use dynamic ssbos this can go away.
                    let intersecting_bricks = (0..max_bricks)
                        .map(|i| {
                            intersecting_bricks
                                .get(i as usize)
                                .unwrap_or(intersecting_bricks.get(0).unwrap())
                        })
                        .collect::<Vec<_>>();
                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config = DescriptorConfig::new([
                            &intersecting_bricks.as_slice(),
                            &kernel_handle,
                            &gpu_brick_out,
                        ]);

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            pipeline.push_constant(consts);
                            pipeline.push_descriptor_set(0, descriptor_config);
                            pipeline.dispatch(global_size);
                        }
                    });

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
    .into()
}

pub fn separable_convolution<'op>(
    v: VolumeOperator<'op>,
    [k0, k1, k2]: [ArrayOperator<'op>; 3],
) -> VolumeOperator<'op> {
    let v = convolution_1d::<2>(v, k2);
    let v = convolution_1d::<1>(v, k1);
    let v = convolution_1d::<0>(v, k0);
    v
}

pub fn mean<'op>(input: VolumeOperator<'op>) -> ScalarOperator<'op, f32> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        mem_dim: cgmath::Vector3<u32>,
        logical_dim: cgmath::Vector3<u32>,
        norm_factor: f32,
    }
    const SHADER: &'static str = r#"
#version 450

#include <util.glsl>
#include <atomic.glsl>

#extension GL_KHR_shader_subgroup_arithmetic : require

layout (local_size_x = 1024) in;

layout(std430, binding = 0) readonly buffer InputBuffer{
    float values[];
} sourceData;

layout(std430, binding = 1) buffer OutputBuffer{
    uint value;
} sum;

declare_push_consts(consts);

shared uint shared_sum;

void main()
{
    uint gID = gl_GlobalInvocationID.x;
    if(gl_LocalInvocationIndex == 0) {
        shared_sum = floatBitsToUint(0.0);
    }
    barrier();

    float val;

    uvec3 local = from_linear3(gID, consts.mem_dim);

    if(all(lessThan(local, consts.logical_dim))) {
        val = sourceData.values[gID] * consts.norm_factor;
    } else {
        val = 0.0;
    }

    float sg_sum = subgroupAdd(val);

    if(gl_SubgroupInvocationID == 0) {
        atomic_add_float(shared_sum, sg_sum);
    }

    barrier();

    if(gl_LocalInvocationIndex == 0) {
        atomic_add_float(sum.value, uintBitsToFloat(shared_sum));
    }
}
"#;

    crate::operators::scalar::scalar(
        OperatorId::new("volume_mean_gpu").dependent_on(&input),
        input,
        move |ctx, input, _| {
            async move {
                let device = ctx.vulkan_device();

                let m = ctx.submit(input.metadata.request_scalar()).await;

                let to_request = m.brick_positions().collect::<Vec<_>>();
                let batch_size = 1024;

                let pipeline =
                    device.request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new().push_const_block::<PushConstants>(),
                            ),
                            true,
                        )
                    });

                let sum = ctx.alloc_scalar_gpu(device)?;

                let normalization_factor = 1.0 / (crate::data::hmul(m.dimensions) as f32);

                device.with_cmd_buffer(|cmd| {
                    unsafe {
                        device.functions().cmd_update_buffer(
                            cmd.raw(),
                            sum.buffer,
                            0,
                            bytemuck::cast_slice(&[0f32]),
                        )
                    };
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

                for chunk in to_request.chunks(batch_size) {
                    let mut stream = ctx.submit_unordered_with_data(chunk.iter().map(|pos| {
                        (
                            input.bricks.request_gpu(
                                device.id,
                                *pos,
                                DstBarrierInfo {
                                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                    access: vk::AccessFlags2::SHADER_READ,
                                },
                            ),
                            *pos,
                        )
                    }));
                    while let Some((gpu_brick_in, pos)) = stream.next().await {
                        let brick_info = m.chunk_info(pos);

                        device.with_cmd_buffer(|cmd| {
                            let descriptor_config = DescriptorConfig::new([&gpu_brick_in, &sum]);

                            let global_size = brick_info.mem_elements();

                            unsafe {
                                let mut pipeline = pipeline.bind(cmd);

                                pipeline.push_constant(PushConstants {
                                    mem_dim: brick_info.mem_dimensions.into_elem::<u32>().into(),
                                    logical_dim: brick_info
                                        .logical_dimensions
                                        .into_elem::<u32>()
                                        .into(),
                                    norm_factor: normalization_factor,
                                });
                                pipeline.push_descriptor_set(0, descriptor_config);
                                pipeline.dispatch(global_size);
                            }
                        });
                    }
                    ctx.submit(device.wait_for_current_cmd_buffer_completion())
                        .await;
                }
                unsafe {
                    sum.initialized(SrcBarrierInfo {
                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::SHADER_WRITE,
                    })
                };

                Ok(())
            }
            .into()
        },
    )
}

pub struct VoxelRasterizerGLSL {
    pub body: String,
    pub metadata: VolumeMetaData,
}

impl super::volume::VolumeOperatorState for VoxelRasterizerGLSL {
    fn operate<'a>(&'a self) -> VolumeOperator<'a> {
        #[derive(Copy, Clone, AsStd140, GlslStruct)]
        struct PushConstants {
            offset: cgmath::Vector3<u32>,
            mem_dim: cgmath::Vector3<u32>,
            logical_dim: cgmath::Vector3<u32>,
            vol_dim: cgmath::Vector3<u32>,
            num_chunk_elems: u32,
        }

        let m = self.metadata;

        let shader = format!(
            "{}{}{}",
            r#"
#version 450

#include <util.glsl>

layout (local_size_x = 256) in;

layout(std430, binding = 0) buffer OutputBuffer{
    float values[];
} outputData;

declare_push_consts(consts);

void main()
{
    uint gID = gl_GlobalInvocationID.x;

    if(gID < consts.num_chunk_elems) {
        uvec3 out_local = from_linear3(gID, consts.mem_dim);
        float result = 0.0;
        uvec3 pos_voxel = out_local + consts.offset;
        vec3 pos_normalized = vec3(pos_voxel)/vec3(consts.vol_dim);

        if(all(lessThan(out_local, consts.logical_dim))) {
        "#,
            self.body,
            r#"
        } else {
            result = NaN;
        }

        outputData.values[gID] = result;
    }
}
"#
        );

        TensorOperator::with_state(
            OperatorId::new("rasterize_gpu")
                .dependent_on(Id::hash(&shader))
                .dependent_on(Id::hash(&m)),
            (),
            shader,
            move |ctx, _, _| async move { ctx.write(m) }.into(),
            move |ctx, positions, shader, _| {
                async move {
                    let device = ctx.vulkan_device();

                    let pipeline = device.request_state(
                        RessourceId::new("pipeline").of(ctx.current_op()),
                        || {
                            ComputePipeline::new(
                                device,
                                (
                                    shader.as_str(),
                                    ShaderDefines::new().push_const_block::<PushConstants>(),
                                ),
                                true,
                            )
                        },
                    );

                    for pos in positions {
                        let brick_info = m.chunk_info(pos);

                        let gpu_brick_out =
                            ctx.alloc_slot_gpu(device, pos, brick_info.mem_elements())?;
                        device.with_cmd_buffer(|cmd| {
                            let descriptor_config = DescriptorConfig::new([&gpu_brick_out]);

                            let global_size = brick_info.mem_elements();

                            unsafe {
                                let mut pipeline = pipeline.bind(cmd);

                                pipeline.push_constant(PushConstants {
                                    offset: brick_info.begin.into_elem::<u32>().into(),
                                    logical_dim: brick_info
                                        .logical_dimensions
                                        .into_elem::<u32>()
                                        .into(),
                                    mem_dim: m.chunk_size.into_elem::<u32>().into(),
                                    vol_dim: m.dimensions.into_elem::<u32>().into(),
                                    num_chunk_elems: global_size as u32,
                                });
                                pipeline.push_descriptor_set(0, descriptor_config);
                                pipeline.dispatch(global_size);
                            }
                        });

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
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        data::{GlobalCoordinate, LocalVoxelPosition, Vector, VoxelPosition},
        operators::volume::VolumeOperatorState,
        test_util::*,
    };

    #[test]
    fn test_rescale_gpu() {
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let (point_vol, center) = center_point_vol(size, brick_size);

        let fill_expected = |comp: &mut ndarray::ArrayViewMut3<f32>| {
            for z in 0..size.z().raw {
                for y in 0..size.y().raw {
                    for x in 0..size.x().raw {
                        let pos = VoxelPosition::from([z, y, x]);
                        if pos != center {
                            comp[pos.as_index()] = 1.0;
                        }
                    }
                }
            }
            comp[center.as_index()] = -1.0;
        };
        let scale = (-2.0).into();
        let offset = (1.0).into();
        let input = point_vol.operate();
        let output = linear_rescale(input, scale, offset);
        compare_volume(output, fill_expected);
    }

    #[test]
    fn test_rechunk_gpu() {
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let input = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            crate::data::to_linear(v, size) as f32
        });

        let fill_expected = |comp: &mut ndarray::ArrayViewMut3<f32>| {
            for z in 0..size.z().raw {
                for y in 0..size.y().raw {
                    for x in 0..size.x().raw {
                        let pos = VoxelPosition::from([z, y, x]);
                        let val = crate::data::to_linear(pos, size) as f32;
                        comp[pos.as_index()] = val
                    }
                }
            }
        };
        let input = input.operate();
        for chunk_size in [[5, 1, 1], [4, 4, 1], [2, 3, 4], [1, 1, 1], [5, 5, 5]] {
            let output = rechunk(
                input.clone(),
                LocalVoxelPosition::from(chunk_size).into_elem(),
            );
            compare_volume(output, fill_expected);
        }
    }

    #[test]
    fn test_rasterize_gpu() {
        let size = VoxelPosition::fill(5.into());

        let fill_expected = |comp: &mut ndarray::ArrayViewMut3<f32>| {
            for z in 0..size.z().raw {
                for y in 0..size.y().raw {
                    for x in 0..size.x().raw {
                        let pos = VoxelPosition::from([z, y, x]);
                        comp[pos.as_index()] = x as f32 + y as f32 + z as f32;
                    }
                }
            }
        };
        for chunk_size in [[5, 1, 1], [4, 4, 1], [2, 3, 4], [1, 1, 1], [5, 5, 5]] {
            let input = VoxelRasterizerGLSL {
                metadata: crate::array::VolumeMetaData {
                    dimensions: size,
                    chunk_size: chunk_size.into(),
                },
                body: r#"result = float(pos_voxel.x + pos_voxel.y + pos_voxel.z);"#.to_owned(),
            };
            let input = input.operate();
            let output = rechunk(input, LocalVoxelPosition::from(chunk_size).into_elem());
            compare_volume(output, fill_expected);
        }
    }

    fn compare_convolution_1d<const DIM: usize>(
        input: &dyn VolumeOperatorState,
        kernel: &[f32],
        fill_expected: impl FnOnce(&mut ndarray::ArrayViewMut3<f32>),
    ) {
        let input = input.operate();
        let output = convolution_1d::<DIM>(input, crate::operators::array::from_static(kernel));
        compare_volume(output, fill_expected);
    }

    fn test_convolution_1d_generic<const DIM: usize>() {
        // Small
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let (point_vol, center) = center_point_vol(size, brick_size);
        compare_convolution_1d::<DIM>(&point_vol, &[1.0, -1.0, 2.0], |comp| {
            comp[center.map_element(DIM, |v| v - 1u32).as_index()] = 1.0;
            comp[center.map_element(DIM, |v| v).as_index()] = -1.0;
            comp[center.map_element(DIM, |v| v + 1u32).as_index()] = 2.0;
        });

        // Larger
        let size = VoxelPosition::fill(13.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let (point_vol, center) = center_point_vol(size, brick_size);
        let kernel_size = 7;
        let extent = kernel_size / 2;
        let mut kernel = vec![0.0; kernel_size];
        kernel[0] = -1.0;
        kernel[1] = -2.0;
        kernel[kernel_size - 1] = 1.0;
        kernel[kernel_size - 2] = 2.0;
        compare_convolution_1d::<DIM>(&point_vol, &kernel, |comp| {
            comp[center.map_element(DIM, |v| v - extent).as_index()] = -1.0;
            comp[center.map_element(DIM, |v| v - extent + 1u32).as_index()] = -2.0;
            comp[center.map_element(DIM, |v| v + extent).as_index()] = 1.0;
            comp[center.map_element(DIM, |v| v + extent - 1u32).as_index()] = 2.0;
        });
    }

    #[test]
    fn test_convolution_1d_x() {
        test_convolution_1d_generic::<2>();
    }
    #[test]
    fn test_convolution_1d_y() {
        test_convolution_1d_generic::<1>();
    }
    #[test]
    fn test_convolution_1d_z() {
        test_convolution_1d_generic::<0>();
    }

    #[test]
    fn test_separable_convolution() {
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let (point_vol, center) = center_point_vol(size, brick_size);

        let kernels = [&[2.0, 1.0, 2.0], &[2.0, 1.0, 2.0], &[2.0, 1.0, 2.0]];
        let kernels = std::array::from_fn(|i| crate::operators::array::from_static(kernels[i]));
        let output = separable_convolution(point_vol.operate(), kernels);
        compare_volume(output, |comp| {
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let offset = Vector::new([dz, dy, dx]);
                        let l1_dist = offset.map(i32::abs).fold(0, std::ops::Add::add);
                        let expected_val = 1 << l1_dist;
                        comp[(center.try_into_elem::<i32>().unwrap() + offset)
                            .try_into_elem::<GlobalCoordinate>()
                            .unwrap()
                            .as_index()] = expected_val as f32;
                    }
                }
            }
        });
    }
}
