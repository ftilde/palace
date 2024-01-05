use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};
use futures::StreamExt;
use itertools::Itertools;

use crate::{
    array::VolumeMetaData,
    data::{ChunkCoordinate, LocalCoordinate, Vector},
    dim::*,
    operator::OperatorDescriptor,
    operators::tensor::TensorOperator,
    storage::{gpu, Element},
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

pub fn linear_rescale<'op, D: Dimension>(
    input: TensorOperator<D, f32>,
    scale: f32,
    offset: f32,
) -> TensorOperator<D, f32> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        scale: f32,
        offset: f32,
    }
    const SHADER: &'static str = r#"
#version 450

#include <util.glsl>

layout (local_size_x = 256) in;

// Note: We cannot use `restrict` here and below since we bind the same buffer to sourceData and
// outputData in the inplace update case.
layout(std430, binding = 0) readonly buffer InputBuffer{
    float values[BRICK_MEM_SIZE];
} sourceData;

layout(std430, binding = 1) buffer OutputBuffer{
    float values[BRICK_MEM_SIZE];
} outputData;

declare_push_consts(consts);

void main()
{
    uint gID = gl_GlobalInvocationID.x;

    if(gID < BRICK_MEM_SIZE) {
        outputData.values[gID] = consts.scale*sourceData.values[gID] + consts.offset;
    }
}
"#;

    TensorOperator::with_state(
        OperatorDescriptor::new("volume_scale_gpu")
            .dependent_on(&input)
            .dependent_on_data(&scale)
            .dependent_on_data(&offset),
        input.metadata,
        (input, scale, offset),
        move |ctx, positions, (input, scale, offset)| {
            async move {
                let device = ctx.vulkan_device();

                let access_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                };
                let m = input.metadata;

                let pipeline =
                    device.request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .add("BRICK_MEM_SIZE", m.chunk_size.hmul())
                                    .push_const_block::<PushConstants>(),
                            ),
                            true,
                        )
                    });

                let mut brick_stream =
                    ctx.submit_unordered_with_data(positions.iter().map(|pos| {
                        (
                            input.chunks.request_inplace_gpu(
                                device.id,
                                *pos,
                                ctx.current_op_desc().unwrap(),
                                access_info,
                            ),
                            *pos,
                        )
                    }));

                while let Some((inplace, pos)) = brick_stream.next().await {
                    let brick_info = m.chunk_info(pos);

                    let inplace = ctx.submit(inplace.alloc()).await;

                    let (gpu_brick_in, gpu_brick_out): (&dyn AsDescriptors, &dyn AsDescriptors) =
                        match &inplace {
                            gpu::InplaceHandle::Inplace(rw, _v) => (rw, rw),
                            gpu::InplaceHandle::New(r, w) => (r, w),
                        };

                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config =
                            DescriptorConfig::new([gpu_brick_in, gpu_brick_out]);

                        let global_size = brick_info.mem_elements();

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            let consts = PushConstants {
                                scale: *scale,
                                offset: *offset,
                            };
                            pipeline.push_constant(consts);

                            pipeline.push_descriptor_set(0, descriptor_config);
                            pipeline.dispatch(global_size);
                        }
                    });

                    unsafe {
                        inplace.initialized(
                            *ctx,
                            SrcBarrierInfo {
                                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                access: vk::AccessFlags2::SHADER_WRITE,
                            },
                        )
                    };
                }

                Ok(())
            }
            .into()
        },
    )
}

pub trait GLSLType {
    const TYPE_NAME: &'static str;
}

impl GLSLType for f32 {
    const TYPE_NAME: &'static str = "float";
}

impl GLSLType for Vector<D4, u8> {
    const TYPE_NAME: &'static str = "uint8_t[4]";
}

pub fn rechunk<D: Dimension, T: Element + GLSLType>(
    input: TensorOperator<D, T>,
    brick_size: Vector<D, ChunkSize>,
) -> TensorOperator<D, T> {
    #[derive(Clone, bytemuck::Zeroable)]
    #[repr(C)]
    #[allow(dead_code)] //It says these fields are not read otherwise?? Why?
    struct PushConstants<D: Dimension> {
        mem_size_in: Vector<D, u32>,
        mem_size_out: Vector<D, u32>,
        begin_in: Vector<D, u32>,
        begin_out: Vector<D, u32>,
        region_size: Vector<D, u32>,
        global_size: u32,
    }
    impl<D: Dimension> Copy for PushConstants<D> where Vector<D, u32>: Copy {}
    //TODO: This is fine for the current layout, but we really want a better general approach
    unsafe impl<D: Dimension> bytemuck::Pod for PushConstants<D> where PushConstants<D>: Copy {}
    const SHADER: &'static str = r#"
#version 450

#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include <util.glsl>
#include <vec.glsl>

layout (local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputBuffer{
    T values[BRICK_MEM_SIZE_IN];
} sourceData;

layout(std430, binding = 1) buffer OutputBuffer{
    T values[];
} outputData;

layout(scalar, push_constant) uniform PushConsts {
    uint[N] mem_size_in;
    uint[N] mem_size_out;
    uint[N] begin_in;
    uint[N] begin_out;
    uint[N] region_size;
    uint global_size;
} constants;

void main() {
    uint gID = gl_GlobalInvocationID.x;

    if(gID < constants.global_size) {
        uint[N] region_pos = from_linear(gID, constants.region_size);

        uint[N] in_pos = add(constants.begin_in, region_pos);
        uint[N] out_pos = add(constants.begin_out, region_pos);

        uint in_index = to_linear(in_pos, constants.mem_size_in);
        uint out_index = to_linear(out_pos, constants.mem_size_out);

        outputData.values[out_index] = sourceData.values[in_index];
    }
}
"#;
    TensorOperator::with_state(
        OperatorDescriptor::new("volume_rechunk_gpu")
            .dependent_on(&input)
            .dependent_on_data(&brick_size)
            .dependent_on_data(T::TYPE_NAME)
            .dependent_on_data(&D::N),
        {
            let mut m = input.metadata;
            m.chunk_size = brick_size.zip(m.dimensions, |v, d| v.apply(d));
            m
        },
        input,
        move |ctx, positions, input| {
            // TODO: optimize case where input.brick_size == output.brick_size
            async move {
                let device = ctx.vulkan_device();

                let m_in = input.metadata;
                let m_out = {
                    let mut m_out = m_in;
                    m_out.chunk_size = brick_size.zip(m_in.dimensions, |v, d| v.apply(d));
                    m_out
                };

                let pipeline = device.request_state(
                    RessourceId::new("pipeline")
                        .of(ctx.current_op())
                        .dependent_on(T::TYPE_NAME)
                        .dependent_on(&D::N),
                    || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .add("BRICK_MEM_SIZE_IN", m_in.chunk_size.hmul())
                                    .add("N", D::N)
                                    .add("T", T::TYPE_NAME),
                            ),
                            true,
                        )
                    },
                );

                let requests = positions.into_iter().map(|pos| {
                    let out_info = m_out.chunk_info(pos);
                    let out_begin = out_info.begin();
                    let out_end = out_info.end();

                    let in_begin_brick = m_in.chunk_pos(out_begin);
                    let in_end_brick = m_in.chunk_pos(out_end.map(|v| v - 1u32));

                    let in_brick_positions = (0..D::N)
                        .into_iter()
                        .map(|i| in_begin_brick[i].raw..=in_end_brick[i].raw)
                        .multi_cartesian_product()
                        .map(|coordinates| {
                            Vector::<D, ChunkCoordinate>::try_from(coordinates).unwrap()
                        })
                        .collect::<Vec<_>>();
                    let intersecting_bricks = ctx.group(in_brick_positions.iter().map(|pos| {
                        input.chunks.request_gpu(
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
                        .submit(ctx.alloc_slot_gpu(device, pos, out_info.mem_elements()))
                        .await;

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

                            let global_size = overlap_size.hmul();

                            //TODO initialization of outside regions
                            unsafe {
                                let mut pipeline = pipeline.bind(cmd);

                                let consts = PushConstants {
                                    mem_size_in: m_in.chunk_size.raw(),
                                    mem_size_out: m_out.chunk_size.raw(),
                                    begin_in: in_chunk_begin.raw(),
                                    begin_out: out_chunk_begin.raw(),
                                    region_size: overlap_size.raw(),
                                    global_size: global_size as _,
                                };
                                pipeline.push_constant_pod(consts);
                                pipeline.push_descriptor_set(0, descriptor_config);
                                pipeline.dispatch(global_size);
                            }
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
                }

                Ok(())
            }
            .into()
        },
    )
}

/// A one dimensional convolution in the specified (constant) axis. Currently, clamping is the only
/// supported (and thus always applied) border handling routine.
//TODO It should be relatively easy to support other strategies now
pub fn convolution_1d<D: Dimension>(
    input: TensorOperator<D, f32>,
    kernel: ArrayOperator<f32>,
    dim: usize,
) -> TensorOperator<D, f32> {
    assert!(dim < D::N);
    #[derive(Clone, bytemuck::Zeroable)]
    #[repr(C)]
    struct PushConstants<D: Dimension> {
        mem_dim: Vector<D, u32>,
        logical_dim_out: Vector<D, u32>,
        out_begin: Vector<D, u32>,
        global_dim: Vector<D, u32>,
        dim_in_chunks: Vector<D, u32>,
        num_chunks: u32,
        first_chunk_pos: u32,
        extent: i32,
    }
    impl<D: Dimension> Copy for PushConstants<D> where Vector<D, u32>: Copy {}
    //TODO: This is fine for the current layout, but we really want a better general approach
    unsafe impl<D: Dimension> bytemuck::Pod for PushConstants<D> where PushConstants<D>: Copy {}
    const SHADER: &'static str = r#"
#version 450

#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <vec.glsl>

layout (local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputBuffer{
    float values[BRICK_MEM_SIZE];
} sourceData[MAX_BRICKS];

layout(std430, binding = 1) readonly buffer KernelBuffer{
    float values[KERNEL_SIZE];
} kernel;

layout(std430, binding = 2) buffer OutputBuffer{
    float values[BRICK_MEM_SIZE];
} outputData;

layout(scalar, push_constant) uniform PushConsts {
    uint[N] mem_dim;
    uint[N] logical_dim_out;
    uint[N] out_begin;
    uint[N] global_dim;
    uint[N] dim_in_chunks;
    uint num_chunks;
    uint first_chunk_pos;
    int extent;
} consts;

float kernel_val(int p) {
    int kernel_buf_index = consts.extent - p;
    return kernel.values[kernel_buf_index];
}

float sample_brick(uint[N] pos, int brick) {
    uint local_index = to_linear(pos, consts.mem_dim);
    return sourceData[brick].values[local_index];
}

void main() {
    uint gID = gl_GlobalInvocationID.x;

    if(gID < BRICK_MEM_SIZE) {
        uint[N] out_local = from_linear(gID, consts.mem_dim);
        float acc = 0.0;

        if(all(less_than(out_local, consts.logical_dim_out))) {

            int out_chunk_to_global = int(consts.out_begin[DIM]);
            int out_global = int(out_local[DIM]) + out_chunk_to_global;

            int begin_ext = -consts.extent;
            int end_ext = consts.extent;

            int last_chunk = int(consts.dim_in_chunks[DIM] - 1);

            for (int i = 0; i<consts.num_chunks; ++i) {
                int chunk_pos = int(consts.first_chunk_pos) + i;
                int global_begin_pos_in = chunk_pos * int(consts.mem_dim[DIM]);

                int logical_dim_in = min(
                    global_begin_pos_in + int(consts.mem_dim[DIM]),
                    int(consts.global_dim[DIM])
                ) - global_begin_pos_in;

                int in_chunk_to_global = global_begin_pos_in;
                int out_chunk_to_in_chunk = out_chunk_to_global - in_chunk_to_global;
                int out_pos_rel_to_in_pos_rel = int(out_local[DIM]) + out_chunk_to_in_chunk;

                int chunk_begin_local = 0;
                int chunk_end_local = logical_dim_in - 1;

                int l_begin_no_clip = begin_ext + out_pos_rel_to_in_pos_rel;
                int l_end_no_clip = end_ext + out_pos_rel_to_in_pos_rel;

                int l_begin = max(l_begin_no_clip, chunk_begin_local);
                int l_end = min(l_end_no_clip, chunk_end_local);

                uint[N] pos = out_local;

                // Border handling for first chunk in dim
                if(chunk_pos == 0) {
                    pos[DIM] = chunk_begin_local; //Clip to volume/chunk
                    float local_val = sample_brick(pos, i);

                    for (int local=l_begin_no_clip; local<chunk_begin_local; ++local) {
                        int kernel_offset = local - out_pos_rel_to_in_pos_rel;
                        acc += kernel_val(kernel_offset) * local_val;
                    }
                }

                for (int local=l_begin; local<=l_end; ++local) {
                    int kernel_offset = local - out_pos_rel_to_in_pos_rel;
                    pos[DIM] = local;
                    acc += kernel_val(kernel_offset) * sample_brick(pos, i);
                }

                // Border handling for last chunk in dim
                if(chunk_pos == last_chunk) {
                    pos[DIM] = chunk_end_local; //Clip to volume/chunk
                    float local_val = sample_brick(pos, i);

                    for (int local=chunk_end_local+1; local<=l_end_no_clip; ++local) {
                        int kernel_offset = local - out_pos_rel_to_in_pos_rel;
                        acc += kernel_val(kernel_offset) * local_val;
                    }
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
        OperatorDescriptor::new("convolution_1d_gpu")
            .dependent_on(&input)
            .dependent_on(&kernel)
            .dependent_on_data(&dim),
        input.metadata,
        (input, kernel),
        move |ctx, positions, (input, kernel)| {
            async move {
                let device = ctx.vulkan_device();

                let m_in = input.metadata;
                let kernel_m = kernel.metadata;
                let kernel_handle = ctx
                    .submit(kernel.chunks.request_gpu(
                        device.id,
                        [0].into(),
                        DstBarrierInfo {
                            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            access: vk::AccessFlags2::SHADER_READ,
                        },
                    ))
                    .await;

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
                        .map_element(dim, |v| (v.raw.saturating_sub(extent as u32)).into());
                    let in_end = out_end
                        .map_element(dim, |v| (v + extent as u32).min(m_out.dimensions[dim]));

                    let in_begin_brick = m_in.chunk_pos(in_begin);
                    let in_end_brick = m_in.chunk_pos(in_end.map(|v| v - 1u32));

                    let in_brick_positions = (0..D::N)
                        .into_iter()
                        .map(|i| in_begin_brick[i].raw..=in_end_brick[i].raw)
                        .multi_cartesian_product()
                        .map(|coordinates| {
                            Vector::<D, ChunkCoordinate>::try_from(coordinates).unwrap()
                        })
                        .collect::<Vec<_>>();

                    let intersecting_bricks = ctx.group(in_brick_positions.iter().map(|pos| {
                        input.chunks.request_gpu(
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
                    2 * crate::util::div_round_up(extent, m_in.chunk_size[dim].raw) + 1;

                //TODO: This is really ugly, we should investigate to use z,y,x ordering in shaders
                //as well.
                let pipeline = device.request_state(
                    RessourceId::new("pipeline")
                        .of(ctx.current_op())
                        .dependent_on(&max_bricks)
                        .dependent_on(&D::N)
                        .dependent_on(&dim),
                    || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .add("MAX_BRICKS", max_bricks)
                                    .add("DIM", dim)
                                    .add("N", D::N)
                                    .add("BRICK_MEM_SIZE", m_in.chunk_size.hmul())
                                    .add("KERNEL_SIZE", kernel_size),
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
                        .submit(ctx.alloc_slot_gpu(device, pos, out_info.mem_elements()))
                        .await;

                    let out_begin = out_info.begin();

                    for window in in_brick_positions.windows(2) {
                        for d in 0..D::N {
                            if d == dim {
                                assert_eq!(window[0][d] + 1u32, window[1][d]);
                            } else {
                                assert_eq!(window[0][d], window[1][d]);
                            }
                        }
                    }

                    let num_chunks = in_brick_positions.len();
                    assert!(num_chunks > 0);

                    assert_eq!(num_chunks, intersecting_bricks.len());

                    let first_chunk_pos = in_brick_positions.first().unwrap()[dim].raw;
                    let global_size = m_out.chunk_size.hmul();

                    let consts = PushConstants {
                        mem_dim: m_out.chunk_size.raw(),
                        logical_dim_out: out_info.logical_dimensions.raw(),
                        out_begin: out_begin.raw(),
                        global_dim: m_out.dimensions.raw(),
                        dim_in_chunks: m_out.dimension_in_chunks().raw(),
                        num_chunks: num_chunks as _,
                        first_chunk_pos,
                        extent: extent as i32,
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

                            pipeline.push_constant_pod(consts);
                            pipeline.push_descriptor_set(0, descriptor_config);
                            pipeline.dispatch(global_size);
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
                }

                Ok(())
            }
            .into()
        },
    )
    .into()
}

//TODO: kind of annoying that we have to use a reference to the operator here, but that is the only way it is copy...
pub fn separable_convolution<D: Dimension>(
    mut v: TensorOperator<D, f32>,
    kernels: Vector<D, &ArrayOperator<f32>>,
) -> TensorOperator<D, f32> {
    for dim in (0..D::N).rev() {
        v = convolution_1d(v, kernels[dim].clone(), dim);
    }
    v
}

pub fn mean<'op>(input: VolumeOperator<f32>) -> ScalarOperator<f32> {
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
    float values[BRICK_MEM_SIZE];
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
        OperatorDescriptor::new("volume_mean_gpu").dependent_on(&input),
        input,
        move |ctx, input| {
            async move {
                let device = ctx.vulkan_device();

                let m = input.metadata;

                let to_request = m.brick_positions().collect::<Vec<_>>();
                let batch_size = 1024;

                let pipeline =
                    device.request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .push_const_block::<PushConstants>()
                                    .add("BRICK_MEM_SIZE", m.chunk_size.hmul()),
                            ),
                            true,
                        )
                    });

                let sum = ctx.submit(ctx.alloc_scalar_gpu(device)).await;

                let normalization_factor = 1.0 / (m.dimensions.hmul() as f32);

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
                            input.chunks.request_gpu(
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
                    sum.initialized(
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

pub fn rasterize_gpu(metadata: VolumeMetaData, body: &str) -> VolumeOperator<f32> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        offset: cgmath::Vector3<u32>,
        mem_dim: cgmath::Vector3<u32>,
        logical_dim: cgmath::Vector3<u32>,
        vol_dim: cgmath::Vector3<u32>,
    }

    let shader = format!(
        "{}{}{}",
        r#"
#version 450

#include <util.glsl>

layout (local_size_x = 256) in;

layout(std430, binding = 0) buffer OutputBuffer{
    float values[BRICK_MEM_SIZE];
} outputData;

declare_push_consts(consts);

void main()
{
    uint gID = gl_GlobalInvocationID.x;

    if(gID < BRICK_MEM_SIZE) {
        uvec3 out_local = from_linear3(gID, consts.mem_dim);
        float result = 0.0;
        uvec3 pos_voxel = out_local + consts.offset;
        vec3 pos_normalized = vec3(pos_voxel)/vec3(consts.vol_dim);

        if(all(lessThan(out_local, consts.logical_dim))) {
        "#,
        body,
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
        OperatorDescriptor::new("rasterize_gpu")
            .dependent_on_data(body)
            .dependent_on_data(&metadata),
        metadata,
        (metadata, shader),
        move |ctx, positions, (metadata, shader)| {
            async move {
                let device = ctx.vulkan_device();

                let m = metadata;

                let pipeline = device.request_state(
                    RessourceId::new("pipeline")
                        .of(ctx.current_op())
                        .dependent_on(&m.chunk_size),
                    || {
                        ComputePipeline::new(
                            device,
                            (
                                shader.as_str(),
                                ShaderDefines::new()
                                    .push_const_block::<PushConstants>()
                                    .add("BRICK_MEM_SIZE", m.chunk_size.hmul()),
                            ),
                            true,
                        )
                    },
                );

                for pos in positions {
                    let brick_info = m.chunk_info(pos);

                    let gpu_brick_out = ctx
                        .submit(ctx.alloc_slot_gpu(device, pos, brick_info.mem_elements()))
                        .await;
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
                            });
                            pipeline.push_descriptor_set(0, descriptor_config);
                            pipeline.dispatch(global_size);
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
                }

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
    fn operate(&self) -> VolumeOperator<f32> {
        rasterize_gpu(self.metadata, &self.body)
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
    fn test_mean_gpu() {
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let input = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            let v = crate::vec::to_linear(v, size);
            v as f32
        });
        let input = input.operate();

        let output = mean(input);

        let mut runtime = crate::runtime::RunTime::new(1 << 30, None, Some(1)).unwrap();

        let output = &output;
        let mean = runtime
            .resolve(None, move |ctx, _| {
                async move {
                    let m = ctx.submit(output.request_scalar()).await;
                    Ok(m)
                }
                .into()
            })
            .unwrap();

        let n = size.hmul();
        let expected = (0..n).into_iter().sum::<usize>() as f32 / n as f32;

        println!("Mean: {}", mean);
        println!("Expected: {}", expected);
        let diff = (mean - expected).abs();
        let rel_diff = diff / (mean.max(expected));
        println!("Rel diff: {}", rel_diff);
        assert!(rel_diff < 0.0001);
    }

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
        compare_tensor_fn(output, fill_expected);
    }

    #[test]
    fn test_rescale_gpu_not_inplace() {
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
        let input = point_vol.operate();
        let l = linear_rescale(input.clone(), -2.0, 1.0);
        let l2 = linear_rescale(input, -2.0, -10.0);
        let output = crate::operators::bin_ops::max(l, l2);
        compare_tensor_fn(output, fill_expected);
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
            compare_tensor_fn(output, fill_expected);
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
            compare_tensor_fn(output, fill_expected);
        }
    }

    fn compare_convolution_1d(
        input: &dyn VolumeOperatorState,
        kernel: &[f32],
        fill_expected: impl FnOnce(&mut ndarray::ArrayViewMut3<f32>),
        dim: usize,
    ) {
        let input = input.operate();
        let output = convolution_1d(input, crate::operators::array::from_rc(kernel.into()), dim);
        compare_tensor_fn(output, fill_expected);
    }

    fn test_convolution_1d_generic(dim: usize) {
        // Small
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let (point_vol, center) = center_point_vol(size, brick_size);
        compare_convolution_1d(
            &point_vol,
            &[1.0, -1.0, 2.0],
            |comp| {
                comp[center.map_element(dim, |v| v - 1u32).as_index()] = 1.0;
                comp[center.map_element(dim, |v| v).as_index()] = -1.0;
                comp[center.map_element(dim, |v| v + 1u32).as_index()] = 2.0;
            },
            dim,
        );

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
        compare_convolution_1d(
            &point_vol,
            &kernel,
            |comp| {
                comp[center.map_element(dim, |v| v - extent).as_index()] = -1.0;
                comp[center.map_element(dim, |v| v - extent + 1u32).as_index()] = -2.0;
                comp[center.map_element(dim, |v| v + extent).as_index()] = 1.0;
                comp[center.map_element(dim, |v| v + extent - 1u32).as_index()] = 2.0;
            },
            dim,
        );
    }

    #[test]
    fn test_convolution_1d_x() {
        test_convolution_1d_generic(2);
    }
    #[test]
    fn test_convolution_1d_y() {
        test_convolution_1d_generic(1);
    }
    #[test]
    fn test_convolution_1d_z() {
        test_convolution_1d_generic(0);
    }

    #[test]
    fn test_convolution_1d_clamp() {
        let size = VoxelPosition::fill(5.into());
        let start = VoxelPosition::fill(0.into());
        let end = size - VoxelPosition::fill(1.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let vol = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            if v == start || v == end {
                1.0
            } else {
                0.0
            }
        });

        compare_convolution_1d(
            &vol,
            &[7.0, 1.0, 3.0],
            |comp| {
                comp[[0, 0, 0]] = 4.0;
                comp[[0, 0, 1]] = 3.0;

                comp[[4, 4, 3]] = 7.0;
                comp[[4, 4, 4]] = 8.0;
            },
            2,
        );
    }

    #[test]
    fn test_separable_convolution() {
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let (point_vol, center) = center_point_vol(size, brick_size);

        let kernels = [&[2.0, 1.0, 2.0], &[2.0, 1.0, 2.0], &[2.0, 1.0, 2.0]];
        let kernels: [_; 3] =
            std::array::from_fn(|i| crate::operators::array::from_static(kernels[i]));
        let kernels = Vector::from_fn(|i| &kernels[i]);
        let output = separable_convolution(point_vol.operate(), kernels);
        compare_tensor_fn(output, |comp| {
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let offset = Vector::<D3, i32>::new([dz, dy, dx]);
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
