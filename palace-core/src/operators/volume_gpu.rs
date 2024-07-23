use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};
use futures::StreamExt;
use itertools::Itertools;

use crate::{
    array::ChunkIndex,
    data::{ChunkCoordinate, LocalCoordinate, Vector},
    dim::*,
    dtypes::{DType, ElementType, StaticElementType},
    operator::OperatorDescriptor,
    operators::tensor::TensorOperator,
    storage::gpu,
    task::RequestStream,
    vulkan::{
        pipeline::{AsDescriptors, ComputePipeline, DescriptorConfig, DynPushConstants},
        shader::{Config, ShaderDefines},
        state::RessourceId,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::{
    array::ArrayOperator,
    raycaster::TransFuncOperator,
    scalar::ScalarOperator,
    volume::{ChunkSize, VolumeOperator},
};

pub fn linear_rescale<'op, D: DynDimension>(
    input: TensorOperator<D, StaticElementType<f32>>,
    scale: f32,
    offset: f32,
) -> TensorOperator<D, StaticElementType<f32>> {
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
        Default::default(),
        input.metadata.clone(),
        (input, scale, offset),
        move |ctx, positions, (input, scale, offset)| {
            async move {
                let device = ctx.preferred_device();

                let access_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                };
                let m = input.metadata.clone().into_dyn();

                let pipeline = device.request_state(
                    RessourceId::new("pipeline").of(ctx.current_op()),
                    || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .add("BRICK_MEM_SIZE", m.num_chunk_elements())
                                    .push_const_block::<PushConstants>(),
                            ),
                            true,
                        )
                    },
                )?;

                let mut brick_stream = ctx
                    .submit_unordered(positions.iter().map(|(pos, _)| {
                        input.chunks.request_inplace_gpu(
                            device.id,
                            *pos,
                            ctx.current_op_desc().unwrap(),
                            DType::F32,
                            access_info,
                        )
                    }))
                    .then_req(*ctx, |inplace| inplace.alloc());

                while let Some(inplace) = brick_stream.next().await {
                    let (gpu_brick_in, gpu_brick_out): (&dyn AsDescriptors, &dyn AsDescriptors) =
                        match &inplace {
                            gpu::InplaceHandle::Inplace(rw, _v) => (rw, rw),
                            gpu::InplaceHandle::New(r, w) => (r, w),
                        };

                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config =
                            DescriptorConfig::new([gpu_brick_in, gpu_brick_out]);

                        let global_size = m.num_chunk_elements();

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

pub fn apply_tf<'op, D: DynDimension>(
    input: TensorOperator<D, StaticElementType<f32>>,
    tf: TransFuncOperator,
) -> TensorOperator<D, StaticElementType<Vector<D4, u8>>> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        tf_min: f32,
        tf_max: f32,
        tf_len: u32,
    }
    const SHADER: &'static str = r#"
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include <util.glsl>

layout (local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputBuffer{
    float values[BRICK_MEM_SIZE];
} sourceData;

layout(std430, binding = 1) buffer TFTableBuffer {
    u8vec4 values[];
} tf_table;

layout(std430, binding = 2) buffer OutputBuffer{
    u8vec4 values[BRICK_MEM_SIZE];
} outputData;

declare_push_consts(consts);

//TODO: deduplicate, move to module
u8vec4 classify(float val) {
    float norm = (val-consts.tf_min)/(consts.tf_max - consts.tf_min);
    uint index = min(uint(max(0.0, norm) * consts.tf_len), consts.tf_len - 1);
    return tf_table.values[index];
}

void main()
{
    uint gID = gl_GlobalInvocationID.x;

    if(gID < BRICK_MEM_SIZE) {
        float v = sourceData.values[gID];
        outputData.values[gID] = classify(v);
    }
}
"#;

    TensorOperator::with_state(
        OperatorDescriptor::new("volume_scale_gpu")
            .dependent_on(&input)
            .dependent_on_data(&tf),
        Default::default(),
        input.metadata.clone(),
        (input, tf),
        move |ctx, positions, (input, tf)| {
            async move {
                let device = ctx.preferred_device();

                let access_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                };
                let m = input.metadata.clone().into_dyn();

                assert_eq!(tf.table.metadata.dimension_in_chunks()[0].raw, 1);
                let tf_data_gpu = ctx
                    .submit(
                        tf.table
                            .chunks
                            .request_gpu(device.id, ChunkIndex(0), access_info),
                    )
                    .await;

                let pipeline = device.request_state(
                    RessourceId::new("pipeline").of(ctx.current_op()),
                    || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .add("BRICK_MEM_SIZE", m.num_chunk_elements())
                                    .push_const_block::<PushConstants>(),
                            ),
                            true,
                        )
                    },
                )?;

                let mut brick_stream = ctx
                    .submit_unordered_with_data(positions.iter().map(|(pos, _)| {
                        (input.chunks.request_gpu(device.id, *pos, access_info), *pos)
                    }))
                    .then_req_with_data(*ctx, |(input, pos)| {
                        let output = ctx.alloc_slot_gpu(device, pos, m.num_chunk_elements());
                        (output, input)
                    });

                while let Some((output_chunk, input_chunk)) = brick_stream.next().await {
                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config =
                            DescriptorConfig::new([&input_chunk, &tf_data_gpu, &output_chunk]);

                        let global_size = m.num_chunk_elements();

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            let tf_data = tf.data();
                            let consts = PushConstants {
                                tf_min: tf_data.min,
                                tf_max: tf_data.max,
                                tf_len: tf_data.len,
                            };
                            pipeline.push_constant(consts);

                            pipeline.push_descriptor_set(0, descriptor_config);
                            pipeline.dispatch(global_size);
                        }
                    });

                    unsafe {
                        output_chunk.initialized(
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

pub fn cast<'op, D: Dimension>(
    input: TensorOperator<D, DType>,
    out_type: DType,
) -> TensorOperator<D, DType> {
    let in_dtype = input.chunks.dtype();

    if in_dtype == out_type {
        return input;
    }

    const SHADER: &'static str = r#"
#include <util.glsl>

layout (local_size_x = 256) in;

// Note: We cannot use `restrict` here and below since we bind the same buffer to sourceData and
// outputData in the inplace update case.
layout(std430, binding = 0) readonly buffer InputBuffer{
    IN_TYPE values[BRICK_MEM_SIZE];
} sourceData;

layout(std430, binding = 1) buffer OutputBuffer{
    OUT_TYPE values[BRICK_MEM_SIZE];
} outputData;

void main()
{
    uint gID = gl_GlobalInvocationID.x;

    if(gID < BRICK_MEM_SIZE) {
        outputData.values[gID] = OUT_TYPE(sourceData.values[gID]);
    }
}
"#;

    TensorOperator::with_state(
        OperatorDescriptor::new("cast_gpu")
            .dependent_on(&input)
            .dependent_on_data(&out_type),
        out_type,
        input.metadata,
        input,
        move |ctx, positions, input| {
            async move {
                let device = ctx.preferred_device();

                let access_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                };
                let m = input.metadata;

                let pipeline = device.request_state(
                    RessourceId::new("pipeline").of(ctx.current_op()),
                    || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .add("BRICK_MEM_SIZE", m.chunk_size.hmul())
                                    .add("IN_TYPE", in_dtype.glsl_type())
                                    .add("OUT_TYPE", out_type.glsl_type()),
                                Config::new()
                                    .ext(in_dtype.glsl_ext())
                                    .ext(out_type.glsl_ext()),
                            ),
                            true,
                        )
                    },
                )?;

                let mut brick_stream = ctx
                    .submit_unordered_with_data(positions.iter().map(|(pos, _)| {
                        (
                            input.chunks.request_inplace_gpu(
                                device.id,
                                *pos,
                                ctx.current_op_desc().unwrap(),
                                out_type,
                                access_info,
                            ),
                            *pos,
                        )
                    }))
                    .then_req_with_data(*ctx, |(inplace, pos)| (inplace.alloc(), pos));

                while let Some((inplace, pos)) = brick_stream.next().await {
                    let brick_info = m.chunk_info(pos);

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

pub fn threshold<'op, D: DynDimension>(
    input: TensorOperator<D, StaticElementType<f32>>,
    threshold: f32,
) -> TensorOperator<D, StaticElementType<f32>> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        threshold: f32,
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
        outputData.values[gID] = sourceData.values[gID] < consts.threshold ? 0.0 : 1.0;
    }
}
"#;

    TensorOperator::with_state(
        OperatorDescriptor::new("threshold_gpu")
            .dependent_on(&input)
            .dependent_on_data(&threshold),
        Default::default(),
        input.metadata.clone(),
        (input, threshold),
        move |ctx, positions, (input, threshold)| {
            async move {
                let device = ctx.preferred_device();

                let access_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                };
                let m = input.metadata.clone().into_dyn();

                let pipeline = device.request_state(
                    RessourceId::new("pipeline").of(ctx.current_op()),
                    || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .add("BRICK_MEM_SIZE", m.num_chunk_elements())
                                    .push_const_block::<PushConstants>(),
                            ),
                            true,
                        )
                    },
                )?;

                let mut brick_stream = ctx
                    .submit_unordered(positions.iter().map(|(pos, _)| {
                        input.chunks.request_inplace_gpu(
                            device.id,
                            *pos,
                            ctx.current_op_desc().unwrap(),
                            DType::F32,
                            access_info,
                        )
                    }))
                    .then_req(*ctx, |inplace| inplace.alloc());

                while let Some(inplace) = brick_stream.next().await {
                    let (gpu_brick_in, gpu_brick_out): (&dyn AsDescriptors, &dyn AsDescriptors) =
                        match &inplace {
                            gpu::InplaceHandle::Inplace(rw, _v) => (rw, rw),
                            gpu::InplaceHandle::New(r, w) => (r, w),
                        };

                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config =
                            DescriptorConfig::new([gpu_brick_in, gpu_brick_out]);

                        let global_size = m.num_chunk_elements();

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            let consts = PushConstants {
                                threshold: *threshold,
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

pub fn rechunk<D: DynDimension, T: ElementType>(
    input: TensorOperator<D, T>,
    chunk_size: Vector<D, ChunkSize>,
) -> TensorOperator<D, T> {
    let md = &input.metadata;

    // Early return in case of matched sizes
    if md.chunk_size == chunk_size.zip(&md.dimensions, |b, d| b.apply(d)) {
        return input;
    }

    let dtype: DType = input.chunks.dtype().into();

    let dim = md.chunk_size.len();

    let push_constants = DynPushConstants::new()
        .array::<u32>(dim, "mem_size_in")
        .array::<u32>(dim, "mem_size_out")
        .array::<u32>(dim, "begin_in")
        .array::<u32>(dim, "begin_out")
        .array::<u32>(dim, "region_size")
        .scalar::<u32>("global_size");

    const SHADER: &'static str = r#"
#include <util.glsl>
#include <vec.glsl>

layout (local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputBuffer{
    T values[BRICK_MEM_SIZE_IN];
} sourceData;

layout(std430, binding = 1) buffer OutputBuffer{
    T values[];
} outputData;

declare_push_consts(constants)

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
            .dependent_on_data(&chunk_size)
            .dependent_on_data(&dtype)
            .dependent_on_data(&dim),
        input.chunks.dtype(),
        {
            let mut m = input.metadata.clone();
            m.chunk_size = chunk_size.zip(&m.dimensions, |v, d| v.apply(d));
            m
        },
        (input, chunk_size, push_constants),
        move |ctx, mut positions, (input, chunk_size, push_constants)| {
            async move {
                let device = ctx.preferred_device();

                let nd = input.metadata.dimensions.len();

                let m_in = input.metadata.clone();
                let m_out = {
                    let mut m_out = m_in.clone();
                    m_out.chunk_size = chunk_size.zip(&m_in.dimensions, |v, d| v.apply(d));
                    m_out
                };

                positions.sort_by_key(|(v, _)| v.0);

                let pipeline = device.request_state(
                    RessourceId::new("pipeline")
                        .of(ctx.current_op())
                        .dependent_on(&dtype)
                        .dependent_on(&dim),
                    || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .add("BRICK_MEM_SIZE_IN", m_in.num_chunk_elements())
                                    .add("N", dim)
                                    .add("T", dtype.glsl_type())
                                    .push_const_block_dyn(&push_constants),
                                Config::new()
                                    .ext(dtype.glsl_ext())
                                    .ext(Some(crate::vulkan::shader::ext::SCALAR_BLOCK_LAYOUT)),
                            ),
                            true,
                        )
                    },
                )?;

                let requests = positions.into_iter().map(|(pos, _)| {
                    let out_info = m_out.chunk_info(pos);
                    let out_begin = out_info.begin();
                    let out_end = out_info.end();

                    let in_begin_brick = m_in.chunk_pos(out_begin);
                    let in_end_brick = m_in.chunk_pos(&out_end.map(|v| v - 1u32));

                    let in_brick_positions = (0..nd)
                        .into_iter()
                        .map(|i| in_begin_brick[i].raw..=in_end_brick[i].raw)
                        .multi_cartesian_product()
                        .map(|coordinates| {
                            m_in.chunk_index(
                                &Vector::<D, ChunkCoordinate>::try_from(coordinates).unwrap(),
                            )
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
                            let overlap_end = in_end.zip(&out_end, |i, o| i.min(o));
                            let overlap_size =
                                (&overlap_end - &overlap_begin).map(LocalCoordinate::interpret_as);

                            let in_chunk_begin = in_info.in_chunk(&overlap_begin);

                            let out_chunk_begin = out_info.in_chunk(&overlap_begin);

                            let descriptor_config =
                                DescriptorConfig::new([gpu_brick_in, &gpu_brick_out]);

                            let global_size = overlap_size.hmul();

                            //TODO initialization of outside regions
                            unsafe {
                                let mut pipeline = pipeline.bind(cmd);

                                pipeline.push_constant_dyn(&push_constants, |consts| {
                                    consts.array(&m_in.chunk_size.raw())?;
                                    consts.array(&m_out.chunk_size.raw())?;
                                    consts.array(&in_chunk_begin.raw())?;
                                    consts.array(&out_chunk_begin.raw())?;
                                    consts.array(&overlap_size.raw())?;
                                    consts.scalar(global_size as u32)?;
                                    Ok(())
                                });

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
pub fn convolution_1d<D: DynDimension, T: ElementType>(
    input: TensorOperator<D, T>,
    kernel: ArrayOperator<T>,
    dim: usize,
) -> TensorOperator<D, T> {
    let nd = input.metadata.dimensions.len();

    assert!(dim < nd);

    let dtype: DType = input.chunks.dtype().into();

    let push_constants = DynPushConstants::new()
        .array::<u32>(nd, "mem_dim")
        .array::<u32>(nd, "logical_dim_out")
        .array::<u32>(nd, "out_begin")
        .array::<u32>(nd, "global_dim")
        .array::<u32>(nd, "dim_in_chunks")
        .scalar::<u32>("num_chunks")
        .scalar::<u32>("first_chunk_pos")
        .scalar::<i32>("extent");

    const SHADER: &'static str = r#"
#include <util.glsl>
#include <vec.glsl>

layout (local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputBuffer{
    T values[BRICK_MEM_SIZE];
} sourceData[MAX_BRICKS];

layout(std430, binding = 1) readonly buffer KernelBuffer{
    T values[KERNEL_SIZE];
} kernel;

layout(std430, binding = 2) buffer OutputBuffer{
    T values[BRICK_MEM_SIZE];
} outputData;

declare_push_consts(consts)

T kernel_val(int p) {
    int kernel_buf_index = consts.extent - p;
    return kernel.values[kernel_buf_index];
}

T sample_brick(uint[N] pos, int brick) {
    uint local_index = to_linear(pos, consts.mem_dim);
    return sourceData[brick].values[local_index];
}

void main() {
    uint gID = gl_GlobalInvocationID.x;

    if(gID < BRICK_MEM_SIZE) {
        uint[N] out_local = from_linear(gID, consts.mem_dim);
        T acc = T(0);

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
                    T local_val = sample_brick(pos, i);

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
                    T local_val = sample_brick(pos, i);

                    for (int local=chunk_end_local+1; local<=l_end_no_clip; ++local) {
                        int kernel_offset = local - out_pos_rel_to_in_pos_rel;
                        acc += kernel_val(kernel_offset) * local_val;
                    }
                }
            }
        } else {
            //acc = NaN;
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
        input.chunks.dtype(),
        input.metadata.clone(),
        (input, kernel, push_constants),
        move |ctx, mut positions, (input, kernel, push_constants)| {
            async move {
                let device = ctx.preferred_device();

                let m_in = &input.metadata;
                let kernel_m = kernel.metadata;
                let kernel_handle = ctx
                    .submit(kernel.chunks.request_gpu(
                        device.id,
                        ChunkIndex(0),
                        DstBarrierInfo {
                            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            access: vk::AccessFlags2::SHADER_READ,
                        },
                    ))
                    .await;

                let m_out = m_in.clone();

                assert_eq!(
                    kernel_m.dimensions.raw(),
                    kernel_m.chunk_size.raw(),
                    "Only unchunked kernels are supported for now"
                );

                let kernel_size = *kernel_m.dimensions.raw();
                assert!(kernel_size % 2 == 1, "Kernel size must be odd");
                let extent = kernel_size / 2;

                positions.sort_by_key(|(v, _)| v.0);

                let requests = positions.into_iter().map(|(pos, _)| {
                    let out_info = m_out.chunk_info(pos);
                    let out_begin = out_info.begin();
                    let out_end = out_info.end();

                    let in_begin = out_begin
                        .clone()
                        .map_element(dim, |v| (v.raw.saturating_sub(extent as u32)).into());
                    let in_end = out_end
                        .clone()
                        .map_element(dim, |v| (v + extent as u32).min(m_out.dimensions[dim]));

                    let in_begin_brick = m_in.chunk_pos(&in_begin);
                    let in_end_brick = m_in.chunk_pos(&in_end.map(|v| v - 1u32));

                    let in_brick_positions = (0..nd)
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
                            m_in.chunk_index(pos),
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
                        .dependent_on(&nd)
                        .dependent_on(&dim)
                        .dependent_on(&dtype),
                    || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .add("MAX_BRICKS", max_bricks)
                                    .add("DIM", dim)
                                    .add("N", nd)
                                    .add("T", dtype.glsl_type())
                                    .add("BRICK_MEM_SIZE", m_in.chunk_size.hmul())
                                    .add("KERNEL_SIZE", kernel_size)
                                    .push_const_block_dyn(&push_constants),
                                Config::new()
                                    .ext(dtype.glsl_ext())
                                    .ext(Some(crate::vulkan::shader::ext::SCALAR_BLOCK_LAYOUT)),
                            ),
                            true,
                        )
                    },
                )?;

                let mut stream = ctx.submit_unordered_with_data(requests).then_req_with_data(
                    *ctx,
                    |(intersecting_bricks, (pos, in_brick_positions))| {
                        let gpu_brick_out =
                            ctx.alloc_slot_gpu(device, pos, m_out.num_chunk_elements());
                        (
                            gpu_brick_out,
                            (intersecting_bricks, pos, in_brick_positions),
                        )
                    },
                );
                while let Some((gpu_brick_out, (intersecting_bricks, pos, in_brick_positions))) =
                    stream.next().await
                {
                    let out_info = m_out.chunk_info(pos);

                    let out_begin = out_info.begin();

                    for window in in_brick_positions.windows(2) {
                        for d in 0..nd {
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

                            pipeline.push_constant_dyn(&push_constants, |w| {
                                w.array(&m_out.chunk_size.raw())?;
                                w.array(&out_info.logical_dimensions.raw())?;
                                w.array(&out_begin.raw())?;
                                w.array(&m_out.dimensions.raw())?;
                                w.array(&m_out.dimension_in_chunks().raw())?;
                                w.scalar(num_chunks as u32)?;
                                w.scalar(first_chunk_pos)?;
                                w.scalar(extent as i32)?;

                                Ok(())
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
    .into()
}

//TODO: kind of annoying that we have to use a reference to the operator here, but that is the only way it is copy...
pub fn separable_convolution<D: Dimension>(
    mut v: TensorOperator<D, StaticElementType<f32>>,
    kernels: Vector<D, &ArrayOperator<StaticElementType<f32>>>,
) -> TensorOperator<D, StaticElementType<f32>> {
    for dim in (0..D::N).rev() {
        v = convolution_1d(v, kernels[dim].clone(), dim);
    }
    v
}

pub fn mean<'op>(
    input: VolumeOperator<StaticElementType<f32>>,
) -> ScalarOperator<StaticElementType<f32>> {
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

    uvec3 local = from_linear(gID, consts.mem_dim);

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
                let device = ctx.preferred_device();

                let m = input.metadata;

                let to_request = m.brick_positions().collect::<Vec<_>>();
                let batch_size = 1024;

                let pipeline = device.request_state(
                    RessourceId::new("pipeline").of(ctx.current_op()),
                    || {
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
                    },
                )?;

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
                        let pos = m.chunk_index(pos);
                        (
                            input.chunks.request_gpu(
                                device.id,
                                pos,
                                DstBarrierInfo {
                                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                    access: vk::AccessFlags2::SHADER_READ,
                                },
                            ),
                            pos,
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        data::{GlobalCoordinate, LocalVoxelPosition, Vector, VoxelPosition},
        test_util::*,
    };

    #[test]
    fn test_mean_gpu() {
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let input = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            let v = crate::vec::to_linear(&v, &size);
            v as f32
        });

        let output = mean(input);

        let mut runtime =
            crate::runtime::RunTime::new(1 << 30, 1 << 30, None, None, None, None).unwrap();

        let output = &output;
        let mean = runtime
            .resolve(None, false, move |ctx, _| {
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
        let output = linear_rescale(point_vol, scale, offset);
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
        let l = linear_rescale(point_vol.clone(), -2.0, 1.0);
        let l2 = linear_rescale(point_vol, -2.0, -10.0);
        let output = crate::operators::bin_ops::max(l, l2);
        compare_tensor_fn(output, fill_expected);
    }

    #[test]
    fn test_rechunk_gpu() {
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let input = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            crate::data::to_linear(&v, &size) as f32
        });

        let fill_expected = |comp: &mut ndarray::ArrayViewMut3<f32>| {
            for z in 0..size.z().raw {
                for y in 0..size.y().raw {
                    for x in 0..size.x().raw {
                        let pos = VoxelPosition::from([z, y, x]);
                        let val = crate::data::to_linear(&pos, &size) as f32;
                        comp[pos.as_index()] = val
                    }
                }
            }
        };
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
            let input = crate::operators::procedural::rasterize(
                crate::array::VolumeMetaData {
                    dimensions: size,
                    chunk_size: chunk_size.into(),
                },
                r#"float run(float[3] pos_normalized, uint[3] pos_voxel) { return float(pos_voxel[0] + pos_voxel[1] + pos_voxel[2]); }"#,
            );
            let output = rechunk(input, LocalVoxelPosition::from(chunk_size).into_elem());
            compare_tensor_fn(output, fill_expected);
        }
    }

    fn compare_convolution_1d(
        input: VolumeOperator<StaticElementType<f32>>,
        kernel: &[f32],
        fill_expected: impl FnOnce(&mut ndarray::ArrayViewMut3<f32>),
        dim: usize,
    ) {
        let output = convolution_1d(input, crate::operators::array::from_rc(kernel.into()), dim);
        compare_tensor_fn(output, fill_expected);
    }

    fn test_convolution_1d_generic(dim: usize) {
        // Small
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let (point_vol, center) = center_point_vol(size, brick_size);
        compare_convolution_1d(
            point_vol,
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
            point_vol,
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
            vol,
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
        let output = separable_convolution(point_vol, kernels);
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
