use std::hash::{DefaultHasher, Hash, Hasher};

use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};
use futures::StreamExt;
use id::Identify;
use itertools::Itertools;
use rand::prelude::*;

use crate::{
    array::ChunkIndex,
    data::{ChunkCoordinate, LocalCoordinate, Vector},
    dim::*,
    dtypes::{DType, ElementType, StaticElementType},
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    operators::tensor::TensorOperator,
    storage::{gpu, DataVersionType},
    task::RequestStream,
    vulkan::{
        pipeline::{
            AsDescriptors, ComputePipelineBuilder, DescriptorConfig, DynPushConstants,
            LocalSizeConfig,
        },
        shader::Shader,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::{
    array::ArrayOperator, raycaster::TransFuncOperator, scalar::ScalarOperator, volume::ChunkSize,
};

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
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include <util.glsl>
#include <size_util.glsl>

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
    uint gID = global_position_linear;

    if(gID < BRICK_MEM_SIZE) {
        float v = sourceData.values[gID];
        outputData.values[gID] = classify(v);
    }
}
"#;

    TensorOperator::with_state(
        op_descriptor!(),
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

                let pipeline =
                    device.request_state(m.num_chunk_elements(), |device, chunk_elements| {
                        ComputePipelineBuilder::new(
                            Shader::new(SHADER)
                                .define("BRICK_MEM_SIZE", chunk_elements)
                                .push_const_block::<PushConstants>(),
                        )
                        .use_push_descriptor(true)
                        .build(device)
                    })?;

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
                            pipeline.dispatch(device, global_size);
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
pub fn threshold<'op, D: DynDimension>(
    input: TensorOperator<D, StaticElementType<f32>>,
    threshold: f32,
) -> TensorOperator<D, StaticElementType<f32>> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        threshold: f32,
    }
    const SHADER: &'static str = r#"
#include <util.glsl>
#include <size_util.glsl>

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
    uint gID = global_position_linear;

    if(gID < BRICK_MEM_SIZE) {
        outputData.values[gID] = sourceData.values[gID] < consts.threshold ? 0.0 : 1.0;
    }
}
"#;

    TensorOperator::with_state(
        op_descriptor!(),
        Default::default(),
        input.metadata.clone(),
        (input, DataParam(threshold)),
        move |ctx, positions, (input, threshold)| {
            async move {
                let device = ctx.preferred_device();

                let access_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                };
                let m = input.metadata.clone().into_dyn();
                let num_chunk_elements = m.num_chunk_elements();

                let pipeline =
                    device.request_state(m.num_chunk_elements(), |device, chunk_elements| {
                        ComputePipelineBuilder::new(
                            Shader::new(SHADER)
                                .define("BRICK_MEM_SIZE", chunk_elements)
                                .push_const_block::<PushConstants>(),
                        )
                        .use_push_descriptor(true)
                        .build(device)
                    })?;

                let mut brick_stream = ctx
                    .submit_unordered(positions.iter().map(|(pos, _)| {
                        input.chunks.request_inplace_gpu(
                            device.id,
                            *pos,
                            ctx.current_op_desc().unwrap(),
                            DType::scalar(crate::dtypes::ScalarType::F32),
                            num_chunk_elements,
                            access_info,
                        )
                    }))
                    .then_req(*ctx, |inplace| inplace.alloc());

                while let Some(inplace) = brick_stream.next().await {
                    let (gpu_brick_in, gpu_brick_out): (&dyn AsDescriptors, &dyn AsDescriptors) =
                        match &inplace {
                            gpu::InplaceHandle::Inplace(rw, _) => (rw, rw),
                            gpu::InplaceHandle::New(r, w) => (r, w),
                        };

                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config =
                            DescriptorConfig::new([gpu_brick_in, gpu_brick_out]);

                        let global_size = m.num_chunk_elements();

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            let consts = PushConstants {
                                threshold: **threshold,
                            };
                            pipeline.push_constant(consts);

                            pipeline.push_descriptor_set(0, descriptor_config);
                            pipeline.dispatch(device, global_size);
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

    let nd = input.dim().n();

    let push_constants = DynPushConstants::new()
        .vec::<u32>(nd, "mem_size_in")
        .vec::<u32>(nd, "mem_size_out")
        .vec::<u32>(nd, "begin_in")
        .vec::<u32>(nd, "begin_out")
        .vec::<u32>(nd, "region_size")
        .scalar::<u32>("global_size");

    const SHADER: &'static str = r#"
#include <util.glsl>
#include <vec.glsl>
#include <size_util.glsl>

layout(std430, binding = 0) readonly buffer InputBuffer{
    T values[BRICK_MEM_SIZE_IN];
} sourceData;

layout(std430, binding = 1) buffer OutputBuffer{
    T values[];
} outputData;

declare_push_consts(constants);

void main() {
    uint gID = global_position_linear;

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
        op_descriptor!(),
        input.chunks.dtype(),
        {
            let mut m = input.metadata.clone();
            m.chunk_size = chunk_size.zip(&m.dimensions, |v, d| v.apply(d));
            m
        },
        (input, DataParam(chunk_size), DataParam(push_constants)),
        |ctx, mut positions, (input, chunk_size, push_constants)| {
            async move {
                let device = ctx.preferred_device();

                let dtype: DType = input.chunks.dtype().into();

                let nd = input.metadata.dimensions.len();

                let m_in = input.metadata.clone();
                let m_out = {
                    let mut m_out = m_in.clone();
                    m_out.chunk_size = chunk_size.zip(&m_in.dimensions, |v, d| v.apply(d));
                    m_out
                };

                positions.sort_by_key(|(v, _)| v.0);

                let pipeline = device.request_state(
                    (&push_constants, &m_in.num_chunk_elements(), &dtype, &nd),
                    |device, (push_constants, num_elements, dtype, nd)| {
                        ComputePipelineBuilder::new(
                            Shader::new(SHADER)
                                .define("BRICK_MEM_SIZE_IN", num_elements)
                                .define("N", nd)
                                .define("T", dtype.glsl_type())
                                .push_const_block_dyn(push_constants)
                                .ext(dtype.glsl_ext())
                                .ext(Some(crate::vulkan::shader::ext::SCALAR_BLOCK_LAYOUT)),
                        )
                        .use_push_descriptor(true)
                        .build(device)
                    },
                )?;

                let caches = positions.into_iter().map(|(pos, _)| {
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

                    let tile_done =
                        ctx.access_state_cache::<u8>(pos, "tile_done", in_brick_positions.len());
                    (tile_done, (pos, in_brick_positions))
                });

                let mut stream = ctx
                    .submit_unordered_with_data(caches)
                    .then_req_with_data(*ctx, |(tile_done, (pos, in_brick_positions))| {
                        let mut tile_done = unsafe {
                            tile_done.init(|r| {
                                crate::data::fill_uninit(r, 0);
                            })
                        };
                        let reuse_res =
                            ctx.alloc_try_reuse_gpu(device, pos, m_out.num_chunk_elements());

                        if reuse_res.new {
                            //println!("lost prev version :(");
                            tile_done.fill(0);
                        }

                        let in_brick_positions_to_fill: Vec<_> = in_brick_positions
                            .into_iter()
                            .enumerate()
                            .filter_map(|(i, position)| {
                                (tile_done[i] == 0).then_some((i, position))
                            })
                            .collect();

                        let brick_requests =
                            ctx.group(in_brick_positions_to_fill.iter().map(|(_, pos)| {
                                input.chunks.request_gpu(
                                    device.id,
                                    *pos,
                                    DstBarrierInfo {
                                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                        access: vk::AccessFlags2::SHADER_READ,
                                    },
                                )
                            }));

                        (
                            brick_requests,
                            (
                                reuse_res.request,
                                pos,
                                in_brick_positions_to_fill,
                                tile_done,
                            ),
                        )
                    })
                    .then_req_with_data(
                        *ctx,
                        |(in_bricks, (out_request, pos, in_brick_positions_to_fill, tile_done))| {
                            (
                                out_request,
                                (in_bricks, pos, in_brick_positions_to_fill, tile_done),
                            )
                        },
                    );

                while let Some((
                    gpu_brick_out,
                    (intersecting_bricks, pos, in_brick_positions, mut tile_done),
                )) = stream.next().await
                {
                    let out_info = m_out.chunk_info(pos);

                    device.with_cmd_buffer(|cmd| {
                        let out_begin = out_info.begin();
                        let out_end = out_info.end();

                        //let skip = tile_done.len() - intersecting_bricks.len();
                        //if skip > 0 {
                        //    println!("skipping {}", skip);
                        //}

                        for (gpu_brick_in, (tile_index, in_brick_pos)) in intersecting_bricks
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
                                    consts.vec(&m_in.chunk_size.raw())?;
                                    consts.vec(&m_out.chunk_size.raw())?;
                                    consts.vec(&in_chunk_begin.raw())?;
                                    consts.vec(&out_chunk_begin.raw())?;
                                    consts.vec(&overlap_size.raw())?;
                                    consts.scalar(global_size as u32)?;
                                    Ok(())
                                });

                                pipeline.push_descriptor_set(0, descriptor_config);
                                pipeline.dispatch(device, global_size);
                            }

                            if gpu_brick_in.version == DataVersionType::Final {
                                tile_done[tile_index] = 1;
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
pub fn convolution_1d<D: DynDimension, T: ElementType, K: ElementType>(
    input: TensorOperator<D, T>,
    kernel: ArrayOperator<K>,
    dim: usize,
) -> TensorOperator<D, T> {
    let nd = input.dim().n();

    assert!(dim < nd);

    let push_constants = DynPushConstants::new()
        .vec::<u32>(nd, "mem_dim")
        .vec::<u32>(nd, "logical_dim_out")
        .vec::<u32>(nd, "out_begin")
        .vec::<u32>(nd, "global_dim")
        .vec::<u32>(nd, "dim_in_chunks")
        .scalar::<u32>("num_chunks")
        .scalar::<u32>("first_chunk_pos")
        .scalar::<i32>("extent");

    const SHADER: &'static str = r#"
#include <util.glsl>
#include <vec.glsl>
#include <size_util.glsl>

layout(std430, binding = 0) readonly buffer InputBuffer{
    T values[BRICK_MEM_SIZE];
} sourceData[MAX_BRICKS];

layout(std430, binding = 1) readonly buffer KernelBuffer{
    K values[KERNEL_SIZE];
} kernel;

layout(std430, binding = 2) buffer OutputBuffer{
    T values[BRICK_MEM_SIZE];
} outputData;

declare_push_consts(consts);

K kernel_val(int p) {
    int kernel_buf_index = consts.extent - p;
    return kernel.values[kernel_buf_index];
}

T sample_brick(uint[N] pos, int brick) {
    uint local_index = to_linear(pos, consts.mem_dim);
    return sourceData[brick].values[local_index];
}

void main() {
    uint gID = global_position_linear;

    if(gID < BRICK_MEM_SIZE) {
        uint[N] out_local = from_linear(gID, consts.mem_dim);
        K acc = K(0);

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
                        acc += kernel_val(kernel_offset) * K(local_val);
                    }
                }

                for (int local=l_begin; local<=l_end; ++local) {
                    int kernel_offset = local - out_pos_rel_to_in_pos_rel;
                    pos[DIM] = local;
                    acc += kernel_val(kernel_offset) * K(sample_brick(pos, i));
                }

                // Border handling for last chunk in dim
                if(chunk_pos == last_chunk) {
                    pos[DIM] = chunk_end_local; //Clip to volume/chunk
                    T local_val = sample_brick(pos, i);

                    for (int local=chunk_end_local+1; local<=l_end_no_clip; ++local) {
                        int kernel_offset = local - out_pos_rel_to_in_pos_rel;
                        acc += kernel_val(kernel_offset) * K(local_val);
                    }
                }
            }
        } else {
            //acc = NaN;
        }

        outputData.values[gID] = T(acc);
    }
}
"#;
    TensorOperator::with_state(
        op_descriptor!(),
        input.chunks.dtype(),
        input.metadata.clone(),
        (input, kernel, DataParam(push_constants), DataParam(dim)),
        |ctx, mut positions, (input, kernel, push_constants, dim)| {
            async move {
                let device = ctx.preferred_device();

                let dtype: DType = input.dtype().into();
                let kernel_dtype: DType = kernel.dtype().into();
                let nd = input.dim().n();
                let dim = **dim;

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

                let pipeline = device.request_state(
                    (
                        push_constants,
                        max_bricks,
                        dim,
                        nd,
                        dtype,
                        kernel_dtype,
                        m_in.chunk_size.hmul(),
                        kernel_size,
                    ),
                    |device,
                     (
                        push_constants,
                        max_bricks,
                        dim,
                        nd,
                        dtype,
                        kernel_dtype,
                        mem_size,
                        kernel_size,
                    )| {
                        ComputePipelineBuilder::new(
                            Shader::new(SHADER)
                                .define("MAX_BRICKS", max_bricks)
                                .define("DIM", dim)
                                .define("N", nd)
                                .define("T", dtype.glsl_type())
                                .define("K", kernel_dtype.glsl_type())
                                .define("BRICK_MEM_SIZE", mem_size)
                                .define("KERNEL_SIZE", kernel_size)
                                .push_const_block_dyn(&push_constants)
                                .ext(dtype.glsl_ext())
                                .ext(kernel_dtype.glsl_ext())
                                .ext(Some(crate::vulkan::shader::ext::SCALAR_BLOCK_LAYOUT)),
                        )
                        .use_push_descriptor(true)
                        .build(device)
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
                                w.vec(&m_out.chunk_size.raw())?;
                                w.vec(&out_info.logical_dimensions.raw())?;
                                w.vec(&out_begin.raw())?;
                                w.vec(&m_out.dimensions.raw())?;
                                w.vec(&m_out.dimension_in_chunks().raw())?;
                                w.scalar(num_chunks as u32)?;
                                w.scalar(first_chunk_pos)?;
                                w.scalar(extent as i32)?;

                                Ok(())
                            });
                            pipeline.push_descriptor_set(0, descriptor_config);
                            pipeline.dispatch(device, global_size);
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
pub fn separable_convolution<D: DynDimension, T: ElementType, K: ElementType>(
    mut v: TensorOperator<D, T>,
    kernels: Vector<D, &ArrayOperator<K>>,
) -> TensorOperator<D, T> {
    assert_eq!(v.dim(), kernels.dim());
    for dim in (0..v.dim().n()).rev() {
        v = convolution_1d(v, kernels[dim].clone(), dim);
    }
    v
}

#[derive(Copy, Clone, Identify)]
pub enum AggretationMethod {
    Mean,
    Max,
    Min,
}

#[derive(Copy, Clone, Identify)]
pub enum SampleMethod {
    All,
    Subset(usize),
}

impl From<Option<usize>> for SampleMethod {
    fn from(value: Option<usize>) -> Self {
        match value {
            Some(n) => Self::Subset(n),
            None => Self::All,
        }
    }
}

impl AggretationMethod {
    fn norm_factor(&self, num_voxels: usize) -> f32 {
        match self {
            AggretationMethod::Mean => 1.0 / num_voxels as f32,
            AggretationMethod::Max | AggretationMethod::Min => 1.0,
        }
    }
    fn aggregration_function_glsl(&self) -> &'static str {
        match self {
            AggretationMethod::Mean => "atomic_add_float",
            AggretationMethod::Min => "atomic_min_float",
            AggretationMethod::Max => "atomic_max_float",
        }
    }
    fn subgroup_aggregration_function_glsl(&self) -> &'static str {
        match self {
            AggretationMethod::Mean => "subgroupAdd",
            AggretationMethod::Min => "subgroupMin",
            AggretationMethod::Max => "subgroupMax",
        }
    }
    fn neutral_val(&self) -> f32 {
        match self {
            AggretationMethod::Mean => 0.0,
            AggretationMethod::Min => f32::INFINITY,
            AggretationMethod::Max => -f32::INFINITY,
        }
    }
}

pub fn mean<'op, D: DynDimension>(
    input: TensorOperator<D, StaticElementType<f32>>,
    sample_method: SampleMethod,
) -> ScalarOperator<StaticElementType<f32>> {
    scalar_aggregation(input, AggretationMethod::Mean, sample_method)
}

pub fn min<'op, D: DynDimension>(
    input: TensorOperator<D, StaticElementType<f32>>,
    sample_method: SampleMethod,
) -> ScalarOperator<StaticElementType<f32>> {
    scalar_aggregation(input, AggretationMethod::Min, sample_method)
}

pub fn max<'op, D: DynDimension>(
    input: TensorOperator<D, StaticElementType<f32>>,
    sample_method: SampleMethod,
) -> ScalarOperator<StaticElementType<f32>> {
    scalar_aggregation(input, AggretationMethod::Max, sample_method)
}

fn scalar_aggregation<'op, D: DynDimension>(
    input: TensorOperator<D, StaticElementType<f32>>,
    method: AggretationMethod,
    sample_method: SampleMethod,
) -> ScalarOperator<StaticElementType<f32>> {
    const SHADER: &'static str = r#"
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <atomic.glsl>
#include <size_util.glsl>
#include <vec.glsl>

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
    uint gID = global_position_linear;
    uint lID = local_index_subgroup_order;
    if(lID == 0) {
        shared_sum = floatBitsToUint(NEUTRAL_VAL);
    }
    barrier();

    float val;

    uint[ND] local = from_linear(gID, consts.mem_dim);

    if(all(less_than(local, consts.logical_dim))) {
        val = sourceData.values[gID] * consts.norm_factor;
    } else {
        val = NEUTRAL_VAL;
    }

    float sg_agg = AGG_FUNCTION_SUBGROUP(val);

    if(gl_SubgroupInvocationID == 0) {
        AGG_FUNCTION(shared_sum, sg_agg);
    }

    barrier();

    if(lID == 0) {
        AGG_FUNCTION(sum.value, uintBitsToFloat(shared_sum));
    }
}
"#;
    crate::operators::scalar::scalar(
        op_descriptor!(),
        (input, DataParam(method), DataParam(sample_method)),
        move |ctx, (input, method, sample_method)| {
            async move {
                let device = ctx.preferred_device();

                let nd = input.dim().n();

                let push_constants = DynPushConstants::new()
                    .vec::<u32>(nd, "mem_dim")
                    .vec::<u32>(nd, "logical_dim")
                    .scalar::<f32>("norm_factor");

                let m = &input.metadata;

                let mut all_chunks = m.chunk_indices().into_iter().collect::<Vec<_>>();
                let (to_request, num_elements) = match **sample_method {
                    SampleMethod::All => (all_chunks.as_slice(), m.dimensions.hmul()),
                    SampleMethod::Subset(n) => {
                        let mut h = DefaultHasher::new();
                        ctx.current_op().inner().hash(&mut h);
                        let seed = h.finish();
                        let mut rng = rand::rngs::SmallRng::seed_from_u64(dbg!(seed));
                        let (ret, _) = all_chunks.partial_shuffle(&mut rng, n);
                        ret.sort();

                        let num_elements = ret
                            .iter()
                            .map(|pos| m.chunk_info(*pos).logical_dimensions.hmul())
                            .sum();
                        (&*ret, num_elements)
                    }
                };

                let pipeline = device.request_state(
                    (&push_constants, m.chunk_size.hmul(), method, nd),
                    |device, (push_constants, mem_size, method, nd)| {
                        let neutral_val_str =
                            format!("uintBitsToFloat({})", method.neutral_val().to_bits());
                        ComputePipelineBuilder::new(
                            Shader::new(SHADER)
                                .push_const_block_dyn(push_constants)
                                .define("BRICK_MEM_SIZE", mem_size)
                                .define("ND", nd)
                                .define("AGG_FUNCTION", method.aggregration_function_glsl())
                                .define(
                                    "AGG_FUNCTION_SUBGROUP",
                                    method.subgroup_aggregration_function_glsl(),
                                )
                                .define("NEUTRAL_VAL", neutral_val_str),
                        )
                        .local_size(LocalSizeConfig::Large)
                        .use_push_descriptor(true)
                        .build(device)
                    },
                )?;

                let sum = ctx.submit(ctx.alloc_scalar_gpu(device)).await;

                let normalization_factor = method.norm_factor(num_elements);

                device.with_cmd_buffer(|cmd| {
                    unsafe {
                        device.functions().cmd_update_buffer(
                            cmd.raw(),
                            sum.buffer,
                            0,
                            bytemuck::cast_slice(&[method.neutral_val()]),
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

                let mut stream = ctx.submit_unordered_with_data(to_request.iter().map(|pos| {
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

                            pipeline.push_constant_dyn(&push_constants, |consts| {
                                consts.vec(&brick_info.mem_dimensions.into_elem::<u32>())?;
                                consts.vec(&brick_info.logical_dimensions.into_elem::<u32>())?;
                                consts.scalar(normalization_factor)?;
                                Ok(())
                            });
                            pipeline.push_descriptor_set(0, descriptor_config);
                            pipeline.dispatch(device, global_size);
                        }
                    });
                }
                ctx.submit(device.wait_for_current_cmd_buffer_completion())
                    .await;

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

        let output = mean(input, SampleMethod::All);

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
        input: crate::operators::volume::VolumeOperator<StaticElementType<f32>>,
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
