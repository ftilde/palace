use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};
use futures::StreamExt;
use itertools::Itertools;

use crate::{
    array::ChunkIndex,
    data::{ChunkCoordinate, Vector},
    dim::*,
    dtypes::{DType, ElementType, StaticElementType},
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    operators::tensor::TensorOperator,
    storage::gpu,
    task::RequestStream,
    vulkan::{
        pipeline::{AsDescriptors, ComputePipelineBuilder, DescriptorConfig, DynPushConstants},
        shader::Shader,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::{array::ArrayOperator, raycaster::TransFuncOperator};

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
                        let output = ctx.alloc_slot_gpu(device, pos, &m.chunk_size);
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

K sample_kernel(int p) {
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

        K[TND] acc;
        for(int j = 0; j < TND; ++j) {
            acc[j] = K(0);
        }

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
                        K kernel_val = sample_kernel(kernel_offset);

                        for(int j = 0; j < TND; ++j) {
                            acc[j] += kernel_val * K(local_val[j]);
                        }
                    }
                }

                for (int local=l_begin; local<=l_end; ++local) {
                    int kernel_offset = local - out_pos_rel_to_in_pos_rel;
                    pos[DIM] = local;
                    T local_val = sample_brick(pos, i);
                    K kernel_val = sample_kernel(kernel_offset);

                    for(int j = 0; j < TND; ++j) {
                        acc[j] += kernel_val * K(local_val[j]);
                    }
                }

                // Border handling for last chunk in dim
                if(chunk_pos == last_chunk) {
                    pos[DIM] = chunk_end_local; //Clip to volume/chunk
                    T local_val = sample_brick(pos, i);

                    for (int local=chunk_end_local+1; local<=l_end_no_clip; ++local) {
                        int kernel_offset = local - out_pos_rel_to_in_pos_rel;
                        K kernel_val = sample_kernel(kernel_offset);

                        for(int j = 0; j < TND; ++j) {
                            acc[j] += kernel_val * K(local_val[j]);
                        }
                    }
                }
            }
        } else {
            //acc = NaN;
        }

        for(int j = 0; j < TND; ++j) {
            outputData.values[gID][j] = T_SCALAR(acc[j]);
        }
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
                assert!(kernel_dtype.is_scalar());

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
                                .define("T", dtype.glsl_type_force_vec())
                                .define("K", kernel_dtype.glsl_type())
                                .define("TND", dtype.size)
                                .define("T_SCALAR", dtype.scalar.glsl_type())
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
                        let gpu_brick_out = ctx.alloc_slot_gpu(device, pos, &m_out.chunk_size);
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        data::{GlobalCoordinate, LocalVoxelPosition, Vector, VoxelPosition},
        operators::rechunk::rechunk,
        test_util::*,
    };

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
        input: crate::operators::tensor::VolumeOperator<StaticElementType<f32>>,
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
