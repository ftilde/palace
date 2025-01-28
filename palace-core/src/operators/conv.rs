use ash::vk;
use futures::StreamExt;

use crate::{
    array::ChunkIndex,
    chunk_utils::ChunkNeighborhood,
    data::GlobalCoordinate,
    dim::DynDimension,
    dtypes::{DType, ElementType},
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    task::RequestStream,
    vec::Vector,
    vulkan::{
        pipeline::{ComputePipelineBuilder, DescriptorConfig, DynPushConstants},
        shader::Shader,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::{array::ArrayOperator, tensor::TensorOperator};

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
                    pos[DIM] = chunk_begin_local; //Clip to tensor/chunk
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
                    pos[DIM] = chunk_end_local; //Clip to tensor/chunk
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
                    let extent_vec = Vector::<_, GlobalCoordinate>::fill_with_len(0.into(), nd)
                        .map_element(dim, |_v| extent.into());
                    let chunk_neighbors =
                        ChunkNeighborhood::around(&m_out, pos, extent_vec.clone(), extent_vec);

                    let in_brick_positions =
                        chunk_neighbors.chunk_positions_linear().collect::<Vec<_>>();

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
        dim::D3,
        dtypes::StaticElementType,
        test_util::*,
    };

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
