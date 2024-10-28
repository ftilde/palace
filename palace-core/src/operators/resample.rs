use ash::vk;
use futures::StreamExt;
use itertools::Itertools;

use crate::{
    array::TensorMetaData,
    coordinate::ChunkCoordinate,
    data::{Matrix, Vector, AABB},
    dim::*,
    dtypes::{DType, ElementType},
    operator::{OpaqueOperator, OperatorDescriptor},
    task::RequestStream,
    vulkan::{
        pipeline::{ComputePipeline, DescriptorConfig, DynPushConstants},
        shader::{Config, ShaderDefines},
        state::RessourceId,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::tensor::{EmbeddedTensorOperator, LODTensorOperator, TensorOperator};

pub fn resample_rescale_mat<'op, D: LargerDim>(
    input_size: TensorMetaData<D>,
    output_size: TensorMetaData<D>,
) -> Matrix<D::Larger, f32> {
    let to_input_center =
        Matrix::from_translation(Vector::<D, f32>::fill_with_len(-0.5, input_size.dim().n()));
    let rescale = Matrix::from_scale(
        &(&input_size.dimensions.raw().f32() / &output_size.dimensions.raw().f32()),
    )
    .to_homogeneous();
    let to_output_center =
        Matrix::from_translation(Vector::<D, f32>::fill_with_len(0.5, input_size.dim().n()));
    let out = to_input_center * &rescale * &to_output_center;

    out
}

pub fn resample<'op, D: LargerDim, T: ElementType>(
    input: EmbeddedTensorOperator<D, T>,
    output_size: TensorMetaData<D>,
) -> EmbeddedTensorOperator<D, T> {
    let mat = resample_rescale_mat(input.metadata.clone(), output_size.clone());
    let inner = resample_transform(input.inner, output_size.clone(), mat.clone());
    let mut embedding_data = input.embedding_data;

    // We know that mat is only an affine transformation, so we can extract the scaling
    // parts directly.
    let scale_mat = mat.to_scaling_part();
    let scale =
        Vector::<D, f32>::try_from_fn_and_len(output_size.dim().n(), |i| scale_mat.at(i, i))
            .unwrap();
    embedding_data.spacing = embedding_data.spacing * scale;

    EmbeddedTensorOperator {
        inner,
        embedding_data,
    }
}

pub fn smooth_downsample<'op, D: LargerDim, T: ElementType>(
    input: EmbeddedTensorOperator<D, T>,
    output_size: TensorMetaData<D>,
) -> EmbeddedTensorOperator<D, T> {
    const DOWNSAMPLE_SCALE_MULT: f32 = 0.5;
    let scale = {
        let s_in = input.metadata.dimensions.raw().f32();
        let s_out = output_size.dimensions.raw().f32();

        let downsample_factor = s_in / s_out;
        let stddev = downsample_factor.scale(DOWNSAMPLE_SCALE_MULT);
        stddev
    };

    let mut v = input;
    for dim in (0..v.dim().n()).rev() {
        let scale = scale[dim];
        if scale != DOWNSAMPLE_SCALE_MULT {
            let kernel = crate::operators::kernels::gauss(scale);
            v = v.map_inner(|v| crate::operators::volume_gpu::convolution_1d(v, kernel, dim));
        }
    }

    resample(v, output_size)
}

pub fn coarser_lod_md<D: LargerDim, E: ElementType>(
    input: &EmbeddedTensorOperator<D, E>,
    step_factor: f32,
) -> TensorMetaData<D> {
    assert!(step_factor > 1.0);

    let dim = input.dim();

    let e = input.embedding_data.clone();
    let m = input.metadata.clone();

    let new_spacing_raw = e.spacing.clone() * Vector::fill_with_len(step_factor, dim.n());
    let smallest_new = new_spacing_raw.fold(f32::MAX, |a, b| a.min(b));
    let new_spacing = e
        .spacing
        .clone()
        .zip(&Vector::fill_with_len(smallest_new, dim.n()), |a, b| {
            a.max(b)
        });
    let element_ratio = e.spacing / new_spacing;
    let new_dimensions = (m.dimensions.raw().f32() * element_ratio)
        .map(|v| v.ceil() as u32)
        .global();
    TensorMetaData {
        dimensions: new_dimensions,
        chunk_size: m.chunk_size,
    }
}

pub fn create_lod<D: LargerDim, T: ElementType>(
    input: EmbeddedTensorOperator<D, T>,
    step_factor: f32,
) -> LODTensorOperator<D, T> {
    assert!(step_factor > 1.0);

    let mut levels = Vec::new();
    let mut current = input;

    levels.push(current.clone());

    while current.metadata.dimension_in_chunks().hmul() != 1 {
        let new_md = coarser_lod_md(&current, step_factor);

        //TODO: Maybe we do not want to hardcode this. It would also be easy to offer something
        //like "cache everything but the highest resolution layer" on LODTensorOperator
        current = smooth_downsample(current, new_md.clone()).cache();
        levels.push(current.clone());
    }

    LODTensorOperator { levels }
}

pub fn resample_transform<D: LargerDim, T: ElementType>(
    input: TensorOperator<D, T>,
    output_size: TensorMetaData<D>,
    element_out_to_in: Matrix<D::Larger, f32>,
) -> TensorOperator<D, T> {
    let nd = input.dim().n();

    let push_constants = DynPushConstants::new()
        .mat::<f32>(nd + 1, "transform")
        .vec::<u32>(nd, "chunk_dim_in")
        .vec::<u32>(nd, "vol_dim_in")
        .vec::<u32>(nd, "mem_size_out")
        .vec::<u32>(nd, "out_begin");

    let dtype_dyn: DType = input.dtype().into();

    const SHADER: &'static str = r#"
#include <util.glsl>
#include <mat.glsl>
#include <vec.glsl>
#include <util.glsl>

#define ChunkValue T

#define BRICK_MEM_SIZE BRICK_MEM_SIZE_IN
#include <sample.glsl>
#undef BRICK_MEM_SIZE

#if N == 1
layout (local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
#elif N == 2
layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
#else
layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
#endif


layout(std430, binding = 0) buffer RefBuffer {
    Chunk values[NUM_CHUNKS];
} bricks;

layout(std430, binding = 1) buffer OutputBuffer{
    T values[];
} outputData;

declare_push_consts(consts);

void main() {

    // Reverse the invocation mapping which squashes higher dims into z

    uint[N] out_brick_pos;
    out_brick_pos[N-1] = gl_GlobalInvocationID.x;
#if N >= 2
    out_brick_pos[N-2] = gl_GlobalInvocationID.y;
#endif
#if N >= 3
    uint[N-2] coarse_dims;
    for(int i = 0; i < N-2; i+=1) {
        coarse_dims[i] = consts.mem_size_out[i];
    }
    uint[N-2] coarse_pos = from_linear(gl_GlobalInvocationID.z, coarse_dims);
    for(int i = 0; i < N-2; i+=1) {
        out_brick_pos[i] = coarse_pos[i];
    }
#endif

    if(all(less_than(out_brick_pos, consts.mem_size_out))) {
        uint[N] global_pos = add(out_brick_pos, consts.out_begin);
        float[N] sample_pos = from_homogeneous(mul(consts.transform, to_homogeneous(to_float(global_pos))));
        map(N, sample_pos, sample_pos, round);

        T default_val = T(0);

        TensorMetaData(N) m_in;
        m_in.dimensions = consts.vol_dim_in;
        m_in.chunk_size = consts.chunk_dim_in;

        int res;
        uint sample_brick_pos_linear;
        T sampled_intensity;
        try_sample(N, sample_pos, m_in, bricks.values, res, sample_brick_pos_linear, sampled_intensity);

        if(res == SAMPLE_RES_FOUND) {
            // Nothing to do!
        } else if(res == SAMPLE_RES_NOT_PRESENT) {
            // This SHOULD not happen...
            sampled_intensity = default_val;
        } else /* SAMPLE_RES_OUTSIDE */ {
            sampled_intensity = default_val;
        }

        uint gID = to_linear(out_brick_pos, consts.mem_size_out);
        outputData.values[gID] = sampled_intensity;
    }
}
"#;

    TensorOperator::with_state(
        OperatorDescriptor::new("resample")
            .dependent_on(&input)
            .dependent_on_data(&output_size)
            .dependent_on_data(&element_out_to_in),
        input.dtype(),
        output_size.clone(),
        (input, output_size, element_out_to_in, push_constants),
        move |ctx, mut positions, (input, output_size, element_out_to_in, push_constants)| {
            async move {
                let device = ctx.preferred_device();

                let dst_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };

                let m_in = &input.metadata;
                let m_out = output_size;

                let num_chunks = m_in.dimension_in_chunks().hmul();

                let pipeline = device.request_state(
                    RessourceId::new("pipeline")
                        .of(ctx.current_op())
                        .dependent_on(&num_chunks)
                        .dependent_on(&m_in.chunk_size)
                        .dependent_on(&nd)
                        .dependent_on(&dtype_dyn),
                    || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .add("NUM_CHUNKS", num_chunks)
                                    .add("BRICK_MEM_SIZE_IN", m_in.chunk_size.hmul())
                                    .add("N", nd)
                                    .add("T", dtype_dyn.glsl_type())
                                    .push_const_block_dyn(&push_constants),
                                Config::new()
                                    .ext(dtype_dyn.glsl_ext())
                                    .ext(Some(crate::vulkan::shader::ext::SCALAR_BLOCK_LAYOUT))
                                    .ext(Some(crate::vulkan::shader::ext::BUFFER_REFERENCE))
                                    .ext(Some(crate::vulkan::shader::ext::INT64_TYPES)),
                            ),
                            true,
                        )
                    },
                )?;
                positions.sort_by_key(|(v, _)| v.0);

                let requests = positions.into_iter().map(|(pos, _)| {
                    let out_info = m_out.chunk_info(pos);
                    let pos_vec = m_out.chunk_pos_from_index(pos);
                    assert!(pos_vec
                        .zip(&m_out.dimension_in_chunks(), |l, r| l < r)
                        .all());
                    let out_begin = out_info.begin();
                    let out_end = out_info.end();

                    let aabb = AABB::new(
                        &out_begin.map(|v| v.raw as f32),
                        &out_end.map(|v| (v.raw - 1) as f32),
                    );
                    let aabb = aabb.transform(&element_out_to_in);

                    let out_begin = aabb.lower().map(|v| v.floor().max(0.0) as u32).global();
                    let out_end = aabb.upper().map(|v| v.ceil() as u32).global();

                    let in_begin_brick = m_in.chunk_pos(&out_begin);
                    let in_end_brick = m_in
                        .chunk_pos(&out_end)
                        .zip(&m_in.dimension_in_chunks(), |l, r| l.min(r - 1u32));

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
                    // TODO: It would be nice to share the chunk_index between requests, but then
                    // we never free any chunks and run out of memory. Maybe when/if we have a
                    // better approach for indices this can be revisited.
                    let chunk_index = device
                        .storage
                        .get_index(
                            *ctx,
                            device,
                            input.chunks.descriptor(),
                            num_chunks,
                            dst_info,
                        )
                        .await;

                    let out_info = m_out.chunk_info(pos);

                    for (gpu_brick_in, in_brick_pos) in intersecting_bricks
                        .into_iter()
                        .zip(in_brick_positions.into_iter())
                    {
                        let brick_pos_linear =
                            crate::data::to_linear(&in_brick_pos, &m_in.dimension_in_chunks());
                        chunk_index.insert(brick_pos_linear as u64, gpu_brick_in);
                    }

                    // Make writes to the index visible
                    ctx.submit(device.barrier(
                        SrcBarrierInfo {
                            stage: vk::PipelineStageFlags2::TRANSFER,
                            access: vk::AccessFlags2::TRANSFER_WRITE,
                        },
                        DstBarrierInfo {
                            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                            access: vk::AccessFlags2::SHADER_READ,
                        },
                    ))
                    .await;

                    let descriptor_config = DescriptorConfig::new([&chunk_index, &gpu_brick_out]);

                    // map [u, w, z, y, x] to [uwz, y, x]
                    // and [y, x] to [1, y, x]
                    let full_size = m_out.chunk_size.raw();
                    let mut size = Vector::<D3, u32>::fill(1);
                    for i in 0..nd {
                        let oi = i.saturating_sub(nd.saturating_sub(D3::N));
                        size[oi] *= full_size[i];
                    }

                    device.with_cmd_buffer(|cmd| unsafe {
                        let mut pipeline = pipeline.bind(cmd);

                        pipeline.push_constant_dyn(&push_constants, |consts| {
                            consts.mat(element_out_to_in)?;
                            consts.vec(&m_in.chunk_size.raw())?;
                            consts.vec(&m_in.dimensions.raw())?;
                            consts.vec(&m_out.chunk_size.raw())?;
                            consts.vec(&out_info.begin().raw())?;
                            Ok(())
                        });

                        pipeline.push_descriptor_set(0, descriptor_config);
                        pipeline.dispatch3d(size);
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
