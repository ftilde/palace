use ash::vk;
use futures::StreamExt;
use itertools::Itertools;

use crate::{
    array::TensorMetaData,
    coordinate::ChunkCoordinate,
    data::{Matrix, Vector, AABB},
    dim::*,
    operator::{OpaqueOperator, OperatorId},
    vulkan::{
        pipeline::{ComputePipeline, DescriptorConfig},
        shader::ShaderDefines,
        state::RessourceId,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::tensor::{EmbeddedTensorOperator, LODTensorOperator, TensorOperator};

pub fn resample_rescale_mat<'op, D: LargerDim>(
    input_size: TensorMetaData<D>,
    output_size: TensorMetaData<D>,
) -> Matrix<D::Larger, f32> {
    let to_input_center = Matrix::from_translation(Vector::<D, f32>::fill(-0.5));
    let rescale =
        Matrix::from_scale(input_size.dimensions.raw().f32() / output_size.dimensions.raw().f32())
            .to_homogeneous();
    let to_output_center = Matrix::from_translation(Vector::<D, f32>::fill(0.5));
    let out = to_input_center * rescale * to_output_center;

    out
}

pub fn resample<'op, D: LargerDim>(
    input: EmbeddedTensorOperator<D, f32>,
    output_size: TensorMetaData<D>,
) -> EmbeddedTensorOperator<D, f32> {
    let mat = resample_rescale_mat(input.metadata.clone(), output_size.clone());
    let inner = resample_transform(input.inner, output_size, mat.clone());
    let mut embedding_data = input.embedding_data;

    // We know that mat is only an affine transformation, so we can extract the scaling
    // parts directly.
    let scale_mat = mat.to_scaling_part();
    let scale = Vector::<D, f32>::from_fn(|i| *scale_mat.at(i, i));
    embedding_data.spacing = embedding_data.spacing * scale;

    EmbeddedTensorOperator {
        inner,
        embedding_data,
    }
}

pub fn smooth_downsample<'op, D: LargerDim>(
    input: EmbeddedTensorOperator<D, f32>,
    output_size: TensorMetaData<D>,
) -> EmbeddedTensorOperator<D, f32> {
    let scale = {
        let s_in = input.metadata.dimensions.raw().f32();
        let s_out = output_size.dimensions.raw().f32();

        let downsample_factor = s_in / s_out;
        let stddev = downsample_factor.scale(0.5);
        stddev
    };

    let kernels = scale
        .into_iter()
        .map(|scale| crate::operators::kernels::gauss(scale))
        .collect::<Vec<_>>();
    let kernel_refs = Vector::from_fn(|i| &kernels[i]);
    let smoothed =
        input.map_inner(|v| crate::operators::volume_gpu::separable_convolution(v, kernel_refs));
    resample(smoothed, output_size)
}

pub fn create_lod<'op, D: LargerDim>(
    input: EmbeddedTensorOperator<D, f32>,
    step_factor: f32,
) -> LODTensorOperator<D, f32> {
    assert!(step_factor > 1.0);

    let mut levels = Vec::new();
    let mut current = input;

    levels.push(current.clone());

    loop {
        let new_md = {
            let e = current.embedding_data;
            let m = current.metadata;

            let new_spacing_raw = e.spacing * Vector::fill(step_factor);
            let smallest_new = new_spacing_raw.fold(f32::MAX, |a, b| a.min(b));
            let new_spacing = e.spacing.zip(Vector::fill(smallest_new), |a, b| a.max(b));
            let element_ratio = e.spacing / new_spacing;
            let new_dimensions = (m.dimensions.raw().f32() * element_ratio)
                .map(|v| v.ceil() as u32)
                .global();
            TensorMetaData {
                dimensions: new_dimensions,
                chunk_size: m.chunk_size,
            }
        };

        current = smooth_downsample(current, new_md);
        levels.push(current.clone());

        if new_md.dimension_in_chunks().hmul() == 1 {
            break;
        }
    }

    LODTensorOperator { levels }
}

pub fn resample_transform<'op, D: LargerDim>(
    input: TensorOperator<D, f32>,
    output_size: TensorMetaData<D>,
    element_out_to_in: Matrix<D::Larger, f32>,
) -> TensorOperator<D, f32> {
    #[derive(Clone, bytemuck::Zeroable)]
    #[repr(C)]
    struct PushConstants<D: LargerDim> {
        transform: Matrix<D::Larger, f32>,
        chunk_dim_in: Vector<D, u32>,
        vol_dim_in: Vector<D, u32>,
        out_begin: Vector<D, u32>,
        mem_size_out: Vector<D, u32>,
    }
    impl<D: LargerDim> Copy for PushConstants<D> where Vector<D, u32>: Copy {}
    //TODO: This is fine for the current layout, but we really want a better general approach
    unsafe impl<D: LargerDim> bytemuck::Pod for PushConstants<D> where PushConstants<D>: Copy {}

    const SHADER: &'static str = r#"
#version 450

#extension GL_EXT_scalar_block_layout : require

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include <util.glsl>
#include <mat.glsl>
#include <vec.glsl>
#include <util.glsl>

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
    BrickType values[NUM_CHUNKS];
} bricks;

layout(std430, binding = 1) buffer OutputBuffer{
    float values[];
} outputData;

layout(scalar, push_constant) uniform PushConsts {
    Mat(N+1) transform;
    uint[N] chunk_dim_in;
    uint[N] vol_dim_in;
    uint[N] out_begin;
    uint[N] mem_size_out;
} consts;

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

        float default_val = 0.5;

        TensorMetaData(N) m_in;
        m_in.dimensions = consts.vol_dim_in;
        m_in.chunk_size = consts.chunk_dim_in;

        int res;
        uint sample_brick_pos_linear;
        float sampled_intensity;
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
        OperatorId::new("resample")
            .dependent_on(&input)
            .dependent_on(&output_size)
            .dependent_on(&element_out_to_in),
        output_size,
        (input, output_size, element_out_to_in),
        move |ctx, positions, (input, output_size, element_out_to_in)| {
            async move {
                let device = ctx.vulkan_device();

                let dst_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                };

                let m_in = input.metadata;
                let m_out = output_size;

                let num_chunks = m_in.dimension_in_chunks().hmul();

                let pipeline = device.request_state(
                    RessourceId::new("pipeline")
                        .of(ctx.current_op())
                        .dependent_on(&D::N),
                    || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .add("NUM_CHUNKS", num_chunks)
                                    .add("BRICK_MEM_SIZE_IN", m_in.chunk_size.hmul())
                                    .add("N", D::N),
                            ),
                            true,
                        )
                    },
                );

                let requests = positions.into_iter().map(|pos| {
                    let out_info = m_out.chunk_info(pos);
                    let out_begin = out_info.begin();
                    let out_end = out_info.end();

                    let aabb = AABB::new(
                        out_begin.map(|v| v.raw as f32),
                        out_end.map(|v| (v.raw - 1) as f32),
                    );
                    let aabb = aabb.transform(&element_out_to_in);

                    let out_begin = aabb.lower().map(|v| v.floor().max(0.0) as u32).global();
                    let out_end = aabb.upper().map(|v| v.ceil() as u32).global();

                    let in_begin_brick = m_in.chunk_pos(out_begin);
                    let in_end_brick = m_in
                        .chunk_pos(out_end)
                        .zip(m_in.dimension_in_chunks(), |l, r| l.min(r - 1u32));

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

                let chunk_index = device
                    .storage
                    .get_index(*ctx, device, input.chunks.id(), num_chunks, dst_info)
                    .await;

                while let Some((intersecting_bricks, (pos, in_brick_positions))) =
                    stream.next().await
                {
                    let out_info = m_out.chunk_info(pos);

                    let gpu_brick_out = ctx
                        .alloc_slot_gpu(device, pos, out_info.mem_elements())
                        .unwrap();

                    for (gpu_brick_in, in_brick_pos) in intersecting_bricks
                        .into_iter()
                        .zip(in_brick_positions.into_iter())
                    {
                        let brick_pos_linear =
                            crate::data::to_linear(in_brick_pos, m_in.dimension_in_chunks());
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
                    for i in 0..D::N {
                        let oi = i.saturating_sub(D::N.saturating_sub(D3::N));
                        size[oi] *= full_size[i];
                    }

                    device.with_cmd_buffer(|cmd| unsafe {
                        let mut pipeline = pipeline.bind(cmd);

                        pipeline.push_constant_pod(PushConstants {
                            transform: (*element_out_to_in),
                            chunk_dim_in: m_in.chunk_size.raw(),
                            vol_dim_in: m_in.dimensions.raw(),

                            mem_size_out: m_out.chunk_size.raw(),
                            out_begin: out_info.begin().raw(),
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
