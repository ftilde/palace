use ash::vk;
use crevice::{glsl::GlslStruct, std140::AsStd140};
use futures::StreamExt;

use crate::{
    array::TensorMetaData,
    data::{hmul, Matrix, Vector, AABB},
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

//TODO: generalize to arbitrary N.
//This is difficult atm because (at least) we cannot specify the matrix size as N+1
type D = D3;

pub fn resample_rescale_mat<'op>(
    input_size: TensorMetaData<D>,
    output_size: TensorMetaData<D>,
) -> Matrix<<D3 as LargerDim>::Larger, f32> {
    let to_input_center = Matrix::from_translation(Vector::<D, f32>::fill(-0.5));
    let rescale =
        Matrix::from_scale(input_size.dimensions.raw().f32() / output_size.dimensions.raw().f32())
            .to_homogeneuous();
    let to_output_center = Matrix::from_translation(Vector::<D, f32>::fill(0.5));
    let out = to_input_center * rescale * to_output_center;

    out
}

pub fn resample<'op>(
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

pub fn smooth_downsample<'op>(
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

    let kernels = std::array::from_fn(|i| crate::operators::kernels::gauss(scale[i]));
    let smoothed =
        input.map_inner(|v| crate::operators::volume_gpu::separable_convolution(v, kernels.into()));
    resample(smoothed, output_size)
}

pub fn create_lod<'op>(
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

        if hmul(new_md.dimension_in_chunks()) == 1 {
            break;
        }
    }

    LODTensorOperator { levels }
}

pub fn resample_transform<'op>(
    input: TensorOperator<D, f32>,
    output_size: TensorMetaData<D>,
    element_out_to_in: Matrix<<D3 as LargerDim>::Larger, f32>,
) -> TensorOperator<D, f32> {
    #[derive(Copy, Clone, AsStd140, GlslStruct)]
    struct PushConstants {
        transform: cgmath::Matrix4<f32>,
        chunk_dim_in: cgmath::Vector3<u32>,
        vol_dim_in: cgmath::Vector3<u32>,
        out_begin: cgmath::Vector3<u32>,
        mem_size_out: cgmath::Vector3<u32>,
    }
    const SHADER: &'static str = r#"
#version 450

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include <util.glsl>
#include <mat.glsl>
#include <util.glsl>

#define BRICK_MEM_SIZE BRICK_MEM_SIZE_IN
#include <sample.glsl>
#undef BRICK_MEM_SIZE

layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(std430, binding = 0) buffer RefBuffer {
    BrickType values[NUM_CHUNKS];
} bricks;

layout(std430, binding = 1) buffer OutputBuffer{
    float values[];
} outputData;

declare_push_consts(consts);

void main() {
    uvec3 out_brick_pos = gl_GlobalInvocationID.xyz;

    if(out_brick_pos.x < consts.mem_size_out.x &&
       out_brick_pos.y < consts.mem_size_out.y &&
       out_brick_pos.z < consts.mem_size_out.z) {
        uvec3 global_pos = out_brick_pos + consts.out_begin;
        vec3 sample_posf = (consts.transform * vec4(vec3(global_pos), 1.0)).xyz;
        ivec3 sample_pos = ivec3(round(sample_posf));
        //ivec3 sample_pos = ivec3(floor(sample_posf) + vec3(0.5));

        ivec3 vol_dim = ivec3(consts.vol_dim_in);

        float default_val = 0.5;

        VolumeMetaData m_in;
        m_in.dimensions = consts.vol_dim_in;
        m_in.chunk_size = consts.chunk_dim_in;

        int res;
        uint sample_brick_pos_linear;
        float sampled_intensity;
        try_sample(sample_pos, m_in, bricks.values, res, sample_brick_pos_linear, sampled_intensity);

        if(res == SAMPLE_RES_FOUND) {
            // Nothing to do!
        } else if(res == SAMPLE_RES_NOT_PRESENT) {
            // This SHOULD not happen...
            sampled_intensity = default_val;
        } else /* SAMPLE_RES_OUTSIDE */ {
            sampled_intensity = default_val;
        }

        uint gID = to_linear3(out_brick_pos, consts.mem_size_out);
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

                let num_chunks = hmul(m_in.dimension_in_chunks());

                let pipeline =
                    device.request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new()
                                    .push_const_block::<PushConstants>()
                                    .add("NUM_CHUNKS", num_chunks)
                                    .add("BRICK_MEM_SIZE_IN", hmul(m_in.chunk_size)),
                            ),
                            true,
                        )
                    });

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

                    let in_brick_positions = itertools::iproduct! {
                        in_begin_brick.z().raw..=in_end_brick.z().raw,
                        in_begin_brick.y().raw..=in_end_brick.y().raw,
                        in_begin_brick.x().raw..=in_end_brick.x().raw
                    }
                    .map(|(z, y, x)| Vector::from([z, y, x]))
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

                    device.with_cmd_buffer(|cmd| unsafe {
                        let mut pipeline = pipeline.bind(cmd);

                        pipeline.push_constant(PushConstants {
                            transform: (*element_out_to_in).into(),
                            chunk_dim_in: m_in.chunk_size.raw().into(),
                            vol_dim_in: m_in.dimensions.raw().into(),

                            mem_size_out: m_out.chunk_size.raw().into(),
                            out_begin: out_info.begin().raw().into(),
                        });
                        pipeline.push_descriptor_set(0, descriptor_config);
                        pipeline.dispatch3d(m_out.chunk_size.raw());
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
