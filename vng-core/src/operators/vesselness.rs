use ash::vk;
use futures::StreamExt;

use crate::dim::*;
use crate::operator::OperatorId;
use crate::operators::array::ArrayOperator;
use crate::vec::Vector;
use crate::vulkan::pipeline::{ComputePipeline, DescriptorConfig};
use crate::vulkan::shader::ShaderDefines;
use crate::vulkan::state::RessourceId;
use crate::vulkan::{DstBarrierInfo, SrcBarrierInfo};

use super::{kernels::*, volume_gpu};
use super::{tensor::TensorOperator, volume::VolumeOperator};

pub fn multiscale_vesselness(
    //TODO: Make this embedding-aware
    input: VolumeOperator<f32>,
    min_scale: f32,
    max_scale: f32,
    num_steps: usize,
) -> VolumeOperator<f32> {
    assert!(num_steps > 0);

    let mut out = vesselness(input.clone(), min_scale);

    let num_reductions = num_steps - 1;
    for step in 1..num_steps {
        let alpha = step as f32 / num_reductions as f32;

        let step_scale = {
            let min_log = min_scale.ln();
            let max_log = max_scale.ln();

            let inter_log = min_log * (1.0 - alpha) + max_log * alpha;

            let scale = inter_log.exp();
            scale
        };

        let vesselness = vesselness(input.clone(), step_scale.clone());
        let step_vesselness =
            crate::operators::volume_gpu::linear_rescale(vesselness, step_scale * step_scale, 0.0);
        out = crate::operators::bin_ops::max(out, step_vesselness);
    }

    out
}

pub fn vesselness(input: VolumeOperator<f32>, scale: f32) -> VolumeOperator<f32> {
    const SHADER: &'static str = r#"
#version 450

#include <eigenvalues.glsl>
#include <vesselness.glsl>

layout (local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer InputBufferXX{
    float values[BRICK_MEM_SIZE];
} b_xx;

layout(std430, binding = 1) readonly buffer InputBufferXY{
    float values[BRICK_MEM_SIZE];
} b_xy;

layout(std430, binding = 2) readonly buffer InputBufferXZ{
    float values[BRICK_MEM_SIZE];
} b_xz;

layout(std430, binding = 3) readonly buffer InputBufferYY{
    float values[BRICK_MEM_SIZE];
} b_yy;

layout(std430, binding = 4) readonly buffer InputBufferYZ{
    float values[BRICK_MEM_SIZE];
} b_yz;

layout(std430, binding = 5) readonly buffer InputBufferZZ{
    float values[BRICK_MEM_SIZE];
} b_zz;

layout(std430, binding = 6) buffer OutputBuffer{
    float values[BRICK_MEM_SIZE];
} outputData;

//declare_push_consts(consts);

void main() {
    uint gID = gl_GlobalInvocationID.x;

    if(gID < BRICK_MEM_SIZE) {

        SymMat3 m;
        m.xx = b_xx.values[gID];
        m.xy = b_xy.values[gID];
        m.xz = b_xz.values[gID];
        m.yy = b_yy.values[gID];
        m.yz = b_yz.values[gID];
        m.zz = b_zz.values[gID];

        float l1, l2, l3;
        eigenvalues(m, l1, l2, l3);

        float vesselness = sato_vesselness(l1, l2, l3, 0.5 /* = 2*0.5*0.5 */, 8.0 /* = 2*2*2 */);

        outputData.values[gID] = vesselness;
    }
}
"#;

    type Conv = fn(f32) -> ArrayOperator<f32>;

    let g = |f1: Conv, f2: Conv, f3: Conv| {
        let kernels = [f1(scale.clone()), f2(scale.clone()), f3(scale.clone())];
        let kernel_refs = Vector::<D3, _>::from_fn(|i| &kernels[i]);
        volume_gpu::separable_convolution(input.clone(), kernel_refs)
    };

    let xx = g(gauss, gauss, ddgauss_dxdx);
    let xy = g(gauss, dgauss_dx, dgauss_dx);
    let xz = g(dgauss_dx, gauss, dgauss_dx);
    let yy = g(gauss, ddgauss_dxdx, gauss);
    let yz = g(dgauss_dx, dgauss_dx, gauss);
    let zz = g(ddgauss_dxdx, gauss, gauss);

    TensorOperator::with_state(
        OperatorId::new("vesselness")
            .dependent_on(&input)
            .dependent_on(&scale),
        input.metadata,
        (input, [xx, xy, xz, yy, yz, zz]),
        move |ctx, positions, (input, hessian)| {
            async move {
                let device = ctx.vulkan_device();

                let m = input.metadata;

                let requests = positions.into_iter().map(|pos| {
                    let hessian_bricks = ctx.group(hessian.into_iter().map(|entry| {
                        entry.chunks.request_gpu(
                            device.id,
                            pos,
                            DstBarrierInfo {
                                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                access: vk::AccessFlags2::SHADER_READ,
                            },
                        )
                    }));

                    (hessian_bricks, pos)
                });

                let pipeline =
                    device.request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(
                            device,
                            (
                                SHADER,
                                ShaderDefines::new().add("BRICK_MEM_SIZE", m.chunk_size.hmul()), //.push_const_block::<PushConstants>(),
                            ),
                            true,
                        )
                    });

                let mut stream = ctx.submit_unordered_with_data(requests);
                while let Some((hessian_bricks, pos)) = stream.next().await {
                    let out_info = m.chunk_info(pos);

                    let global_size = out_info.mem_dimensions.hmul();

                    let gpu_brick_out = ctx
                        .alloc_slot_gpu(device, pos, out_info.mem_elements())
                        .unwrap();

                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config = DescriptorConfig::new([
                            &hessian_bricks[0],
                            &hessian_bricks[1],
                            &hessian_bricks[2],
                            &hessian_bricks[3],
                            &hessian_bricks[4],
                            &hessian_bricks[5],
                            &gpu_brick_out,
                        ]);

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

                            //pipeline.push_constant(consts);
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
