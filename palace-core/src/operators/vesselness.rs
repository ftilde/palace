use ash::vk;
use futures::StreamExt;

use crate::dtypes::StaticElementType;
use crate::jit::jit;
use crate::operator::OperatorDescriptor;
use crate::operators::array::ArrayOperator;
use crate::vec::Vector;
use crate::vulkan::pipeline::{ComputePipelineBuilder, DescriptorConfig};
use crate::vulkan::shader::Shader;
use crate::vulkan::{DstBarrierInfo, SrcBarrierInfo};
use crate::{dim::*, op_descriptor};

use super::tensor::TensorOperator;
use super::volume::EmbeddedVolumeOperator;
use super::{kernels::*, volume_gpu};

pub fn multiscale_vesselness(
    input: EmbeddedVolumeOperator<StaticElementType<f32>>,
    min_scale: f32,
    max_scale: f32,
    num_steps: usize,
) -> EmbeddedVolumeOperator<StaticElementType<f32>> {
    assert!(num_steps > 0);

    let step_scale = min_scale;
    let out = vesselness(input.clone(), step_scale);
    let spacing_diag = input.embedding_data.spacing.length();
    let c = 1.0 / (spacing_diag * spacing_diag);

    let mut out = jit(out.inner.into())
        .mul((c * step_scale * step_scale).into())
        .unwrap();

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
        let step_vesselness = jit(vesselness.inner.into())
            .mul((c * step_scale * step_scale).into())
            .unwrap();

        out = out.max(step_vesselness).unwrap();
    }

    out.compile()
        .unwrap()
        .embedded(input.embedding_data)
        .try_into()
        .unwrap()
}

pub fn vesselness(
    input: EmbeddedVolumeOperator<StaticElementType<f32>>,
    scale: f32,
) -> EmbeddedVolumeOperator<StaticElementType<f32>> {
    const SHADER: &'static str = r#"
#include <eigenvalues.glsl>
#include <vesselness.glsl>
#include <size_util.glsl>

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
    uint gID = global_position_linear;

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

    type Conv = fn(f32) -> ArrayOperator<StaticElementType<f32>>;

    let spacing = input.embedding_data.spacing;
    let g = |f1: Conv, f2: Conv, f3: Conv| {
        let kernels = [
            f1(scale / spacing[0]),
            f2(scale / spacing[1]),
            f3(scale / spacing[2]),
        ];
        let kernel_refs = Vector::<D3, _>::from_fn(|i| &kernels[i]);
        volume_gpu::separable_convolution(input.inner.clone(), kernel_refs)
    };

    let xx = g(gauss, gauss, ddgauss_dxdx);
    let xy = g(gauss, dgauss_dx, dgauss_dx);
    let xz = g(dgauss_dx, gauss, dgauss_dx);
    let yy = g(gauss, ddgauss_dxdx, gauss);
    let yz = g(dgauss_dx, dgauss_dx, gauss);
    let zz = g(ddgauss_dxdx, gauss, gauss);

    let embedding_data = input.embedding_data;
    TensorOperator::with_state(
        op_descriptor!(),
        Default::default(),
        input.metadata,
        (input, [xx, xy, xz, yy, yz, zz]),
        move |ctx, positions, (input, hessian)| {
            async move {
                let device = ctx.preferred_device();

                let m = input.metadata;

                let requests = positions.into_iter().map(|(pos, _)| {
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

                let pipeline = device.request_state(m.chunk_size.hmul(), |device, mem_size| {
                    ComputePipelineBuilder::new(
                        Shader::new(SHADER).define("BRICK_MEM_SIZE", mem_size),
                    )
                    .use_push_descriptor(true)
                    .build(device)
                })?;

                let mut stream = ctx.submit_unordered_with_data(requests);
                while let Some((hessian_bricks, pos)) = stream.next().await {
                    let out_info = m.chunk_info(pos);

                    let global_size = out_info.mem_dimensions.hmul();

                    let gpu_brick_out = ctx
                        .submit(ctx.alloc_slot_gpu(device, pos, &out_info.mem_dimensions))
                        .await;

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
    .embedded(embedding_data)
}
