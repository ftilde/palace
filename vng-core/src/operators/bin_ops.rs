use ash::vk;
use futures::StreamExt;

use crate::{
    data::hmul,
    id::Id,
    operator::OperatorId,
    vulkan::{
        pipeline::{ComputePipeline, DescriptorConfig},
        shader::ShaderDefines,
        state::RessourceId,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::tensor::TensorOperator;

fn bin_op<const N: usize>(
    input1: TensorOperator<N>,
    input2: TensorOperator<N>,
    body: &'static str,
) -> TensorOperator<N> {
    let shader = format!(
        "{}{}{}",
        r#"
#version 450

#include <util.glsl>

layout (local_size_x = 256) in;

// Note: We cannot use `restrict` here and below since we bind the same buffer to sourceData and
// outputData in the inplace update case.
layout(std430, binding = 0) readonly buffer InputBuffer1{
    float values[BRICK_MEM_SIZE];
} sourceData1;

layout(std430, binding = 1) readonly buffer InputBuffer2{
    float values[BRICK_MEM_SIZE];
} sourceData2;

layout(std430, binding = 2) buffer OutputBuffer{
    float values[BRICK_MEM_SIZE];
} outputData;

void main()
{
    uint gID = gl_GlobalInvocationID.x;

    if(gID < BRICK_MEM_SIZE) {
        float v1 = sourceData1.values[gID];
        float v2 = sourceData2.values[gID];
        float res;
        "#,
        body,
        r#"
        outputData.values[gID] = res;
    }
}
"#
    );

    TensorOperator::with_state(
        OperatorId::new("bin_op")
            .dependent_on(&input1)
            .dependent_on(&input2)
            .dependent_on(Id::hash(body)),
        (input1.clone(), input2.clone()),
        (input1.clone(), input2.clone(), shader),
        move |ctx, (input1, input2)| {
            async move {
                let (m1, m2) = futures::join! {
                    ctx.submit(input1.metadata.request_scalar()),
                    ctx.submit(input2.metadata.request_scalar()),
                };
                assert_eq!(m1, m2);
                ctx.write(m1)
            }
            .into()
        },
        move |ctx, positions, (input1, input2, shader)| {
            async move {
                let device = ctx.vulkan_device();

                let access_info = DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                };
                let m = ctx.submit(input1.metadata.request_scalar()).await;

                let pipeline =
                    device.request_state(RessourceId::new("pipeline").of(ctx.current_op()), || {
                        ComputePipeline::new(
                            device,
                            (
                                shader.as_str(),
                                ShaderDefines::new().add("BRICK_MEM_SIZE", hmul(m.chunk_size)),
                            ),
                            true,
                        )
                    });

                let mut brick_stream =
                    ctx.submit_unordered_with_data(positions.iter().map(|pos| {
                        (
                            ctx.group([
                                input1.bricks.request_gpu(device.id, *pos, access_info),
                                input2.bricks.request_gpu(device.id, *pos, access_info),
                            ]),
                            *pos,
                        )
                    }));

                while let Some((mut in_bricks, pos)) = brick_stream.next().await {
                    let brick_info = m.chunk_info(pos);
                    let mut in_bricks = in_bricks.drain(..);
                    let gpu_brick_in1 = in_bricks.next().unwrap();
                    let gpu_brick_in2 = in_bricks.next().unwrap();

                    let gpu_brick_out =
                        ctx.alloc_slot_gpu(device, pos, brick_info.mem_elements())?;

                    device.with_cmd_buffer(|cmd| {
                        let descriptor_config =
                            DescriptorConfig::new([&gpu_brick_in1, &gpu_brick_in2, &gpu_brick_out]);

                        let global_size = brick_info.mem_elements();

                        unsafe {
                            let mut pipeline = pipeline.bind(cmd);

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

pub fn add<const N: usize>(
    input1: TensorOperator<N>,
    input2: TensorOperator<N>,
) -> TensorOperator<N> {
    bin_op(input1, input2, "res = v1 + v2;")
}

pub fn sub<const N: usize>(
    input1: TensorOperator<N>,
    input2: TensorOperator<N>,
) -> TensorOperator<N> {
    bin_op(input1, input2, "res = v1 - v2;")
}

pub fn mul<const N: usize>(
    input1: TensorOperator<N>,
    input2: TensorOperator<N>,
) -> TensorOperator<N> {
    bin_op(input1, input2, "res = v1 * v2;")
}

pub fn min<const N: usize>(
    input1: TensorOperator<N>,
    input2: TensorOperator<N>,
) -> TensorOperator<N> {
    bin_op(input1, input2, "res = min(v1, v2);")
}

pub fn max<const N: usize>(
    input1: TensorOperator<N>,
    input2: TensorOperator<N>,
) -> TensorOperator<N> {
    bin_op(input1, input2, "res = max(v1, v2);")
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{operators::array::from_static, test_util::compare_tensor};

    #[test]
    fn test_add_gpu() {
        let input1 = &[1.0, 2.0, 3.0, 4.0];
        let input2 = &[0.0, 0.0, 7.0, -4.0];
        let expected = &[1.0, 2.0, 10.0, 0.0];

        let output = add(from_static(input1), from_static(input2));

        compare_tensor(output, from_static(expected));
    }

    #[test]
    fn test_sub_gpu() {
        let input1 = &[1.0, 2.0, 3.0, 4.0];
        let input2 = &[0.0, 0.0, 7.0, -4.0];
        let expected = &[1.0, 2.0, -4.0, 8.0];

        let output = sub(from_static(input1), from_static(input2));

        compare_tensor(output, from_static(expected));
    }

    #[test]
    fn test_mul_gpu() {
        let input1 = &[1.0, 2.0, 3.0, 4.0];
        let input2 = &[0.0, 0.0, 7.0, -4.0];
        let expected = &[0.0, 0.0, 21.0, -16.0];

        let output = mul(from_static(input1), from_static(input2));

        compare_tensor(output, from_static(expected));
    }

    #[test]
    fn test_min_gpu() {
        let input1 = &[1.0, 2.0, 3.0, 4.0];
        let input2 = &[0.0, 0.0, 7.0, -4.0];
        let expected = &[0.0, 0.0, 3.0, -4.0];

        let output = min(from_static(input1), from_static(input2));

        compare_tensor(output, from_static(expected));
    }

    #[test]
    fn test_max_gpu() {
        let input1 = &[1.0, 2.0, 3.0, 4.0];
        let input2 = &[0.0, 0.0, 7.0, -4.0];
        let expected = &[1.0, 2.0, 7.0, 4.0];

        let output = max(from_static(input1), from_static(input2));

        compare_tensor(output, from_static(expected));
    }
}
