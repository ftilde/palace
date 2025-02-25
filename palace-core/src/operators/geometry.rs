use ash::vk;

use crate::{
    dim::{DynDimension, D1},
    dtypes::{DType, ScalarType},
    mat::Matrix,
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    vulkan::{
        pipeline::{ComputePipelineBuilder, DescriptorConfig, DynPushConstants},
        shader::Shader,
        DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::tensor::TensorOperator;

pub fn transform<D: DynDimension>(
    points: TensorOperator<D1, DType>,
    matrix: Matrix<D, f32>,
) -> TensorOperator<D1, DType> {
    let dtype = points.dtype();

    assert_eq!(dtype.scalar, ScalarType::F32);
    assert_eq!(dtype.size as usize, matrix.dim().n() - 1);

    TensorOperator::with_state(
        op_descriptor!(),
        dtype,
        points.metadata.clone(),
        (points, DataParam(matrix)),
        |ctx, positions, loc, (points, matrix)| {
            async move {
                let device = ctx.preferred_device(loc);

                let md = &points.metadata;
                let dtype = points.dtype();
                let nd = dtype.size;

                let push_constants = DynPushConstants::new().mat::<f32>(matrix.dim().n(), "matrix");

                let pipeline = device.request_state(
                    (&push_constants, md.chunk_size.hmul(), nd),
                    |device, (push_constants, mem_size, nd)| {
                        ComputePipelineBuilder::new(
                            Shader::new(include_str!("transform_points.glsl"))
                                .push_const_block_dyn(push_constants)
                                .define("MEM_SIZE", mem_size)
                                .define("ND", nd),
                        )
                        .build(device)
                    },
                )?;

                let push_constants = &push_constants;

                let _ = ctx
                    .run_unordered(positions.into_iter().map(move |pos| {
                        async move {
                            let read_info = DstBarrierInfo {
                                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                access: vk::AccessFlags2::SHADER_READ,
                            };
                            let points_chunk = ctx
                                .submit(points.chunks.request_gpu(device.id, pos, read_info))
                                .await;

                            let out_chunk = ctx
                                .submit(ctx.alloc_slot_gpu(&device, pos, &md.chunk_size))
                                .await;

                            let descriptor_config =
                                DescriptorConfig::new([&points_chunk, &out_chunk]);

                            let global_size = md.chunk_size.raw();

                            device.with_cmd_buffer(|cmd| unsafe {
                                let mut pipeline = pipeline.bind(cmd);

                                pipeline.push_constant_dyn(&push_constants, |consts| {
                                    consts.mat(&matrix)?;
                                    Ok(())
                                });
                                pipeline.write_descriptor_set(0, descriptor_config);
                                pipeline.dispatch_dyn(device, global_size);
                            });

                            unsafe {
                                out_chunk.initialized(
                                    *ctx,
                                    SrcBarrierInfo {
                                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                        access: vk::AccessFlags2::SHADER_WRITE,
                                    },
                                )
                            };
                        }
                        .into()
                    }))
                    .await;

                Ok(())
            }
            .into()
        },
    )
}
