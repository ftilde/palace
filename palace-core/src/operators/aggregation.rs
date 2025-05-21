use std::hash::{DefaultHasher, Hash, Hasher};

use ash::vk;
use futures::StreamExt;
use id::Identify;
use rand::{seq::SliceRandom, SeedableRng};

use crate::{
    array::ChunkInfo,
    dim::DynDimension,
    dtypes::StaticElementType,
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    vulkan::{
        pipeline::{
            AsBufferDescriptor, ComputePipelineBuilder, DescriptorConfig, DynPushConstants,
            LocalSizeConfig,
        },
        shader::Shader,
        DeviceContext, DstBarrierInfo, SrcBarrierInfo,
    },
};

use super::{scalar::ScalarOperator, tensor::TensorOperator};

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
    pub fn neutral_val(&self) -> f32 {
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

pub async unsafe fn aggregation_on_chunk<D: DynDimension>(
    buf: &impl AsBufferDescriptor,
    result: &impl AsBufferDescriptor,
    device: &DeviceContext,
    chunk_info: ChunkInfo<D>,
    method: AggretationMethod,
    normalization_factor: f32,
) -> Result<(), crate::Error> {
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

    if(all(less_than(local, consts.logical_dim)) && gID < BRICK_MEM_SIZE) {
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
    let nd = chunk_info.mem_dimensions.len();
    let push_constants = DynPushConstants::new()
        .vec::<u32>(nd, "mem_dim")
        .vec::<u32>(nd, "logical_dim")
        .scalar::<f32>("norm_factor");

    let pipeline = device.request_state(
        (
            &push_constants,
            chunk_info.mem_dimensions.hmul(),
            method,
            nd,
        ),
        |device, (push_constants, mem_size, method, nd)| {
            let neutral_val_str = format!("uintBitsToFloat({})", method.neutral_val().to_bits());
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

    device.with_cmd_buffer(|cmd| {
        let descriptor_config = DescriptorConfig::new([buf, result]);

        let global_size = chunk_info.mem_elements();

        unsafe {
            let mut pipeline = pipeline.bind(cmd);

            pipeline.push_constant_dyn(&push_constants, |consts| {
                consts.vec(&chunk_info.mem_dimensions.into_elem::<u32>())?;
                consts.vec(&chunk_info.logical_dimensions.into_elem::<u32>())?;
                consts.scalar(normalization_factor)?;
                Ok(())
            });
            pipeline.push_descriptor_set(0, descriptor_config);
            pipeline.dispatch(device, global_size);
        }
    });

    Ok(())
}

fn scalar_aggregation<'op, D: DynDimension>(
    input: TensorOperator<D, StaticElementType<f32>>,
    method: AggretationMethod,
    sample_method: SampleMethod,
) -> ScalarOperator<StaticElementType<f32>> {
    crate::operators::scalar::scalar(
        op_descriptor!(),
        (input, DataParam(method), DataParam(sample_method)),
        move |ctx, loc, (input, method, sample_method)| {
            async move {
                let device = ctx.preferred_device(loc);

                let m = &input.metadata;

                let mut all_chunks = m.chunk_indices().into_iter().collect::<Vec<_>>();
                let (to_request, num_elements) = match **sample_method {
                    SampleMethod::All => (all_chunks.as_slice(), m.dimensions.hmul()),
                    SampleMethod::Subset(n) => {
                        let mut h = DefaultHasher::new();
                        ctx.current_op().inner().hash(&mut h);
                        let seed = h.finish();
                        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
                        let (ret, _) = all_chunks.partial_shuffle(&mut rng, n);
                        ret.sort();

                        let num_elements = ret
                            .iter()
                            .map(|pos| m.chunk_info(*pos).logical_dimensions.hmul())
                            .sum();
                        (&*ret, num_elements)
                    }
                };

                let sum = ctx.submit(ctx.alloc_scalar_gpu(device)).await;

                let normalization_factor = method.norm_factor(num_elements);

                device.with_cmd_buffer(|cmd| unsafe {
                    cmd.functions().cmd_fill_buffer(
                        cmd.raw(),
                        sum.buffer,
                        0,
                        std::mem::size_of::<f32>() as _,
                        method.neutral_val().to_bits(),
                    );
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

                    unsafe {
                        aggregation_on_chunk(
                            &gpu_brick_in,
                            &sum,
                            device,
                            brick_info,
                            **method,
                            normalization_factor,
                        )
                        .await?;
                    }
                }
                //TODO: why this?
                ctx.submit(device.wait_for_current_cmd_buffer_completion())
                    .await;

                //TODO: maybe this is the source of the write after write hazards? do we need
                //transfer write here, too?
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
    use crate::data::{LocalVoxelPosition, VoxelPosition};

    #[test]
    fn test_mean_gpu() {
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let input = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            let v = crate::vec::to_linear(&v, &size);
            v as f32
        });

        let output = mean(input, SampleMethod::All);

        let mut runtime = crate::runtime::RunTime::build()
            .finish(1 << 30, 1 << 30)
            .unwrap();

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
}
