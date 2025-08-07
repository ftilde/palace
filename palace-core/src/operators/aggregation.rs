use std::hash::{DefaultHasher, Hash, Hasher};

use ash::vk;
use futures::StreamExt;
use id::Identify;
use itertools::Itertools;
use rand::SeedableRng;

use crate::{
    array::{ChunkIndex, ChunkInfo, TensorMetaData},
    data::{ChunkCoordinate, LocalCoordinate},
    dim::DynDimension,
    dtypes::{DType, ElementType, StaticElementType},
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    vec::Vector,
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
    output_element_offset: u32,
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
    uint value[];
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
        AGG_FUNCTION(sum.value[consts.output_element_offset], uintBitsToFloat(shared_sum));
    }
}
"#;
    let nd = chunk_info.mem_dimensions.len();
    let push_constants = DynPushConstants::new()
        .vec::<u32>(nd, "mem_dim")
        .vec::<u32>(nd, "logical_dim")
        .scalar::<u32>("output_element_offset")
        .scalar::<f32>("norm_factor");

    let pipeline = device
        .request_state(
            (
                &push_constants,
                chunk_info.mem_dimensions.hmul(),
                method,
                nd,
            ),
            |device, (push_constants, mem_size, method, nd)| {
                let neutral_val_str =
                    format!("uintBitsToFloat({})", method.neutral_val().to_bits());
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
        )
        .unwrap();

    device.with_cmd_buffer(|cmd| {
        let descriptor_config = DescriptorConfig::new([buf, result]);

        let global_size = chunk_info.mem_elements();

        unsafe {
            let mut pipeline = pipeline.bind(cmd);

            pipeline.push_constant_dyn(&push_constants, |consts| {
                consts.vec(&chunk_info.mem_dimensions.into_elem::<u32>())?;
                consts.vec(&chunk_info.logical_dimensions.into_elem::<u32>())?;
                consts.scalar(output_element_offset)?;
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

                let num_chunks = m.num_chunks();
                let (to_request, num_elements) = match **sample_method {
                    SampleMethod::All => {
                        (m.chunk_indices().collect::<Vec<_>>(), m.dimensions.hmul())
                    }
                    SampleMethod::Subset(n) => {
                        let n = n.min(num_chunks);
                        let mut h = DefaultHasher::new();
                        ctx.current_op().inner().hash(&mut h);
                        let seed = h.finish();
                        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

                        let to_sample = rand::seq::index::sample(&mut rng, num_chunks, n);
                        let mut ret = (0..n)
                            .into_iter()
                            .map(|i| ChunkIndex(to_sample.index(i) as u64))
                            .collect::<Vec<_>>();
                        ret.sort();

                        let num_elements = ret
                            .iter()
                            .map(|pos| m.chunk_info(*pos).logical_dimensions.hmul())
                            .sum();
                        (ret, num_elements)
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
                            0,
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

pub fn chunk_aggregation<'op, D: DynDimension>(
    input: TensorOperator<D, StaticElementType<f32>>,
    out_chunk_size: Vector<D, LocalCoordinate>,
    method: AggretationMethod,
) -> TensorOperator<D, StaticElementType<f32>> {
    let md = TensorMetaData {
        dimensions: input.metadata.dimension_in_chunks().raw().global(),
        chunk_size: out_chunk_size,
    };
    crate::operators::tensor::TensorOperator::with_state(
        op_descriptor!(),
        Default::default(),
        md.clone(),
        (input, DataParam(md), DataParam(method)),
        move |ctx, positions, loc, (input, md_out, method)| {
            <crate::task::Task<'_>>::from(async move {
                let device = ctx.preferred_device(loc);

                let m_in = &input.metadata;
                let nd = input.metadata.dimensions.len();
                let dtype: DType = input.dtype().into();

                for pos in positions {
                    let out_chunk_md = md_out.chunk_info(pos);

                    let begin = out_chunk_md.begin.raw().chunk();
                    let end = out_chunk_md.end();

                    let in_chunk_positions = (0..nd)
                        .into_iter()
                        .map(|i| begin[i].raw..end[i].raw)
                        .multi_cartesian_product()
                        .map(|coordinates| {
                            m_in.chunk_index(
                                &Vector::<D, ChunkCoordinate>::try_from(coordinates).unwrap(),
                            )
                        })
                        .collect::<Vec<_>>();

                    let mut stream =
                        ctx.submit_unordered_with_data(in_chunk_positions.iter().map(|pos| {
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
                    let out_buf = ctx
                        .submit(ctx.alloc_slot_gpu(device, pos, &md_out.chunk_size))
                        .await;

                    let elm_size = dtype.element_layout().size() as u64;

                    device.with_cmd_buffer(|cmd| unsafe {
                        cmd.functions().cmd_fill_buffer(
                            cmd.raw(),
                            out_buf.buffer,
                            0,
                            md_out.num_chunk_elements() as u64 * elm_size,
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

                    while let Some((gpu_brick_in, in_pos)) = stream.next().await {
                        let brick_info = m_in.chunk_info(in_pos);

                        let normalization_factor =
                            method.norm_factor(brick_info.logical_dimensions.hmul());

                        let global_pos_out = m_in.chunk_pos_from_index(in_pos).raw().global();
                        let pos_in_chunk_out = &out_chunk_md.in_chunk(&global_pos_out);
                        let offset =
                            crate::vec::to_linear(pos_in_chunk_out, &out_chunk_md.mem_dimensions)
                                as u64;
                        unsafe {
                            super::aggregation::aggregation_on_chunk(
                                &gpu_brick_in,
                                &out_buf,
                                offset.try_into().unwrap(),
                                device,
                                brick_info.clone(),
                                **method,
                                normalization_factor,
                            )
                            .await?;
                        }
                    }
                    unsafe {
                        out_buf.initialized(
                            *ctx,
                            SrcBarrierInfo {
                                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                                access: vk::AccessFlags2::SHADER_WRITE,
                            },
                        )
                    };
                }

                Ok(())
            })
        },
    )
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        data::{LocalVoxelPosition, VoxelPosition},
        dim::D3,
        test_util::compare_tensor_fn,
    };

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

    #[test]
    fn test_chunk_agg() {
        let size = VoxelPosition::fill(7.into());
        let brick_size = LocalVoxelPosition::fill(2.into());
        let dim_in_chunks = TensorMetaData {
            dimensions: size,
            chunk_size: brick_size,
        }
        .dimension_in_chunks();

        let input = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
            let v = crate::vec::to_linear(&v, &size);
            v as f32
        });

        let output = chunk_aggregation(input, brick_size, AggretationMethod::Min);

        compare_tensor_fn(output, |comp| {
            for z in 0..dim_in_chunks.z().raw {
                for y in 0..dim_in_chunks.y().raw {
                    for x in 0..dim_in_chunks.x().raw {
                        let pos = Vector::<D3, u32>::new([x, y, z]);
                        let mut min = f32::MAX;
                        for dz in 0..brick_size.z().raw {
                            for dy in 0..brick_size.y().raw {
                                for dx in 0..brick_size.x().raw {
                                    let offset = Vector::<D3, u32>::new([dx, dy, dz]);
                                    let input_pos = pos * brick_size.raw() + offset;

                                    let v = crate::vec::to_linear(&input_pos.global(), &size);
                                    min = min.min(v as f32)
                                }
                            }
                        }
                        comp[pos.global().as_index()] = min;
                    }
                }
            }
        });
    }
}
