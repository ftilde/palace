use ash::vk;
use itertools::Itertools;

use crate::{
    array::{ChunkIndex, TensorMetaData},
    data::{ChunkCoordinate, GlobalCoordinate, LocalCoordinate},
    dim::{Dimension, DynDimension},
    dtypes::{DType, StaticElementType},
    operators::tensor::TensorOperator,
    storage::{
        gpu::{IndexHandle, StateCacheHandle, StateCacheResult},
        Element,
    },
    task::OpaqueTaskContext,
    vec::Vector,
    vulkan::{
        pipeline::{
            AsDescriptors, ComputePipeline, ComputePipelineBuilder, DescriptorConfig,
            DynPushConstants,
        },
        shader::Shader,
        state::VulkanState,
        CommandBuffer, DeviceContext, DstBarrierInfo,
    },
};

type RTElement = u32;

pub struct ChunkFeedbackTable<'a> {
    inner: StateCacheHandle<'a>,
    pub newly_initialized: bool,
}

impl<'a> ChunkFeedbackTable<'a> {
    // Note: a barrier is needed after initialization to make values visible
    pub fn new(device: &DeviceContext, inner: StateCacheResult<'a>) -> Self {
        let mut newly_initialized = false;
        let inner = inner.init(|v| {
            device.with_cmd_buffer(|cmd| unsafe {
                device.functions().cmd_fill_buffer(
                    cmd.raw(),
                    v.buffer,
                    0,
                    vk::WHOLE_SIZE,
                    RTElement::max_value(),
                );
            });
            newly_initialized = true;
        });
        Self {
            inner,
            newly_initialized,
        }
    }
    // Note: a barrier is needed after clearing to make values visible
    pub fn clear(&self, cmd: &mut CommandBuffer) {
        unsafe {
            cmd.functions().cmd_fill_buffer(
                cmd.raw(),
                self.inner.buffer,
                0,
                vk::WHOLE_SIZE,
                RTElement::max_value(),
            )
        };
    }

    fn num_elements(&self) -> usize {
        crate::util::num_elms_in_array::<RTElement>(self.inner.size as usize)
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.inner.buffer
    }

    pub fn inner(&self) -> &StateCacheHandle<'a> {
        &self.inner
    }

    // Note: any changes to the buffer have to be made visible to the cpu side via a barrier first
    pub async fn download_inserted<'cref, 'inv>(
        &self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        device: &'cref DeviceContext,
    ) -> Vec<RTElement> {
        let num_elements = self.num_elements();
        let layout = std::alloc::Layout::array::<RTElement>(num_elements).unwrap();
        let mut request_table_cpu = vec![0u32; num_elements];
        let request_table_cpu_bytes: &mut [u8] =
            bytemuck::cast_slice_mut(request_table_cpu.as_mut_slice());
        unsafe {
            crate::vulkan::memory::copy_to_cpu(
                ctx,
                device,
                self.inner.buffer,
                layout,
                request_table_cpu_bytes.as_mut_ptr().cast(),
            )
            .await
        };

        let to_request_linear = request_table_cpu
            .into_iter()
            .filter(|v| *v != RTElement::max_value())
            .collect::<Vec<RTElement>>();

        to_request_linear
    }
}

pub struct Timeout;

pub async fn request_to_index_with_timeout<'cref, 'inv, D: Dimension, E: Element>(
    ctx: &OpaqueTaskContext<'cref, 'inv>,
    device: &DeviceContext,
    to_request_linear: &mut [RTElement],
    vol: &'inv TensorOperator<D, StaticElementType<E>>,
    index: &IndexHandle<'_>,
    batch_size: &mut usize,
    interactive: bool,
) -> Result<(), Timeout> {
    let dim_in_bricks = vol.metadata.dimension_in_chunks();
    let num_bricks = dim_in_bricks.hmul();

    // Sort to get at least some benefit from spatial neighborhood
    to_request_linear.sort_unstable();

    let max_batch_size = to_request_linear.len().max(*batch_size);

    // Fulfill requests
    let mut to_request_linear = &to_request_linear[..];
    while !to_request_linear.is_empty() {
        let batch;
        (batch, to_request_linear) =
            to_request_linear.split_at((*batch_size).min(to_request_linear.len()));

        let to_request = batch.iter().map(|v| {
            assert!(*v < num_bricks as _);
            vol.chunks.request_gpu(
                device.id,
                ChunkIndex((*v).into()),
                DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                },
            )
        });
        let requested_bricks = ctx.submit(ctx.group(to_request)).await;

        for (brick, brick_linear_pos) in requested_bricks.into_iter().zip(batch.into_iter()) {
            index.insert(*brick_linear_pos as u64, brick);
        }

        if let Some(lateness) = ctx.past_deadline(interactive) {
            if lateness > 2.0 {
                *batch_size = (*batch_size / 2).max(1);
            }
            return Err(Timeout);
        }

        *batch_size = (*batch_size * 4).min(max_batch_size);
    }

    Ok(())
}

pub struct ChunkCopyPipeline<D: DynDimension> {
    pipeline: ComputePipeline,
    push_constants: DynPushConstants,
    source_size: Vector<D, LocalCoordinate>,
}

impl<D: DynDimension> VulkanState for ChunkCopyPipeline<D> {
    unsafe fn deinitialize(&mut self, context: &DeviceContext) {
        self.pipeline.deinitialize(context)
    }
}
impl<D: DynDimension> ChunkCopyPipeline<D> {
    pub fn new(
        device: &DeviceContext,
        dtype: DType,
        source_size: Vector<D, LocalCoordinate>,
    ) -> Result<Self, crate::Error> {
        const SHADER: &'static str = r#"
#include <util.glsl>
#include <vec.glsl>
#include <size_util.glsl>

layout(std430, binding = 0) readonly buffer InputBuffer{
    T values[BRICK_MEM_SIZE_IN];
} sourceData;

layout(std430, binding = 1) buffer OutputBuffer{
    T values[];
} outputData;

declare_push_consts(constants);

void main() {
    uint gID = global_position_linear;

    if(gID < constants.global_size) {
        uint[N] region_pos = from_linear(gID, constants.region_size);

        uint[N] in_pos = add(constants.begin_in, region_pos);
        uint[N] out_pos = add(constants.begin_out, region_pos);

        uint in_index = to_linear(in_pos, constants.mem_size_in);
        uint out_index = to_linear(out_pos, constants.mem_size_out);

        outputData.values[out_index] = sourceData.values[in_index];
    }
}
"#;
        let nd = source_size.len();
        let push_constants = DynPushConstants::new()
            .vec::<u32>(nd, "mem_size_in")
            .vec::<u32>(nd, "mem_size_out")
            .vec::<u32>(nd, "begin_in")
            .vec::<u32>(nd, "begin_out")
            .vec::<u32>(nd, "region_size")
            .scalar::<u32>("global_size");

        let pipeline = ComputePipelineBuilder::new(
            Shader::new(SHADER)
                .define("BRICK_MEM_SIZE_IN", source_size.hmul())
                .define("N", nd)
                .define("T", dtype.glsl_type())
                .push_const_block_dyn(&push_constants)
                .ext(dtype.glsl_ext())
                .ext(Some(crate::vulkan::shader::ext::SCALAR_BLOCK_LAYOUT)),
        )
        .use_push_descriptor(true)
        .build(device)?;

        Ok(Self {
            pipeline,
            push_constants,
            source_size,
        })
    }

    // Safety: gpu out and in buffers must be large enough to perform the copies as defined by
    // offsets and overlap_size
    pub unsafe fn run(
        &self,
        device: &DeviceContext,
        gpu_chunk_in: &dyn AsDescriptors,
        gpu_chunk_out: &dyn AsDescriptors,
        src_offset: &Vector<D, LocalCoordinate>,
        dst_offset: &Vector<D, LocalCoordinate>,
        dst_size: &Vector<D, LocalCoordinate>,
        overlap_size: &Vector<D, LocalCoordinate>,
    ) {
        let descriptor_config = DescriptorConfig::new([gpu_chunk_in, gpu_chunk_out]);
        device.with_cmd_buffer(|cmd| unsafe {
            let mut pipeline = self.pipeline.bind(cmd);

            let global_size = overlap_size.hmul();

            pipeline.push_constant_dyn(&self.push_constants, |consts| {
                consts.vec(&self.source_size.raw())?;
                consts.vec(&dst_size.raw())?;
                consts.vec(&src_offset.raw())?;
                consts.vec(&dst_offset.raw())?;
                consts.vec(&overlap_size.raw())?;
                consts.scalar(global_size as u32)?;
                Ok(())
            });

            pipeline.push_descriptor_set(0, descriptor_config);
            pipeline.dispatch(device, global_size);
        });
    }
}

pub struct ChunkNeighborhood<'a, D: DynDimension> {
    pub begin_chunk: Vector<D, ChunkCoordinate>,
    pub end_chunk: Vector<D, ChunkCoordinate>, //inclusive!
    md: &'a TensorMetaData<D>,
}

impl<'a, D: DynDimension> ChunkNeighborhood<'a, D> {
    pub fn around(
        md: &'a TensorMetaData<D>,
        pos: ChunkIndex,
        expand_minus: Vector<D, GlobalCoordinate>,
        expand_plus: Vector<D, GlobalCoordinate>,
    ) -> Self {
        let out_info = md.chunk_info(pos);
        let out_begin = out_info.begin();
        let out_last = out_info.end().map(|v| v - 1u32);

        let in_begin = out_begin
            .clone()
            .zip(&expand_minus, |l, r| l.raw.saturating_sub(r.raw).into());
        let in_end = out_last
            .clone()
            .zip(&expand_plus, |l, r| l.raw + r.raw)
            .zip(&md.dimensions, |l, r| {
                GlobalCoordinate::from(l.min(r.raw - 1))
            });

        let begin_chunk = md.chunk_pos(&in_begin);
        let end_chunk = md.chunk_pos(&in_end);

        Self {
            begin_chunk,
            end_chunk,
            md,
        }
    }

    pub fn dim_in_chunks(&self) -> Vector<D, ChunkCoordinate> {
        self.end_chunk.clone() - self.begin_chunk.clone()
            + Vector::fill_with_len(ChunkCoordinate::from(1), self.begin_chunk.len())
    }

    pub fn chunk_indices_linear<'b>(&'b self) -> impl Iterator<Item = ChunkIndex> + 'b {
        self.chunk_positions_linear()
            .map(|p| self.md.chunk_index(&p))
    }

    pub fn chunk_positions_linear<'b>(&self) -> impl Iterator<Item = Vector<D, ChunkCoordinate>> {
        let nd = self.md.dim().n();

        (0..nd)
            .into_iter()
            .map(|i| self.begin_chunk[i].raw..=self.end_chunk[i].raw)
            .multi_cartesian_product()
            .map(|coordinates| Vector::<D, ChunkCoordinate>::try_from(coordinates).unwrap())
    }
}
