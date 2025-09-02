use std::collections::BTreeMap;

use ash::vk;
use itertools::Itertools;

use crate::{
    array::{ChunkIndex, TensorMetaData},
    data::{ChunkCoordinate, GlobalCoordinate, LocalCoordinate},
    dim::{Dimension, DynDimension},
    dtypes::{DType, ElementType},
    operators::tensor::TensorOperator,
    storage::gpu::{BufferAddress, PageTableHandle, StateCacheHandle, StateCacheResult},
    task::OpaqueTaskContext,
    vec::Vector,
    vulkan::{
        pipeline::{
            AsBufferDescriptor, AsDescriptors, ComputePipeline, ComputePipelineBuilder,
            DescriptorConfig, DynPushConstants,
        },
        shader::Shader,
        state::VulkanState,
        CommandBuffer, DeviceContext, DstBarrierInfo,
    },
};

pub type FeedbackTableElement = u64;

const CHUNK_INDEX_BITS: u64 = 48;

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TensorQueryValue(u64);

impl From<u64> for TensorQueryValue {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl TensorQueryValue {
    pub fn chunk_index(&self) -> ChunkIndex {
        ChunkIndex(self.0 & ((1 << CHUNK_INDEX_BITS) - 1))
    }

    pub fn level(&self) -> usize {
        (self.0 >> CHUNK_INDEX_BITS) as usize
    }
}

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
                    0xffff_ffff,
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
                0xffff_ffff,
            )
        };
    }

    fn num_elements(&self) -> usize {
        crate::util::num_elms_in_array::<FeedbackTableElement>(self.inner.size as usize)
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.inner.buffer
    }

    pub fn inner(&self) -> &StateCacheHandle<'a> {
        &self.inner
    }

    // Note: any changes to the buffer have to be made visible to the cpu side via a barrier first
    pub async fn download_inserted<'cref, 'inv, T: From<FeedbackTableElement>>(
        &self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        device: &'cref DeviceContext,
    ) -> Vec<T> {
        let num_elements = self.num_elements();
        let layout = std::alloc::Layout::array::<FeedbackTableElement>(num_elements).unwrap();
        let mut request_table_cpu = vec![0u64; num_elements];
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
            .filter(|v| *v != FeedbackTableElement::max_value())
            .map(|v| v.into())
            .collect::<Vec<T>>();

        to_request_linear
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum RequestTableResult {
    #[default]
    Done = 0,
    Continue = 1,
    Timeout = 2,
}

impl RequestTableResult {
    pub fn combine(&mut self, other: RequestTableResult) {
        *self = (*self).max(other);
    }
}

pub struct RequestTable<'a>(ChunkFeedbackTable<'a>);

impl<'a> RequestTable<'a> {
    pub fn new(device: &DeviceContext, inner: StateCacheResult<'a>) -> Self {
        Self(ChunkFeedbackTable::new(device, inner))
    }

    /// Download the table results, try to insert the requested chunks into page table, respect
    /// timeout
    pub async fn download_and_insert<'cref, 'inv, D: Dimension, E: ElementType>(
        &mut self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        device: &'cref DeviceContext,
        tensor_and_pt: Vec<(&'inv TensorOperator<D, E>, &PageTableHandle<'_>)>,
        batch_size: &mut usize,
        interactive: bool,
        force_reset: bool,
    ) -> RequestTableResult {
        if !self.0.newly_initialized && !force_reset {
            let mut to_request_linear = self.0.download_inserted(ctx, device).await;

            if to_request_linear.is_empty() {
                return RequestTableResult::Done;
            }

            let res = request_to_page_table_with_timeout(
                &ctx,
                device,
                &mut to_request_linear,
                tensor_and_pt,
                batch_size,
                interactive,
            )
            .await;

            // Clear request table for the next iteration
            device.with_cmd_buffer(|cmd| self.0.clear(cmd));

            if let Err(crate::chunk_utils::Timeout) = res {
                return RequestTableResult::Timeout;
            }
        } else {
            self.0.newly_initialized = false;
        }
        RequestTableResult::Continue
    }

    pub fn buffer_address(&self) -> BufferAddress {
        self.0.inner.buffer_address()
    }
}

impl<'a> AsBufferDescriptor for RequestTable<'a> {
    fn gen_buffer_info(&self) -> vk::DescriptorBufferInfo {
        AsBufferDescriptor::gen_buffer_info(self.0.inner())
    }
}

pub struct UseTable<'a>(ChunkFeedbackTable<'a>);

impl<'a> UseTable<'a> {
    pub fn new(device: &DeviceContext, inner: StateCacheResult<'a>) -> Self {
        Self(ChunkFeedbackTable::new(device, inner))
    }

    pub fn buffer_address(&self) -> BufferAddress {
        self.0.inner.buffer_address()
    }

    /// Download the table results, note the buffers uses in the page table lru.
    pub async fn download_and_note_use<'cref, 'inv>(
        &mut self,
        ctx: OpaqueTaskContext<'cref, 'inv>,
        device: &'cref DeviceContext,
    ) {
        if !self.0.newly_initialized {
            let used_linear = self.0.download_inserted(ctx, device).await;
            let used_linear = used_linear
                .into_iter()
                .map(BufferAddress)
                .collect::<Vec<_>>();

            if !used_linear.is_empty() {
                device.storage.note_use(used_linear);

                device.with_cmd_buffer(|cmd| self.0.clear(cmd));
            }
        } else {
            self.0.newly_initialized = false;
        }
    }
}

pub struct Timeout;

pub async fn request_to_page_table_with_timeout<'cref, 'inv, D: Dimension, E: ElementType>(
    ctx: &OpaqueTaskContext<'cref, 'inv>,
    device: &DeviceContext,
    to_request_linear: &mut [TensorQueryValue],
    tensor_and_pt: Vec<(&'inv TensorOperator<D, E>, &PageTableHandle<'_>)>,
    batch_size: &mut usize,
    interactive: bool,
) -> Result<(), Timeout> {
    // Sort to get at least some benefit from spatial neighborhood
    to_request_linear.sort_unstable();

    let max_batch_size = to_request_linear.len().max(*batch_size);

    // Fulfill requests
    let mut to_request_linear = &to_request_linear[..];

    let mut res = Ok(());

    let mut pos_and_chunk = (0..tensor_and_pt.len())
        .into_iter()
        .map(|i| (i, Vec::new()))
        .collect::<BTreeMap<_, _>>();
    while !to_request_linear.is_empty() {
        let batch;
        (batch, to_request_linear) =
            to_request_linear.split_at((*batch_size).min(to_request_linear.len()));
        //println!("Batch: {} | {:?}", batch.len(), batch[0]);

        let to_request = batch.iter().map(|v| {
            let (tensor, _pt) = tensor_and_pt[v.level()];
            //println!("request: {:?} -> {} | {:?}", v, v.level(), v.chunk_index());
            let dim_in_bricks = tensor.metadata.dimension_in_chunks();
            let num_bricks = dim_in_bricks.hmul();
            assert!(
                (num_bricks as u64) < crate::storage::gpu::page_table::MAX_NUM_CHUNKS,
                "Cannot create page table for tensor with {} chunks. Maximum number of chunks is {}.", num_bricks, crate::storage::gpu::page_table::MAX_NUM_CHUNKS,
            );

            assert!(v.chunk_index().0 < num_bricks as _);
            tensor.chunks.request_gpu(
                device.id,
                v.chunk_index(),
                DstBarrierInfo {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_READ,
                },
            )
        });
        let requested_bricks = ctx.submit(ctx.group(to_request)).await;

        for (chunk, tqv) in requested_bricks.into_iter().zip(batch.iter()) {
            pos_and_chunk
                .get_mut(&tqv.level())
                .unwrap()
                .push((tqv.chunk_index(), chunk));
        }

        if let Some(lateness) = ctx.past_deadline(interactive) {
            if lateness > 2.0 {
                *batch_size = (*batch_size / 2).max(1);
            }
            res = Err(Timeout);
            break;
        }

        *batch_size = (*batch_size * 4).min(max_batch_size);
    }

    //let total_to_insert = pos_and_chunk.values().map(|v| v.len()).sum::<usize>();
    //println!("Inserting {} total", total_to_insert);

    let tensor_and_pt = &tensor_and_pt;
    futures::future::join_all(
        pos_and_chunk
            .into_iter()
            .map(move |(level, pac)| async move {
                tensor_and_pt[level].1.insert(*ctx, pac).await;
            }),
    )
    .await;

    res
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
