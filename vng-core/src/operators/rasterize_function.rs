use futures::StreamExt;

use crate::{
    array::VolumeMetaData,
    data::{BrickPosition, LocalVoxelPosition, Vector, VoxelPosition},
    operator::OperatorId,
    task::TaskContext,
    Error,
};

use super::{
    tensor::TensorOperator,
    volume::{VolumeOperator, VolumeOperatorState},
};

#[derive(Clone)]
pub struct VoxelPosRasterizer<F> {
    function: F,
    metadata: VolumeMetaData,
}

pub fn voxel<F: 'static + Fn(VoxelPosition) -> f32 + Sync>(
    dimensions: VoxelPosition,
    brick_size: LocalVoxelPosition,
    f: F,
) -> VoxelPosRasterizer<F> {
    VoxelPosRasterizer {
        function: f,
        metadata: VolumeMetaData {
            dimensions,
            chunk_size: brick_size,
        },
    }
}

pub fn normalized(
    dimensions: VoxelPosition,
    brick_size: LocalVoxelPosition,
    f: impl 'static + Fn(Vector<3, f32>) -> f32 + Sync + Clone,
) -> VoxelPosRasterizer<impl 'static + Fn(VoxelPosition) -> f32 + Sync + Clone> {
    let dim_f = dimensions.map(|v| v.raw as f32);
    voxel(dimensions, brick_size, move |pos: VoxelPosition| {
        f(pos.map(|v| v.raw as f32) / dim_f)
    })
}

async fn rasterize<'cref, 'inv, F: 'static + Fn(VoxelPosition) -> f32 + Sync>(
    metadata: &VolumeMetaData,
    function: &F,
    ctx: TaskContext<'cref, 'inv, BrickPosition, f32>,
    positions: Vec<BrickPosition>,
) -> Result<(), Error> {
    let work = positions.into_iter().map(|pos| {
        let chunk = metadata.chunk_info(pos);

        let brick_handle = ctx.alloc_slot(pos, chunk.mem_elements()).unwrap();
        let mut brick_handle = brick_handle.into_thread_handle();
        ctx.spawn_compute(move || {
            crate::data::init_non_full(&mut brick_handle, &chunk, f32::NAN);

            let chunk_info = metadata.chunk_info(pos);

            let mut out_chunk = crate::data::chunk_mut(&mut brick_handle, &chunk_info);
            let begin = chunk_info.begin();

            for ((z, y, x), v) in out_chunk.indexed_iter_mut() {
                let pos: LocalVoxelPosition = [z as u32, y as u32, x as u32].into();
                let pos = begin + pos;
                v.write(function(pos));
            }

            brick_handle
        })
    });

    let stream = ctx.submit_unordered(work);

    futures::pin_mut!(stream);
    while let Some(handle) = stream.next().await {
        let handle = handle.into_main_handle(ctx.storage());
        unsafe { handle.initialized(*ctx) };
    }

    Ok(())
}

impl<F: 'static + Fn(VoxelPosition) -> f32 + Sync + Clone> VolumeOperatorState
    for VoxelPosRasterizer<F>
{
    fn operate(&self) -> VolumeOperator<f32> {
        TensorOperator::with_state(
            OperatorId::new("ImplicitFunctionRasterizer::operate")
                //TODO: Not sure if using func id is entirely correct: One may create a wrapper that
                //creates a `|_| var` closure based on a parameter `var`. All of those would have the
                //same type!
                .dependent_on(&crate::id::func_id::<F>())
                .dependent_on(&self.metadata),
            self.metadata.clone(),
            self.clone(),
            move |ctx, m| async move { ctx.write(*m) }.into(),
            move |ctx, positions, this| {
                async move { rasterize(&this.metadata, &this.function, ctx, positions).await }
                    .into()
            },
        )
    }
}
