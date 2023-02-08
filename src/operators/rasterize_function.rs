use futures::StreamExt;

use crate::{
    array::VolumeMetaData,
    data::{BrickPosition, LocalVoxelPosition, Vector, VoxelPosition},
    id::Id,
    operator::OperatorId,
    task::TaskContext,
    Error,
};

use super::volume::{VolumeOperator, VolumeOperatorState};

pub struct VoxelPosRasterizer<F> {
    function: F,
    metadata: VolumeMetaData,
}

fn func_id<F: 'static>() -> Id {
    // TODO: One problem with this during development: The id of a closure may not change between
    // compilations, which may result in confusion when old values are reported when loaded from a
    // persistent cache even though the closure code was changed in the meantime.
    let id = std::any::TypeId::of::<F>();
    Id::hash(&id)
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
    f: impl 'static + Fn(Vector<3, f32>) -> f32 + Sync,
) -> VoxelPosRasterizer<impl 'static + Fn(VoxelPosition) -> f32 + Sync> {
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
            brick_handle.iter_mut().for_each(|v| {
                v.write(f32::NAN);
            });

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
        unsafe { handle.initialized() };
    }

    Ok(())
}

impl<F: 'static + Fn(VoxelPosition) -> f32 + Sync> VolumeOperatorState for VoxelPosRasterizer<F> {
    fn operate<'a>(&'a self) -> VolumeOperator<'a> {
        VolumeOperator::new(
            OperatorId::new("ImplicitFunctionRasterizer::operate")
                .dependent_on(func_id::<F>())
                .dependent_on(Id::hash(&self.metadata)),
            move |ctx, _| async move { ctx.write(self.metadata) }.into(),
            move |ctx, positions, _| {
                async move { rasterize(&self.metadata, &self.function, ctx, positions).await }
                    .into()
            },
        )
    }
}
