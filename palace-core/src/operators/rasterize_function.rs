use futures::StreamExt;

use crate::{
    array::VolumeMetaData,
    data::{BrickPosition, LocalVoxelPosition, Vector, VoxelPosition},
    dim::*,
    dtypes::StaticElementType,
    operator::OperatorDescriptor,
    storage::DataLocation,
    task::{RequestStream, TaskContext},
    Error,
};

use super::{tensor::TensorOperator, volume::VolumeOperator};

#[derive(Clone)]
pub struct VoxelPosRasterizer<F> {
    function: F,
    metadata: VolumeMetaData,
}

pub fn voxel<F: 'static + Fn(VoxelPosition) -> f32 + Sync + Clone>(
    dimensions: VoxelPosition,
    brick_size: LocalVoxelPosition,
    f: F,
) -> VolumeOperator<StaticElementType<f32>> {
    let r = VoxelPosRasterizer {
        function: f,
        metadata: VolumeMetaData {
            dimensions,
            chunk_size: brick_size,
        },
    };
    r.operate()
}

pub fn normalized(
    dimensions: VoxelPosition,
    brick_size: LocalVoxelPosition,
    f: impl 'static + Fn(Vector<D3, f32>) -> f32 + Sync + Clone,
) -> VolumeOperator<StaticElementType<f32>> {
    let dim_f = dimensions.map(|v| v.raw as f32);
    voxel(dimensions, brick_size, move |pos: VoxelPosition| {
        f(pos.map(|v| v.raw as f32) / dim_f)
    })
}

async fn rasterize<'cref, 'inv, F: 'static + Fn(VoxelPosition) -> f32 + Sync>(
    metadata: &VolumeMetaData,
    function: &F,
    ctx: TaskContext<'cref, 'inv, BrickPosition, StaticElementType<f32>>,
    positions: Vec<(BrickPosition, DataLocation)>,
) -> Result<(), Error> {
    let allocs = positions.into_iter().map(|(pos, _)| {
        let brick_handle_req = ctx.alloc_slot(pos, metadata.num_chunk_elements());
        (brick_handle_req, pos)
    });

    let stream = ctx
        .submit_unordered_with_data(allocs)
        .then_req(*ctx, |(brick_handle, pos)| {
            let mut brick_handle = brick_handle.into_thread_handle();
            ctx.spawn_compute(move || {
                let chunk_info = metadata.chunk_info(pos);

                crate::data::init_non_full(&mut brick_handle, &chunk_info, f32::NAN);

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

    futures::pin_mut!(stream);
    while let Some(handle) = stream.next().await {
        let handle = handle.into_main_handle(ctx.storage());
        unsafe { handle.initialized(*ctx) };
    }

    Ok(())
}

impl<F: 'static + Fn(VoxelPosition) -> f32 + Sync + Clone> VoxelPosRasterizer<F> {
    fn operate(&self) -> VolumeOperator<StaticElementType<f32>> {
        TensorOperator::with_state(
            OperatorDescriptor::new("ImplicitFunctionRasterizer::operate")
                //TODO: Not sure if using func id is entirely correct: One may create a wrapper that
                //creates a `|_| var` closure based on a parameter `var`. All of those would have the
                //same type!
                .dependent_on_data(&id::func_id::<F>())
                .dependent_on_data(&self.metadata),
            Default::default(),
            self.metadata,
            self.clone(),
            move |ctx, positions, this| {
                async move { rasterize(&this.metadata, &this.function, ctx, positions).await }
                    .into()
            },
        )
    }
}
