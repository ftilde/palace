use crate::{
    data::{BrickPosition, LocalVoxelPosition, VoxelPosition},
    operators::volume::{rechunk, VolumeOperator, VolumeOperatorState},
    runtime::RunTime,
};

pub fn compare_volume(
    vol: VolumeOperator,
    size: VoxelPosition,
    fill_expected: impl FnOnce(&mut ndarray::ArrayViewMut3<f32>),
) {
    let mut runtime = RunTime::new(1 << 30, Some(1)).unwrap();

    let full_vol = rechunk(vol, size.local());
    let full_vol = &full_vol;

    let mut c = runtime.context_anchor();
    let mut executor = c.executor();

    executor
        .resolve(|ctx| {
            async move {
                let pos = BrickPosition::from([0, 0, 0]);
                let m = ctx.submit(full_vol.metadata.request_scalar()).await;
                let info = m.chunk_info(pos);
                let vol = ctx.submit(full_vol.bricks.request(pos)).await;
                let vol = crate::data::chunk(&vol, &info);

                let mut comp = vec![0.0; info.mem_elements()];
                let mut comp = crate::data::chunk_mut(&mut comp, &info);
                fill_expected(&mut comp);
                assert_eq!(vol, comp);
                Ok(())
            }
            .into()
        })
        .unwrap();
}

pub fn center_point_vol(
    size: VoxelPosition,
    brick_size: LocalVoxelPosition,
) -> (impl VolumeOperatorState, VoxelPosition) {
    let center = size.map(|v| v / 2u32);

    let point_vol = crate::operators::rasterize_function::voxel(size, brick_size, move |v| {
        if v == center {
            1.0
        } else {
            0.0
        }
    });

    (point_vol, center)
}
