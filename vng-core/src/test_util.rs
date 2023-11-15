use crate::{
    data::{from_linear, hmul, BrickPosition, LocalVoxelPosition, Vector, VoxelPosition},
    operators::{
        tensor::TensorOperator,
        volume::{rechunk, ChunkSize, VolumeOperator, VolumeOperatorState},
    },
    runtime::RunTime,
};

pub fn compare_volume(
    vol: VolumeOperator<f32>,
    fill_expected: impl FnOnce(&mut ndarray::ArrayViewMut3<f32>),
) {
    let mut runtime = RunTime::new(1 << 30, None, Some(1)).unwrap();

    let full_vol = rechunk(vol, Vector::fill(ChunkSize::Full));
    let full_vol = &full_vol;

    runtime
        .resolve(None, |ctx, _| {
            async move {
                let pos = BrickPosition::from([0, 0, 0]);
                let m = ctx.submit(full_vol.metadata.request_scalar()).await;
                let info = m.chunk_info(pos);
                let vol = ctx.submit(full_vol.chunks.request(pos)).await;
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

pub fn compare_tensor<const N: usize>(
    result: TensorOperator<N, f32>,
    expected: TensorOperator<N, f32>,
) {
    let mut runtime = RunTime::new(1 << 30, None, Some(1)).unwrap();

    runtime
        .resolve(None, |ctx, _| {
            let result = &result;
            let expected = &expected;
            async move {
                let m = {
                    let m_r = ctx.submit(result.metadata.request_scalar()).await;
                    let m_e = ctx.submit(expected.metadata.request_scalar()).await;

                    assert_eq!(m_r, m_e);

                    m_r
                };
                let dib = m.dimension_in_chunks();
                let n_chunks = hmul(dib);
                for i in 0..n_chunks {
                    let pos = from_linear(i, dib);

                    let b_l = ctx.submit(result.chunks.request(pos)).await;
                    let b_r = ctx.submit(expected.chunks.request(pos)).await;

                    assert_eq!(&*b_l, &*b_r);
                }
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
