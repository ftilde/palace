use crate::{
    array::ChunkIndex,
    data::{LocalVoxelPosition, Vector, VoxelPosition},
    dim::*,
    dtypes::{DType, StaticElementType},
    operators::{
        rechunk::{rechunk, ChunkSize},
        tensor::{TensorOperator, VolumeOperator},
    },
    storage::Element,
};

pub fn compare_tensor_fn<D: Dimension, T: PartialEq + Element + Default + std::fmt::Debug>(
    vol: TensorOperator<D, StaticElementType<T>>,
    fill_expected: impl FnOnce(&mut ndarray::ArrayViewMut<T, D::NDArrayDim>),
) where
    StaticElementType<T>: Into<DType>,
{
    let mut runtime =
        crate::runtime::RunTime::new(1 << 30, 1 << 30, None, None, None, vec![]).unwrap();

    let full_vol = rechunk(vol, Vector::fill(ChunkSize::Full));
    let full_vol = &full_vol;

    runtime
        .resolve(None, false, |ctx, _| {
            async move {
                let pos = ChunkIndex(0);
                let m = full_vol.metadata;
                let info = m.chunk_info(pos);
                let vol = ctx.submit(full_vol.chunks.request(pos)).await;
                let vol = crate::data::chunk(&vol, &info);

                let mut comp = vec![T::default(); info.mem_elements()];
                let mut comp = crate::data::chunk_mut(&mut comp, &info);
                fill_expected(&mut comp);
                assert_eq!(vol, comp);
                Ok(())
            }
            .into()
        })
        .unwrap();
}

pub fn compare_tensor<D: Dimension>(
    result: TensorOperator<D, StaticElementType<f32>>,
    expected: TensorOperator<D, StaticElementType<f32>>,
) {
    compare_tensor_approx::<D>(result, expected, 0.0);
}

pub fn compare_tensor_approx<D: Dimension>(
    result: TensorOperator<D, StaticElementType<f32>>,
    expected: TensorOperator<D, StaticElementType<f32>>,
    max_diff: f32,
) {
    let mut runtime =
        crate::runtime::RunTime::new(1 << 30, 1 << 30, None, None, None, vec![]).unwrap();

    runtime
        .resolve(None, false, |ctx, _| {
            let result = &result;
            let expected = &expected;
            async move {
                let m = {
                    let m_r = result.metadata;
                    let m_e = expected.metadata;

                    assert_eq!(m_r, m_e);

                    m_r
                };
                let dib = m.dimension_in_chunks();
                let n_chunks = dib.hmul();
                for i in 0..n_chunks {
                    let pos = ChunkIndex(i as _);

                    let b_l = ctx.submit(result.chunks.request(pos)).await;
                    let b_r = ctx.submit(expected.chunks.request(pos)).await;

                    let b_l = &*b_l;
                    let b_r = &*b_r;
                    for (i, (l, r)) in b_l.iter().zip(b_r.iter()).enumerate() {
                        let diff = (l - r).abs();
                        if diff > max_diff {
                            panic!(
                                "{:?}\nand\n{:?}\ndiffer by {}, i.e. more than {} at position {}: {} vs. {}",
                                b_l, b_r, diff, max_diff, i, l, r
                            );
                        }
                    }
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
) -> (VolumeOperator<StaticElementType<f32>>, VoxelPosition) {
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
