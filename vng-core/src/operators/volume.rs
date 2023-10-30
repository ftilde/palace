use derive_more::From;
use futures::stream::StreamExt;

use crate::{
    array::VolumeEmbeddingData,
    data::{
        chunk, chunk_mut, slice_range, BrickPosition, GlobalCoordinate, LocalCoordinate, Vector,
    },
    id::Id,
    operator::OperatorId,
    task::RequestStream,
};

use super::{
    array::ArrayOperator,
    scalar::ScalarOperator,
    tensor::{EmbeddedTensorOperator, TensorOperator},
};

pub trait VolumeOperatorState {
    fn operate(&self) -> VolumeOperator;
}
pub trait EmbeddedVolumeOperatorState {
    fn operate(&self) -> EmbeddedVolumeOperator;
}

impl<O: EmbeddedVolumeOperatorState> VolumeOperatorState for O {
    fn operate(&self) -> VolumeOperator {
        self.operate().into()
    }
}
impl<O: VolumeOperatorState> EmbeddedVolumeOperatorState for (O, VolumeEmbeddingData) {
    fn operate(&self) -> EmbeddedVolumeOperator {
        self.0.operate().embedded(self.1.clone())
    }
}

pub type VolumeOperator = TensorOperator<3>;
pub type EmbeddedVolumeOperator = EmbeddedTensorOperator<3>;

#[allow(unused)]
pub fn mean(input: VolumeOperator) -> ScalarOperator<f32> {
    crate::operators::scalar::scalar(
        OperatorId::new("volume_mean").dependent_on(&input),
        input,
        move |ctx, input| {
            async move {
                let vol = ctx.submit(input.metadata.request_scalar()).await;

                let to_request = vol.brick_positions().collect::<Vec<_>>();
                let batch_size = 1024;

                let mut sum = 0.0;
                for chunk in to_request.chunks(batch_size) {
                    let mut stream = ctx
                        .submit_unordered_with_data(
                            chunk.iter().map(|pos| (input.chunks.request(*pos), *pos)),
                        )
                        .then_req(ctx.into(), |(brick_handle, brick_pos)| {
                            let chunk_info = vol.chunk_info(brick_pos);
                            let brick_handle = brick_handle.into_thread_handle();
                            ctx.spawn_compute(move || {
                                let sum = if chunk_info.is_full() {
                                    brick_handle.iter().sum::<f32>()
                                } else {
                                    let brick = crate::data::chunk(&brick_handle, &chunk_info);
                                    brick.iter().sum::<f32>()
                                };
                                (brick_handle, sum)
                            })
                        });

                    while let Some((brick_handle, part_sum)) = stream.next().await {
                        sum += part_sum;
                        brick_handle.into_main_handle(ctx.storage());
                    }
                }

                let v = sum / vol.num_elements() as f32;

                ctx.write(v)
            }
            .into()
        },
    )
}

#[derive(Copy, Clone, From, Hash)]
pub enum ChunkSize {
    Fixed(LocalCoordinate),
    //Relative(f32),
    Full,
}

impl ChunkSize {
    pub fn apply(self, global_size: GlobalCoordinate) -> LocalCoordinate {
        match self {
            ChunkSize::Fixed(l) => l,
            //RechunkArgument::Relative(f) => ((global_size.raw as f32 * f).round() as u32).into(),
            ChunkSize::Full => global_size.local(),
        }
    }
}

#[allow(unused)]
pub fn rechunk(input: VolumeOperator, brick_size: Vector<3, ChunkSize>) -> VolumeOperator {
    TensorOperator::with_state(
        OperatorId::new("volume_rechunk")
            .dependent_on(&input)
            .dependent_on(Id::hash(&brick_size)),
        input.clone(),
        input,
        move |ctx, input| {
            async move {
                let req = input.metadata.request_scalar();
                let mut m = ctx.submit(req).await;
                m.chunk_size = brick_size.zip(m.dimensions, |v, d| v.apply(d));
                ctx.write(m)
            }
            .into()
        },
        move |ctx, positions, input| {
            // TODO: optimize case where input.brick_size == output.brick_size
            async move {
                let m_in = ctx.submit(input.metadata.request_scalar()).await;
                let m_out = {
                    let mut m_out = m_in;
                    m_out.chunk_size = brick_size.zip(m_in.dimensions, |v, d| v.apply(d));
                    m_out
                };

                let requests = positions.into_iter().map(|pos| {
                    let out_info = m_out.chunk_info(pos);
                    let out_begin = out_info.begin();
                    let out_end = out_info.end();

                    let in_begin_brick = m_in.chunk_pos(out_begin);
                    let in_end_brick = m_in.chunk_pos(out_end.map(|v| v - 1u32));

                    let in_brick_positions = itertools::iproduct! {
                        in_begin_brick.z().raw..=in_end_brick.z().raw,
                        in_begin_brick.y().raw..=in_end_brick.y().raw,
                        in_begin_brick.x().raw..=in_end_brick.x().raw
                    }
                    .map(|(z, y, x)| BrickPosition::from([z, y, x]))
                    .collect::<Vec<_>>();
                    let intersecting_bricks = ctx.group(
                        in_brick_positions
                            .iter()
                            .map(|pos| input.chunks.request(*pos)),
                    );

                    (intersecting_bricks, (pos, in_brick_positions))
                });

                let mut stream = ctx.submit_unordered_with_data(requests).then_req(
                    ctx.into(),
                    |(intersecting_bricks, (pos, in_brick_positions))| {
                        let out_info = m_out.chunk_info(pos);
                        let brick_handle = ctx.alloc_slot(pos, out_info.mem_elements()).unwrap();
                        let mut brick_handle = brick_handle.into_thread_handle();
                        let intersecting_bricks = intersecting_bricks
                            .into_iter()
                            .map(|v| v.into_thread_handle())
                            .collect::<Vec<_>>();

                        ctx.spawn_compute(move || {
                            let out_data = &mut *brick_handle;
                            let out_begin = out_info.begin();
                            let out_end = out_info.end();

                            crate::data::init_non_full(out_data, &out_info, f32::NAN);

                            let mut out_chunk = crate::data::chunk_mut(out_data, &out_info);
                            for (in_data_handle, in_brick_pos) in intersecting_bricks
                                .iter()
                                .zip(in_brick_positions.into_iter())
                            {
                                let in_data = &*in_data_handle;
                                let in_info = m_in.chunk_info(in_brick_pos);
                                let in_chunk = crate::data::chunk(in_data, &in_info);

                                let in_begin = in_info.begin();
                                let in_end = in_info.end();

                                let overlap_begin = in_begin.zip(out_begin, |i, o| i.max(o));
                                let overlap_end = in_end.zip(out_end, |i, o| i.min(o));
                                let overlap_size = (overlap_end - overlap_begin)
                                    .map(LocalCoordinate::interpret_as);

                                let in_chunk_begin = in_info.in_chunk(overlap_begin);
                                let in_chunk_end = in_chunk_begin + overlap_size;

                                let out_chunk_begin = out_info.in_chunk(overlap_begin);
                                let out_chunk_end = out_chunk_begin + overlap_size;

                                let mut o = out_chunk
                                    .slice_mut(slice_range(out_chunk_begin, out_chunk_end));
                                let i = in_chunk.slice(slice_range(in_chunk_begin, in_chunk_end));

                                ndarray::azip!((o in &mut o, i in &i) { o.write(*i); });
                            }
                            (brick_handle, intersecting_bricks)
                        })
                    },
                );

                while let Some((brick_handle, intersecting_bricks)) = stream.next().await {
                    let brick_handle = brick_handle.into_main_handle(ctx.storage());
                    unsafe { brick_handle.initialized(*ctx) };
                    for i in intersecting_bricks {
                        i.into_main_handle(ctx.storage());
                    }
                }
                Ok(())
            }
            .into()
        },
    )
}

/// A one dimensional convolution in the specified (constant) axis. Currently zero padding is the
/// only supported (and thus always applied) border handling routine.
#[allow(unused)]
pub fn convolution_1d<const DIM: usize>(
    input: VolumeOperator,
    kernel: ArrayOperator,
) -> VolumeOperator {
    TensorOperator::with_state(
        OperatorId::new("convolution_1d")
            .dependent_on(&input)
            .dependent_on(&kernel),
        input.clone(),
        (input, kernel),
        move |ctx, input| {
            async move {
                let req = input.metadata.request_scalar();
                let m = ctx.submit(req).await;
                ctx.write(m)
            }
            .into()
        },
        move |ctx, positions, (input, kernel)| {
            // TODO: optimize case where input.brick_size == output.brick_size
            async move {
                let (m_in, kernel_m, kernel_handle) = futures::join!(
                    ctx.submit(input.metadata.request_scalar()),
                    ctx.submit(kernel.metadata.request_scalar()),
                    ctx.submit(kernel.chunks.request([0].into())),
                );

                assert_eq!(
                    kernel_m.dimensions.raw(),
                    kernel_m.chunk_size.raw(),
                    "Only unchunked kernels are supported for now"
                );

                let kernel_size = *kernel_m.dimensions.raw();
                assert!(kernel_size % 2 == 1, "Kernel size must be odd");
                let extent = kernel_size / 2;

                let kernel = &*kernel_handle;
                let m_out = m_in;

                let requests = positions.into_iter().map(|pos| {
                    let out_info = m_out.chunk_info(pos);
                    let out_begin = out_info.begin();
                    let out_end = out_info.end();

                    let in_begin = out_begin
                        .map_element(DIM, |v| (v.raw.saturating_sub(extent as u32)).into());
                    let in_end = out_end
                        .map_element(DIM, |v| (v + extent as u32).min(m_out.dimensions[DIM]));

                    let in_begin_brick = m_in.chunk_pos(in_begin);
                    let in_end_brick = m_in.chunk_pos(in_end.map(|v| v - 1u32));

                    let in_brick_positions = itertools::iproduct! {
                        in_begin_brick.z().raw..=in_end_brick.z().raw,
                        in_begin_brick.y().raw..=in_end_brick.y().raw,
                        in_begin_brick.x().raw..=in_end_brick.x().raw
                    }
                    .map(|(z, y, x)| BrickPosition::from([z, y, x]))
                    .collect::<Vec<_>>();
                    let intersecting_bricks = ctx.group(
                        in_brick_positions
                            .iter()
                            .map(|pos| input.chunks.request(*pos)),
                    );

                    (intersecting_bricks, (pos, in_brick_positions))
                });

                let mut stream = ctx.submit_unordered_with_data(requests).then_req(
                    ctx.into(),
                    |(intersecting_bricks, (pos, in_brick_positions))| {
                        let out_info = m_out.chunk_info(pos);
                        let brick_handle = ctx.alloc_slot(pos, out_info.mem_elements()).unwrap();
                        let mut brick_handle = brick_handle.into_thread_handle();
                        let intersecting_bricks = intersecting_bricks
                            .into_iter()
                            .map(|v| v.into_thread_handle())
                            .collect::<Vec<_>>();

                        ctx.spawn_compute(move || {
                            let out_data = &mut *brick_handle;
                            let out_begin = out_info.begin();
                            let out_end = out_info.end();

                            let out_data = crate::data::fill_uninit(out_data, 0.0);
                            let mut out_chunk = chunk_mut(out_data, &out_info);

                            for (in_data_handle, in_brick_pos) in intersecting_bricks
                                .iter()
                                .zip(in_brick_positions.into_iter())
                            {
                                let in_data = &*in_data_handle;
                                let in_info = m_in.chunk_info(in_brick_pos);
                                let in_chunk = chunk(in_data, &in_info);

                                // Logical dimensions should be equal except possibly in DIM (if we
                                // are at the border)
                                assert!(out_info
                                    .logical_dimensions
                                    .zip_enumerate(in_info.logical_dimensions, |i, a, b| {
                                        i == DIM || a.raw == b.raw
                                    })
                                    .fold(true, std::ops::BitAnd::bitand));

                                let in_begin = in_info.begin();
                                let in_end = in_info.end();

                                let begin_i_global = in_begin[DIM].raw as i32;
                                let end_i_global = in_end[DIM].raw as i32;
                                let begin_o_global = out_begin[DIM].raw as i32;
                                let end_o_global = out_end[DIM].raw as i32;
                                let extent = extent as i32;

                                let begin_ext = (begin_i_global - end_o_global).max(-extent);
                                let end_ext = (end_i_global - begin_o_global).min(extent);

                                for offset in begin_ext..=end_ext {
                                    let kernel_buf_index = (extent - offset) as usize;
                                    let kernel_val = kernel[kernel_buf_index];

                                    let begin_i_local = (begin_o_global + offset) - begin_i_global;
                                    let end_i_local = (end_o_global + offset) - begin_i_global;

                                    let iter_i_begin = Vector::<3, i32>::fill(0)
                                        .map_element(DIM, |a| a.max(begin_i_local))
                                        .map(|v| v as usize);
                                    let iter_i_end = in_info
                                        .logical_dimensions
                                        .map(|v| v.raw as i32)
                                        .map_element(DIM, |a| a.min(end_i_local))
                                        .map(|v| v as usize);

                                    let begin_o_local = (begin_i_global - offset) - begin_o_global;
                                    let end_o_local = (end_i_global - offset) - begin_o_global;

                                    let iter_o_begin = Vector::<3, i32>::fill(0)
                                        .map_element(DIM, |a| a.max(begin_o_local))
                                        .map(|v| v as usize);
                                    let iter_o_end = out_info
                                        .logical_dimensions
                                        .map(|v| v.raw as i32)
                                        .map_element(DIM, |a| a.min(end_o_local))
                                        .map(|v| v as usize);

                                    assert!(iter_i_end - iter_i_begin == iter_o_end - iter_o_begin);

                                    let in_chunk_active =
                                        in_chunk.slice(slice_range(iter_i_begin, iter_i_end));
                                    let in_lines = in_chunk_active.rows();

                                    let mut out_chunk_active =
                                        out_chunk.slice_mut(slice_range(iter_o_begin, iter_o_end));
                                    let out_lines = out_chunk_active.rows_mut();

                                    for (mut ol, il) in
                                        out_lines.into_iter().zip(in_lines.into_iter())
                                    {
                                        let ol = ol.as_slice_mut().unwrap();
                                        let il = il.as_slice().unwrap();
                                        for (o, i) in ol.iter_mut().zip(il.iter()) {
                                            let v = kernel_val * i;
                                            *o += v;
                                        }
                                    }
                                }
                            }
                            (brick_handle, intersecting_bricks)
                        })
                    },
                );

                while let Some((brick_handle, intersecting_bricks)) = stream.next().await {
                    let brick_handle = brick_handle.into_main_handle(ctx.storage());
                    unsafe { brick_handle.initialized(*ctx) };
                    for i in intersecting_bricks {
                        i.into_main_handle(ctx.storage());
                    }
                }
                Ok(())
            }
            .into()
        },
    )
    .into()
}

#[allow(unused)]
pub fn separable_convolution(
    v: VolumeOperator,
    [k0, k1, k2]: [ArrayOperator; 3],
) -> VolumeOperator {
    let v = convolution_1d::<2>(v, k2);
    let v = convolution_1d::<1>(v, k1);
    let v = convolution_1d::<0>(v, k0);
    v
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        data::{GlobalCoordinate, LocalVoxelPosition, VoxelPosition},
        test_util::*,
    };

    fn compare_convolution_1d<const DIM: usize>(
        input: &dyn VolumeOperatorState,
        kernel: &[f32],
        fill_expected: impl FnOnce(&mut ndarray::ArrayViewMut3<f32>),
    ) {
        let input = input.operate();
        let kernel = crate::operators::array::from_rc(kernel.into());
        let output = convolution_1d::<DIM>(input, kernel);
        compare_volume(output, fill_expected);
    }

    fn test_convolution_1d_generic<const DIM: usize>() {
        // Small
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let (point_vol, center) = center_point_vol(size, brick_size);
        compare_convolution_1d::<DIM>(&point_vol, &[1.0, -1.0, 2.0], |comp| {
            comp[center.map_element(DIM, |v| v - 1u32).as_index()] = 1.0;
            comp[center.map_element(DIM, |v| v).as_index()] = -1.0;
            comp[center.map_element(DIM, |v| v + 1u32).as_index()] = 2.0;
        });

        // Larger
        let size = VoxelPosition::fill(13.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let (point_vol, center) = center_point_vol(size, brick_size);
        let kernel_size = 7;
        let extent = kernel_size / 2;
        let mut kernel = vec![0.0; kernel_size];
        kernel[0] = -1.0;
        kernel[1] = -2.0;
        kernel[kernel_size - 1] = 1.0;
        kernel[kernel_size - 2] = 2.0;
        compare_convolution_1d::<DIM>(&point_vol, &kernel, |comp| {
            comp[center.map_element(DIM, |v| v - extent).as_index()] = -1.0;
            comp[center.map_element(DIM, |v| v - extent + 1u32).as_index()] = -2.0;
            comp[center.map_element(DIM, |v| v + extent).as_index()] = 1.0;
            comp[center.map_element(DIM, |v| v + extent - 1u32).as_index()] = 2.0;
        });
    }

    #[test]
    fn test_convolution_1d_x() {
        test_convolution_1d_generic::<2>();
    }
    #[test]
    fn test_convolution_1d_y() {
        test_convolution_1d_generic::<1>();
    }
    #[test]
    fn test_convolution_1d_z() {
        test_convolution_1d_generic::<0>();
    }

    #[test]
    fn test_separable_convolution() {
        let size = VoxelPosition::fill(5.into());
        let brick_size = LocalVoxelPosition::fill(2.into());

        let (point_vol, center) = center_point_vol(size, brick_size);

        let kernels = [&[2.0, 1.0, 2.0], &[2.0, 1.0, 2.0], &[2.0, 1.0, 2.0]];
        let kernels = std::array::from_fn(|i| crate::operators::array::from_static(kernels[i]));
        let output = separable_convolution(point_vol.operate(), kernels);
        compare_volume(output, |comp| {
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let offset = Vector::new([dz, dy, dx]);
                        let l1_dist = offset.map(i32::abs).fold(0, std::ops::Add::add);
                        let expected_val = 1 << l1_dist;
                        comp[(center.try_into_elem::<i32>().unwrap() + offset)
                            .try_into_elem::<GlobalCoordinate>()
                            .unwrap()
                            .as_index()] = expected_val as f32;
                    }
                }
            }
        });
    }
}
