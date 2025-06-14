use id::Identify;

use crate::{
    array::{ArrayMetaData, ChunkIndex},
    data::Vector,
    dtypes::StaticElementType,
    op_descriptor,
    operator::{DataParam, OperatorDescriptor},
    storage::Element,
};

use super::{array::ArrayOperator, tensor::TensorOperator};

struct Extent(usize);

impl Extent {
    fn size(&self) -> usize {
        2 * self.0 + 1
    }
}

fn suitable_extent(stddev: f32) -> Extent {
    assert!(stddev >= 0.0);
    Extent((stddev * 3.0).round() as usize)
}

fn gauss_func(stdev: f32) -> impl Fn(f32) -> f32 {
    move |x| (-0.5 * x * x / (stdev * stdev)).exp() / (std::f32::consts::TAU.sqrt() * stdev)
}

fn dgauss_dx_func(stdev: f32) -> impl Fn(f32) -> f32 {
    move |x| {
        -(-0.5 * x * x / (stdev * stdev)).exp() * x
            / (std::f32::consts::TAU.sqrt() * stdev * stdev * stdev)
    }
}

fn ddgauss_dxdx_func(stdev: f32) -> impl Fn(f32) -> f32 {
    move |x| {
        (-0.5 * x * x / (stdev * stdev)).exp() * (x * x - stdev * stdev)
            / (std::f32::consts::TAU.sqrt() * stdev * stdev * stdev * stdev * stdev)
    }
}

fn smooth_sample(f: impl Fn(f32) -> f32, i: f32) -> f32 {
    let radius = 5;
    let mut sum = 0.0;
    for d in -radius..=radius {
        let df = d as f32 / (radius as f32);
        let sample_pos = i - 0.5 * df;
        sum += f(sample_pos);
    }
    sum / (2.0 * radius as f32 + 1.0)
}

pub fn comp_kernel_gauss(stddev: f32) -> Vec<f32> {
    let extent = suitable_extent(stddev);
    let func = gauss_func(stddev);
    let mut kernel = Vec::with_capacity(extent.size());
    let mut sum = 0.0;
    let extent = extent.0 as i32;
    for i in -extent..=extent {
        let val = smooth_sample(&func, i as f32);
        sum += val;
        kernel.push(val);
    }
    // Normalize so that all values sum to 1
    for v in kernel.as_mut_slice() {
        *v /= sum;
    }
    kernel
}

fn gen_kernel_sum_zero(func: impl Fn(f32) -> f32, norm_nominator: f32, extent: Extent) -> Vec<f32> {
    let mut kernel = Vec::with_capacity(extent.size());

    let extent = extent.0 as i32;

    let mut positive_sum = 0.0;
    let mut negative_sum = 0.0;
    for i in -extent..=extent {
        let val = smooth_sample(&func, i as f32);
        if val > 0.0 {
            positive_sum += val;
        } else {
            negative_sum += val;
        }
        kernel.push(val);
    }
    let positive_normalization_factor = (norm_nominator / positive_sum).abs();
    let negative_normalization_factor = (norm_nominator / negative_sum).abs();
    // Normalize so that all values sum to 0
    for v in kernel.as_mut_slice() {
        if *v > 0.0 {
            *v *= positive_normalization_factor;
        } else {
            *v *= negative_normalization_factor;
        }
    }
    kernel
}

pub fn comp_kernel_dgauss_dx(stddev: f32) -> Vec<f32> {
    let extent = suitable_extent(stddev);
    let norm_nom = gauss_func(stddev)(0.0);
    let func = dgauss_dx_func(stddev);
    gen_kernel_sum_zero(func, norm_nom, extent)
}

pub fn comp_kernel_ddgauss_dxdx(stddev: f32) -> Vec<f32> {
    let extent = suitable_extent(stddev);
    let norm_nom = 2.0 * dgauss_dx_func(stddev)(stddev);
    let func = ddgauss_dxdx_func(stddev);
    gen_kernel_sum_zero(func, norm_nom, extent)
}

pub fn gauss<'a>(stddev: f32) -> ArrayOperator<StaticElementType<f32>> {
    gen_kernel_operator(
        op_descriptor!(),
        stddev,
        |stddev| suitable_extent(*stddev).size(),
        comp_kernel_gauss,
    )
}
pub fn dgauss_dx<'a>(stddev: f32) -> ArrayOperator<StaticElementType<f32>> {
    gen_kernel_operator(
        op_descriptor!(),
        stddev,
        |stddev| suitable_extent(*stddev).size(),
        comp_kernel_dgauss_dx,
    )
}
pub fn ddgauss_dxdx<'a>(stddev: f32) -> ArrayOperator<StaticElementType<f32>> {
    gen_kernel_operator(
        op_descriptor!(),
        stddev,
        |stddev| suitable_extent(*stddev).size(),
        comp_kernel_ddgauss_dxdx,
    )
}

fn gen_kernel_operator<Params: Element + Identify>(
    descriptor: OperatorDescriptor,
    params: Params,
    get_size: fn(&Params) -> usize,
    gen_kernel: fn(Params) -> Vec<f32>,
) -> ArrayOperator<StaticElementType<f32>> {
    TensorOperator::unbatched(
        descriptor,
        Default::default(),
        {
            let size = get_size(&params) as u32;
            ArrayMetaData {
                dimensions: Vector::fill(size.into()),
                chunk_size: Vector::fill(size.into()),
            }
        },
        (
            DataParam(params),
            //Note: This only works because we are controlling the parameter functions and we take
            //in the descriptor separately
            DataParam(id::identify_source_location(get_size)),
            DataParam(id::identify_source_location(gen_kernel)),
        ),
        move |ctx, pos, _, (params, get_size, gen_kernel)| {
            assert_eq!(pos, ChunkIndex(0));
            async move {
                let size = get_size(&params);

                let mut out = ctx
                    .submit(ctx.alloc_slot_num_elements(pos, size).unwrap_value())
                    .await;
                let mut out_data = &mut *out;
                ctx.submit(ctx.spawn_compute(move || {
                    let kernel = gen_kernel(**params);
                    assert_eq!(kernel.len(), size);
                    crate::data::write_slice_uninit(&mut out_data, &kernel);
                }))
                .await;

                // Safety: slot and kernel are of the exact same size. Thus all values are
                // initialized.
                unsafe { out.initialized(*ctx) };
                Ok(())
            }
            .into()
        },
    )
}
