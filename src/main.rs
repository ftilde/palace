use std::path::PathBuf;

use clap::{Parser, Subcommand};
use data::{LocalVoxelPosition, VoxelPosition};
use operators::{
    reader::{Hdf5VolumeSourceState, NiftiVolumeSourceState, VvdVolumeSourceState},
    volume::VolumeOperatorState,
};
use runtime::RunTime;

use crate::operators::{volume, volume_gpu};

mod array;
mod data;
mod id;
mod operator;
mod operators;
mod runtime;
mod storage;
mod task;
mod task_graph;
mod task_manager;
#[cfg(test)]
mod test_util;
mod threadpool;
mod util;
mod vulkan;

// TODO look into thiserror/anyhow
type Error = Box<(dyn std::error::Error + 'static)>;

#[derive(Parser, Clone)]
struct SyntheticArgs {
    #[arg()]
    size: u32,
}

#[derive(Parser, Clone)]
struct FileArgs {
    #[arg()]
    vol: PathBuf,
}

#[derive(Subcommand, Clone)]
enum Input {
    File(FileArgs),
    Synthetic(SyntheticArgs),
}

#[derive(Parser)]
struct CliArgs {
    #[command(subcommand)]
    input: Input,

    #[arg(short, long, default_value = "1.0")]
    factor: f32,

    /// Size of the memory pool that will be allocated in gigabytes.
    #[arg(short, long, default_value = "8")]
    mem_size: usize,

    /// Force a specific size for the compute task pool [default: number of cores]
    #[arg(short, long)]
    compute_pool_size: Option<usize>,
}

fn open_volume(
    path: PathBuf,
    brick_size_hint: LocalVoxelPosition,
) -> Result<Box<dyn VolumeOperatorState>, Box<dyn std::error::Error>> {
    let Some(file) = path.file_name() else {
        return Err("No file name in path".into());
    };
    let file = file.to_string_lossy();
    let segments = file.split('.').collect::<Vec<_>>();

    Ok(match segments[..] {
        [.., "vvd"] => Box::new(VvdVolumeSourceState::open(&path, brick_size_hint)?),
        [.., "nii"] | [.., "nii", "gz"] => Box::new(NiftiVolumeSourceState::open_single(path)?),
        [.., "hdr"] => {
            let data = path.with_extension("img");
            Box::new(NiftiVolumeSourceState::open_separate(path, data)?)
        }
        [.., "h5"] => Box::new(Hdf5VolumeSourceState::open(path, "/volume".to_string())?),
        _ => {
            return Err(format!("Unknown volume format for file {}", path.to_string_lossy()).into())
        }
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse();

    let storage_size = args.mem_size << 30; //in gigabyte

    let mut runtime = RunTime::new(storage_size, args.compute_pool_size)?;

    let brick_size = LocalVoxelPosition::fill(64.into());

    let vol_state = match args.input {
        Input::File(path) => open_volume(path.vol, brick_size)?,
        Input::Synthetic(args) => Box::new(operators::rasterize_function::normalized(
            VoxelPosition::fill(args.size.into()),
            brick_size,
            |v| {
                let r2 = v
                    .map(|v| v - 0.5)
                    .map(|v| v * v)
                    .fold(0.0f32, std::ops::Add::add);
                r2.sqrt()
            },
        )),
    };

    eval_network(&mut runtime, &*vol_state, args.factor)
}

fn eval_network(
    runtime: &mut RunTime,
    vol: &dyn VolumeOperatorState,
    factor: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let vol = vol.operate();

    let rechunked = volume_gpu::rechunk(vol, LocalVoxelPosition::fill(48.into()));

    let smoothing_kernel = &[1.0 / 4.0, 2.0 / 4.0, 1.0 / 4.0];
    let convolved = volume::separable_convolution(
        rechunked.clone(),
        [smoothing_kernel, smoothing_kernel, smoothing_kernel],
    );
    let mapped = volume::map(convolved, |v| v.min(0.5));

    let scaled1 = volume_gpu::linear_rescale(mapped, factor.into(), 0.0.into());
    let scaled2 = volume_gpu::linear_rescale(scaled1, factor.into(), 0.0.into());
    let scaled2 = volume_gpu::linear_rescale(scaled2, (-1.0).into(), 0.0.into());

    let mean = volume::mean(scaled2);
    let mean_unscaled = volume::mean(rechunked);

    let mut c = runtime.context_anchor();
    let mut executor = c.executor();

    // TODO: it's slightly annoying that we have to construct the reference here (because of async
    // move). Is there a better way, i.e. to only move some values into the future?
    let mean_ref = &mean;
    let mean_val = executor
        .resolve(|ctx| async move { Ok(ctx.submit(mean_ref.request_scalar()).await) }.into())?;

    let tasks_executed = executor.statistics().tasks_executed;
    println!(
        "Computed scaled mean val: {} ({} tasks)",
        mean_val, tasks_executed
    );

    // Neat: We can even write to references in the closure/future below to get results out.
    let mean_unscaled_ref = &mean_unscaled;
    let mut mean_val_unscaled = 0.0;
    let muv_ref = &mut mean_val_unscaled;
    let tasks_executed_prev = executor.statistics().tasks_executed;
    let id = executor.resolve(|ctx| {
        async move {
            let req = mean_unscaled_ref.request_scalar();
            let id = req.id().unwrap_data();
            *muv_ref = ctx.submit(req).await;
            Ok(id)
        }
        .into()
    })?;
    let tasks_executed = executor.statistics().tasks_executed - tasks_executed_prev;
    println!(
        "Computed unscaled mean val: {} ({} tasks)",
        mean_val_unscaled, tasks_executed
    );

    executor.data.storage.try_free(id.id).unwrap();

    let tasks_executed_prev = executor.statistics().tasks_executed;
    let muv_ref = &mut mean_val_unscaled;
    executor.resolve(|ctx| {
        async move {
            *muv_ref = ctx.submit(mean_unscaled_ref.request_scalar()).await;
            Ok(())
        }
        .into()
    })?;
    let tasks_executed = executor.statistics().tasks_executed - tasks_executed_prev;
    println!(
        "Computed unscaled mean val again, after deletion: {} ({} tasks)",
        mean_val_unscaled, tasks_executed
    );

    Ok(())
}
