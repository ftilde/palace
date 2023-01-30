use std::path::PathBuf;

use clap::Parser;
use operators::{Hdf5VolumeSourceState, VolumeOperatorState, VvdVolumeSourceState};
use runtime::RunTime;

use crate::{data::VoxelPosition, operators::volume};

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
mod threadpool;
mod util;
#[allow(unused)] //TODO: Remove once warnings are resolved
mod vulkan;

// TODO look into thiserror/anyhow
type Error = Box<(dyn std::error::Error + 'static)>;

#[derive(Parser)]
struct CliArgs {
    #[arg()]
    vvd_vol: PathBuf,

    #[arg(short, long, default_value = "1.0")]
    factor: f32,

    #[arg(long, default_value = "false")]
    with_vulkan: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse();

    if args.with_vulkan {
        let _vulkan_manager = vulkan::VulkanManager::new();
    }

    let storage_size = 1 << 30; //One gigabyte
    let thread_pool_size = 4;

    let mut runtime = RunTime::new(storage_size, thread_pool_size);

    let brick_size = VoxelPosition(cgmath::vec3(32, 32, 32));

    let extension = args
        .vvd_vol
        .extension()
        .map(|v| v.to_string_lossy().to_string());
    let vol_state: Box<dyn VolumeOperatorState> = match extension.as_deref() {
        Some("vvd") => Box::new(VvdVolumeSourceState::open(&args.vvd_vol, brick_size)?),
        Some("h5") => Box::new(Hdf5VolumeSourceState::open(
            args.vvd_vol,
            "/volume".to_string(),
        )?),
        _ => {
            return Err(format!(
                "Unknown volume format for file {}",
                args.vvd_vol.to_string_lossy()
            )
            .into())
        }
    };

    eval_network(&mut runtime, &*vol_state, &args.factor)
}

fn eval_network(
    runtime: &mut RunTime,
    vol: &dyn VolumeOperatorState,
    factor: &f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let vol = vol.operate();
    let factor = factor.into();
    let offset = (&0.0).into();

    let rechunked = volume::rechunk(&vol, VoxelPosition((32, 32, 32).into()));

    let scaled1 = volume::linear_rescale(&rechunked, &factor, &offset);
    let scaled2 = volume::linear_rescale(&scaled1, &factor, &offset);

    let mean = volume::mean(&scaled2);
    let mean_unscaled = volume::mean(&rechunked);

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

    executor.data.storage.try_free(id).unwrap();

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
