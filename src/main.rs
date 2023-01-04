use std::path::PathBuf;

use clap::Parser;

use crate::{
    data::VoxelPosition,
    operators::{request_value, LinearRescale, Mean, VvdVolumeSource},
    storage::Storage,
};

mod array;
mod data;
mod id;
mod operator;
mod operators;
mod runtime;
mod storage;
mod task;
mod task_manager;
mod threadpool;
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
    let storage = Storage::new(storage_size);
    let thread_pool_size = 4;

    let factor = args.factor;

    let brick_size = VoxelPosition(cgmath::vec3(32, 32, 23));

    let vol = VvdVolumeSource::open(&args.vvd_vol, brick_size)?;

    let scaled1 = LinearRescale {
        vol: &vol,
        factor: &factor,
        offset: &0.0,
    };

    let scaled2 = LinearRescale {
        vol: &scaled1,
        factor: &factor,
        offset: &0.0,
    };

    let mean = Mean::new(&scaled2);
    let mean_unscaled = Mean::new(&vol);

    let request_queue = runtime::RequestQueue::new();
    let hints = runtime::TaskHints::new();
    let (thread_pool, thread_spawner) = task_manager::create_task_manager(thread_pool_size);
    let mut rt = runtime::RunTime::new(
        &storage,
        thread_pool,
        &thread_spawner,
        &request_queue,
        &hints,
    );

    // TODO: it's slightly annoying that we have to construct the reference here (because of async
    // move). Is there a better way, i.e. to only move some values into the future?
    let mean = &mean;
    let mean_val =
        rt.resolve(|ctx| async move { Ok(*ctx.submit(request_value(mean)).await) }.into())?;

    let tasks_executed = rt.statistics().tasks_executed;
    println!(
        "Computed scaled mean val: {} ({} tasks)",
        mean_val, tasks_executed
    );

    // Neat: We can even write to references in the closure/future below to get results out.
    let mean_unscaled = &mean_unscaled;
    let mut mean_val_unscaled = 0.0;
    let muv_ref = &mut mean_val_unscaled;
    let tasks_executed_prev = rt.statistics().tasks_executed;
    let id = rt.resolve(|ctx| {
        async move {
            let req = request_value(mean_unscaled);
            let id = req.id;
            *muv_ref = *ctx.submit(req).await;
            Ok(id)
        }
        .into()
    })?;
    let tasks_executed = rt.statistics().tasks_executed - tasks_executed_prev;
    println!(
        "Computed unscaled mean val: {} ({} tasks)",
        mean_val_unscaled, tasks_executed
    );

    storage.try_free(id).unwrap();

    let tasks_executed_prev = rt.statistics().tasks_executed;
    let muv_ref = &mut mean_val_unscaled;
    rt.resolve(|ctx| {
        async move {
            *muv_ref = *ctx.submit(request_value(mean_unscaled)).await;
            Ok(())
        }
        .into()
    })?;
    let tasks_executed = rt.statistics().tasks_executed - tasks_executed_prev;
    println!(
        "Computed unscaled mean val again, after deletion: {} ({} tasks)",
        mean_val_unscaled, tasks_executed
    );

    Ok(())
}
