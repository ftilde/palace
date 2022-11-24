use std::path::PathBuf;

use clap::Parser;

use crate::{
    data::VoxelPosition,
    operators::{request_value, Mean, Scale, VvdVolumeSource},
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
mod vulkan;

// TODO look into thiserror/anyhow
type Error = Box<(dyn std::error::Error + 'static)>;

#[derive(Parser)]
struct CliArgs {
    #[arg()]
    vvd_vol: PathBuf,

    #[arg(short, long, default_value = "1.0")]
    factor: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _vulkan_manager = vulkan::VulkanManager::new();

    let args = CliArgs::parse();

    let storage_size = 1 << 30; //One gigabyte
    let storage = Storage::new(storage_size);

    let request_queue = Box::leak(Box::new(runtime::RequestQueue::new()));

    let factor = args.factor;

    let brick_size = VoxelPosition(cgmath::vec3(32, 32, 32));

    let vol = VvdVolumeSource::open(&args.vvd_vol, brick_size)?;

    let scaled1 = Scale {
        vol: &vol,
        factor: &factor,
    };

    let scaled2 = Scale {
        vol: &scaled1,
        factor: &factor,
    };

    let mean = Mean::new(&scaled2);
    let mean_unscaled = Mean::new(&vol);

    let mut rt = runtime::RunTime::new(&storage, &request_queue);

    // TODO: it's slightly annoying that we have to construct the reference here (because of async
    // move). Is there a better way, i.e. to only move some values into the future?
    let mean = &mean;
    let mean_val = rt.resolve(|ctx| async move { Ok(*request_value(mean, ctx).await?) }.into())?;

    let tasks_executed = rt.statistics().tasks_executed;
    println!(
        "Computed scaled mean val: {} ({} tasks)",
        mean_val, tasks_executed
    );

    // Neat: We can even write to references in the closure/future below to get results out.
    let mean_unscaled = &mean_unscaled;
    let mut mean_val_unscaled = 0.0;
    let muv_ref = &mut mean_val_unscaled;
    rt.resolve(|ctx| {
        async move {
            *muv_ref = *request_value(mean_unscaled, ctx).await?;
            Ok(())
        }
        .into()
    })?;
    let tasks_executed = rt.statistics().tasks_executed - tasks_executed;
    println!(
        "Computed unscaled mean val: {} ({} tasks)",
        mean_val_unscaled, tasks_executed
    );

    Ok(())
}
