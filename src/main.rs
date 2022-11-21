use std::path::PathBuf;

use clap::Parser;

use crate::{
    data::VoxelPosition,
    operator::Network,
    operators::{Mean, Scale, VvdVolumeSource},
    storage::Storage,
    task::DatumRequest,
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

    let mut network = Network::new();

    let brick_size = VoxelPosition(cgmath::vec3(32, 32, 32));

    let vol = network.add(VvdVolumeSource::open(&args.vvd_vol, brick_size)?);

    let factor = network.add(args.factor);

    let scaled1 = network.add(Scale { vol, factor });

    let scaled2 = network.add(Scale {
        vol: scaled1,
        factor,
    });

    let mean = network.add(Mean::new(scaled2));
    let mean_unscaled = network.add(Mean::new(vol));

    let storage_size = 1 << 30; //One gigabyte
    let storage = Storage::new(storage_size);
    let request_queue = runtime::RequestQueue::new();
    let mut rt = runtime::RunTime::new(&network, &storage, &request_queue);

    let mean_val = unsafe { rt.request_blocking::<f32>(mean, DatumRequest::Value)? };

    let tasks_executed = rt.statistics().tasks_executed;
    println!(
        "Computed scaled mean val: {} ({} tasks)",
        mean_val, tasks_executed
    );
    let mean_val_unscaled =
        unsafe { rt.request_blocking::<f32>(mean_unscaled, DatumRequest::Value)? };
    let tasks_executed = rt.statistics().tasks_executed - tasks_executed;
    println!(
        "Computed unscaled mean val: {} ({} tasks)",
        mean_val_unscaled, tasks_executed
    );

    Ok(())
}
