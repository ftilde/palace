use std::path::PathBuf;

use clap::Parser;

use crate::{
    data::{Storage, VoxelPosition},
    operator::Network,
    operators::{Mean, Scale, VvdVolumeSource},
    task::DatumRequest,
};

mod array;
mod data;
mod id;
mod operator;
mod operators;
mod runtime;
mod task;
mod vulkan;

// TODO look into thiserror/anyhow
type Error = Box<(dyn std::error::Error + 'static)>;

#[derive(Parser)]
struct CliArgs {
    #[arg()]
    raw_vol: PathBuf,

    #[arg(short, long, default_value = "1.0")]
    factor: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _vulkan_manager = vulkan::VulkanManager::new();
    
    let args = CliArgs::parse();

    let mut network = Network::new();

    let brick_size = VoxelPosition(cgmath::vec3(32, 32, 32));

    let vol = network.add(VvdVolumeSource::open(&args.raw_vol, brick_size)?);

    let factor = network.add(args.factor);

    let scaled1 = network.add(Scale { vol, factor });

    let scaled2 = network.add(Scale {
        vol: scaled1,
        factor,
    });

    let mean = network.add(Mean::new(scaled2));
    let mean_unscaled = network.add(Mean::new(vol));

    let storage = Storage::new();
    let request_queue = runtime::RequestQueue::new();
    let mut rt = runtime::RunTime::new(&network, &storage, &request_queue);

    let mean_val = rt.request_blocking(mean, DatumRequest::Value)?.float()?;

    let tasks_executed = rt.statistics().tasks_executed;
    println!(
        "Computed scaled mean val: {} ({} tasks)",
        mean_val, tasks_executed
    );
    let mean_val_unscaled = rt
        .request_blocking(mean_unscaled, DatumRequest::Value)?
        .float()?;
    let tasks_executed = rt.statistics().tasks_executed - tasks_executed;
    println!(
        "Computed unscaled mean val: {} ({} tasks)",
        mean_val_unscaled, tasks_executed
    );

    Ok(())
}
