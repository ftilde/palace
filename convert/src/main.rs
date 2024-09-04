use std::path::PathBuf;

use clap::{Parser, Subcommand};
use palace_core::runtime::RunTime;

#[derive(Subcommand, Clone)]
enum Output {
    Zarr,
}

#[derive(Parser)]
struct CliArgs {
    #[arg()]
    input: PathBuf,

    #[command(subcommand)]
    output_type: Output,

    #[arg()]
    output_path: PathBuf,

    /// Size of the memory pool that will be allocated
    #[arg(short, long, default_value = "8G")]
    mem_size: bytesize::ByteSize,

    /// Size of the gpu memory pool that will be allocated
    #[arg(short, long, default_value = "8G")]
    gpu_mem_size: bytesize::ByteSize,

    /// Size of the disk cache that will be allocated
    #[arg(short, long)]
    disk_cache_size: Option<bytesize::ByteSize>,

    /// Force a specific size for the compute task pool [default: number of cores]
    #[arg(short, long)]
    compute_pool_size: Option<usize>,

    /// Use the vulkan device with the specified id
    #[arg(long, default_value = "0")]
    device: usize,
}

fn main() {
    let args = CliArgs::parse();

    let storage_size = args.mem_size.0 as _;
    let gpu_storage_size = args.gpu_mem_size.0 as _;
    let disk_cache_size = args.disk_cache_size.map(|v| v.0 as _);

    let mut runtime = RunTime::new(
        storage_size,
        gpu_storage_size,
        args.compute_pool_size,
        disk_cache_size,
        None,
        Some(args.device),
    )
    .unwrap();

    let input = palace_volume::open(args.input, palace_volume::Hints::default()).unwrap();
    let input = input.inner.into_dyn();
    let input = &input;

    runtime
        .resolve(None, false, |ctx, _| {
            async move {
                match args.output_type {
                    Output::Zarr => palace_zarr::save(ctx, &args.output_path, input).await,
                }
            }
            .into()
        })
        .unwrap();
}
