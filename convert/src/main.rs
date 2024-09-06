use std::path::PathBuf;

use clap::{Parser, Subcommand};
use palace_core::runtime::RunTime;
use palace_volume::LodOrigin;
use palace_zarr::WriteHints;

#[derive(Subcommand, Clone)]
enum Output {
    Zarr,
    ZarrLod,
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
    #[arg(long)]
    compute_pool_size: Option<usize>,

    /// Use the vulkan device with the specified id
    #[arg(long, default_value = "0")]
    device: usize,

    /// Use the vulkan device with the specified id
    #[arg(short, long, default_value = "1")]
    compression_level: i32,
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

    let (input_lod, lod_origin) =
        palace_volume::open_or_create_lod(args.input, palace_volume::Hints::default()).unwrap();
    let input_lod = input_lod.into_dyn();

    let input = input_lod.levels[0].clone().into_dyn();

    let input = &input;
    let input_lod = &input_lod;

    let write_hints = WriteHints {
        compression_level: args.compression_level,
    };

    match args.output_type {
        Output::Zarr => runtime.resolve(None, false, |ctx, _| {
            async move {
                palace_zarr::save_embedded_tensor(ctx, &args.output_path, input, write_hints).await
            }
            .into()
        }),
        Output::ZarrLod => palace_zarr::save_lod_tensor(
            &mut runtime,
            &args.output_path,
            input_lod,
            write_hints,
            matches!(lod_origin, LodOrigin::Dynamic),
        ),
    }
    .unwrap();
}
