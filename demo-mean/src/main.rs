use std::path::PathBuf;

use clap::{Parser, Subcommand};
use palace_core::array;
use palace_core::data::{LocalVoxelPosition, VoxelPosition};
use palace_core::dim::DDyn;
use palace_core::dtypes::{ScalarType, StaticElementType};
use palace_core::operators;
use palace_core::operators::rechunk::ChunkSize;
use palace_core::operators::tensor::TensorOperator;
use palace_core::runtime::RunTime;
use palace_io::Hints;

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
    SyntheticCpu(SyntheticArgs),
}

#[derive(Parser)]
struct CliArgs {
    #[command(subcommand)]
    input: Input,

    #[arg(short, long, default_value = "0")]
    slice_num: u32,

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

    /// Use the vulkan devices with the specified ids
    #[arg(long, value_delimiter = ',', num_args=1..)]
    devices: Vec<usize>,

    /// Number of chunks to sample. [default: all]
    #[arg(long)]
    sample_chunks: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        args.devices,
    )?;

    let chunk_size = LocalVoxelPosition::fill(64.into());
    let chunk_size_hint = chunk_size.map(ChunkSize::Fixed).into_dyn();

    let vol = match args.input {
        Input::File(path) => {
            let v = palace_io::open(path.vol, Hints::new().chunk_size(chunk_size_hint))?.inner;
            palace_core::jit::jit(v)
                .cast(ScalarType::F32.into())
                .unwrap()
                .compile()
                .unwrap()
        }
        Input::SyntheticCpu(args) => operators::rasterize_function::normalized(
            VoxelPosition::fill(args.size.into()),
            chunk_size,
            |v| {
                let r2 = v
                    .map(|v| v - 0.5)
                    .map(|v| v * v)
                    .fold(0.0f32, std::ops::Add::add);
                r2.sqrt()
            },
        )
        .into_dyn()
        .into(),
        Input::Synthetic(args) => operators::procedural::rasterize(
            array::VolumeMetaData {
                dimensions: VoxelPosition::fill(args.size.into()),
                chunk_size,
            },
            r#"float run(float[3] p, uint[3] pos_voxel) {
                vec3 centered = vec3(p[2], p[1], p[0])-vec3(0.5);
                vec3 sq = centered*centered;
                float d_sq = sq.x + sq.y + sq.z;
                return sqrt(d_sq);
            }"#,
        )
        .into_dyn()
        .into(),
    };

    let vol = vol.try_into().unwrap();

    eval_network(&mut runtime, vol, args.sample_chunks)
}

fn eval_network(
    runtime: &mut RunTime,
    vol: TensorOperator<DDyn, StaticElementType<f32>>,
    num_chunks: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    //let vol = palace_core::jit::jit(vol.into())
    //    .add(10.0.into())
    //    .unwrap()
    //    .compile()
    //    .unwrap()
    //    .try_into()
    //    .unwrap();

    let mean_unscaled = operators::aggregation::mean(vol, num_chunks.into());

    // Neat: We can even write to references in the closure/future below to get results out.
    let mean_unscaled_ref = &mean_unscaled;
    let mut mean_val_unscaled = 0.0;
    let muv_ref = &mut mean_val_unscaled;
    runtime.resolve(None, false, |ctx, _| {
        async move {
            let req = mean_unscaled_ref.request_scalar();
            *muv_ref = ctx.submit(req).await;
            Ok(())
        }
        .into()
    })?;

    println!("Computed mean: {}", mean_val_unscaled);

    Ok(())
}
