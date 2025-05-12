use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use palace_core::{
    dim::{DDyn, DynDimension},
    dtypes::ScalarType,
    operators::{rechunk::ChunkSize, resample::DownsampleStep},
    runtime::RunTime,
    vec::Vector,
};
use palace_io::LodOrigin;

#[derive(Subcommand, Clone)]
enum FileMode {
    Single,
    Lod {
        /// Use the vulkan device with the specified id
        #[arg(long)]
        lod_steps: Option<String>,
    },
}

#[derive(Clone)]
enum FileFormat {
    Zarr,
    HDF5,
}

#[derive(Parser)]
struct CliArgs {
    #[arg()]
    input: PathBuf,

    #[command(subcommand)]
    mode: FileMode,

    #[arg()]
    output_path: PathBuf,

    // Location hint for input volume
    #[arg(short, long)]
    location: Option<String>,

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

    /// Use the vulkan devices with the specified ids
    #[arg(long, value_delimiter = ',', num_args=1..)]
    devices: Vec<usize>,

    /// Use the vulkan device with the specified id
    #[arg(short, long, default_value = "1")]
    compression_level: i32,

    /// Cast to specified scalar data type before saving
    #[arg(long, value_parser(ScalarType::try_from_pretty))]
    cast_dtype: Option<ScalarType>,

    /// Rechunk tensor to specified chunk size
    #[arg(long)]
    chunk_size: Option<String>,
}

fn parse_lod_steps(s: String) -> Vector<DDyn, DownsampleStep> {
    let vec = s
        .split(",")
        .map(|v| {
            let mut c = v.chars();
            match c.next().unwrap() {
                'i' => DownsampleStep::Ignore,
                'f' => DownsampleStep::Fixed(c.as_str().parse::<f32>().unwrap()),
                _ => DownsampleStep::Synchronized(v.parse::<f32>().unwrap()),
            }
        })
        .collect::<Vec<_>>();
    Vector::<DDyn, _>::new(vec)
}

fn parse_chunk_size(s: &str) -> Vector<DDyn, ChunkSize> {
    let vec = s
        .split(",")
        .map(|v| match v {
            "full" => ChunkSize::Full,
            o => ChunkSize::Fixed(o.parse::<u32>().unwrap().into()),
        })
        .collect::<Vec<_>>();
    Vector::<DDyn, _>::new(vec)
}

fn parse_file_type(path: &Path) -> Result<FileFormat, String> {
    let path = path.to_string_lossy();
    let segments = path.split('.').collect::<Vec<_>>();

    match segments[..] {
        [.., "zarr"] | [.., "zarr", "zip"] => Ok(FileFormat::Zarr),
        [.., "h5"] | [.., "hdf5"] => Ok(FileFormat::HDF5),
        _ => Err(format!("Unknown file format in file {}", path).into()),
    }
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
        args.devices,
    )
    .unwrap();

    let mut input_hints = palace_io::Hints::default();
    if let Some(location) = args.location {
        input_hints = input_hints.location(location);
    }

    let nd = {
        // open once just to get dimensionality
        let (input_lod, _lod_origin) =
            palace_io::open_or_create_lod(args.input.clone(), input_hints.clone()).unwrap();
        input_lod.levels[0].metadata.dim().n()
    };

    let chunk_size = args.chunk_size.map(|chunk_size_str| {
        let chunk_size = parse_chunk_size(&chunk_size_str);
        match (chunk_size.dim().n(), nd) {
            (1, n) => Vector::fill_with_len(chunk_size[0], n),
            (l, r) if l == r => chunk_size,
            (_l, _r) => panic!(
                "Invalid chunk specification for tensor of dim {}: {}",
                nd, chunk_size_str
            ),
        }
    });

    if let Some(chunk_size) = chunk_size {
        input_hints = input_hints.chunk_size(chunk_size.clone());
        input_hints = input_hints.rechunk(true);
    }

    let (mut input_lod, lod_origin) =
        palace_io::open_or_create_lod(args.input, input_hints).unwrap();

    if let Some(dtype) = args.cast_dtype {
        input_lod = input_lod.map(|v| {
            v.map_inner(|v| {
                palace_core::jit::jit(v)
                    .cast(dtype.vec(1))
                    .unwrap()
                    .compile()
                    .unwrap()
            })
        });
    }

    let input = input_lod.levels[0].clone().into_dyn();

    let input = &input;
    let input_lod = &input_lod;

    let format = parse_file_type(&args.output_path).unwrap();

    match format {
        FileFormat::Zarr => {
            let mut write_hints = palace_zarr::WriteHints {
                compression_level: args.compression_level,
                lod_downsample_steps: None,
            };

            match args.mode {
                FileMode::Single => runtime.resolve(None, false, |ctx, _| {
                    async move {
                        palace_zarr::save_embedded_tensor(
                            ctx,
                            &args.output_path,
                            input,
                            write_hints,
                        )
                        .await
                    }
                    .into()
                }),

                FileMode::Lod { lod_steps } => {
                    write_hints.lod_downsample_steps = lod_steps.map(parse_lod_steps);
                    palace_zarr::save_lod_tensor(
                        &mut runtime,
                        &args.output_path,
                        input_lod,
                        write_hints,
                        matches!(lod_origin, LodOrigin::Dynamic),
                    )
                }
            }
        }
        FileFormat::HDF5 => {
            let mut write_hints = palace_hdf5::WriteHints {
                compression_level: args.compression_level.try_into().unwrap(),
                lod_downsample_steps: None,
            };

            match args.mode {
                FileMode::Single => runtime.resolve(None, false, |ctx, _| {
                    async move {
                        palace_hdf5::save_embedded_tensor(
                            ctx,
                            &args.output_path,
                            input,
                            &write_hints,
                        )
                        .await
                    }
                    .into()
                }),

                FileMode::Lod { lod_steps } => {
                    write_hints.lod_downsample_steps = lod_steps.map(parse_lod_steps);
                    palace_hdf5::save_lod_tensor(
                        &mut runtime,
                        &args.output_path,
                        input_lod,
                        &write_hints,
                        matches!(lod_origin, LodOrigin::Dynamic),
                    )
                }
            }
        }
    }
    .unwrap();
}
