use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use palace_core::{
    dim::{DDyn, DynDimension},
    dtypes::{DType, ScalarType},
    operators::{
        procedural,
        rechunk::ChunkSize,
        resample::DownsampleStep,
        tensor::{EmbeddedTensorOperator, LODTensorOperator},
    },
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
    VVD,
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

    /// Convert vector datatype value into outer dimension
    #[arg(long)]
    unfold_dtype: bool,

    /// Rechunk tensor to specified chunk size
    #[arg(long)]
    chunk_size: Option<String>,

    /// Size for procedurally generated volumes
    #[arg(long)]
    size_hint: Option<u32>,

    /// Generate a const chunk table (instead of full tensor) with the specified chunk size
    #[arg(long)]
    gen_const_chunk_table: Option<String>,

    #[arg(long)]
    const_chunk_table_max_diff: Option<f32>,

    #[arg(long)]
    max_parallel_tasks: Option<usize>,

    #[arg(long)]
    max_requests_per_task: Option<usize>,
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
        [.., "vvd"] => Ok(FileFormat::VVD),
        _ => Err(format!("Unknown file format in file {}", path).into()),
    }
}

fn open(
    path: PathBuf,
    hints: palace_io::Hints,
    size_hint: Option<u32>,
) -> Result<(LODTensorOperator<DDyn, DType>, LodOrigin), Box<dyn std::error::Error>> {
    fn gen_md(
        hints: palace_io::Hints,
        size_hint: Option<u32>,
    ) -> Result<palace_core::array::VolumeMetaData, Box<dyn std::error::Error>> {
        let dimensions = palace_core::vec::VoxelPosition::fill(
            size_hint
                .ok_or_else(|| format!("Procedurally generated volumes need a size hint"))?
                .into(),
        );
        Ok(palace_core::array::VolumeMetaData {
            dimensions,
            chunk_size: hints
                .chunk_size
                .map(|v| {
                    v.try_into_static()
                        .unwrap()
                        .zip(&dimensions, |l, r| l.apply(r))
                })
                .unwrap_or(Vector::<palace_core::dim::D3, _>::fill(64.into())),
        })
    }
    let ed = Default::default();

    match path.to_string_lossy().as_ref() {
        "mandelbulb" => Ok((
            procedural::mandelbulb(gen_md(hints, size_hint)?, ed)
                .into_dyn()
                .into(),
            LodOrigin::Existing,
        )),
        "ball" => Ok((
            procedural::ball(gen_md(hints, size_hint)?, ed)
                .into_dyn()
                .into(),
            LodOrigin::Existing,
        )),
        "full" => Ok((
            procedural::full(gen_md(hints, size_hint)?, ed)
                .into_dyn()
                .into(),
            LodOrigin::Existing,
        )),
        _ => palace_io::open_or_create_lod(path, hints),
    }
}

fn main() {
    let args = CliArgs::parse();

    let mut runtime = RunTime::build()
        .disk_cache_size_opt(args.disk_cache_size.map(|v| v.0 as _))
        .devices(args.devices)
        .num_compute_threads_opt(args.compute_pool_size)
        .max_parallel_tasks_opt(args.max_parallel_tasks)
        .max_requests_per_task_opt(args.max_requests_per_task)
        .finish(args.mem_size.0 as _, args.gpu_mem_size.0 as _)
        .unwrap();

    let mut input_hints = palace_io::Hints::default();
    if let Some(location) = args.location {
        input_hints = input_hints.location(location);
    }

    let nd = {
        // open once just to get dimensionality
        let (input_lod, _lod_origin) =
            open(args.input.clone(), input_hints.clone(), args.size_hint).unwrap();
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

    let (mut input_lod, lod_origin) = open(args.input, input_hints, args.size_hint).unwrap();

    if args.unfold_dtype {
        input_lod = input_lod.map(|v| v.unfold_dtype(1.0).unwrap());
    }

    if let Some(dtype) = args.cast_dtype {
        input_lod = input_lod.map(|v| {
            v.map_inner(|v| {
                palace_core::jit::jit(v)
                    //.mul(255.0.into())
                    //.unwrap()
                    .cast(dtype.vec(1))
                    .unwrap()
                    .compile()
                    .unwrap()
            })
        });
    }

    if let Some(chunk_size_str) = args.gen_const_chunk_table {
        let chunk_size = parse_chunk_size(&chunk_size_str);
        let chunk_size = match (chunk_size.dim().n(), nd) {
            (1, n) => Vector::fill_with_len(chunk_size[0], n),
            (l, r) if l == r => chunk_size,
            (_l, _r) => panic!(
                "Invalid chunk specification for tensor of dim {}: {}",
                nd, chunk_size_str
            ),
        };

        input_lod = palace_core::operators::const_chunks::const_chunk_table(
            input_lod.try_into().unwrap(),
            chunk_size,
            args.const_chunk_table_max_diff.unwrap_or(0.0),
        )
        .into();
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
                    let recreate_lod =
                        matches!(lod_origin, LodOrigin::Dynamic) || lod_steps.is_some();
                    write_hints.lod_downsample_steps = lod_steps.map(parse_lod_steps);
                    palace_zarr::save_lod_tensor(
                        &mut runtime,
                        &args.output_path,
                        input_lod,
                        write_hints,
                        recreate_lod,
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
                    let recreate_lod =
                        matches!(lod_origin, LodOrigin::Dynamic) || lod_steps.is_some();
                    write_hints.lod_downsample_steps = lod_steps.map(parse_lod_steps);
                    palace_hdf5::save_lod_tensor(
                        &mut runtime,
                        &args.output_path,
                        input_lod,
                        &write_hints,
                        recreate_lod,
                    )
                }
            }
        }
        FileFormat::VVD => match args.mode {
            FileMode::Single => {
                let v: EmbeddedTensorOperator<palace_core::dim::D3, _> = input
                    .clone()
                    .try_into_static()
                    .ok_or_else(|| format!("vvd format only supports volumes (3D tensors)."))
                    .unwrap();
                let v = &v;
                runtime.resolve(None, false, |ctx, _| {
                    async move { palace_vvd::save_embedded_tensor(ctx, &args.output_path, v).await }
                        .into()
                })
            }

            FileMode::Lod { .. } => {
                panic!("vvd format does not support lod volumes.");
            }
        },
    }
    .unwrap();
}
