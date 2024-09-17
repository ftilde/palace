use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, Subcommand};
use palace_core::data::{LocalVoxelPosition, Vector, VoxelPosition};
use palace_core::dtypes::StaticElementType;
use palace_core::event::{EventStream, Key, MouseButton, OnKeyPress, OnMouseDrag, OnWheelMove};
use palace_core::jit::jit;
use palace_core::operators::raycaster::{
    CameraState, CompositingMode, RaycasterConfig, Shading, TransFuncOperator,
};
use palace_core::operators::volume::{ChunkSize, LODVolumeOperator};
use palace_core::operators::volume_gpu;
use palace_core::operators::{self};
use palace_core::runtime::{Deadline, RunTime};
use palace_core::storage::DataVersionType;
use palace_core::vulkan::window::Window;

use palace_core::array::{self, ImageMetaData};

#[derive(Subcommand, Clone)]
enum Type {
    Ball,
    Full,
    Mandelbulb,
}

#[derive(Parser, Clone)]
struct SyntheticArgs {
    #[arg()]
    size: u32,
    #[command(subcommand)]
    scenario: Type,
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
}

#[derive(Parser)]
struct CliArgs {
    #[command(subcommand)]
    input: Input,

    /// Size of the memory pool that will be allocated
    #[arg(short, long, default_value = "8G")]
    mem_size: bytesize::ByteSize,

    /// Size of the gpu memory pool that will be allocated
    #[arg(short, long, default_value = "8G")]
    gpu_mem_size: bytesize::ByteSize,

    /// Size of the disk cache that will be allocated
    #[arg(short, long)]
    disk_cache_size: Option<bytesize::ByteSize>,

    /// Stop after rendering a complete frame
    #[arg(short, long)]
    bench: bool,

    /// Force a specific size for the compute task pool [default: number of cores]
    #[arg(short, long)]
    compute_pool_size: Option<usize>,

    /// Transfer function (voreen .tfi file)
    #[arg(short, long)]
    transfunc_path: Option<PathBuf>,

    /// Use the vulkan device with the specified id
    #[arg(long, default_value = "0")]
    device: usize,
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
        Some(args.device),
    )?;

    let brick_size = LocalVoxelPosition::fill(64.into());

    let vol = match args.input {
        Input::File(path) => {
            let base =
                palace_volume::open(path.vol, palace_volume::Hints::new().brick_size(brick_size))?;
            palace_core::operators::resample::create_lod(base.try_into()?, 2.0)
        }
        Input::Synthetic(args) => {
            let md = array::VolumeMetaData {
                dimensions: VoxelPosition::fill(args.size.into()),
                chunk_size: brick_size,
            };
            match args.scenario {
                Type::Ball => operators::procedural::ball(md),
                Type::Full => operators::procedural::full(md),
                Type::Mandelbulb => operators::procedural::mandelbulb(md),
            }
        }
    };

    let mut camera_state =
        CameraState::for_volume(vol.fine_metadata(), vol.fine_embedding_data(), 30.0);
    let mut scale = 1.0;
    let mut offset: f32 = 0.0;
    let mut stddev = 5.0;

    let mut tf = if let Some(path) = args.transfunc_path {
        palace_vvd::load_tfi(&path).unwrap()
    } else {
        TransFuncOperator::red_ramp(0.0, 1.0)
    };

    palace_winit::run_with_window(
        &mut runtime,
        Duration::from_millis(10),
        |_event_loop, window, rt, events, timeout| {
            let version = eval_network(
                rt,
                window,
                vol.clone(),
                &mut camera_state,
                &mut scale,
                &mut offset,
                &mut stddev,
                &mut tf,
                events,
                timeout,
            )
            .unwrap();
            if args.bench && version == DataVersionType::Final {
                _event_loop.exit();
            }
            Ok(version)
        },
    )
}

fn eval_network(
    runtime: &mut RunTime,
    window: &mut Window,
    vol: LODVolumeOperator<StaticElementType<f32>>,
    camera_state: &mut CameraState,
    scale: &mut f32,
    offset: &mut f32,
    stddev: &mut f32,
    tf: &mut TransFuncOperator,
    mut events: EventStream,
    deadline: Deadline,
) -> Result<DataVersionType, Box<dyn std::error::Error>> {
    events.act(|c| {
        c.chain(OnKeyPress(Key::Digit1, || *scale *= 1.10))
            .chain(OnKeyPress(Key::Digit2, || *scale /= 1.10))
            .chain(OnKeyPress(Key::Digit3, || *offset += 0.01))
            .chain(OnKeyPress(Key::Digit4, || *offset -= 0.01))
            .chain(OnKeyPress(Key::Equal, || *stddev *= 1.10))
            .chain(OnKeyPress(Key::Minus, || *stddev /= 1.10))
            //.chain(OnWheelMove(|delta, _| *fov -= delta))
            .chain(OnWheelMove(|delta, _| {
                camera_state.trackball.move_inout(delta);
            }))
            .chain(OnMouseDrag(MouseButton::Left, |_, delta| {
                camera_state.trackball.pan_around(delta);
            }))
    });

    let vol = vol.map(|vol| {
        vol.map_inner(|vol| {
            //volume_gpu::rechunk(vol.clone(), LocalVoxelPosition::fill(48.into()).into_elem());

            //let kernel = operators::kernels::gauss(*stddev);
            //let after_kernel =
            //    volume_gpu::separable_convolution(vol, Vector::from([&kernel, &kernel, &kernel]));
            //let after_kernel = operators::vesselness::vesselness(vol, *stddev);
            let after_kernel = vol;

            let scaled = jit(after_kernel.into())
                .mul((*scale).into())
                .unwrap()
                .add((*offset).into())
                .unwrap()
                .compile()
                .unwrap()
                .try_into()
                .unwrap();
            scaled
        })
    });

    let md = ImageMetaData {
        dimensions: window.size(),
        chunk_size: window.size().local(),
        //chunk_size: Vector::fill(512.into()),
    };

    let matrix = camera_state.projection_mat(md.dimensions);
    let eep = palace_core::operators::raycaster::entry_exit_points(
        vol.fine_metadata(),
        vol.fine_embedding_data(),
        md.into(),
        matrix.into(),
    );
    let mut config = RaycasterConfig::default();
    config.compositing_mode = CompositingMode::DVR;
    config.shading = Shading::Phong;
    let frame = palace_core::operators::raycaster::raycast(vol, eep, tf.clone(), config);
    let frame = volume_gpu::rechunk(frame, Vector::fill(ChunkSize::Full));

    let slice_ref = &frame;
    let version = runtime.resolve(Some(deadline), false, |ctx, _| {
        async move { window.render(ctx, slice_ref).await }.into()
    })?;
    //let tasks_executed = executor.statistics().tasks_executed;
    //println!("Rendering done ({} tasks)", tasks_executed);

    Ok(version)
}
