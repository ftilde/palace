use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use palace_core::data::{LocalVoxelPosition, Vector, VoxelPosition};
use palace_core::event::{
    EventSource, EventStream, Key, MouseButton, OnKeyPress, OnMouseDrag, OnWheelMove,
};
use palace_core::operators::raycaster::{
    CameraState, CompositingMode, RaycasterConfig, Shading, TransFuncOperator,
};
use palace_core::operators::volume::{ChunkSize, EmbeddedVolumeOperator};
use palace_core::operators::volume_gpu;
use palace_core::operators::{self};
use palace_core::runtime::RunTime;
use palace_core::storage::DataVersionType;
use palace_core::vulkan::window::Window;
use winit::event::{Event, WindowEvent};
use winit::platform::run_return::EventLoopExtRunReturn;

use palace_core::array::{self, ImageMetaData, VolumeEmbeddingData};

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

fn open_volume(
    path: PathBuf,
    brick_size_hint: LocalVoxelPosition,
) -> Result<EmbeddedVolumeOperator<f32>, Box<dyn std::error::Error>> {
    let Some(file) = path.file_name() else {
        return Err("No file name in path".into());
    };
    let file = file.to_string_lossy();
    let segments = file.split('.').collect::<Vec<_>>();

    match segments[..] {
        [.., "vvd"] => palace_vvd::open(&path, brick_size_hint),
        [.., "nii"] | [.., "nii", "gz"] => palace_nifti::open_single(path),
        [.., "hdr"] => {
            let data = path.with_extension("img");
            palace_nifti::open_separate(path, data)
        }
        [.., "h5"] => palace_hdf5::open(path, "/volume".to_string()),
        _ => Err(format!("Unknown volume format for file {}", path.to_string_lossy()).into()),
    }
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
        Input::File(path) => open_volume(path.vol, brick_size)?,
        Input::SyntheticCpu(args) => operators::rasterize_function::normalized(
            VoxelPosition::fill(args.size.into()),
            brick_size,
            |v| {
                let r2 = v
                    .map(|v| v - 0.5)
                    .map(|v| v * v)
                    .fold(0.0f32, std::ops::Add::add);
                r2.sqrt()
            },
        )
        .embedded(VolumeEmbeddingData {
            spacing: Vector::fill(1.0),
        }),
        Input::Synthetic(args) => operators::volume_gpu::rasterize(
            array::VolumeMetaData {
                dimensions: VoxelPosition::fill(args.size.into()),
                chunk_size: brick_size,
            },
            r#"{

                vec3 centered = pos_normalized-vec3(0.5);
                vec3 sq = centered*centered;
                float d_sq = sq.x + sq.y + sq.z;
                result = clamp(10*(0.4 - sqrt(d_sq)), 0.0, 1.0);
            }"#,
        )
        .embedded(VolumeEmbeddingData {
            spacing: Vector::fill(1.0),
        }),
    };

    let mut camera_state = CameraState::for_volume(vol.metadata, vol.embedding_data, 30.0);
    let mut scale = 1.0;
    let mut offset: f32 = 0.0;
    let mut stddev = 5.0;

    let mut tf = if let Some(path) = args.transfunc_path {
        palace_vvd::load_tfi(&path).unwrap()
    } else {
        TransFuncOperator::red_ramp(0.0, 1.0)
    };

    let mut event_loop = winit::event_loop::EventLoop::new();

    let mut window = Window::new(&runtime.vulkan, &event_loop).unwrap();
    let mut events = EventSource::default();

    let mut next_timeout = Instant::now() + Duration::from_millis(10);

    event_loop.run_return(|event, _, control_flow| {
        control_flow.set_poll();

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                control_flow.set_exit();
            }
            Event::MainEventsCleared => {
                // Application update code.
                window.inner().request_redraw();
            }
            Event::WindowEvent {
                window_id: _,
                event: winit::event::WindowEvent::Resized(new_size),
            } => {
                window.resize(new_size, &runtime.vulkan);
            }
            Event::WindowEvent {
                window_id: _,
                event,
            } => {
                events.add(event);
            }
            Event::RedrawRequested(_) => {
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in MainEventsCleared, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.
                next_timeout = Instant::now() + Duration::from_millis(10);
                let version = eval_network(
                    &mut runtime,
                    &mut window,
                    vol.clone(),
                    &mut camera_state,
                    &mut scale,
                    &mut offset,
                    &mut stddev,
                    &mut tf,
                    events.current_batch(),
                    next_timeout,
                )
                .unwrap();
                if args.bench && version == DataVersionType::Final {
                    control_flow.set_exit();
                }
            }
            _ => (),
        }
    });

    unsafe { window.deinitialize(&runtime.vulkan) };

    Ok(())
}

fn eval_network(
    runtime: &mut RunTime,
    window: &mut Window,
    vol: EmbeddedVolumeOperator<f32>,
    camera_state: &mut CameraState,
    scale: &mut f32,
    offset: &mut f32,
    stddev: &mut f32,
    tf: &mut TransFuncOperator,
    mut events: EventStream,
    deadline: Instant,
) -> Result<DataVersionType, Box<dyn std::error::Error>> {
    events.act(|c| {
        c.chain(OnKeyPress(Key::Key1, || *scale *= 1.10))
            .chain(OnKeyPress(Key::Key2, || *scale /= 1.10))
            .chain(OnKeyPress(Key::Key3, || *offset += 0.01))
            .chain(OnKeyPress(Key::Key4, || *offset -= 0.01))
            .chain(OnKeyPress(Key::Plus, || *stddev *= 1.10))
            .chain(OnKeyPress(Key::Minus, || *stddev /= 1.10))
            //.chain(OnWheelMove(|delta, _| *fov -= delta))
            .chain(OnWheelMove(|delta, _| {
                camera_state.trackball.move_inout(delta);
            }))
            .chain(OnMouseDrag(MouseButton::Left, |_, delta| {
                camera_state.trackball.pan_around(delta);
            }))
    });

    let vol = vol.map_inner(|vol| {
        //volume_gpu::rechunk(vol.clone(), LocalVoxelPosition::fill(48.into()).into_elem());

        //let kernel = operators::kernels::gauss(*stddev);
        //let after_kernel =
        //    volume_gpu::separable_convolution(vol, Vector::from([&kernel, &kernel, &kernel]));
        //let after_kernel = operators::vesselness::vesselness(vol, *stddev);
        let after_kernel = vol;

        let scaled = volume_gpu::linear_rescale(after_kernel, (*scale).into(), (*offset).into());
        scaled
    });

    //let gen = volume_gpu::rasterize_gpu(
    //    vol.metadata.clone(),
    //    r#"{
    //        vec3 cs = pos_normalized-vec3(0.5);
    //        float d_sq = dot(cs, cs);
    //        result = sqrt(d_sq);
    //        //result = pos_normalized.x;
    //        //vec3 c = abs(cs);
    //        //result = min(min(c.x, c.y), c.z);
    //    }"#,
    //);
    //let vol = bin_ops::sub(vol, gen);

    let vol = palace_core::operators::resample::create_lod(vol, 2.0);
    //let vol = vol.single_level_lod();

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
