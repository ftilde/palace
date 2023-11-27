use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use vng_core::data::{LocalVoxelPosition, Matrix, Vector, VoxelPosition};
use vng_core::event::{
    EventSource, EventStream, Key, MouseButton, OnKeyPress, OnMouseDrag, OnWheelMove,
};
use vng_core::operators::volume::{ChunkSize, EmbeddedVolumeOperatorState};
use vng_core::operators::volume_gpu;
use vng_core::operators::{self};
use vng_core::runtime::RunTime;
use vng_core::storage::DataVersionType;
use vng_core::vulkan::window::Window;
//use vng_hdf5::Hdf5VolumeSourceState;
use vng_nifti::NiftiVolumeSourceState;
use vng_vvd::VvdVolumeSourceState;
use winit::event::{Event, WindowEvent};
use winit::platform::run_return::EventLoopExtRunReturn;

use vng_core::array::{self, ImageMetaData, VolumeEmbeddingData};

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

    /// Size of the memory pool that will be allocated in gigabytes.
    #[arg(short, long, default_value = "8")]
    mem_size: usize,

    /// Size of the gpu memory pool that will be allocated in gigabytes.
    #[arg(short, long)]
    gpu_mem_size: Option<u64>,

    /// Force a specific size for the compute task pool [default: number of cores]
    #[arg(short, long)]
    compute_pool_size: Option<usize>,
}

fn open_volume(
    path: PathBuf,
    brick_size_hint: LocalVoxelPosition,
) -> Result<Box<dyn EmbeddedVolumeOperatorState>, Box<dyn std::error::Error>> {
    let Some(file) = path.file_name() else {
        return Err("No file name in path".into());
    };
    let file = file.to_string_lossy();
    let segments = file.split('.').collect::<Vec<_>>();

    Ok(match segments[..] {
        [.., "vvd"] => Box::new(VvdVolumeSourceState::open(&path, brick_size_hint)?),
        [.., "nii"] | [.., "nii", "gz"] => Box::new(NiftiVolumeSourceState::open_single(path)?),
        [.., "hdr"] => {
            let data = path.with_extension("img");
            Box::new(NiftiVolumeSourceState::open_separate(path, data)?)
        }
        //[.., "h5"] => Box::new(Hdf5VolumeSourceState::open(path, "/volume".to_string())?),
        _ => {
            return Err(format!("Unknown volume format for file {}", path.to_string_lossy()).into())
        }
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse();

    let storage_size = args.mem_size << 30; //in gigabyte
    let gpu_storage_size = args.gpu_mem_size.map(|s| s << 30); // also in gigabyte

    let mut runtime = RunTime::new(storage_size, gpu_storage_size, args.compute_pool_size)?;

    let brick_size = LocalVoxelPosition::fill(64.into());

    let vol_state = match args.input {
        Input::File(path) => open_volume(path.vol, brick_size)?,
        Input::SyntheticCpu(args) => Box::new((
            operators::rasterize_function::normalized(
                VoxelPosition::fill(args.size.into()),
                brick_size,
                |v| {
                    let r2 = v
                        .map(|v| v - 0.5)
                        .map(|v| v * v)
                        .fold(0.0f32, std::ops::Add::add);
                    r2.sqrt()
                },
            ),
            VolumeEmbeddingData {
                spacing: Vector::fill(1.0),
            },
        )),
        Input::Synthetic(args) => Box::new((
            operators::volume_gpu::VoxelRasterizerGLSL {
                metadata: array::VolumeMetaData {
                    dimensions: VoxelPosition::fill(args.size.into()),
                    chunk_size: brick_size,
                },
                body: r#"{

                vec3 centered = pos_normalized-vec3(0.5);
                vec3 sq = centered*centered;
                float d_sq = sq.x + sq.y + sq.z;
                result = sqrt(d_sq) * 0.5 + (centered.x*centered.x - abs(centered.z))*0.5;

            }"#
                .to_owned(),
            },
            VolumeEmbeddingData {
                spacing: Vector::fill(1.0),
            },
        )),
    };

    let mut fov: f32 = 30.0;
    let mut eye = [5.0, 0.0, 0.0].into();
    let mut center = [0.0, 0.0, 0.0].into();
    let mut up = [1.0, 1.0, 0.0].into();
    let mut scale = 1.0;
    let mut offset: f32 = 0.0;
    let mut stddev = 1.0;

    let mut event_loop = winit::event_loop::EventLoop::new();

    let mut window = Window::new(&runtime.vulkan, &event_loop).unwrap();
    let mut events = EventSource::default();

    let mut next_timeout = Instant::now() + Duration::from_millis(10);

    event_loop.run_return(|event, _, control_flow| {
        control_flow.set_wait();

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
                    &*vol_state,
                    &mut fov,
                    &mut eye,
                    &mut center,
                    &mut up,
                    &mut scale,
                    &mut offset,
                    &mut stddev,
                    events.current_batch(),
                    next_timeout,
                )
                .unwrap();
                if version == DataVersionType::Final {
                    //control_flow.set_exit();
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
    vol: &dyn EmbeddedVolumeOperatorState,
    fov: &mut f32,
    eye: &mut Vector<3, f32>,
    center: &mut Vector<3, f32>,
    up: &mut Vector<3, f32>,
    scale: &mut f32,
    offset: &mut f32,
    stddev: &mut f32,
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
                let look = *center - *eye;
                let new_look = look.scale(1.0 - delta * 0.1);
                *eye = *center - new_look;
            }))
            .chain(OnMouseDrag(MouseButton::Left, |_, delta| {
                let look = *center - *eye;
                let look_len = look.length();
                let left = up.cross(look).normalized();
                let move_factor = 0.005;
                let delta = delta.map(|v| v as f32 * move_factor);

                let new_look = (look.normalized() + up.scale(delta.y()) + left.scale(-delta.x()))
                    .normalized()
                    .scale(look_len);

                *eye = *center - new_look;
                let left = up.cross(new_look);
                *up = new_look.cross(left).normalized();
            }))
    });

    let vol = vol.operate();

    let vol = vol.map_inner(|vol| {
        volume_gpu::rechunk(vol.clone(), LocalVoxelPosition::fill(48.into()).into_elem());

        //let kernel = operators::kernels::gauss(scalar::constant_pod(*stddev));
        //let after_kernel =
        //    volume_gpu::separable_convolution(vol, [kernel.clone(), kernel.clone(), kernel]);
        let after_kernel = operators::vesselness::vesselness(vol, *stddev);

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

    let vol = vng_core::operators::resample::create_lod(vol, 2.0);

    let md = ImageMetaData {
        dimensions: window.size(),
        //chunk_size: window.size().local(),
        chunk_size: Vector::fill(512.into()),
    };

    let perspective: Matrix<4, f32> = cgmath::perspective(
        cgmath::Deg(*fov),
        md.dimensions.x().raw as f32 / md.dimensions.y().raw as f32,
        0.001, //TODO:
        100.0,
    )
    .into();
    let look_at: Matrix<4, f32> =
        cgmath::Matrix4::look_at_rh((*eye).into(), (*center).into(), (*up).into()).into();
    let matrix = perspective * look_at;
    let eep = vng_core::operators::raycaster::entry_exit_points(
        vol.fine_metadata(),
        vol.fine_embedding_data(),
        md.into(),
        matrix.into(),
    );
    let frame = vng_core::operators::raycaster::raycast(vol, eep);
    let frame = volume_gpu::rechunk(frame, Vector::fill(ChunkSize::Full));

    let slice_ref = &frame;
    let version = runtime.resolve(Some(deadline), |ctx, _| {
        async move { window.render(ctx, slice_ref).await }.into()
    })?;
    //let tasks_executed = executor.statistics().tasks_executed;
    //println!("Rendering done ({} tasks)", tasks_executed);

    Ok(version)
}
