use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use vng_core::data::{LocalVoxelPosition, Vector, VoxelPosition};
use vng_core::event::{
    EventSource, EventStream, Key, MouseButton, OnKeyPress, OnMouseDrag, OnWheelMove,
};
use vng_core::operators::gui::{egui, GuiState};
use vng_core::operators::volume::{ChunkSize, VolumeOperator};
use vng_core::operators::{self, volume::VolumeOperatorState};
use vng_core::operators::{scalar, volume_gpu};
use vng_core::runtime::RunTime;
use vng_core::storage::DataVersionType;
use vng_core::vulkan::window::Window;
//use vng_hdf5::Hdf5VolumeSourceState;
use vng_nifti::NiftiVolumeSourceState;
use vng_vvd::VvdVolumeSourceState;
use winit::event::{Event, WindowEvent};
use winit::platform::run_return::EventLoopExtRunReturn;

use vng_core::array::{self, ImageMetaData};

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
) -> Result<Box<dyn VolumeOperatorState>, Box<dyn std::error::Error>> {
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
        Input::SyntheticCpu(args) => Box::new(operators::rasterize_function::normalized(
            VoxelPosition::fill(args.size.into()),
            brick_size,
            |v| {
                let r2 = v
                    .map(|v| v - 0.5)
                    .map(|v| v * v)
                    .fold(0.0f32, std::ops::Add::add);
                r2.sqrt()
            },
        )),
        Input::Synthetic(args) => Box::new(operators::volume_gpu::VoxelRasterizerGLSL {
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
        }),
    };

    let mut angle: f32 = 0.0;
    let mut slice_num = 0;
    let mut slice_offset = [0.0, 0.0].into();
    let mut slice_zoom_level = 1.0;
    let mut scale = 1.0;
    let mut offset: f32 = 0.0;
    let mut stddev: f32 = 5.0;
    let mut gui = GuiState::default();

    let mut event_loop = EventLoop::new();

    let mut window = Window::new(&runtime.vulkan, &event_loop).unwrap();
    let mut events = EventSource::default();

    let mut next_timeout = Instant::now() + Duration::from_millis(10);

    event_loop.run_return(|event, _, control_flow| {
        control_flow.set_wait();

        let Some(event) = event.to_static() else {
            return;
        };
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
                    &mut angle,
                    &mut slice_num,
                    &mut slice_offset,
                    &mut slice_zoom_level,
                    &mut scale,
                    &mut offset,
                    &mut stddev,
                    &mut gui,
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

pub type EventLoop<T> = winit::event_loop::EventLoop<T>;

fn slice_viewer_z<'op>(
    slice_input: VolumeOperator<'op>,
    md: ImageMetaData,
    slice_num: &mut i32,
    offset: &mut Vector<2, f32>,
    zoom_level: &mut f32,
    events: &mut EventStream,
) -> VolumeOperator<'op> {
    events.act(|c| {
        c.chain(offset.drag(MouseButton::Left))
            .chain(OnMouseDrag(MouseButton::Right, |_pos, delta| {
                *slice_num += delta.y();
            }))
            //.chain(OnWheelMove(|delta| *slice_num += delta as i32))
            .chain(OnWheelMove(|delta, state| {
                if let Some(state) = &state.mouse_state {
                    let zoom_change = (-delta * 0.05).exp();
                    *zoom_level *= zoom_change;

                    let pos = state.pos.map(|v| v as f32);

                    *offset = (*offset - pos) / Vector::fill(zoom_change) + pos;
                }
            }))
    });

    let md = ImageMetaData {
        dimensions: md.dimensions,
        chunk_size: Vector::fill(512.into()),
    };

    let slice_num_g = ((*slice_num).max(0) as u32).into();
    let slice_proj_z = crate::operators::sliceviewer::slice_projection_mat_z(
        slice_input.metadata.clone(),
        crate::operators::scalar::constant_hash(md),
        crate::operators::scalar::constant_hash(slice_num_g),
        crate::operators::scalar::constant_pod(*offset),
        crate::operators::scalar::constant_pod(*zoom_level),
    );
    let slice = crate::operators::sliceviewer::render_slice(
        slice_input,
        crate::operators::scalar::constant_hash(md),
        slice_proj_z,
    );
    let slice = volume_gpu::rechunk(slice, Vector::fill(ChunkSize::Full));

    slice
}

fn slice_viewer_rot<'op>(
    slice_input: VolumeOperator<'op>,
    md: ImageMetaData,
    angle: &mut f32,
    mut events: EventStream,
) -> VolumeOperator<'op> {
    events.act(|c| {
        c.chain(OnMouseDrag(MouseButton::Right, |_pos, delta| {
            *angle += delta.x() as f32 * 0.01;
        }))
        .chain(OnWheelMove(|delta, _| *angle += delta * 0.05))
    });

    let md = ImageMetaData {
        dimensions: md.dimensions,
        chunk_size: Vector::fill(512.into()),
    };

    let slice_proj_rot = crate::operators::sliceviewer::slice_projection_mat_centered_rotate(
        slice_input.metadata.clone(),
        crate::operators::scalar::constant_hash(md),
        crate::operators::scalar::constant_pod(*angle),
    );
    let slice = crate::operators::sliceviewer::render_slice(
        slice_input,
        crate::operators::scalar::constant_hash(md),
        slice_proj_rot,
    );
    let slice = volume_gpu::rechunk(slice, Vector::fill(ChunkSize::Full));
    slice
}

fn eval_network(
    runtime: &mut RunTime,
    window: &mut Window,
    vol: &dyn VolumeOperatorState,
    angle: &mut f32,
    slice_num: &mut i32,
    slice_offset: &mut Vector<2, f32>,
    slice_zoom_level: &mut f32,
    scale: &mut f32,
    offset: &mut f32,
    stddev: &mut f32,
    gui: &mut GuiState,
    mut events: EventStream,
    deadline: Instant,
) -> Result<DataVersionType, Box<dyn std::error::Error>> {
    events.act(|c| {
        c.chain(OnKeyPress(Key::Key9, || *slice_num += 1))
            .chain(OnKeyPress(Key::Key0, || *slice_num -= 1))
            .chain(OnKeyPress(Key::Key1, || *scale *= 1.10))
            .chain(OnKeyPress(Key::Key2, || *scale /= 1.10))
            .chain(OnKeyPress(Key::Key3, || *offset += 0.01))
            .chain(OnKeyPress(Key::Key4, || *offset -= 0.01))
            .chain(OnKeyPress(Key::Plus, || *stddev *= 1.10))
            .chain(OnKeyPress(Key::Minus, || *stddev /= 1.10))
    });

    let mut splitter = operators::splitter::Splitter::new(window.size(), 0.5);

    let (mut events_l, events_r) = splitter.split_events(&mut events);

    let gui = gui.setup(&mut events_l, |ctx| {
        egui::CentralPanel::default().show(&ctx, |ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label("Slice: ");
                    if ui.button("-").clicked() {
                        *slice_num -= 1;
                    }
                    ui.label(slice_num.to_string());
                    if ui.button("+").clicked() {
                        *slice_num += 1;
                    }
                });
                ui.add(egui::Slider::new(scale, 0.1..=10.0).text("Scale"));
                ui.add(egui::Slider::new(offset, -10.0..=10.0).text("Offset"));
            });
        });
        //dbg!(egui::Window::new("My Window").show(ctx, |ui| {
        //    ui.label("Hello World!");
        //}));
    });

    let vol = vol.operate();

    let vol = volume_gpu::rechunk(vol, LocalVoxelPosition::fill(48.into()).into_elem());

    let after_kernel = operators::vesselness::multiscale_vesselness(
        vol,
        scalar::constant_pod(3.0),
        scalar::constant_pod(*stddev),
        3,
    );
    //let after_kernel = operators::vesselness::vesselness(vol, scalar::constant_pod(*stddev));
    let scaled = volume_gpu::linear_rescale(after_kernel, (*scale).into(), (*offset).into());

    let left = slice_viewer_z(
        scaled.clone(),
        splitter.metadata_l(),
        slice_num,
        slice_offset,
        slice_zoom_level,
        &mut events_l,
    );

    let left = gui.render(left);
    let right = slice_viewer_rot(scaled, splitter.metadata_r(), angle, events_r);
    let frame = splitter.operate(left, right);

    let mut c = runtime.context_anchor();
    let mut executor = c.executor(Some(deadline));

    let slice_ref = &frame;
    let version =
        executor.resolve(|ctx| async move { window.render(ctx, slice_ref).await }.into())?;
    //let tasks_executed = executor.statistics().tasks_executed;
    //println!("Rendering done ({} tasks)", tasks_executed);

    Ok(version)
}
