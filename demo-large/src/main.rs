use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use vng_core::cgmath;
use vng_core::data::{LocalVoxelPosition, Vector, VoxelPosition};
use vng_core::event::{EventSource, EventStream, MouseButton, OnMouseDrag, OnWheelMove};
use vng_core::operators::gui::{egui, GuiState};
use vng_core::operators::volume::{ChunkSize, VolumeOperator};
use vng_core::operators::{self, volume::VolumeOperatorState};
use vng_core::operators::{scalar, volume_gpu};
use vng_core::runtime::RunTime;
use vng_core::storage::DataVersionType;
use vng_core::vulkan::state::VulkanState;
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

struct VesselnessState {
    min_rad: f32,
    max_rad: f32,
    steps: usize,
}

struct RaycastingState {
    fov: f32,
    eye: Vector<3, f32>,
    center: Vector<3, f32>,
    up: Vector<3, f32>,
}

struct SliceviewState {
    selected: u32,
    offset: Vector<2, f32>,
    zoom_level: f32,
}

#[derive(Debug, PartialEq)]
enum ProcessState {
    PassThrough,
    Vesselness,
}

#[derive(Debug, PartialEq)]
enum RenderingState {
    Slice,
    Raycasting,
}

struct State {
    gui: GuiState,
    process: ProcessState,
    rendering: RenderingState,
    vesselness: VesselnessState,
    raycasting: RaycastingState,
    sliceview: SliceviewState,
    rotslice: RotSliceState,
}

struct RotSliceState {
    gui: GuiState,
    angle: f32,
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

    let mut state = State {
        gui: GuiState::default(),
        process: ProcessState::PassThrough,
        rendering: RenderingState::Slice,
        vesselness: VesselnessState {
            min_rad: 1.0,
            max_rad: 5.0,
            steps: 3,
        },
        raycasting: RaycastingState {
            fov: 30.0,
            eye: [5.5, 0.5, 0.5].into(),
            center: [0.5, 0.5, 0.5].into(),
            up: [1.0, 1.0, 0.0].into(),
        },
        sliceview: SliceviewState {
            selected: 0,
            offset: [0.0, 0.0].into(),
            zoom_level: 1.0,
        },
        rotslice: RotSliceState {
            gui: GuiState::default(),
            angle: 0.0,
        },
    };

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
                    &mut state,
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

    //TODO: Hm, not sure if this works out to well in a multi-device scenario... We have to
    //investigate how to fix that.
    unsafe { state.gui.deinitialize(&runtime.vulkan.device_contexts()[0]) };
    unsafe { window.deinitialize(&runtime.vulkan) };

    Ok(())
}

pub type EventLoop<T> = winit::event_loop::EventLoop<T>;

fn slice_viewer_z<'op>(
    slice_input: VolumeOperator<'op>,
    md: ImageMetaData,
    state: &mut SliceviewState,
    events: &mut EventStream,
) -> VolumeOperator<'op> {
    events.act(|c| {
        c.chain(state.offset.drag(MouseButton::Left))
            .chain(OnMouseDrag(MouseButton::Right, |_pos, delta| {
                state.selected = (state.selected as i32 + delta.y()).max(0) as u32;
            }))
            //.chain(OnWheelMove(|delta| *slice_num += delta as i32))
            .chain(OnWheelMove(|delta, e_state| {
                if let Some(m_state) = &e_state.mouse_state {
                    let zoom_change = (-delta * 0.05).exp();
                    state.zoom_level *= zoom_change;

                    let pos = m_state.pos.map(|v| v as f32);

                    state.offset = (state.offset - pos) / Vector::fill(zoom_change) + pos;
                }
            }))
    });

    let md = ImageMetaData {
        dimensions: md.dimensions,
        chunk_size: Vector::fill(512.into()),
    };

    let slice_proj_z = crate::operators::sliceviewer::slice_projection_mat_z(
        slice_input.metadata.clone(),
        crate::operators::scalar::constant_hash(md),
        crate::operators::scalar::constant_hash(state.selected.into()),
        crate::operators::scalar::constant_pod(state.offset),
        crate::operators::scalar::constant_pod(state.zoom_level),
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
    runtime: &mut RunTime,
    slice_input: VolumeOperator<'op>,
    md: ImageMetaData,
    state: &'op mut RotSliceState,
    mut events: EventStream,
) -> VolumeOperator<'op> {
    events.act(|c| {
        c.chain(OnMouseDrag(MouseButton::Right, |_pos, delta| {
            state.angle += delta.x() as f32 * 0.01;
        }))
        .chain(OnWheelMove(|delta, _| state.angle += delta * 0.05))
    });

    let md = ImageMetaData {
        dimensions: md.dimensions,
        chunk_size: Vector::fill(512.into()),
    };

    let slice_proj_rot = crate::operators::sliceviewer::slice_projection_mat_centered_rotate(
        slice_input.metadata.clone(),
        crate::operators::scalar::constant_hash(md),
        crate::operators::scalar::constant_pod(state.angle),
    );

    let mat_ref = &slice_proj_rot;
    let vol_ref = &slice_input;

    let info = if let Some(mouse_state) = &events.latest_state().mouse_state {
        let mouse_pos = mouse_state.pos;
        runtime
            .resolve(None, move |ctx, _| {
                async move {
                    let mat = ctx.submit(mat_ref.request_scalar()).await;
                    let m_in = ctx.submit(vol_ref.metadata.request_scalar()).await;

                    let mouse_pos = Vector::<4, f32>::from([
                        1.0,
                        0.0,
                        mouse_pos.y() as f32,
                        mouse_pos.x() as f32,
                    ]);
                    let vol_pos = mat * mouse_pos;
                    let vol_pos = vol_pos.drop_dim(0).map(|v| v.round() as i32);

                    let dim = m_in.dimensions.raw();

                    let inside = vol_pos
                        .zip(dim, |p, d| 0 <= p && p < d as i32)
                        .fold(true, |a, b| a && b);
                    let ret = if inside {
                        let vol_pos = vol_pos.map(|v| (v as u32)).global();
                        let chunk_pos = m_in.chunk_pos(vol_pos);
                        let chunk_info = m_in.chunk_info(chunk_pos);

                        let brick = ctx.submit(vol_ref.bricks.request(chunk_pos)).await;

                        let local_pos = chunk_info.in_chunk(vol_pos);
                        let brick = vng_core::data::chunk(&brick, &chunk_info);

                        let val = *brick.get(local_pos.as_index()).unwrap();
                        Some((val, vol_pos))
                    } else {
                        None
                    };

                    Ok(ret)
                }
                .into()
            })
            .unwrap()
    } else {
        None
    };

    let gui = state.gui.setup(&mut events, |ctx| {
        egui::Window::new("Info").show(ctx, |ui| {
            let s = if let Some((val, pos)) = info {
                format!(
                    "Pos: [{}, {}, {}]\n Value: {}",
                    pos.x().raw,
                    pos.y().raw,
                    pos.z().raw,
                    val
                )
            } else {
                format!("Outside the volume")
            };
            ui.label(s);
        });
    });
    let slice = crate::operators::sliceviewer::render_slice(
        slice_input,
        crate::operators::scalar::constant_hash(md),
        slice_proj_rot,
    );
    let frame = volume_gpu::rechunk(slice, Vector::fill(ChunkSize::Full));
    let frame = gui.render(frame);
    frame
}

fn eval_network(
    runtime: &mut RunTime,
    window: &mut Window,
    vol: &dyn VolumeOperatorState,
    app_state: &mut State,
    mut events: EventStream,
    deadline: Instant,
) -> Result<DataVersionType, Box<dyn std::error::Error>> {
    //events.act(|c| {
    //    c.chain(OnKeyPress(Key::Key9, || *slice_num += 1))
    //        .chain(OnKeyPress(Key::Key0, || *slice_num -= 1))
    //        .chain(OnKeyPress(Key::Key1, || *scale *= 1.10))
    //        .chain(OnKeyPress(Key::Key2, || *scale /= 1.10))
    //        .chain(OnKeyPress(Key::Key3, || *offset += 0.01))
    //        .chain(OnKeyPress(Key::Key4, || *offset -= 0.01))
    //        .chain(OnKeyPress(Key::Plus, || *stddev *= 1.10))
    //        .chain(OnKeyPress(Key::Minus, || *stddev /= 1.10))
    //});

    let vol = vol.operate();

    //let vol = volume_gpu::rechunk(vol, LocalVoxelPosition::fill(48.into()).into_elem());

    let processed = match app_state.process {
        ProcessState::PassThrough => vol,
        ProcessState::Vesselness => operators::vesselness::multiscale_vesselness(
            vol,
            scalar::constant_pod(app_state.vesselness.min_rad),
            scalar::constant_pod(app_state.vesselness.max_rad),
            app_state.vesselness.steps,
        ),
    };

    let gui = app_state.gui.setup(&mut events, |ctx| {
        egui::Window::new("Settings").show(ctx, |ui| {
            ui.vertical(|ui| {
                egui::ComboBox::from_label("Processing")
                    .selected_text(format!("{:?}", app_state.process))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut app_state.process,
                            ProcessState::PassThrough,
                            "None",
                        );
                        ui.selectable_value(
                            &mut app_state.process,
                            ProcessState::Vesselness,
                            "Vesselness",
                        );
                    });
                match app_state.process {
                    ProcessState::PassThrough => {}
                    ProcessState::Vesselness => {
                        ui.add(
                            egui::Slider::new(&mut app_state.vesselness.min_rad, 0.5..=100.0)
                                .text("Min Radius")
                                .logarithmic(true),
                        );
                        ui.add(
                            egui::Slider::new(&mut app_state.vesselness.max_rad, 0.5..=100.0)
                                .text("Max Radius")
                                .logarithmic(true),
                        );
                        ui.add(
                            egui::Slider::new(&mut app_state.vesselness.steps, 2..=10)
                                .text("Scale space steps"),
                        );
                    }
                }

                egui::ComboBox::from_label("Rendering")
                    .selected_text(format!("{:?}", app_state.rendering))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut app_state.rendering,
                            RenderingState::Slice,
                            "Slice",
                        );
                        ui.selectable_value(
                            &mut app_state.rendering,
                            RenderingState::Raycasting,
                            "Raycasting",
                        );
                    });
                match app_state.rendering {
                    RenderingState::Slice => {
                        let vol_ref = &processed;
                        let metadata = runtime
                            .resolve(Some(deadline), move |ctx, _| {
                                async move {
                                        Ok(ctx.submit(vol_ref.metadata.request_scalar()).await)
                                    }
                                    .into()
                            })
                            .unwrap();
                        let num_slices = metadata.dimensions.z().raw;
                        let max_slice = num_slices - 1;

                        ui.horizontal(|ui| {
                            ui.label("Slice (z): ");
                            if ui.button("-").clicked() {
                                app_state.sliceview.selected =
                                    app_state.sliceview.selected.saturating_sub(1);
                            }
                            ui.add(egui::Slider::new(
                                &mut app_state.sliceview.selected,
                                0..=max_slice,
                            ));
                            if ui.button("+").clicked() {
                                app_state.sliceview.selected =
                                    (app_state.sliceview.selected + 1).min(max_slice);
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Angle: ");
                            ui.add(egui::Slider::new(
                                &mut app_state.rotslice.angle,
                                0.0..=std::f32::consts::TAU,
                            ));
                        });
                    }
                    RenderingState::Raycasting => {
                        ui.add(
                            egui::Slider::new(&mut app_state.raycasting.fov, 10.0..=100.0)
                                .text("FOV")
                                .logarithmic(true),
                        );
                    }
                }
            });
        });
    });

    let frame = match app_state.rendering {
        RenderingState::Slice => {
            let mut splitter = operators::splitter::Splitter::new(window.size(), 0.5);

            let (mut events_l, events_r) = splitter.split_events(&mut events);

            let left = slice_viewer_z(
                processed.clone(),
                splitter.metadata_l(),
                &mut app_state.sliceview,
                &mut events_l,
            );

            let right = slice_viewer_rot(
                runtime,
                processed,
                splitter.metadata_r(),
                &mut app_state.rotslice,
                events_r,
            );

            splitter.render(left, right)
        }
        RenderingState::Raycasting => {
            let md = ImageMetaData {
                dimensions: window.size(),
                //chunk_size: window.size().local(),
                chunk_size: Vector::fill(512.into()),
            };

            let perspective = cgmath::perspective(
                cgmath::Deg(app_state.raycasting.fov),
                md.dimensions.x().raw as f32 / md.dimensions.y().raw as f32,
                0.001,
                100.0,
            );
            let matrix = perspective
                * cgmath::Matrix4::look_at_rh(
                    app_state.raycasting.eye.into(),
                    app_state.raycasting.center.into(),
                    app_state.raycasting.up.into(),
                );
            let eep = vng_core::operators::raycaster::entry_exit_points(
                processed.metadata.clone(),
                scalar::constant_hash(md),
                scalar::constant_as_array(matrix),
            );
            vng_core::operators::raycaster::raycast(processed, eep)
        }
    };

    let frame = volume_gpu::rechunk(frame, Vector::fill(ChunkSize::Full));
    let frame = gui.render(frame);

    let slice_ref = &frame;
    let version = runtime.resolve(Some(deadline), |ctx, _| {
        async move { window.render(ctx, slice_ref).await }.into()
    })?;

    Ok(version)
}
