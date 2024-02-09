use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use vng_core::data::{GlobalCoordinate, LocalVoxelPosition, Vector};
use vng_core::dim::*;
use vng_core::event::{EventSource, EventStream, MouseButton, OnMouseDrag, OnWheelMove};
use vng_core::operators::gui::{egui, GuiState};
use vng_core::operators::raycaster::{CameraState, RaycasterConfig};
use vng_core::operators::sliceviewer::SliceviewState;
use vng_core::operators::tensor::FrameOperator;
use vng_core::operators::volume::{ChunkSize, EmbeddedVolumeOperatorState, LODVolumeOperator};
use vng_core::operators::{self, volume_gpu};
use vng_core::runtime::RunTime;
use vng_core::storage::DataVersionType;
use vng_core::vulkan::state::VulkanState;
use vng_core::vulkan::window::Window;
use vng_hdf5::Hdf5VolumeSourceState;
use vng_nifti::NiftiVolumeSourceState;
use vng_vvd::VvdVolumeSourceState;
use winit::event::{Event, WindowEvent};
use winit::platform::run_return::EventLoopExtRunReturn;

use vng_core::array::ImageMetaData;

#[derive(Parser, Clone)]
struct SyntheticArgs {
    #[arg()]
    size: u32,
}

#[derive(Parser)]
struct CliArgs {
    #[arg()]
    vol: PathBuf,

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
        [.., "h5"] => Box::new(Hdf5VolumeSourceState::open(path, "/volume".to_string())?),
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

#[derive(Debug, PartialEq)]
enum ProcessState {
    PassThrough,
    Smooth,
    Vesselness,
}

#[derive(Debug, PartialEq)]
enum RenderingState {
    Slice,
    Raycasting,
}

struct RaycastingState {
    camera: CameraState,
    config: RaycasterConfig,
}

struct State {
    gui: GuiState,
    process: ProcessState,
    rendering: RenderingState,
    vesselness: VesselnessState,
    raycasting: RaycastingState,
    sliceview: SliceState,
    smoothing_std: f32,
}

struct SliceState {
    inner: SliceviewState,
    gui: GuiState,
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
    )?;

    let brick_size = LocalVoxelPosition::fill(32.into());

    let vol_state = open_volume(args.vol, brick_size)?;

    let vol = vol_state.operate();
    let vol = &vol;

    let vol_diag = vol.real_dimensions().length();
    let voxel_diag = vol.embedding_data.spacing.length();

    let mut state = State {
        gui: GuiState::default(),
        process: ProcessState::PassThrough,
        rendering: RenderingState::Slice,
        vesselness: VesselnessState {
            min_rad: voxel_diag * 2.0,
            max_rad: vol_diag * 0.01,
            steps: 3,
        },
        raycasting: RaycastingState {
            camera: CameraState::for_volume(vol.metadata, vol.embedding_data, 30.0),
            config: RaycasterConfig::default(),
        },
        sliceview: SliceState {
            inner: SliceviewState::for_volume(vol.metadata, vol.embedding_data, 0),
            gui: GuiState::default(),
        },
        smoothing_std: voxel_diag,
    };

    let mut event_loop = EventLoop::new();

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
                    &mut state,
                    events.current_batch(),
                    next_timeout,
                )
                .unwrap();
                if version == DataVersionType::Final {
                    //control_flow.set_exit();
                }
                //std::thread::sleep(dbg!(
                //    next_timeout.saturating_duration_since(std::time::Instant::now())
                //));
            }
            _ => (),
        }
    });

    //TODO: Hm, not sure if this works out to well in a multi-device scenario... We have to
    //investigate how to fix that.
    unsafe {
        state
            .sliceview
            .gui
            .deinitialize(&runtime.vulkan.device_contexts()[0])
    };
    unsafe { state.gui.deinitialize(&runtime.vulkan.device_contexts()[0]) };
    unsafe { window.deinitialize(&runtime.vulkan) };

    Ok(())
}

pub type EventLoop<T> = winit::event_loop::EventLoop<T>;

fn slice_viewer_z(
    runtime: &mut RunTime,
    vol: LODVolumeOperator<f32>,
    size: Vector<D2, GlobalCoordinate>,
    state: &mut SliceState,
    events: &mut EventStream,
) -> FrameOperator {
    let md = ImageMetaData {
        dimensions: size,
        chunk_size: size.local(),
        //chunk_size: Vector::fill(512.into()),
    };

    let slice_proj_z =
        state
            .inner
            .projection_mat(vol.fine_metadata(), vol.fine_embedding_data(), size);

    let mat_ref = &slice_proj_z;
    let vol_ref = &vol;
    let info = if let Some(mouse_state) = &events.latest_state().mouse_state {
        let mouse_pos = mouse_state.pos;
        let md = vol.fine_metadata();
        let md_ref = &md;
        runtime
            .resolve(None, move |ctx, _| {
                async move {
                    let mat = mat_ref;
                    let m_in = md_ref;

                    let mouse_pos = Vector::<D4, f32>::from([
                        1.0,
                        0.0,
                        mouse_pos.y() as f32,
                        mouse_pos.x() as f32,
                    ]);
                    let vol_pos = *mat * mouse_pos;
                    let vol_pos = vol_pos.drop_dim(0).map(|v| v.round() as i32);

                    let dim = m_in.dimensions.raw();

                    let inside = vol_pos
                        .zip(dim, |p, d| 0 <= p && p < d as i32)
                        .fold(true, |a, b| a && b);
                    let ret = if inside {
                        let vol_pos = vol_pos.map(|v| (v as u32)).global();
                        let chunk_pos = m_in.chunk_pos(vol_pos);
                        let chunk_info = m_in.chunk_info(chunk_pos);

                        let brick = ctx
                            .submit(vol_ref.levels[0].chunks.request(chunk_pos))
                            .await;

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

    let gui = state.gui.setup(events, |ctx| {
        egui::Window::new("Info")
            .anchor(egui::Align2::RIGHT_TOP, [0.0; 2])
            .show(ctx, |ui| {
                let s = if let Some((val, pos)) = info {
                    let pos = pos.raw().f32() * vol.fine_embedding_data().spacing;
                    format!(
                        "Pos: [{}, {}, {}]\n Value: {}",
                        pos.x(),
                        pos.y(),
                        pos.z(),
                        val
                    )
                } else {
                    format!("Outside the volume")
                };
                ui.label(s);
            });
    });

    events.act(|c| {
        c.chain(state.inner.offset.drag(MouseButton::Left))
            .chain(OnMouseDrag(MouseButton::Right, |_pos, delta| {
                state.inner.scroll(delta.y());
            }))
            .chain(OnWheelMove(|delta, e_state| {
                if let Some(m_state) = &e_state.mouse_state {
                    let pos = m_state.pos.map(|v| v as f32);
                    state.inner.zoom(delta, pos);
                }
            }))
    });

    let slice = crate::operators::sliceviewer::render_slice(vol, md.into(), slice_proj_z);
    let slice = volume_gpu::rechunk(slice, Vector::fill(ChunkSize::Full));
    let frame = gui.render(slice);

    frame
}

fn raycaster(
    vol: LODVolumeOperator<f32>,
    size: Vector<D2, GlobalCoordinate>,
    state: &mut RaycastingState,
    mut events: EventStream,
) -> FrameOperator {
    events.act(|c| {
        c.chain(OnWheelMove(|delta, _| {
            state.camera.trackball.move_inout(delta);
        }))
        .chain(OnMouseDrag(MouseButton::Left, |_, delta| {
            state.camera.trackball.pan_around(delta);
        }))
    });

    let md = ImageMetaData {
        dimensions: size,
        chunk_size: size.local(),
        //chunk_size: Vector::fill(512.into()),
    };

    let matrix = state.camera.projection_mat(md.dimensions);
    let eep = vng_core::operators::raycaster::entry_exit_points(
        vol.fine_metadata(),
        vol.fine_embedding_data(),
        md.into(),
        matrix.into(),
    );
    vng_core::operators::raycaster::raycast(vol, eep, state.config)
}

fn eval_network(
    runtime: &mut RunTime,
    window: &mut Window,
    vol: &dyn EmbeddedVolumeOperatorState,
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

    let volume_diag = vol.real_dimensions().length();
    let voxel_diag = vol.embedding_data.spacing.length();
    let radius_range = voxel_diag..=volume_diag * 0.1;
    let smoothing_range = voxel_diag..=volume_diag * 0.1;

    let vol = vng_core::operators::resample::create_lod(vol, 2.0);

    //let vol = volume_gpu::rechunk(vol, LocalVoxelPosition::fill(48.into()).into_elem());

    let processed = match app_state.process {
        ProcessState::PassThrough => vol,
        ProcessState::Smooth => vol.map(|evol| {
            let spacing = evol.embedding_data.spacing;
            evol.map_inner(|vol| {
                let factor = app_state.smoothing_std;
                let kernels: [_; 3] =
                    std::array::from_fn(|i| operators::kernels::gauss(factor / spacing[i]));
                let kernel_refs = Vector::<D3, _>::from_fn(|i| &kernels[i]);
                operators::volume_gpu::separable_convolution(vol, kernel_refs)
            })
        }),
        ProcessState::Vesselness => vol.map(|vol| {
            operators::vesselness::multiscale_vesselness(
                vol,
                app_state.vesselness.min_rad.into(),
                app_state.vesselness.max_rad.into(),
                app_state.vesselness.steps,
            )
        }),
    };
    let mut take_screenshot = false;

    let gui = app_state.gui.setup(&mut events, |ctx| {
        egui::Window::new("Settings").show(ctx, |ui| {
            ui.vertical(|ui| {
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
                egui::ComboBox::from_label("Processing")
                    .selected_text(format!("{:?}", app_state.process))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut app_state.process,
                            ProcessState::PassThrough,
                            "Passthrough",
                        );
                        ui.selectable_value(&mut app_state.process, ProcessState::Smooth, "Smooth");
                        ui.selectable_value(
                            &mut app_state.process,
                            ProcessState::Vesselness,
                            "Vesselness",
                        );
                    });
                match app_state.process {
                    ProcessState::PassThrough => {}
                    ProcessState::Smooth => {
                        ui.add(
                            egui::Slider::new(&mut app_state.smoothing_std, smoothing_range)
                                .text("Standard deviation")
                                .logarithmic(true),
                        );
                    }
                    ProcessState::Vesselness => {
                        ui.add(
                            egui::Slider::new(
                                &mut app_state.vesselness.min_rad,
                                radius_range.clone(),
                            )
                            .text("Min Radius")
                            .logarithmic(true),
                        );
                        ui.add(
                            egui::Slider::new(&mut app_state.vesselness.max_rad, radius_range)
                                .text("Max Radius")
                                .logarithmic(true),
                        );
                        ui.add(
                            egui::Slider::new(&mut app_state.vesselness.steps, 1..=10)
                                .text("Scale space steps"),
                        );
                    }
                }

                match app_state.rendering {
                    RenderingState::Slice => {
                        ui.horizontal(|ui| {
                            ui.label("Slice (z): ");
                            if ui.button("-").clicked() {
                                app_state.sliceview.inner.scroll(-1);
                            }
                            ui.add(egui::Slider::new(
                                &mut app_state.sliceview.inner.depth,
                                0.0..=app_state.sliceview.inner.dim_end,
                            ));
                            if ui.button("+").clicked() {
                                app_state.sliceview.inner.scroll(1);
                            }
                        });
                    }
                    RenderingState::Raycasting => {
                        ui.add(
                            egui::Slider::new(&mut app_state.raycasting.camera.fov, 10.0..=100.0)
                                .text("FOV")
                                .logarithmic(true),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut app_state.raycasting.config.lod_coarseness,
                                1.0..=100.0,
                            )
                            .text("LOD coarseness")
                            .logarithmic(true),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut app_state.raycasting.config.oversampling_factor,
                                0.01..=10.0,
                            )
                            .text("Oversampling factor")
                            .logarithmic(true),
                        );
                    }
                }
                if ui.button("Save Screenshot").clicked() {
                    take_screenshot = true;
                }
            });
        });
    });

    let frame = match app_state.rendering {
        RenderingState::Slice => slice_viewer_z(
            runtime,
            processed.clone(),
            window.size(),
            &mut app_state.sliceview,
            &mut events,
        ),
        RenderingState::Raycasting => raycaster(
            processed.into(),
            window.size(),
            &mut app_state.raycasting,
            events,
        ),
    };

    let frame = volume_gpu::rechunk(frame, Vector::fill(ChunkSize::Full));
    let frame = gui.render(frame);

    let slice_ref = &frame;
    let version = runtime.resolve(Some(deadline), |ctx, _| {
        async move { window.render(ctx, slice_ref).await }.into()
    })?;

    if take_screenshot {
        runtime
            .resolve(Some(deadline), |ctx, _| {
                async move {
                    vng_core::operators::png_writer::write(ctx, slice_ref, "screenshot.png".into())
                        .await
                }
                .into()
            })
            .unwrap();
    }

    Ok(version)
}
