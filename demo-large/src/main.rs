use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use palace_core::data::{GlobalCoordinate, Vector};
use palace_core::dim::*;
use palace_core::dtypes::{ScalarType, StaticElementType};
use palace_core::event::{EventStream, MouseButton, OnMouseDrag, OnWheelMove};
use palace_core::operators::gui::{egui, GuiState};
use palace_core::operators::raycaster::{CameraState, CompositingMode, RaycasterConfig, Shading};
use palace_core::operators::rechunk::ChunkSize;
use palace_core::operators::sliceviewer::SliceviewState;
use palace_core::operators::tensor::{FrameOperator, LODVolumeOperator, VolumeOperator};
use palace_core::operators::{self, aggregation};
use palace_core::runtime::{Deadline, RunTime};
use palace_core::storage::DataVersionType;
use palace_core::transfunc::TransFuncOperator;
use palace_core::vulkan::window::Window;

use palace_core::array::ImageMetaData;

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

    /// Use the vulkan devices with the specified ids
    #[arg(long, value_delimiter = ',', num_args=1..)]
    devices: Vec<usize>,

    /// Transfer function (voreen .tfi file)
    #[arg(short, long)]
    transfunc_path: Option<PathBuf>,

    /// Initially fit transfer function to volume values
    #[arg(long)]
    fit_transfunc: bool,

    /// Stop after rendering a complete frame
    #[arg(short, long)]
    bench: bool,
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
    tf: TransFuncOperator,
    smoothing_std: f32,
}

struct SliceState {
    inner: SliceviewState,
    gui: GuiState,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse();

    let mut runtime = RunTime::build()
        .disk_cache_size_opt(args.disk_cache_size.map(|v| v.0 as _))
        .devices(args.devices)
        .num_compute_threads_opt(args.compute_pool_size)
        .finish(args.mem_size.0 as _, args.gpu_mem_size.0 as _)?;

    let hints = palace_io::Hints::new();
    let vol = palace_io::open_or_create_lod(args.vol, hints).unwrap();

    let vol: LODVolumeOperator<StaticElementType<f32>> = vol
        .0
        .map(|v| {
            v.map_inner(|v| {
                palace_core::jit::jit(v)
                    .cast(ScalarType::F32.into())
                    .unwrap()
                    .compile()
                    .unwrap()
            })
        })
        .try_into_static::<D3>()
        .unwrap()
        .try_into()
        .unwrap();
    let l0 = vol.levels[0].clone();

    let vol_diag = l0.real_dimensions().length();
    let voxel_diag = l0.embedding_data.spacing.length();

    let mut state = State {
        gui: GuiState::new(),
        process: ProcessState::PassThrough,
        rendering: RenderingState::Raycasting,
        vesselness: VesselnessState {
            min_rad: voxel_diag * 2.0,
            max_rad: vol_diag * 0.01,
            steps: 3,
        },
        raycasting: RaycastingState {
            camera: CameraState::for_volume(l0.metadata, l0.embedding_data, 30.0),
            config: RaycasterConfig::default(),
        },
        sliceview: SliceState {
            inner: SliceviewState::for_volume(l0.metadata, l0.embedding_data, 0),
            gui: GuiState::new(),
        },
        tf: if let Some(path) = args.transfunc_path {
            palace_vvd::load_tfi(&path).unwrap()
        } else {
            TransFuncOperator::grey_ramp(0.0, 1.0)
        },
        smoothing_std: voxel_diag,
    };

    if args.fit_transfunc {
        fit_transfer_function(&mut runtime, &mut state, &l0.inner);
    }

    let res = palace_winit::run_with_window(
        &mut runtime,
        Duration::from_millis(10),
        |event_loop, window, rt, events, timeout| {
            let version =
                eval_network(rt, window, vol.clone(), &mut state, events, timeout).unwrap();
            if args.bench && version == DataVersionType::Final {
                event_loop.exit();
            }
            Ok(version)
        },
    );

    state.sliceview.gui.destroy(&runtime);
    state.gui.destroy(&runtime);

    res
}

fn slice_viewer_z(
    runtime: &mut RunTime,
    vol: LODVolumeOperator<StaticElementType<f32>>,
    size: Vector<D2, GlobalCoordinate>,
    state: &mut SliceState,
    tf: &TransFuncOperator,
    events: &mut EventStream,
) -> FrameOperator {
    let md = ImageMetaData {
        dimensions: size,
        chunk_size: size.local(),
        //chunk_size: Vector::fill(512.into()),
    };

    let info = if let Some(mouse_state) = &events.latest_state().mouse_state {
        let mouse_pos = mouse_state.pos;
        let level = &vol.levels[0];
        let slice_proj_z = state
            .inner
            .projection_mat(level.metadata, level.embedding_data, size);
        let mat_ref = &slice_proj_z;
        runtime
            .resolve(None, false, move |ctx, _| {
                async move {
                    let mat = mat_ref;
                    let md = &level.metadata;

                    let mouse_pos = Vector::<D4, f32>::from([
                        1.0,
                        0.0,
                        mouse_pos.y() as f32,
                        mouse_pos.x() as f32,
                    ]);
                    let vol_pos = *mat * &mouse_pos;
                    let vol_pos = vol_pos.drop_dim(0).map(|v| v.round() as i32);

                    let dim = md.dimensions.raw();

                    let inside = vol_pos
                        .zip(&dim, |p, d| 0 <= p && p < d as i32)
                        .fold(true, |a, b| a && b);
                    let ret = if inside {
                        let vol_pos = vol_pos.map(|v| (v as u32)).global();
                        let chunk_pos = md.chunk_pos(&vol_pos);
                        let chunk_info = md.chunk_info_vec(&chunk_pos);

                        let brick = ctx
                            .submit(level.chunks.request(md.chunk_index(&chunk_pos)))
                            .await;

                        let local_pos = chunk_info.in_chunk(&vol_pos);
                        let brick = palace_core::data::chunk(&brick, &chunk_info);

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

    let slice_proj_z =
        state
            .inner
            .projection_mat(vol.fine_metadata(), vol.fine_embedding_data(), size);
    let slice = crate::operators::sliceviewer::render_slice(
        vol,
        md.into(),
        slice_proj_z,
        tf.clone(),
        Default::default(),
    )
    .unwrap();
    let slice = operators::rechunk::rechunk(slice, Vector::fill(ChunkSize::Full));
    let frame = gui.render(slice);

    frame
}

fn raycaster(
    vol: LODVolumeOperator<StaticElementType<f32>>,
    size: Vector<D2, GlobalCoordinate>,
    state: &mut RaycastingState,
    tf: &TransFuncOperator,
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
        //chunk_size: size.local(),
        chunk_size: Vector::fill(512.into()),
    };

    let matrix = state.camera.projection_mat(md.dimensions);
    let eep = palace_core::operators::raycaster::entry_exit_points(
        vol.fine_metadata(),
        vol.fine_embedding_data(),
        md.into(),
        matrix.into(),
    );
    palace_core::operators::raycaster::raycast(vol, eep, tf.clone(), state.config).unwrap()
}

fn fit_transfer_function(
    runtime: &mut RunTime,
    app_state: &mut State,
    vol: &VolumeOperator<StaticElementType<f32>>,
) {
    let min_max_sample_bricks = 10;

    let min = aggregation::min(
        vol.clone(),
        aggregation::SampleMethod::Subset(min_max_sample_bricks),
    );
    let min = &min;
    let max = aggregation::max(
        vol.clone(),
        aggregation::SampleMethod::Subset(min_max_sample_bricks),
    );
    let max = &max;

    let (min, max) = runtime
        .resolve(None, false, |ctx, _| {
            async move {
                Ok(palace_core::task::join! {
                    ctx.submit(min.request_scalar()),
                    ctx.submit(max.request_scalar())
                })
            }
            .into()
        })
        .unwrap();

    app_state.tf.min = min;
    app_state.tf.max = max;
}

fn eval_network(
    runtime: &mut RunTime,
    window: &mut Window,
    vol: LODVolumeOperator<StaticElementType<f32>>,
    app_state: &mut State,
    mut events: EventStream,
    deadline: Deadline,
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

    let l0 = vol.levels[0].clone();
    let volume_diag = l0.real_dimensions().length();
    let voxel_diag = l0.embedding_data.spacing.length();
    let radius_range = voxel_diag..=volume_diag * 0.1;
    let smoothing_range = voxel_diag..=volume_diag * 0.1;

    //let vol = vol.map_inner(|v| {
    //    let v = palace_core::jit::jit(v.into());
    //    //let v = v.add((-1.0).into()).unwrap().abs().unwrap();
    //    v.try_into().unwrap()
    //});

    //let vol: EmbeddedVolumeOperator<StaticElementType<f32>> = vol.try_into().unwrap();

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
                operators::conv::separable_convolution(
                    vol,
                    kernel_refs,
                    operators::conv::BorderHandling::Repeat,
                )
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
    let mut fit_tf = false;
    let mut save_task_stream = false;

    let all_devices = runtime.all_devices();

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
                                0.01..=100.0,
                            )
                            .text("LOD coarseness")
                            .logarithmic(true),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut app_state.raycasting.config.preview_lod_coarseness_modifier,
                                1.0..=100.0,
                            )
                            .text("Preview detail reduction")
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
                        egui::ComboBox::from_label("Compositing")
                            .selected_text(format!(
                                "{:?}",
                                app_state.raycasting.config.compositing_mode
                            ))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut app_state.raycasting.config.compositing_mode,
                                    CompositingMode::MOP,
                                    "MOP",
                                );
                                ui.selectable_value(
                                    &mut app_state.raycasting.config.compositing_mode,
                                    CompositingMode::DVR,
                                    "DVR",
                                );
                            });
                        egui::ComboBox::from_label("Shading")
                            .selected_text(format!("{:?}", app_state.raycasting.config.shading))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut app_state.raycasting.config.shading,
                                    Shading::None,
                                    "None",
                                );
                                ui.selectable_value(
                                    &mut app_state.raycasting.config.shading,
                                    Shading::Phong,
                                    "Phong",
                                );
                            });
                    }
                }
                if ui.button("Fit Transfer Function").clicked() {
                    fit_tf = true;
                }
                if ui.button("Save Screenshot").clicked() {
                    take_screenshot = true;
                }
                if ui.button("Save Task Stream").clicked() {
                    save_task_stream = true;
                }
            });
        });
    });

    if fit_tf {
        fit_transfer_function(runtime, app_state, &processed.levels[0].inner);
    }

    let frame = match app_state.rendering {
        RenderingState::Slice => slice_viewer_z(
            runtime,
            processed.clone(),
            window.size(),
            &mut app_state.sliceview,
            &app_state.tf,
            &mut events,
        ),
        RenderingState::Raycasting => raycaster(
            processed.into(),
            window.size(),
            &mut app_state.raycasting,
            &app_state.tf,
            events,
        )
        .distribute_on_gpus(all_devices),
    };

    let frame = operators::rechunk::rechunk(frame, Vector::fill(ChunkSize::Full));
    let frame = gui.render(frame);

    let slice_ref = &frame;
    let version = runtime.resolve(Some(deadline), save_task_stream, |ctx, _| {
        async move { window.render(ctx, slice_ref).await }.into()
    })?;

    if take_screenshot {
        runtime
            .resolve(Some(deadline), false, |ctx, _| {
                async move { palace_png::write(ctx, slice_ref, "screenshot.png".into()).await }
                    .into()
            })
            .unwrap();
    }

    Ok(version)
}
