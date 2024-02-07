use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use vng_core::data::{LocalVoxelPosition, Vector, VoxelPosition};
use vng_core::dim::*;
use vng_core::event::{
    EventSource, EventStream, Key, MouseButton, OnKeyPress, OnMouseDrag, OnWheelMove,
};
use vng_core::operators::gui::{egui, GuiState};
use vng_core::operators::tensor::FrameOperator;
use vng_core::operators::volume::{ChunkSize, EmbeddedVolumeOperatorState, LODVolumeOperator};
use vng_core::operators::{self, volume_gpu};
use vng_core::runtime::RunTime;
use vng_core::storage::DataVersionType;
use vng_core::vulkan::state::VulkanState;
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
        //[.., "h5"] => Box::new(Hdf5VolumeSourceState::open(path, "/volume".to_string())?),
        _ => {
            return Err(format!("Unknown volume format for file {}", path.to_string_lossy()).into())
        }
    })
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

    //TODO: Hm, not sure if this works out to well in a multi-device scenario... We have to
    //investigate how to fix that.
    unsafe { gui.deinitialize(&runtime.vulkan.device_contexts()[0]) };
    unsafe { window.deinitialize(&runtime.vulkan) };

    Ok(())
}

pub type EventLoop<T> = winit::event_loop::EventLoop<T>;

fn slice_viewer_z(
    vol: LODVolumeOperator<f32>,
    md: ImageMetaData,
    slice_num: &mut i32,
    offset: &mut Vector<D2, f32>,
    zoom_level: &mut f32,
    events: &mut EventStream,
) -> FrameOperator {
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
        chunk_size: md.chunk_size,
        //chunk_size: Vector::fill(512.into()),
    };

    let slice_num_g = ((*slice_num).max(0) as u32).into();
    let slice_proj_z = crate::operators::sliceviewer::slice_projection_mat(
        0,
        vol.fine_metadata(),
        vol.fine_embedding_data(),
        md.dimensions,
        slice_num_g,
        *offset,
        *zoom_level,
    );

    let slice = crate::operators::sliceviewer::render_slice(vol, md, slice_proj_z);
    let slice = volume_gpu::rechunk(slice, Vector::fill(ChunkSize::Full));

    slice
}

fn slice_viewer_rot(
    vol: LODVolumeOperator<f32>,
    md: ImageMetaData,
    angle: &mut f32,
    mut events: EventStream,
) -> FrameOperator {
    events.act(|c| {
        c.chain(OnMouseDrag(MouseButton::Right, |_pos, delta| {
            *angle += delta.x() as f32 * 0.01;
        }))
        .chain(OnWheelMove(|delta, _| *angle += delta * 0.05))
    });

    let md = ImageMetaData {
        dimensions: md.dimensions,
        chunk_size: md.chunk_size,
        //chunk_size: Vector::fill(512.into()),
    };

    let slice_proj_rot = crate::operators::sliceviewer::slice_projection_mat_centered_rotate(
        vol.fine_metadata(),
        vol.fine_embedding_data(),
        md.into(),
        (*angle).into(),
    );

    let slice = crate::operators::sliceviewer::render_slice(vol, md.into(), slice_proj_rot);
    let slice = volume_gpu::rechunk(slice, Vector::fill(ChunkSize::Full));
    slice
}

fn eval_network(
    runtime: &mut RunTime,
    window: &mut Window,
    vol: &dyn EmbeddedVolumeOperatorState,
    angle: &mut f32,
    slice_num: &mut i32,
    slice_offset: &mut Vector<D2, f32>,
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

    let mut splitter = operators::splitter::Splitter::new(
        window.size(),
        0.5,
        operators::splitter::SplitDirection::Horizontal,
    );

    let (mut events_l, events_r) = splitter.split_events(&mut events);

    let gui = gui.setup(&mut events_l, |ctx| {
        egui::Window::new("Settings").show(ctx, |ui| {
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
                ui.add(
                    egui::Slider::new(scale, 0.01..=100.0)
                        .text("Scale")
                        .logarithmic(true),
                );
                ui.add(egui::Slider::new(offset, -10.0..=10.0).text("Offset"));
            });
        });
    });

    let vol = vol.operate();

    let vol = vol.map_inner(|vol| {
        //let vol = volume_gpu::rechunk(vol.into(), LocalVoxelPosition::fill(10.into()).into_elem());

        //    let after_kernel =
        //        operators::vesselness::multiscale_vesselness(vol, 3.0.into(), (*stddev).into(), 3);
        //    //let after_kernel = operators::vesselness::vesselness(vol, scalar::constant_pod(*stddev));
        let scaled = volume_gpu::linear_rescale(vol, (*scale).into(), (*offset).into());
        scaled
    });
    let vol = vng_core::operators::resample::create_lod(vol, 2.0);

    let left = slice_viewer_z(
        vol.clone(),
        splitter.metadata_first(),
        slice_num,
        slice_offset,
        slice_zoom_level,
        &mut events_l,
    );

    let left = gui.render(left);
    let right = slice_viewer_rot(vol, splitter.metadata_last(), angle, events_r);
    let frame = splitter.render(left, right);

    let slice_ref = &frame;
    let version = runtime.resolve(Some(deadline), |ctx, _| {
        async move { window.render(ctx, slice_ref).await }.into()
    })?;

    Ok(version)
}
