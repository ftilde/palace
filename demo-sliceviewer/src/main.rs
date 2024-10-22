use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, Subcommand};
use palace_core::data::{LocalVoxelPosition, Vector, VoxelPosition};
use palace_core::dim::*;
use palace_core::dtypes::StaticElementType;
use palace_core::event::{EventStream, Key, MouseButton, OnKeyPress, OnMouseDrag, OnWheelMove};
use palace_core::jit::jit;
use palace_core::operators::gui::{egui, GuiState};
use palace_core::operators::raycaster::TransFuncOperator;
use palace_core::operators::tensor::FrameOperator;
use palace_core::operators::volume::{ChunkSize, LODVolumeOperator};
use palace_core::operators::{self, volume_gpu};
use palace_core::runtime::{Deadline, RunTime};
use palace_core::storage::DataVersionType;
use palace_core::vulkan::window::Window;

use palace_core::array::{self, ImageMetaData};

#[derive(Subcommand, Clone)]
enum Type {
    Ball,
    Full,
    Mandelbulb,
    RandomWalker,
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

    /// Use the vulkan device with the specified id
    #[arg(long, default_value = "0")]
    device: usize,

    /// Force a specific size for the compute task pool [default: number of cores]
    #[arg(short, long)]
    compute_pool_size: Option<usize>,
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

    let tf = TransFuncOperator::grey_ramp(0.0, 1.0);

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
                Type::RandomWalker => {
                    let md = array::VolumeMetaData {
                        dimensions: VoxelPosition::fill(args.size.into()),
                        chunk_size: LocalVoxelPosition::fill(args.size.into()),
                    };
                    let ball = operators::procedural::ball(md).levels[0].clone();
                    let seeds = operators::procedural::rasterize(
                        md,
                        r#"float run(float[3] pos_normalized, uint[3] pos_voxel) {
                            float center_dist = length(to_glsl(pos_normalized)-vec3(0.5));
                            if (center_dist < 0.1) {
                                return 1.0;
                            } else if (center_dist > 0.75) {
                                return 0.0;
                            } else {
                                return -2.0;
                            }
                        }"#,
                    );
                    operators::randomwalker::random_walker(
                        ball.into(),
                        seeds,
                        operators::randomwalker::WeightFunction::Grady { beta: 1000.0 },
                        Default::default(),
                    )
                    .embedded(Default::default())
                    .single_level_lod()
                }
            }
        }
    };

    let mut angle: f32 = 0.0;
    let mut slice_num = 0;
    let mut slice_offset = [0.0, 0.0].into();
    let mut slice_zoom_level = 1.0;
    let mut scale = 1.0;
    let mut offset: f32 = 0.0;
    let mut stddev: f32 = 5.0;
    let mut gui = GuiState::on_device(args.device);

    let res = palace_winit::run_with_window(
        &mut runtime,
        Duration::from_millis(10),
        |_event_loop, window, rt, events, timeout| {
            let version = eval_network(
                rt,
                window,
                vol.clone(),
                &mut angle,
                &mut slice_num,
                &mut slice_offset,
                &mut slice_zoom_level,
                &mut scale,
                &mut offset,
                &mut stddev,
                &mut gui,
                &tf,
                events,
                timeout,
            )
            .unwrap();
            if version == DataVersionType::Final {
                //_event_loop.exit();
            }
            Ok(version)
        },
    );

    gui.destroy(&runtime);

    res
}

fn slice_viewer_z(
    vol: LODVolumeOperator<StaticElementType<f32>>,
    md: ImageMetaData,
    slice_num: &mut i32,
    offset: &mut Vector<D2, f32>,
    zoom_level: &mut f32,
    tf: &TransFuncOperator,
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

    let slice = crate::operators::sliceviewer::render_slice(vol, md, slice_proj_z, tf.clone());
    let slice = volume_gpu::rechunk(slice, Vector::fill(ChunkSize::Full));

    slice
}

fn slice_viewer_rot(
    vol: LODVolumeOperator<StaticElementType<f32>>,
    md: ImageMetaData,
    angle: &mut f32,
    tf: &TransFuncOperator,
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

    let slice =
        crate::operators::sliceviewer::render_slice(vol, md.into(), slice_proj_rot, tf.clone());
    let slice = volume_gpu::rechunk(slice, Vector::fill(ChunkSize::Full));
    slice
}

fn eval_network(
    runtime: &mut RunTime,
    window: &mut Window,
    vol: LODVolumeOperator<StaticElementType<f32>>,
    angle: &mut f32,
    slice_num: &mut i32,
    slice_offset: &mut Vector<D2, f32>,
    slice_zoom_level: &mut f32,
    scale: &mut f32,
    offset: &mut f32,
    stddev: &mut f32,
    gui: &mut GuiState,
    tf: &TransFuncOperator,
    mut events: EventStream,
    deadline: Deadline,
) -> Result<DataVersionType, Box<dyn std::error::Error>> {
    events.act(|c| {
        c.chain(OnKeyPress(Key::Digit9, || *slice_num += 1))
            .chain(OnKeyPress(Key::Digit0, || *slice_num -= 1))
            .chain(OnKeyPress(Key::Digit1, || *scale *= 1.10))
            .chain(OnKeyPress(Key::Digit2, || *scale /= 1.10))
            .chain(OnKeyPress(Key::Digit3, || *offset += 0.01))
            .chain(OnKeyPress(Key::Digit4, || *offset -= 0.01))
            .chain(OnKeyPress(Key::Equal, || *stddev *= 1.10))
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

    let vol = vol.map(|vol| {
        vol.map_inner(|vol| {
            //let vol = volume_gpu::rechunk(vol.into(), LocalVoxelPosition::fill(10.into()).into_elem());

            //    let after_kernel =
            //        operators::vesselness::multiscale_vesselness(vol, 3.0.into(), (*stddev).into(), 3);
            //    //let after_kernel = operators::vesselness::vesselness(vol, scalar::constant_pod(*stddev));
            let scaled = jit(vol.into())
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

    let left = slice_viewer_z(
        vol.clone(),
        splitter.metadata_first(),
        slice_num,
        slice_offset,
        slice_zoom_level,
        tf,
        &mut events_l,
    );

    let left = gui.render(left);
    let right = slice_viewer_rot(vol, splitter.metadata_last(), angle, tf, events_r);
    let frame = splitter.render(left, right);

    let slice_ref = &frame;
    let version = runtime.resolve(Some(deadline), false, |ctx, _| {
        async move { window.render(ctx, slice_ref).await }.into()
    })?;

    Ok(version)
}
