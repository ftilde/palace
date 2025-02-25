use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use palace_core::data::{LocalVoxelPosition, Vector, VoxelPosition};
use palace_core::dim::*;
use palace_core::dtypes::{ScalarType, StaticElementType};
use palace_core::event::{EventStream, Key, MouseButton, OnKeyPress, OnMouseDrag, OnWheelMove};
use palace_core::jit::jit;
use palace_core::operators::array::from_rc;
use palace_core::operators::gui::{egui, GuiState};
use palace_core::operators::rechunk::ChunkSize;
use palace_core::operators::sliceviewer::RenderConfig2D;
use palace_core::operators::tensor::{FrameOperator, LODVolumeOperator};
use palace_core::operators::{self};
use palace_core::runtime::{Deadline, RunTime};
use palace_core::storage::DataVersionType;
use palace_core::transfunc::TransFuncOperator;
use palace_core::vulkan::window::Window;

use palace_core::array::{self, ImageMetaData, TensorEmbeddingData};

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

    /// Use the vulkan devices with the specified ids
    #[arg(long, value_delimiter = ',', num_args=1..)]
    devices: Vec<usize>,

    /// Force a specific size for the compute task pool [default: number of cores]
    #[arg(short, long)]
    compute_pool_size: Option<usize>,

    /// Stop after rendering a complete frame
    #[arg(short, long)]
    bench: bool,
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
        args.devices,
    )?;

    let brick_size = LocalVoxelPosition::fill(32.into());

    let tf = TransFuncOperator::grey_ramp(0.0, 1.0);

    let vol = match args.input {
        Input::File(path) => {
            let vol = palace_io::open_or_create_lod(
                path.vol,
                palace_io::Hints::new().chunk_size(brick_size.into_dyn()),
            )?
            .0;
            vol.map(|v| {
                v.map_inner(|v| {
                    palace_core::jit::jit(v)
                        .cast(ScalarType::F32.into())
                        .unwrap()
                        .compile()
                        .unwrap()
                })
            })
            .try_into_static()
            .unwrap()
            .try_into()
            .unwrap()
        }
        Input::Synthetic(args) => {
            let md = array::VolumeMetaData {
                dimensions: VoxelPosition::fill(args.size.into()),
                chunk_size: brick_size,
            };
            let ed = TensorEmbeddingData {
                spacing: md.dimensions.map(|v| 1.0 / v.raw as f32),
            };
            match args.scenario {
                Type::Ball => operators::procedural::ball(md, ed),
                Type::Full => operators::procedural::full(md, ed),
                Type::Mandelbulb => operators::procedural::mandelbulb(md, ed),
                Type::RandomWalker => {
                    //let md = array::VolumeMetaData {
                    //    dimensions: VoxelPosition::fill(args.size.into()),
                    //    chunk_size: LocalVoxelPosition::fill(args.size.into()),
                    //};
                    //let ball = operators::procedural::ball(md, ed).levels[0].clone();
                    let ball = operators::procedural::ball(md, ed);
                    let seeds_fg = from_rc(Rc::new([Vector::<D3, f32>::fill(0.5)])).into();
                    let seeds_bg =
                        from_rc(Rc::new([Vector::<D3, f32>::fill(0.1), Vector::fill(0.9)])).into();

                    //let seeds = operators::randomwalker::rasterize_seed_points(
                    //    seeds_fg,
                    //    seeds_bg,
                    //    ball.metadata,
                    //    TensorEmbeddingData {
                    //        spacing: md.dimensions.map(|v| 1.0 / v.raw as f32),
                    //    },
                    //);
                    //operators::randomwalker::random_walker(
                    //    ball.into(),
                    //    seeds.inner,
                    //    operators::randomwalker::WeightFunction::Grady { beta: 1000.0 },
                    //    1e-6,
                    //    Default::default(),
                    //)
                    //.embedded(Default::default())
                    //.single_level_lod()
                    let res = operators::randomwalker::hierarchical_random_walker(
                        ball,
                        seeds_fg,
                        seeds_bg,
                        operators::randomwalker::WeightFunction::BhattacharyyaVarGaussian {
                            extent: 1,
                        },
                        1e-6,
                        Default::default(),
                    );
                    res
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
    let mut coarse_lod_factor: f32 = 1.0;
    let mut gui = GuiState::new();

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
                &mut coarse_lod_factor,
                &mut gui,
                &tf,
                events,
                timeout,
            )
            .unwrap();
            if args.bench && version == DataVersionType::Final {
                _event_loop.exit();
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
    render_config: RenderConfig2D,
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

    let slice = crate::operators::sliceviewer::render_slice(
        vol,
        md,
        slice_proj_z,
        tf.clone(),
        render_config,
    );
    let slice = operators::rechunk::rechunk(slice, Vector::fill(ChunkSize::Full));

    slice
}

fn slice_viewer_rot(
    vol: LODVolumeOperator<StaticElementType<f32>>,
    md: ImageMetaData,
    render_config: RenderConfig2D,
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

    let slice = crate::operators::sliceviewer::render_slice(
        vol,
        md.into(),
        slice_proj_rot,
        tf.clone(),
        render_config,
    );
    let slice = operators::rechunk::rechunk(slice, Vector::fill(ChunkSize::Full));
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
    coarse_lod_factor: &mut f32,
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
                ui.add(
                    egui::Slider::new(coarse_lod_factor, 0.1..=100.0)
                        .text("Coarse LOD factor")
                        .logarithmic(true),
                );
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
                .add((*offset).into())
                .unwrap()
                .mul((*scale).into())
                .unwrap()
                .compile()
                .unwrap()
                .try_into()
                .unwrap();
            scaled
        })
    });
    let render_config = RenderConfig2D {
        coarse_lod_factor: *coarse_lod_factor,
    };

    let left = slice_viewer_z(
        vol.clone(),
        splitter.metadata_first(),
        render_config,
        slice_num,
        slice_offset,
        slice_zoom_level,
        tf,
        &mut events_l,
    );

    let left = gui.render(left);
    let right = slice_viewer_rot(
        vol,
        splitter.metadata_last(),
        render_config,
        angle,
        tf,
        events_r,
    );
    let frame = splitter.render(left, right);

    let slice_ref = &frame;
    let version = runtime.resolve(Some(deadline), false, |ctx, _| {
        async move { window.render(ctx, slice_ref).await }.into()
    })?;

    Ok(version)
}
