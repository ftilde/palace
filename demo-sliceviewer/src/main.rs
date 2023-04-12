use std::path::PathBuf;

use clap::{Parser, Subcommand};
use vng_core::data::{LocalVoxelPosition, VoxelPosition};
use vng_core::operators::volume_gpu;
use vng_core::operators::{self, volume::VolumeOperatorState};
use vng_core::runtime::RunTime;
use vng_core::vulkan::window::Window;
//use vng_hdf5::Hdf5VolumeSourceState;
use vng_nifti::NiftiVolumeSourceState;
use vng_vvd::VvdVolumeSourceState;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::platform::run_return::EventLoopExtRunReturn;

use vng_core::array;

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

    let mut runtime = RunTime::new(storage_size, args.compute_pool_size)?;

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
                result = sqrt(d_sq);

            }"#
            .to_owned(),
        }),
    };

    let mut slice_num = 0;
    let mut scale = 1.0;
    let mut offset: f32 = 0.0;

    let mut event_loop = EventLoop::new();

    let mut window = Window::new(&runtime.vulkan, &event_loop).unwrap();

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
                event: winit::event::WindowEvent::MouseWheel { delta, .. },
            } => {
                let motion = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y.signum(),
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y.signum() as f32,
                };
                slice_num += motion as i32;
                println!("Slice: {}", slice_num);
            }
            Event::WindowEvent {
                window_id: _,
                event: winit::event::WindowEvent::KeyboardInput { input, .. },
            } => {
                if input.state == ElementState::Pressed {
                    if let Some(k) = input.virtual_keycode {
                        let mut something_happened = true;
                        match k {
                            VirtualKeyCode::R => {
                                slice_num = 0;
                            }
                            VirtualKeyCode::Key1 => {
                                scale += 0.01;
                            }
                            VirtualKeyCode::Key2 => {
                                scale -= 0.01;
                            }
                            VirtualKeyCode::Key3 => {
                                offset = (offset + 0.01).clamp(0.0, 1.0);
                            }
                            VirtualKeyCode::Key4 => {
                                offset = (offset - 0.01).clamp(0.0, 1.0);
                            }
                            VirtualKeyCode::Plus => {
                                slice_num += 1;
                            }
                            VirtualKeyCode::Minus => {
                                slice_num -= 1;
                            }
                            _ => {
                                something_happened = false;
                            }
                        }
                        if something_happened {
                            println!("Slice: {}, Scale: {}, Offset: {}", slice_num, scale, offset);
                        }
                    }
                }
            }
            Event::RedrawRequested(_) => {
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in MainEventsCleared, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.
                eval_network(
                    &mut runtime,
                    &mut window,
                    &*vol_state,
                    slice_num,
                    scale,
                    offset,
                )
                .unwrap();
            }
            _ => (),
        }
    });

    unsafe { window.deinitialize(&runtime.vulkan) };

    Ok(())
}

pub type EventLoop<T> = winit::event_loop::EventLoop<T>;

fn eval_network(
    runtime: &mut RunTime,
    window: &mut Window,
    vol: &dyn VolumeOperatorState,
    slice_num: i32,
    scale: f32,
    offset: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let vol = vol.operate();

    let rechunked = volume_gpu::rechunk(vol, LocalVoxelPosition::fill(48.into()).into_elem());

    let slice_input = volume_gpu::linear_rescale(rechunked, scale.into(), offset.into());

    let splitter = operators::splitter::Splitter::new(window.size(), 0.5);

    let slice_num_g = (slice_num.max(0) as u32).into();
    let slice_proj_z = crate::operators::sliceviewer::slice_projection_mat_z(
        slice_input.metadata.clone(),
        crate::operators::scalar::constant_hash(splitter.metadata_l()),
        crate::operators::scalar::constant_hash(slice_num_g),
    );
    let angle = slice_num as f32 * 0.05;
    let slice_proj_rot = crate::operators::sliceviewer::slice_projection_mat_centered_rotate(
        slice_input.metadata.clone(),
        crate::operators::scalar::constant_hash(splitter.metadata_r()),
        crate::operators::scalar::constant_pod(angle),
    );
    let slice_z = crate::operators::sliceviewer::render_slice(
        slice_input.clone(),
        crate::operators::scalar::constant_hash(splitter.metadata_l()),
        slice_proj_z,
    );
    //let slice_z = volume_gpu::rechunk(slice_z, Vector::fill(ChunkSize::Full));
    let slice_rot = crate::operators::sliceviewer::render_slice(
        slice_input,
        crate::operators::scalar::constant_hash(splitter.metadata_r()),
        slice_proj_rot,
    );
    //let slice_rot = volume_gpu::rechunk(slice_rot, Vector::fill(ChunkSize::Full));

    let frame = splitter.operate(slice_z, slice_rot);

    let mut c = runtime.context_anchor();
    let mut executor = c.executor();

    let slice_ref = &frame;
    executor.resolve(|ctx| async move { window.render(ctx, slice_ref).await }.into())?;
    //let tasks_executed = executor.statistics().tasks_executed;
    //println!("Rendering done ({} tasks)", tasks_executed);

    Ok(())
}
