use crate::map_err;
use pyo3::prelude::*;
use winit::{
    event::{Event, WindowEvent},
    platform::run_return::EventLoopExtRunReturn,
};

use super::{Events, ScalarOperatorF32, VolumeOperator};

#[pyclass(unsendable)]
pub struct RunTime {
    inner: vng_core::runtime::RunTime,
}

#[pymethods]
impl RunTime {
    #[new]
    pub fn new(
        storage_size: usize,
        gpu_storage_size: Option<u64>,
        num_compute_threads: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: map_err(vng_core::runtime::RunTime::new(
                storage_size,
                gpu_storage_size,
                num_compute_threads,
            ))?,
        })
    }

    fn resolve(&mut self, v: &ScalarOperatorF32) -> PyResult<f32> {
        map_err(self.inner.resolve(None, |ctx, _| {
            async move { Ok(ctx.submit(v.0.request_scalar()).await) }.into()
        }))
    }
}

#[pyclass(unsendable)]
pub struct Window {
    event_loop: winit::event_loop::EventLoop<()>,
    window: vng_core::vulkan::window::Window,
    runtime: Py<RunTime>,
}

impl Drop for Window {
    fn drop(&mut self) {
        Python::with_gil(|py| {
            let rt = self.runtime.borrow(py);

            unsafe { self.window.deinitialize(&rt.inner.vulkan) };
        });
    }
}

#[pymethods]
impl Window {
    #[new]
    fn new(py: Python, runtime: Py<RunTime>) -> PyResult<Self> {
        let event_loop = winit::event_loop::EventLoop::new();

        let window = {
            let rt = runtime.borrow(py);

            map_err(vng_core::vulkan::window::Window::new(
                &rt.inner.vulkan,
                &event_loop,
            ))?
        };

        Ok(Self {
            event_loop,
            window,
            runtime,
        })
    }
    fn run(
        &mut self,
        runtime: &mut RunTime,
        gen_frame: &pyo3::types::PyFunction,
        timeout_ms: Option<u64>,
    ) {
        let mut events = vng_core::event::EventSource::default();

        self.event_loop.run_return(|event, _, control_flow| {
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
                    self.window.inner().request_redraw();
                }
                Event::WindowEvent {
                    window_id: _,
                    event: winit::event::WindowEvent::Resized(new_size),
                } => {
                    self.window.resize(new_size, &runtime.inner.vulkan);
                }
                Event::WindowEvent {
                    window_id: _,
                    event,
                } => {
                    events.add(event);
                }
                Event::RedrawRequested(_) => {
                    let end = timeout_ms
                        .map(|to| std::time::Instant::now() + std::time::Duration::from_millis(to));
                    let size = self.window.size();
                    let size = [size.y().raw, size.x().raw];
                    let events = Events(events.current_batch());
                    let frame = gen_frame.call((size, events), None).unwrap();
                    let frame = frame.extract::<VolumeOperator>().unwrap().into();

                    let frame_ref = &frame;
                    let window = &mut self.window;
                    runtime
                        .inner
                        .resolve(end, |ctx, _| {
                            async move { window.render(ctx, frame_ref).await }.into()
                        })
                        .unwrap();
                }
                _ => (),
            }
        });
    }
}
