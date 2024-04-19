use crate::map_err;
use palace_core::{storage::StaticElementType, vec::Vector};
use pyo3::{exceptions::PyException, prelude::*};
use winit::{
    event::{Event, WindowEvent},
    platform::run_return::EventLoopExtRunReturn,
};

use super::{Events, MaybeEmbeddedTensorOperator, ScalarOperator, TensorOperator};

#[pyclass(unsendable)]
pub struct RunTime {
    pub inner: palace_core::runtime::RunTime,
}

macro_rules! match_dim {
    ($n:expr, $call:expr, $call_err:expr) => {
        match $n {
            1 => {
                type D = palace_core::dim::D1;
                $call()
            }
            2 => {
                type D = palace_core::dim::D2;
                $call()
            }
            3 => {
                type D = palace_core::dim::D3;
                $call()
            }
            4 => {
                type D = palace_core::dim::D4;
                $call()
            }
            5 => {
                type D = palace_core::dim::D5;
                $call()
            }
            n => $call_err(n),
        }
    };
}

#[pymethods]
impl RunTime {
    #[new]
    pub fn new(
        storage_size: usize,
        gpu_storage_size: u64,
        disk_cache_size: Option<usize>,
        num_compute_threads: Option<usize>,
        device: Option<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: map_err(palace_core::runtime::RunTime::new(
                storage_size,
                gpu_storage_size,
                num_compute_threads,
                disk_cache_size,
                None,
                device,
            ))?,
        })
    }

    fn resolve(
        &mut self,
        py: Python,
        v: MaybeEmbeddedTensorOperator,
        pos: Vec<u32>,
    ) -> PyResult<PyObject> {
        let v = v.inner();
        match_dim!(
            pos.len(),
            || {
                let op: palace_core::operators::tensor::TensorOperator<D, StaticElementType<f32>> =
                    v.try_into()?;
                let op_ref = &op;
                map_err(self.inner.resolve(None, false, |ctx, _| {
                    async move {
                        let pos: Vector<D, u32> = pos.try_into().unwrap();
                        let chunk = ctx.submit(op_ref.chunks.request(pos.chunk())).await;
                        let chunk_info = op.metadata.chunk_info(pos.chunk());
                        let chunk = palace_core::data::chunk(&chunk, &chunk_info);
                        Ok(chunk.to_owned())
                    }
                    .into()
                }))
                .map(|v| numpy::PyArray::from_owned_array(py, v).into_py(py))
            },
            |n| Err(PyErr::new::<PyException, _>(format!(
                "{}-dimensional tensor resolving not yet implemented.",
                n
            )))
        )
    }

    fn resolve_scalar(&mut self, v: ScalarOperator) -> PyResult<f32> {
        let op: palace_core::operators::scalar::ScalarOperator<StaticElementType<f32>> =
            v.try_into()?;
        let op_ref = &op;
        map_err(self.inner.resolve(None, false, |ctx, _| {
            async move { Ok(ctx.submit(op_ref.request_scalar()).await) }.into()
        }))
    }
}

#[pyclass(unsendable)]
pub struct Window {
    event_loop: winit::event_loop::EventLoop<()>,
    window: palace_core::vulkan::window::Window,
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

            map_err(palace_core::vulkan::window::Window::new(
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
        py: Python,
        gen_frame: &pyo3::types::PyFunction,
        timeout_ms: Option<u64>,
    ) -> PyResult<()> {
        let mut events = palace_core::event::EventSource::default();

        let mut rt = self.runtime.borrow_mut(py);

        let mut res = Ok(());
        self.event_loop.run_return(|event, _, control_flow| {
            let call_res: PyResult<()> = (|| {
                control_flow.set_poll();

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
                        self.window.resize(new_size, &rt.inner.vulkan);
                    }
                    Event::WindowEvent {
                        window_id: _,
                        event,
                    } => {
                        events.add(event);
                    }
                    Event::RedrawRequested(_) => {
                        let end = timeout_ms.map(|to| {
                            std::time::Instant::now() + std::time::Duration::from_millis(to)
                        });
                        let size = self.window.size();
                        let size = [size.y().raw, size.x().raw];
                        let events = Events(events.current_batch());
                        let frame = gen_frame.call((size, events), None)?;
                        let frame = frame.extract::<TensorOperator>()?.try_into()?;

                        let frame_ref = &frame;
                        let window = &mut self.window;
                        rt.inner
                            .resolve(end, false, |ctx, _| {
                                async move { window.render(ctx, frame_ref).await }.into()
                            })
                            .unwrap();
                    }
                    _ => (),
                }
                Ok(())
            })();

            if call_res.is_err() {
                res = call_res;
                control_flow.set_exit();
            }
        });
        res
    }
}
