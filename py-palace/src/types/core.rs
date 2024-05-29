use std::time::Duration;

use crate::map_err;
use palace_core::{dtypes::StaticElementType, vec::Vector};
use pyo3::{exceptions::PyException, prelude::*};

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

    fn run_with_window(
        &mut self,
        gen_frame: &pyo3::types::PyFunction,
        timeout_ms: u64,
    ) -> PyResult<()> {
        crate::map_err(palace_winit::run_with_window(
            &mut self.inner,
            Duration::from_millis(timeout_ms),
            |_event_loop, window, rt, events, timeout| {
                let size = window.size();
                let size = [size.y().raw, size.x().raw];
                let events = Events(events);
                let frame = gen_frame.call((size, events), None)?;
                let frame = frame.extract::<TensorOperator>()?.try_into()?;

                let frame_ref = &frame;
                let version = rt.resolve(Some(timeout), false, |ctx, _| {
                    async move { window.render(ctx, frame_ref).await }.into()
                })?;

                Ok(version)
            },
        ))
    }
}
