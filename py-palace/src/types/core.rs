use std::{cell::RefCell, rc::Rc, time::Duration};

use crate::map_result;
use palace_core::{
    array::ChunkIndex,
    dim::{DDyn, DynDimension},
    dtypes::{DType, ScalarType, StaticElementType},
    storage::DataVersionType,
    vec::Vector,
};
use pyo3::{exceptions::PyException, prelude::*, IntoPyObjectExt};
use pyo3_stub_gen::derive::gen_stub_pyclass;

use super::{Events, MaybeEmbeddedTensorOperatorArg, ScalarOperator, TensorOperator};

#[gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct RunTime {
    pub inner: Rc<RefCell<palace_core::runtime::RunTime>>,
}

impl RunTime {
    fn resolve_static<T: palace_core::storage::Element + numpy::Element>(
        &self,
        py: Python,
        op_ref: &palace_core::operators::tensor::TensorOperator<DDyn, StaticElementType<T>>,
        chunk_indices: Vec<ChunkIndex>,
    ) -> PyResult<PyObject> {
        let mut rt = self.inner.borrow_mut();
        let chunk_data = map_result(rt.resolve(None, false, |ctx, _| {
            async move {
                let mut chunks = ctx
                    .run_unordered(chunk_indices.into_iter().enumerate().map(
                        move |(i, chunk_id)| {
                            {
                                async move {
                                    let chunk_handle =
                                        ctx.submit(op_ref.chunks.request(chunk_id)).await;
                                    let chunk_info = op_ref.metadata.chunk_info(chunk_id);
                                    let chunk =
                                        palace_core::data::chunk(&chunk_handle, &chunk_info);
                                    let chunk = ctx
                                        .submit(ctx.spawn_compute(move || chunk.to_owned()))
                                        .await;
                                    (i, chunk)
                                }
                            }
                            .into()
                        },
                    ))
                    .await;

                chunks.sort_by_key(|v| v.0);
                Ok(chunks.into_iter().map(|v| v.1).collect::<Vec<_>>())
            }
            .into()
        }))?;
        chunk_data
            .into_iter()
            .map(|data| numpy::PyArray::from_owned_array(py, data))
            .collect::<Vec<_>>()
            .into_py_any(py)
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(Copy, Clone)]
pub struct DeviceId(pub palace_core::vulkan::DeviceId);

#[pymethods]
impl RunTime {
    #[new]
    #[pyo3(signature = (storage_size, gpu_storage_size, disk_cache_size=None, num_compute_threads=None, devices=Vec::new()))]
    pub fn new(
        storage_size: usize,
        gpu_storage_size: u64,
        disk_cache_size: Option<usize>,
        num_compute_threads: Option<usize>,
        devices: Vec<usize>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Rc::new(
                map_result(
                    palace_core::runtime::RunTime::build()
                        .disk_cache_size_opt(disk_cache_size)
                        .devices(devices)
                        .num_compute_threads_opt(num_compute_threads)
                        .finish(storage_size, gpu_storage_size),
                )?
                .into(),
            ),
        })
    }

    fn resolve(
        &self,
        py: Python,
        v: MaybeEmbeddedTensorOperatorArg,
        positions: Vec<Vec<u32>>,
    ) -> PyResult<PyObject> {
        let v = v.unpack().into_inner();
        let op: palace_core::operators::tensor::TensorOperator<DDyn, DType> = v.try_into()?;
        let dim_in_chunks = op.metadata.dimension_in_chunks();

        let dtype = op.dtype();
        if dtype.size != 1 {
            return Err(PyErr::new::<PyException, _>(format!(
                "Expected scalar dtype, but got with size {}",
                dtype.size
            )));
        }

        let chunk_ids = positions
            .into_iter()
            .map(|pos| {
                if op.dim().n() != pos.len() {
                    return Err(PyErr::new::<PyException, _>(format!(
                        "Expected {}-dimensional chunk position for tensor, but got {}",
                        op.dim().n(),
                        pos.len()
                    )));
                }

                let pos: Vector<DDyn, u32> = pos.try_into().unwrap();

                if !pos.zip(&dim_in_chunks, |l, r| l < r.raw).hand() {
                    return Err(PyErr::new::<PyException, _>(format!(
                        "Chunk position {:?} out of range for tensor dimension-in-chunks {:?}",
                        pos,
                        dim_in_chunks.raw(),
                    )));
                }

                Ok(op.metadata.chunk_index(&pos.chunk()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        match dtype.scalar {
            ScalarType::U8 => self.resolve_static::<u8>(py, &op.try_into().unwrap(), chunk_ids),
            ScalarType::I8 => self.resolve_static::<i8>(py, &op.try_into().unwrap(), chunk_ids),
            ScalarType::U16 => self.resolve_static::<u16>(py, &op.try_into().unwrap(), chunk_ids),
            ScalarType::I16 => self.resolve_static::<i16>(py, &op.try_into().unwrap(), chunk_ids),
            ScalarType::F32 => self.resolve_static::<f32>(py, &op.try_into().unwrap(), chunk_ids),
            ScalarType::U32 => self.resolve_static::<u32>(py, &op.try_into().unwrap(), chunk_ids),
            ScalarType::I32 => self.resolve_static::<i32>(py, &op.try_into().unwrap(), chunk_ids),
            ScalarType::U64 => self.resolve_static::<u64>(py, &op.try_into().unwrap(), chunk_ids),
            ScalarType::I64 => self.resolve_static::<i64>(py, &op.try_into().unwrap(), chunk_ids),
        }
    }

    fn resolve_scalar(&self, v: ScalarOperator) -> PyResult<f32> {
        let op: palace_core::operators::scalar::ScalarOperator<StaticElementType<f32>> =
            v.try_into()?;
        let op_ref = &op;
        let mut rt = self.inner.borrow_mut();
        map_result(rt.resolve(None, false, |ctx, _| {
            async move { Ok(ctx.submit(op_ref.request_scalar()).await) }.into()
        }))
    }

    fn all_devices(&self) -> Vec<DeviceId> {
        let rt = self.inner.borrow();
        rt.all_devices().into_iter().map(DeviceId).collect()
    }

    fn checked_device_id(&self, raw_id: usize) -> Option<DeviceId> {
        let rt = self.inner.borrow();
        rt.checked_device_id(raw_id).map(DeviceId)
    }

    #[pyo3(signature=(gen_frame, timeout_ms, record_task_stream=false, bench=false, display_device=None))]
    fn run_with_window(
        &self,
        gen_frame: &Bound<pyo3::types::PyFunction>,
        timeout_ms: u64,
        record_task_stream: bool,
        bench: bool,
        display_device: Option<usize>,
    ) -> PyResult<()> {
        let mut rt = self.inner.clone();
        let display_device = {
            let rt = rt.borrow();
            if let Some(display_device) = display_device {
                Some(rt.checked_device_id(display_device).ok_or_else(|| {
                    crate::map_err(format!("Invalid device id: {}", display_device).into())
                })?)
            } else {
                None
            }
        };
        palace_winit::run_with_window_wrapper(
            &mut rt,
            Duration::from_millis(timeout_ms),
            display_device,
            |_event_loop, window, rt, events, timeout| {
                let size = window.size();
                let size = [size.y().raw, size.x().raw];
                let events = Events(events);
                let frame = gen_frame.call((size, events), None)?;
                let frame = frame.extract::<TensorOperator>()?.try_into_core_static()?;
                let frame = frame.try_into()?;

                let frame_ref = &frame;
                let mut rt = rt.borrow_mut();
                let version = rt
                    .resolve(Some(timeout), record_task_stream, |ctx, _| {
                        async move { window.render(ctx, frame_ref).await }.into()
                    })
                    .map_err(crate::map_err)?;

                if bench && version == DataVersionType::Final {
                    _event_loop.exit();
                }

                Ok(version)
            },
        )
    }
}
