use std::time::Duration;

use crate::map_result;
use palace_core::{
    array::ChunkIndex,
    dim::{DDyn, DynDimension},
    dtypes::{DType, ScalarType, StaticElementType},
    vec::Vector,
};
use pyo3::{exceptions::PyException, prelude::*};

use super::{Events, MaybeEmbeddedTensorOperator, ScalarOperator, TensorOperator};

#[pyclass(unsendable)]
pub struct RunTime {
    pub inner: palace_core::runtime::RunTime,
}

impl RunTime {
    fn resolve_static<T: palace_core::storage::Element + numpy::Element>(
        &mut self,
        py: Python,
        op_ref: &palace_core::operators::tensor::TensorOperator<DDyn, StaticElementType<T>>,
        chunk_i: ChunkIndex,
    ) -> PyResult<PyObject> {
        map_result(self.inner.resolve(None, false, |ctx, _| {
            async move {
                let chunk = ctx.submit(op_ref.chunks.request(chunk_i)).await;
                let chunk_info = op_ref.metadata.chunk_info(chunk_i);
                let chunk = palace_core::data::chunk(&chunk, &chunk_info);
                Ok(chunk.to_owned())
            }
            .into()
        }))
        .map(|v| numpy::PyArray::from_owned_array(py, v).into_py(py))
    }
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
            inner: map_result(palace_core::runtime::RunTime::new(
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
        let v = v.into_inner();
        let op: palace_core::operators::tensor::TensorOperator<DDyn, DType> = v.try_into()?;
        if op.dim().n() != pos.len() {
            return Err(PyErr::new::<PyException, _>(format!(
                "Expected {}-dimensional chunk position for tensor, but got {}",
                op.dim().n(),
                pos.len()
            )));
        }

        let dim_in_chunks = op.metadata.dimension_in_chunks();
        let pos: Vector<DDyn, u32> = pos.try_into().unwrap();

        if !pos.zip(&dim_in_chunks, |l, r| l < r.raw).hand() {
            return Err(PyErr::new::<PyException, _>(format!(
                "Chunk position {:?} out of range for tensor dimension-in-chunks {:?}",
                pos,
                dim_in_chunks.raw(),
            )));
        }

        let dtype = op.dtype();
        if dtype.size != 1 {
            return Err(PyErr::new::<PyException, _>(format!(
                "Expected scalar dtype, but got with size {}",
                dtype.size
            )));
        }

        let chunk_id = op.metadata.chunk_index(&pos.chunk());

        match dtype.scalar {
            ScalarType::U8 => self.resolve_static::<u8>(py, &op.try_into().unwrap(), chunk_id),
            ScalarType::I8 => self.resolve_static::<i16>(py, &op.try_into().unwrap(), chunk_id),
            ScalarType::U16 => self.resolve_static::<u16>(py, &op.try_into().unwrap(), chunk_id),
            ScalarType::I16 => self.resolve_static::<i16>(py, &op.try_into().unwrap(), chunk_id),
            ScalarType::F32 => self.resolve_static::<f32>(py, &op.try_into().unwrap(), chunk_id),
            ScalarType::U32 => self.resolve_static::<u32>(py, &op.try_into().unwrap(), chunk_id),
            ScalarType::I32 => self.resolve_static::<i32>(py, &op.try_into().unwrap(), chunk_id),
        }
    }

    fn resolve_scalar(&mut self, v: ScalarOperator) -> PyResult<f32> {
        let op: palace_core::operators::scalar::ScalarOperator<StaticElementType<f32>> =
            v.try_into()?;
        let op_ref = &op;
        map_result(self.inner.resolve(None, false, |ctx, _| {
            async move { Ok(ctx.submit(op_ref.request_scalar()).await) }.into()
        }))
    }

    fn run_with_window(
        &mut self,
        gen_frame: &pyo3::types::PyFunction,
        timeout_ms: u64,
    ) -> PyResult<()> {
        crate::map_result(palace_winit::run_with_window(
            &mut self.inner,
            Duration::from_millis(timeout_ms),
            |_event_loop, window, rt, events, timeout| {
                let size = window.size();
                let size = [size.y().raw, size.x().raw];
                let events = Events(events);
                let frame = gen_frame.call((size, events), None)?;
                let frame = frame.extract::<TensorOperator>()?.try_into_core_static()?;
                let frame = frame.try_into()?;

                let frame_ref = &frame;
                let version = rt.resolve(Some(timeout), false, |ctx, _| {
                    async move { window.render(ctx, frame_ref).await }.into()
                })?;

                Ok(version)
            },
        ))
    }
}
