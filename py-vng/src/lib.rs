use derive_more::{From, Into};
use numpy::PyReadonlyArray2;
use pyo3::{
    exceptions::{PyException, PyIOError},
    prelude::*,
};
use vng_core::{
    array::{ArrayMetaData, ImageMetaData, VolumeMetaData},
    cgmath::Matrix,
    data::{ChunkCoordinate, LocalVoxelPosition, Vector},
    operator::Operator,
    operators::{
        array::ArrayOperator,
        volume::{ChunkSize, VolumeOperator},
    },
};
use vng_vvd::VvdVolumeSourceState;
use winit::{
    event::{Event, WindowEvent},
    platform::run_return::EventLoopExtRunReturn,
};

mod conversion;
use conversion::ToOperator;

#[pyclass(unsendable)]
struct RunTime {
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
            inner: vng_core::runtime::RunTime::new(
                storage_size,
                gpu_storage_size,
                num_compute_threads,
            )
            .map_err(|e| PyErr::new::<PyIOError, _>(format!("{}", e)))?,
        })
    }

    fn resolve(&mut self, v: &ScalarOperatorF32) -> PyResult<f32> {
        self.inner
            .resolve(None, |ctx, _| {
                async move { Ok(ctx.submit(v.0.request_scalar()).await) }.into()
            })
            .map_err(|e| PyErr::new::<PyIOError, _>(format!("{}", e)))
    }
}

fn map_err<T>(e: Result<T, vng_core::Error>) -> PyResult<T> {
    e.map_err(|e| PyErr::new::<PyException, _>(format!("{}", e)))
}

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
struct ScalarOperatorF32(Operator<(), f32>);
impl conversion::FromPyValue<f32> for ScalarOperatorF32 {
    fn from_py(v: f32) -> PyResult<Self> {
        Ok(ScalarOperatorF32(
            vng_core::operators::scalar::constant_pod(v),
        ))
    }
}
impl<'source> conversion::FromPyValues<'source> for ScalarOperatorF32 {
    type Converter = conversion::ToOperatorFrom<Self, (f32,)>;
}

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
struct ArrayMetadataOperator(Operator<(), ArrayMetaData>);

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
struct Mat4Operator(Operator<(), vng_core::cgmath::Matrix4<f32>>);

impl<'a> conversion::FromPyValue<PyReadonlyArray2<'a, f32>> for Mat4Operator {
    fn from_py(v: PyReadonlyArray2<'a, f32>) -> PyResult<Self> {
        if v.shape() != [4, 4] {
            return Err(PyException::new_err(format!(
                "Array must be of size [4, 4], but is of size {:?}",
                v.shape()
            )));
        }

        let vals: [f32; 16] = v.as_slice()?.try_into().unwrap();
        let mat: &vng_core::cgmath::Matrix4<f32> = (&vals).into();

        assert!(v.is_c_contiguous());
        // Array is in row major order, but cgmath matrices are column major, so we need to
        // transpose
        let mat = mat.transpose();
        Ok(Mat4Operator(
            vng_core::operators::scalar::constant_as_array(mat),
        ))
    }
}
impl<'source> conversion::FromPyValues<'source> for Mat4Operator {
    type Converter = conversion::ToOperatorFrom<Self, (PyReadonlyArray2<'source, f32>,)>;
}

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
struct VolumeMetadataOperator(Operator<(), VolumeMetaData>);

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
struct ImageMetadataOperator(Operator<(), ImageMetaData>);

impl<'a> conversion::FromPyValue<PyImageMetadata> for ImageMetadataOperator {
    fn from_py(v: PyImageMetadata) -> PyResult<Self> {
        Ok(vng_core::operators::scalar::constant_hash(v.0).into())
    }
}
impl<'source> conversion::FromPyValues<'source> for ImageMetadataOperator {
    type Converter = conversion::ToOperatorFrom<Self, (PyImageMetadata,)>;
}

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
struct PyImageMetadata(ImageMetaData);

#[pymethods]
impl PyImageMetadata {
    #[new]
    fn new(dimensions: [u32; 2], chunk_size: [u32; 2]) -> Self {
        Self(ImageMetaData {
            dimensions: dimensions.into(),
            chunk_size: chunk_size.into(),
        })
    }
}

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
struct ArrayValueOperator(Operator<Vector<1, ChunkCoordinate>, f32>);

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
struct VolumeValueOperator(Operator<Vector<3, ChunkCoordinate>, f32>);

#[pyclass(unsendable)]
#[derive(Clone)]
struct PyVolumeOperator {
    #[pyo3(get, set)]
    pub metadata: VolumeMetadataOperator,
    #[pyo3(get, set)]
    pub bricks: VolumeValueOperator,
}

impl Into<VolumeOperator> for PyVolumeOperator {
    fn into(self) -> VolumeOperator {
        VolumeOperator {
            metadata: self.metadata.into(),
            bricks: self.bricks.into(),
        }
    }
}

impl From<VolumeOperator> for PyVolumeOperator {
    fn from(value: VolumeOperator) -> Self {
        Self {
            metadata: value.metadata.into(),
            bricks: value.bricks.into(),
        }
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
struct PyArrayOperator {
    pub metadata: ArrayMetadataOperator,
    pub bricks: ArrayValueOperator,
}

impl Into<ArrayOperator> for PyArrayOperator {
    fn into(self) -> ArrayOperator {
        ArrayOperator {
            metadata: self.metadata.into(),
            bricks: self.bricks.into(),
        }
    }
}

impl From<ArrayOperator> for PyArrayOperator {
    fn from(value: ArrayOperator) -> Self {
        Self {
            metadata: value.metadata.into(),
            bricks: value.bricks.into(),
        }
    }
}

impl<'a> conversion::FromPyValue<numpy::borrow::PyReadonlyArray1<'a, f32>> for PyArrayOperator {
    fn from_py(v: numpy::borrow::PyReadonlyArray1<'a, f32>) -> PyResult<Self> {
        Ok(vng_core::operators::array::from_rc(v.as_slice()?.into()).into())
    }
}
impl<'source> conversion::FromPyValues<'source> for PyArrayOperator {
    type Converter =
        conversion::ToOperatorFrom<Self, (numpy::borrow::PyReadonlyArray1<'source, f32>,)>;
}

#[pyclass(unsendable)]
struct Window {
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
                    event: _,
                } => {
                    //events.add(event);
                }
                Event::RedrawRequested(_) => {
                    let end = timeout_ms
                        .map(|to| std::time::Instant::now() + std::time::Duration::from_millis(to));
                    let size = self.window.size();
                    let size = [size.y().raw, size.x().raw];
                    let frame = gen_frame.call((size,), None).unwrap();
                    let frame = frame.extract::<PyVolumeOperator>().unwrap().into();

                    let frame_ref = &frame;
                    let window = &mut self.window;
                    runtime
                        .inner
                        .resolve(end, |ctx, _| {
                            async move { window.render(ctx, frame_ref).await }.into()
                        })
                        .unwrap();
                    // Redraw the application.
                    //
                    // It's preferable for applications that do not render continuously to render in
                    // this event rather than in MainEventsCleared, since rendering in here allows
                    // the program to gracefully handle redraws requested by the OS.
                    //next_timeout = Instant::now() + Duration::from_millis(10);
                    //let version = eval_network(
                    //    &mut runtime,
                    //    &mut window,
                    //    &*vol_state,
                    //    &mut state,
                    //    events.current_batch(),
                    //    next_timeout,
                    //)
                    //.unwrap();
                    //if version == DataVersionType::Final {
                    //    //control_flow.set_exit();
                    //}
                    //std::thread::sleep(dbg!(
                    //    next_timeout.saturating_duration_since(std::time::Instant::now())
                    //));
                }
                _ => (),
            }
        });
    }
}

#[pyfunction]
fn open_volume(path: std::path::PathBuf) -> PyResult<PyVolumeOperator> {
    let brick_size_hint = LocalVoxelPosition::fill(32.into());

    let Some(file) = path.file_name() else {
        return map_err(Err("No file name in path".into()));
    };
    let file = file.to_string_lossy();
    let segments = file.split('.').collect::<Vec<_>>();

    let vol_source = match segments[..] {
        [.., "vvd"] => Box::new(map_err(VvdVolumeSourceState::open(&path, brick_size_hint))?),
        //[.., "nii"] | [.., "nii", "gz"] => {
        //    Box::new(map_err(NiftiVolumeSourceState::open_single(path)?))
        //}
        //[.., "hdr"] => {
        //    let data = path.with_extension("img");
        //    Box::new(NiftiVolumeSourceState::open_separate(path, data)?)
        //}
        //[.., "h5"] => Box::new(Hdf5VolumeSourceState::open(path, "/volume".to_string())?),
        _ => {
            return map_err(Err(format!(
                "Unknown volume format for file {}",
                path.to_string_lossy()
            )
            .into()))
        }
    };

    use vng_core::operators::volume::VolumeOperatorState;
    let vol = vol_source.operate();

    Ok(PyVolumeOperator {
        bricks: VolumeValueOperator(vol.bricks),
        metadata: VolumeMetadataOperator(vol.metadata),
    })
}

#[pyfunction]
fn mean(vol: PyVolumeOperator) -> ScalarOperatorF32 {
    vng_core::operators::volume_gpu::mean(vol.into()).into()
}

#[pyclass]
#[derive(Clone)]
struct ChunkSizeFull;

#[derive(Copy, Clone)]
struct PyChunkSize(ChunkSize);

impl<'source> FromPyObject<'source> for PyChunkSize {
    fn extract(val: &'source PyAny) -> PyResult<Self> {
        Ok(PyChunkSize(if let Ok(_) = val.extract::<ChunkSizeFull>() {
            ChunkSize::Full
        } else {
            ChunkSize::Fixed(val.extract::<u32>()?.into())
        }))
    }
}

#[pyfunction]
fn rechunk(vol: PyVolumeOperator, size: [PyChunkSize; 3]) -> PyVolumeOperator {
    let size = Vector::from(size).map(|s: PyChunkSize| s.0);
    vng_core::operators::volume_gpu::rechunk(vol.into(), size).into()
}

#[pyfunction]
fn linear_rescale(
    vol: PyVolumeOperator,
    scale: ToOperator<ScalarOperatorF32>,
    offset: ToOperator<ScalarOperatorF32>,
) -> PyResult<PyVolumeOperator> {
    Ok(
        vng_core::operators::volume_gpu::linear_rescale(
            vol.into(),
            scale.0.into(),
            offset.0.into(),
        )
        .into(),
    )
}

#[pyfunction]
fn separable_convolution<'py>(
    vol: PyVolumeOperator,
    zyx: [ToOperator<PyArrayOperator>; 3],
) -> PyResult<PyVolumeOperator> {
    let [z, y, x] = zyx;
    Ok(vng_core::operators::volume_gpu::separable_convolution(
        vol.into(),
        [z.0.into(), y.0.into(), x.0.into()],
    )
    .into())
}

#[pyfunction]
fn entry_exit_points(
    input_md: VolumeMetadataOperator,
    output_md: ToOperator<ImageMetadataOperator>,
    projection: ToOperator<Mat4Operator>,
) -> PyVolumeOperator {
    vng_core::operators::raycaster::entry_exit_points(
        input_md.into(),
        output_md.0.into(),
        projection.0.into(),
    )
    .into()
}

#[pyfunction]
fn raycast(vol: PyVolumeOperator, entry_exit_points: PyVolumeOperator) -> PyVolumeOperator {
    vng_core::operators::raycaster::raycast(vol.into(), entry_exit_points.into()).into()
}

fn cgmat4_to_numpy(py: Python, mat: vng_core::cgmath::Matrix4<f32>) -> &numpy::PyArray2<f32> {
    // cgmath matrices are row major, but we want column major, so we flip the indices (i, j) below:
    numpy::PyArray2::from_owned_array(
        py,
        numpy::ndarray::Array::from_shape_fn((4, 4), |(j, i)| mat[i][j]),
    )
}

#[pyfunction]
fn look_at(py: Python, eye: [f32; 3], center: [f32; 3], up: [f32; 3]) -> &numpy::PyArray2<f32> {
    let mat = vng_core::cgmath::Matrix4::look_at_rh(eye.into(), center.into(), up.into());
    cgmat4_to_numpy(py, mat)
}

#[pyfunction]
fn perspective(
    py: Python,
    md: PyImageMetadata,
    fov_degree: f32,
    near: f32,
    far: f32,
) -> &numpy::PyArray2<f32> {
    let size = md.0.dimensions;
    let mat = vng_core::cgmath::perspective(
        vng_core::cgmath::Deg(fov_degree),
        size.x().raw as f32 / size.y().raw as f32,
        near,
        far,
    );
    cgmat4_to_numpy(py, mat)
}

#[pymodule]
fn vng(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open_volume, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(linear_rescale, m)?)?;
    m.add_function(wrap_pyfunction!(rechunk, m)?)?;
    m.add_function(wrap_pyfunction!(separable_convolution, m)?)?;
    m.add_function(wrap_pyfunction!(entry_exit_points, m)?)?;
    m.add_function(wrap_pyfunction!(raycast, m)?)?;
    m.add_function(wrap_pyfunction!(look_at, m)?)?;
    m.add_function(wrap_pyfunction!(perspective, m)?)?;
    m.add("chunk_size_full", ChunkSizeFull)?;
    m.add_class::<ScalarOperatorF32>()?;
    m.add_class::<PyImageMetadata>()?;
    m.add_class::<RunTime>()?;
    m.add_class::<Window>()?;
    Ok(())
}
