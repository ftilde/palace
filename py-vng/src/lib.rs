use derive_more::{From, Into};
use numpy::PyReadonlyArray1;
use pyo3::{exceptions::PyIOError, prelude::*};
use vng_core::{
    array::{ArrayMetaData, VolumeMetaData},
    data::{ChunkCoordinate, LocalVoxelPosition, Vector},
    operator::Operator,
    operators::{array::ArrayOperator, volume::VolumeOperator},
};
use vng_vvd::VvdVolumeSourceState;

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
    e.map_err(|e| PyErr::new::<PyIOError, _>(format!("{}", e)))
}

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
struct ScalarOperatorF32(Operator<(), f32>);

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
struct ArrayMetadataOperator(Operator<(), ArrayMetaData>);

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
struct VolumeMetadataOperator(Operator<(), VolumeMetaData>);

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
struct ArrayValueOperator(Operator<Vector<1, ChunkCoordinate>, f32>);

#[pyclass(unsendable)]
#[derive(Clone, From, Into)]
struct VolumeValueOperator(Operator<Vector<3, ChunkCoordinate>, f32>);

#[pyclass(unsendable)]
#[derive(Clone)]
struct PyVolumeOperator {
    pub metadata: VolumeMetadataOperator,
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

//fn unpack_f32(val: PyObject, py: Python) -> PyResult<Operator<(), f32>> {
//    if let Ok(v) = val.extract::<f32>(py) {
//        Ok(vng_core::operators::scalar::constant_pod(v))
//    } else {
//        val.extract::<ScalarOperatorF32>(py).map(|v| v.0)
//    }
//}

trait PyPresentValue<'a>: FromPyObject<'a> {
    type Operator: FromPyObject<'a>;

    fn make_operator(v: Self) -> Self::Operator;
}

impl<'a> PyPresentValue<'a> for f32 {
    type Operator = ScalarOperatorF32;

    fn make_operator(v: Self) -> Self::Operator {
        ScalarOperatorF32(vng_core::operators::scalar::constant_pod(v))
    }
}

impl<'a> PyPresentValue<'a> for numpy::borrow::PyReadonlyArray1<'a, f32> {
    type Operator = PyArrayOperator;

    fn make_operator(v: Self) -> Self::Operator {
        vng_core::operators::array::from_rc(v.as_slice().unwrap().into()).into()
    }
}

struct ToOperator<'a, S: PyPresentValue<'a>>(S::Operator);

impl<'source, S: PyPresentValue<'source>> FromPyObject<'source> for ToOperator<'source, S> {
    fn extract(val: &'source PyAny) -> PyResult<Self> {
        let v = if let Ok(v) = val.extract::<S::Operator>() {
            v
        } else {
            S::make_operator(val.extract::<S>()?)
        };
        Ok(ToOperator(v))
    }
}

#[pyfunction]
fn mean(vol: PyVolumeOperator) -> ScalarOperatorF32 {
    vng_core::operators::volume_gpu::mean(vol.into()).into()
}

#[pyfunction]
fn linear_rescale(
    vol: PyVolumeOperator,
    scale: ToOperator<f32>,
    offset: ToOperator<f32>,
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
    zyx: [ToOperator<'py, PyReadonlyArray1<'py, f32>>; 3],
) -> PyResult<PyVolumeOperator> {
    let [z, y, x] = zyx;
    Ok(vng_core::operators::volume_gpu::separable_convolution(
        vol.into(),
        [z.0.into(), y.0.into(), x.0.into()],
    )
    .into())
}

/// A Python module implemented in Rust.
#[pymodule]
fn vng(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open_volume, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(linear_rescale, m)?)?;
    m.add_function(wrap_pyfunction!(separable_convolution, m)?)?;
    m.add_class::<ScalarOperatorF32>()?;
    m.add_class::<RunTime>()?;
    Ok(())
}
