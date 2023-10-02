use super::{core::RunTime, Events, VolumeOperator};
use numpy::PyArray0;
use vng_core::{operators::gui as c, vulkan::state::VulkanState};

use pyo3::{exceptions::PyException, prelude::*, types::PyFunction};

#[pyclass(unsendable)]
pub struct GuiState {
    inner: c::GuiState,
    runtime: Py<RunTime>,
}

#[pymethods]
impl GuiState {
    #[new]
    fn new(runtime: Py<RunTime>) -> Self {
        Self {
            inner: c::GuiState::default(),
            runtime,
        }
    }

    fn setup(
        &self,
        py: Python,
        events: &mut Events,
        mut window_content: GuiNode,
    ) -> PyResult<GuiRenderState> {
        let mut err = None;
        let grs = self.inner.setup(&mut events.0, |ctx| {
            let res =
                c::egui::Window::new("Settings").show(ctx, |ui| window_content.render(py, ui));

            if let Some(res) = res {
                if let Some(res) = res.inner {
                    if let Err(e) = res {
                        err = Some(e);
                    }
                }
            }
        });

        // Due to... philosophical differences in API design between python and rust (imgui) we
        // need to progagate state changed via gui in a separate step _after_ all the egui
        // machinery.
        window_content.propagate_values(py);

        if let Some(err) = err {
            Err(err)
        } else {
            Ok(GuiRenderState(Some(grs)))
        }
    }
}

impl Drop for GuiState {
    fn drop(&mut self) {
        Python::with_gil(|py| {
            let rt = self.runtime.borrow(py);

            //TODO: Hm, not sure if this works out to well in a multi-device scenario... We have to
            //investigate how to fix that.
            unsafe {
                self.inner
                    .deinitialize(&rt.inner.vulkan.device_contexts()[0])
            };
        });
    }
}

#[pyclass(unsendable)]
pub struct GuiRenderState(Option<c::GuiRenderState>);

#[pymethods]
impl GuiRenderState {
    pub fn render(&mut self, input: VolumeOperator) -> PyResult<VolumeOperator> {
        if let Some(grs) = self.0.take() {
            Ok(grs.render(input.into()).into())
        } else {
            Err(PyErr::new::<PyException, _>("GuiRenderState::render() was already called previously. Call GuiState::setup first to obtain a new GuiRenderState."))
        }
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Button {
    text: String,
    action: Py<PyFunction>,
}
#[pymethods]
impl Button {
    #[new]
    fn new(text: String, action: Py<PyFunction>) -> Self {
        Self { text, action }
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Label {
    text: String,
}
#[pymethods]
impl Label {
    #[new]
    fn new(text: String) -> Self {
        Self { text }
    }
}

#[derive(Clone, FromPyObject)]
enum SliderVal {
    Array0(Py<PyArray0<f64>>),
}

impl SliderVal {
    fn get(&self, py: Python) -> f64 {
        match self {
            SliderVal::Array0(ref v) => unsafe { *v.as_ref(py).get(()).unwrap() },
        }
    }
    fn set(&self, py: Python, v: f64) {
        *match self {
            SliderVal::Array0(ref v) => unsafe { v.as_ref(py).get_mut(()).unwrap() },
        } = v;
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Slider {
    val: SliderVal,
    proxy: f64,
    proxy_orig: f64,
    min: f64,
    max: f64,
}

#[pymethods]
impl Slider {
    #[new]
    fn new(py: Python, val: SliderVal, min: f64, max: f64) -> Self {
        let proxy = val.get(py);
        Self {
            val,
            proxy,
            proxy_orig: proxy,
            min,
            max,
        }
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Horizontal {
    nodes: Vec<GuiNode>,
}
#[pymethods]
impl Horizontal {
    #[new]
    fn new(nodes: Vec<GuiNode>) -> Self {
        Self { nodes }
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Vertical {
    nodes: Vec<GuiNode>,
}
#[pymethods]
impl Vertical {
    #[new]
    fn new(nodes: Vec<GuiNode>) -> Self {
        Self { nodes }
    }
}

#[derive(FromPyObject, Clone)]
enum GuiNode {
    Button(Button),
    Slider(Slider),
    Label(Label),
    Horizontal(Horizontal),
    Vertical(Vertical),
}

impl GuiNode {
    fn render(&mut self, py: Python, ui: &mut c::egui::Ui) -> PyResult<()> {
        match self {
            GuiNode::Button(b) => {
                if ui.button(&b.text).clicked() {
                    b.action.call0(py)?;
                }
            }
            GuiNode::Slider(b) => {
                let range = b.min..=b.max;
                ui.add(c::egui::Slider::new(&mut b.proxy, range));
            }
            GuiNode::Label(b) => {
                ui.label(&b.text);
            }
            GuiNode::Horizontal(h) => {
                ui.horizontal(|ui| {
                    for n in &mut h.nodes {
                        n.render(py, ui)?;
                    }
                    PyResult::Ok(())
                })
                .inner?
            }
            GuiNode::Vertical(h) => {
                ui.vertical(|ui| {
                    for n in &mut h.nodes {
                        n.render(py, ui)?;
                    }
                    PyResult::Ok(())
                })
                .inner?
            }
        }
        Ok(())
    }

    fn propagate_values(&self, py: Python) {
        match self {
            GuiNode::Button(_) => {}
            GuiNode::Slider(v) => {
                if v.proxy != v.proxy_orig {
                    v.val.set(py, v.proxy);
                }
            }
            GuiNode::Label(_) => {}
            GuiNode::Horizontal(h) => {
                for n in &h.nodes {
                    n.propagate_values(py)
                }
            }
            GuiNode::Vertical(h) => {
                for n in &h.nodes {
                    n.propagate_values(py)
                }
            }
        }
    }
}
