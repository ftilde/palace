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
        window_content: GuiNode,
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

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Slider {
    val: Py<PyArray0<f64>>,
    min: f64,
    max: f64,
}
#[pymethods]
impl Slider {
    #[new]
    fn new(val: Py<PyArray0<f64>>, min: f64, max: f64) -> Self {
        Self { val, min, max }
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
    Button(Py<Button>),
    Slider(Py<Slider>),
    Label(Py<Label>),
    Horizontal(Py<Horizontal>),
    Vertical(Py<Vertical>),
}

impl GuiNode {
    fn render(&self, py: Python, ui: &mut c::egui::Ui) -> PyResult<()> {
        match self {
            GuiNode::Button(b) => {
                let b = b.borrow(py);
                if ui.button(&b.text).clicked() {
                    b.action.call0(py)?;
                }
            }
            GuiNode::Slider(b) => {
                let b = b.borrow(py);
                let range = b.min..=b.max;
                let v = unsafe { b.val.as_ref(py).get_mut(()).unwrap() };
                ui.add(c::egui::Slider::new(v, range));
            }
            GuiNode::Label(b) => {
                let b = b.borrow(py);
                ui.label(&b.text);
            }
            GuiNode::Horizontal(h) => {
                let h = h.borrow(py);
                ui.horizontal(|ui| {
                    for n in &h.nodes {
                        n.render(py, ui)?;
                    }
                    PyResult::Ok(())
                })
                .inner?
            }
            GuiNode::Vertical(h) => {
                let h = h.borrow(py);
                ui.vertical(|ui| {
                    for n in &h.nodes {
                        n.render(py, ui)?;
                    }
                    PyResult::Ok(())
                })
                .inner?
            }
        }
        Ok(())
    }
}
