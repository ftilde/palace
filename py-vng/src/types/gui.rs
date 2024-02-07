use super::{core::RunTime, Events, TensorOperator};
use numpy::PyArray0;
use state_link::py::NodeHandleF32;
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
    pub fn render(&mut self, input: TensorOperator) -> PyResult<TensorOperator> {
        if let Some(grs) = self.0.take() {
            grs.render(input.try_into()?).try_into()
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
    StoreRef(NodeHandleF32),
}

impl SliderVal {
    fn get(&self, py: Python) -> f64 {
        match self {
            SliderVal::Array0(h) => unsafe { *h.as_ref(py).get(()).unwrap() },
            SliderVal::StoreRef(h) => h.load(py) as f64,
        }
    }
    fn set(&self, py: Python, v: f64) {
        match self {
            SliderVal::Array0(h) => *unsafe { h.as_ref(py).get_mut(()).unwrap() } = v,
            SliderVal::StoreRef(h) => h.write(py, v as f32).unwrap(),
        };
    }
}

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Slider {
    val: SliderVal,
    min: f64,
    max: f64,
}

#[pymethods]
impl Slider {
    #[new]
    fn new(val: SliderVal, min: f64, max: f64) -> Self {
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
                ui.add(c::egui::Slider::from_get_set(range, |new| {
                    if let Some(new) = new {
                        b.val.set(py, new);
                    }
                    b.val.get(py)
                }));
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
}
