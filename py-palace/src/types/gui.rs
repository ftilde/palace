use super::{core::RunTime, Events, TensorOperator};
use numpy::{PyArray0, PyArrayMethods};
use palace_core::operators::gui as c;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum};
use state_link::py::{NodeHandleF32, NodeHandleString, NodeHandleU32};

use pyo3::{exceptions::PyException, prelude::*, types::PyFunction};

#[gen_stub_pyclass]
#[pyclass(unsendable)]
pub struct GuiState {
    inner: c::GuiState,
    runtime: Py<RunTime>,
}

#[pymethods]
impl GuiState {
    #[new]
    fn new(python: Python, runtime: Py<RunTime>) -> Self {
        Self {
            inner: {
                let rt = runtime.borrow(python);
                c::GuiState::on_device(rt.inner.preferred_device)
            },
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

            unsafe { self.inner.deinit(&rt.inner) };
        });
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable)]
pub struct GuiRenderState(Option<c::GuiRenderState>);

#[pymethods]
impl GuiRenderState {
    pub fn render(&mut self, input: TensorOperator) -> PyResult<TensorOperator> {
        if let Some(grs) = self.0.take() {
            Ok(grs
                .render(input.try_into_core_static()?.try_into()?)
                .into_dyn()
                .into())
        } else {
            Err(PyErr::new::<PyException, _>("GuiRenderState::render() was already called previously. Call GuiState::setup first to obtain a new GuiRenderState."))
        }
    }
}

#[gen_stub_pyclass]
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

#[gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct ComboBox {
    title: String,
    alternatives: Vec<String>,
    current: NodeHandleString,
}
#[pymethods]
impl ComboBox {
    #[new]
    fn new(title: String, current: NodeHandleString, alternatives: Vec<String>) -> Self {
        Self {
            title,
            alternatives,
            current,
        }
    }
}

#[gen_stub_pyclass]
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

#[gen_stub_pyclass_enum]
#[derive(Clone, FromPyObject)]
enum SliderVal {
    Array0(Py<PyArray0<f64>>),
    StoreRefF32(NodeHandleF32),
    StoreRefU32(NodeHandleU32),
}

enum ValType {
    Float,
    Int,
}

impl SliderVal {
    fn type_(&self) -> ValType {
        match self {
            SliderVal::Array0(_) | SliderVal::StoreRefF32(_) => ValType::Float,
            SliderVal::StoreRefU32(_) => ValType::Int,
        }
    }

    fn get(&self, py: Python) -> f64 {
        match self {
            SliderVal::Array0(h) => unsafe { *h.bind(py).get(()).unwrap() },
            SliderVal::StoreRefF32(h) => h.load(py) as f64,
            SliderVal::StoreRefU32(h) => h.load(py) as f64,
        }
    }
    fn set(&self, py: Python, v: f64) {
        match self {
            SliderVal::Array0(h) => *unsafe { h.bind(py).get_mut(()).unwrap() } = v,
            SliderVal::StoreRefF32(h) => h.write(py, v as f32).unwrap(),
            SliderVal::StoreRefU32(h) => h.write(py, v as u32).unwrap(),
        };
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Slider {
    val: SliderVal,
    min: f64,
    max: f64,
    logarithmic: bool,
}

#[pymethods]
impl Slider {
    #[new]
    fn new(val: SliderVal, min: f64, max: f64, logarithmic: Option<bool>) -> Self {
        Self {
            val,
            min,
            max,
            logarithmic: logarithmic.unwrap_or(false),
        }
    }
}

#[gen_stub_pyclass]
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

#[gen_stub_pyclass]
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

#[gen_stub_pyclass_enum]
#[derive(FromPyObject, Clone)]
enum GuiNode {
    Button(Button),
    Slider(Slider),
    Label(Label),
    ComboBox(ComboBox),
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
                let mut slider = c::egui::Slider::from_get_set(range, |new| {
                    if let Some(new) = new {
                        b.val.set(py, new);
                    }
                    b.val.get(py)
                })
                .logarithmic(b.logarithmic);

                if let ValType::Int = b.val.type_() {
                    slider = slider.integer();
                }
                ui.add(slider);
            }
            GuiNode::Label(b) => {
                ui.label(&b.text);
            }
            GuiNode::ComboBox(b) => {
                let mut c: String = b.current.load(py);
                c::egui::ComboBox::from_label(&b.title)
                    .selected_text(&c)
                    .show_ui(ui, |ui| {
                        for v in &b.alternatives {
                            ui.selectable_value(&mut c, v.to_owned(), v);
                        }
                    });
                b.current.write(py, c).unwrap();
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
