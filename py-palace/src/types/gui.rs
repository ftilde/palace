use super::{core::RunTime, Events, TensorOperator};
use numpy::{PyArray0, PyArrayMethods};
use palace_core::operators::gui as c;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_complex_enum, gen_stub_pymethods};
use state_link::py::{NodeHandleF32, NodeHandleString, NodeHandleU32};

use pyo3::{exceptions::PyException, prelude::*, types::PyFunction};

#[gen_stub_pyclass]
#[pyclass(unsendable)]
pub struct GuiState {
    inner: c::GuiState,
    runtime: RunTime,
}

#[gen_stub_pymethods]
#[pymethods]
impl GuiState {
    #[new]
    fn new(runtime: RunTime) -> Self {
        Self {
            inner: { c::GuiState::new() },
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
        let rt = self.runtime.inner.borrow();

        unsafe { self.inner.deinit(&rt) };
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable)]
pub struct GuiRenderState(Option<c::GuiRenderState>);

#[gen_stub_pymethods]
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
#[gen_stub_pymethods]
#[pymethods]
impl Button {
    #[new]
    fn new(py: Python, text: String, action: PyObject) -> PyResult<Self> {
        let action = action.extract(py)?;
        Ok(Self { text, action })
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
#[gen_stub_pymethods]
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
#[gen_stub_pymethods]
#[pymethods]
impl Label {
    #[new]
    fn new(text: String) -> Self {
        Self { text }
    }
}

#[gen_stub_pyclass_complex_enum]
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

#[gen_stub_pymethods]
#[pymethods]
impl Slider {
    #[new]
    #[pyo3(signature = (val, min, max, logarithmic=false))]
    fn new(val: SliderVal, min: f64, max: f64, logarithmic: bool) -> Self {
        Self {
            val,
            min,
            max,
            logarithmic,
        }
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Horizontal {
    nodes: Vec<GuiNode>,
}
#[gen_stub_pymethods]
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
#[gen_stub_pymethods]
#[pymethods]
impl Vertical {
    #[new]
    fn new(nodes: Vec<GuiNode>) -> Self {
        Self { nodes }
    }
}

#[gen_stub_pyclass_complex_enum]
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
