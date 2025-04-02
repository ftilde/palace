use derive_more::{From, Into};
use palace_core::{
    dim::{D2, D3},
    event as c,
    mat::Matrix,
    vec::Vector,
};
use pyo3::{exceptions::PyException, prelude::*, types::PyFunction};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};

#[gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(From, Into)]
pub struct Events(pub palace_core::event::EventStream);

#[gen_stub_pymethods]
#[pymethods]
impl Events {
    #[staticmethod]
    fn none() -> Self {
        Events(palace_core::event::EventStream::empty())
    }
    fn act(&mut self, py: Python, behaviours: Vec<Behaviour>) -> PyResult<()> {
        let mut err = None;
        self.0.act(|e| {
            let mut e: c::EventChain = e.into();
            for behaviour in &behaviours {
                let (ec, error) = behaviour.chain(e, py);
                e = ec;
                collect_err(&mut err, error);
            }
            e
        });
        if let Some(err) = err {
            Err(err)
        } else {
            Ok(())
        }
    }

    fn transform(&mut self, t: Matrix<D3, f32>) -> Self {
        let transform = |pos: Vector<D2, i32>| {
            t.transform(&pos.map(|v| v as f32))
                .map(|v| v.round() as i32)
        };
        let mut inner = palace_core::event::EventStream::with_state(
            self.0.latest_state().clone().transform(transform),
        );

        self.0.act(|e| {
            inner.add(e.clone().transform(transform));
            c::EventChain::Available(e)
        });

        Self(inner)
    }

    fn latest_state(&self) -> EventState {
        EventState(self.0.latest_state().clone())
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(From, Into)]
pub struct EventState(pub palace_core::event::EventState);

#[gen_stub_pymethods]
#[pymethods]
impl EventState {
    fn mouse_pos(&self) -> Option<Vec<i32>> {
        self.0
            .mouse_state
            .as_ref()
            .map(|m| m.pos.into_iter().collect())
    }

    fn is_down(&self, key: &str) -> PyResult<bool> {
        let key = translate_to_key_err(&key)?;
        Ok(self.0.key(key).down())
    }
}

#[gen_stub_pyclass_enum]
#[pyclass(unsendable, eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Middle,
    Right,
}

impl Into<palace_core::event::MouseButton> for MouseButton {
    fn into(self) -> palace_core::event::MouseButton {
        match self {
            MouseButton::Left => palace_core::event::MouseButton::Left,
            MouseButton::Middle => palace_core::event::MouseButton::Middle,
            MouseButton::Right => palace_core::event::MouseButton::Right,
        }
    }
}

#[gen_stub_pyclass_enum]
#[derive(FromPyObject, Clone, Debug, From)]
enum Behaviour {
    OnMouseDrag(OnMouseDrag),
    OnMouseClick(OnMouseClick),
    OnWheelMove(OnWheelMove),
    OnKeyPress(OnKeyPress),
    Conditional(Conditional),
}

fn collect_err(c: &mut Option<PyErr>, n: Option<PyErr>) {
    if c.is_none() {
        *c = n;
    }
}

impl Behaviour {
    pub fn chain(&self, e: c::EventChain, py: Python) -> (c::EventChain, Option<PyErr>) {
        let mut err = None;
        (
            match self {
                Behaviour::OnMouseDrag(b) => e.chain(c::OnMouseDrag(b.0.into(), |p, d| {
                    let p = [p.y(), p.x()];
                    let d = [d.y(), d.x()];
                    collect_err(&mut err, b.1.call(py, (p, d), None).err());
                })),
                Behaviour::OnMouseClick(b) => e.chain(c::OnMouseClick(b.0.into(), |p| {
                    let p = [p.y(), p.x()];
                    collect_err(&mut err, b.1.call(py, (p,), None).err());
                })),
                Behaviour::OnWheelMove(b) => e.chain(c::OnWheelMove(|d, s| {
                    if let Some(m_state) = &s.mouse_state {
                        let pos = m_state.pos.map(|v| v as f32);
                        collect_err(&mut err, b.0.call(py, (d, [pos.y(), pos.x()]), None).err());
                    }
                })),
                Behaviour::OnKeyPress(b) => e.chain(c::OnKeyPress(b.0, || {
                    collect_err(&mut err, b.1.call0(py).err());
                })),
                Behaviour::Conditional(b) => e.chain(|e: c::Event| {
                    match b
                        .1
                        .call(py, (EventState(e.state.clone()),), None)
                        .and_then(|r| r.extract::<bool>(py))
                    {
                        Ok(res) => {
                            if res {
                                let (c, e) = b.0.chain(c::EventChain::Available(e), py);
                                collect_err(&mut err, e);
                                c
                            } else {
                                c::EventChain::Available(e)
                            }
                        }
                        Err(er) => {
                            collect_err(&mut err, Some(er));
                            c::EventChain::Available(e)
                        }
                    }
                }),
            },
            err,
        )
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(Clone, Debug)]
pub struct Conditional(Box<Behaviour>, Py<PyFunction>);

#[gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(Clone, Debug)]
pub struct OnMouseDrag(MouseButton, Py<PyFunction>);

#[pymethods]
impl OnMouseDrag {
    #[new]
    fn new(b: MouseButton, f: Py<PyFunction>) -> Self {
        Self(b, f)
    }

    fn when(&self, c: Py<PyFunction>) -> Conditional {
        Conditional(Box::new(self.clone().into()), c)
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(Clone, Debug)]
pub struct OnMouseClick(MouseButton, Py<PyFunction>);

#[pymethods]
impl OnMouseClick {
    #[new]
    fn new(b: MouseButton, f: Py<PyFunction>) -> Self {
        Self(b, f)
    }

    fn when(&self, c: Py<PyFunction>) -> Conditional {
        Conditional(Box::new(self.clone().into()), c)
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(Clone, Debug)]
pub struct OnWheelMove(Py<PyFunction>);

#[pymethods]
impl OnWheelMove {
    #[new]
    fn new(f: Py<PyFunction>) -> Self {
        Self(f)
    }

    fn when(&self, c: Py<PyFunction>) -> Conditional {
        Conditional(Box::new(self.clone().into()), c)
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable)]
#[derive(Clone, Debug)]
pub struct OnKeyPress(c::Key, Py<PyFunction>);

#[pymethods]
impl OnKeyPress {
    #[new]
    fn new(key: String, f: Py<PyFunction>) -> PyResult<Self> {
        let key = translate_to_key_err(&key)?;
        Ok(Self(key, f))
    }

    fn when(&self, c: Py<PyFunction>) -> Conditional {
        Conditional(Box::new(self.clone().into()), c)
    }
}

fn translate_to_key_err(key: &str) -> PyResult<c::Key> {
    translate_to_key(&key)
        .ok_or_else(|| PyErr::new::<PyException, _>(format!("Unknown key identifier {}", key)))
}
fn translate_to_key(s: &str) -> Option<c::Key> {
    use c::Key::*;

    Some(match s {
        "Key1" => Digit1,
        "Key2" => Digit2,
        "Key3" => Digit3,
        "Key4" => Digit4,
        "Key5" => Digit5,
        "Key6" => Digit6,
        "Key7" => Digit7,
        "Key8" => Digit8,
        "Key9" => Digit9,
        "Key0" => Digit0,
        "A" => KeyA,
        "B" => KeyB,
        "C" => KeyC,
        "D" => KeyD,
        "E" => KeyE,
        "F" => KeyF,
        "G" => KeyG,
        "H" => KeyH,
        "I" => KeyI,
        "J" => KeyJ,
        "K" => KeyK,
        "L" => KeyL,
        "M" => KeyM,
        "N" => KeyN,
        "O" => KeyO,
        "P" => KeyP,
        "Q" => KeyQ,
        "R" => KeyR,
        "S" => KeyS,
        "T" => KeyT,
        "U" => KeyU,
        "V" => KeyV,
        "W" => KeyW,
        "X" => KeyX,
        "Y" => KeyY,
        "Z" => KeyZ,
        "Escape" => Escape,
        "F1" => F1,
        "F2" => F2,
        "F3" => F3,
        "F4" => F4,
        "F5" => F5,
        "F6" => F6,
        "F7" => F7,
        "F8" => F8,
        "F9" => F9,
        "F10" => F10,
        "F11" => F11,
        "F12" => F12,
        "F13" => F13,
        "F14" => F14,
        "F15" => F15,
        "F16" => F16,
        "F17" => F17,
        "F18" => F18,
        "F19" => F19,
        "F20" => F20,
        "F21" => F21,
        "F22" => F22,
        "F23" => F23,
        "F24" => F24,
        "Pause" => Pause,
        "Insert" => Insert,
        "Home" => Home,
        "Delete" => Delete,
        "End" => End,
        "PageDown" => PageDown,
        "PageUp" => PageUp,
        "Left" => ArrowLeft,
        "Up" => ArrowUp,
        "Right" => ArrowRight,
        "Down" => ArrowDown,
        "Back" => Backspace,
        "Return" => Enter,
        "Space" => Space,
        "Numpad0" => Numpad0,
        "Numpad1" => Numpad1,
        "Numpad2" => Numpad2,
        "Numpad3" => Numpad3,
        "Numpad4" => Numpad4,
        "Numpad5" => Numpad5,
        "Numpad6" => Numpad6,
        "Numpad7" => Numpad7,
        "Numpad8" => Numpad8,
        "Numpad9" => Numpad9,
        "NumpadAdd" => NumpadAdd,
        "NumpadDivide" => NumpadDivide,
        "NumpadDecimal" => NumpadDecimal,
        "NumpadComma" => NumpadComma,
        "NumpadEnter" => NumpadEnter,
        "NumpadEquals" => NumpadEqual,
        "NumpadMultiply" => NumpadMultiply,
        "NumpadSubtract" => NumpadSubtract,
        "Backslash" => Backslash,
        "Comma" => Comma,
        "Convert" => Convert,
        "MediaSelect" => MediaSelect,
        "MediaStop" => MediaStop,
        "Minus" => Minus,
        "Period" => Period,
        "Power" => Power,
        "Semicolon" => Semicolon,
        "Slash" => Slash,
        "Sleep" => Sleep,
        "Tab" => Tab,
        "Copy" => Copy,
        "Paste" => Paste,
        "Cut" => Cut,
        "ShiftLeft" => ShiftLeft,
        "ShiftRight" => ShiftRight,
        "ControlLeft" => ControlLeft,
        "ControlRight" => ControlRight,
        "AltLeft" => AltLeft,
        "AltRight" => AltRight,
        _ => return None,
    })
}
