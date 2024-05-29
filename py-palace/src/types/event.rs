use derive_more::{From, Into};
use palace_core::event as c;
use pyo3::{exceptions::PyException, prelude::*, types::PyFunction};

#[pyclass(unsendable)]
#[derive(From, Into)]
pub struct Events(pub palace_core::event::EventStream);

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
                e = match behaviour {
                    Behaviour::OnMouseDrag(b) => e.chain(c::OnMouseDrag(b.0.into(), |p, d| {
                        let p = [p.y(), p.x()];
                        let d = [d.y(), d.x()];
                        err = b.1.call(py, (p, d), None).err();
                    })),
                    Behaviour::OnMouseClick(b) => e.chain(c::OnMouseClick(b.0.into(), |p| {
                        let p = [p.y(), p.x()];
                        err = b.1.call(py, (p,), None).err();
                    })),
                    Behaviour::OnWheelMove(b) => e.chain(c::OnWheelMove(|d, s| {
                        if let Some(m_state) = &s.mouse_state {
                            let pos = m_state.pos.map(|v| v as f32);
                            err = b.0.call(py, (d, [pos.y(), pos.x()]), None).err();
                        }
                    })),
                    Behaviour::OnKeyPress(b) => e.chain(c::OnKeyPress(b.0, || {
                        err = b.1.call0(py).err();
                    })),
                };
            }
            e
        });
        if let Some(err) = err {
            Err(err)
        } else {
            Ok(())
        }
    }
}

#[pyclass(unsendable)]
#[derive(Clone, Copy, Debug)]
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

#[derive(FromPyObject)]
enum Behaviour {
    OnMouseDrag(OnMouseDrag),
    OnMouseClick(OnMouseClick),
    OnWheelMove(OnWheelMove),
    OnKeyPress(OnKeyPress),
}

#[pyclass(unsendable)]
#[derive(Clone, Debug)]
pub struct OnMouseDrag(MouseButton, Py<PyFunction>);

#[pymethods]
impl OnMouseDrag {
    #[new]
    fn new(b: MouseButton, f: Py<PyFunction>) -> Self {
        Self(b, f)
    }
}

#[pyclass(unsendable)]
#[derive(Clone, Debug)]
pub struct OnMouseClick(MouseButton, Py<PyFunction>);

#[pymethods]
impl OnMouseClick {
    #[new]
    fn new(b: MouseButton, f: Py<PyFunction>) -> Self {
        Self(b, f)
    }
}

#[pyclass(unsendable)]
#[derive(Clone, Debug)]
pub struct OnWheelMove(Py<PyFunction>);

#[pymethods]
impl OnWheelMove {
    #[new]
    fn new(f: Py<PyFunction>) -> Self {
        Self(f)
    }
}

#[pyclass(unsendable)]
#[derive(Clone, Debug)]
pub struct OnKeyPress(c::Key, Py<PyFunction>);

#[pymethods]
impl OnKeyPress {
    #[new]
    fn new(key: String, f: Py<PyFunction>) -> PyResult<Self> {
        let key = translate_to_key(&key).ok_or_else(|| {
            PyErr::new::<PyException, _>(format!("Unknown key identifier {}", key))
        })?;
        Ok(Self(key, f))
    }
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
        _ => return None,
    })
}
