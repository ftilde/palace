use derive_more::{From, Into};
use pyo3::{exceptions::PyException, prelude::*, types::PyFunction};
use vng_core::event as c;

#[pyclass(unsendable)]
#[derive(From, Into)]
pub struct Events(pub vng_core::event::EventStream);

#[pymethods]
impl Events {
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
                    Behaviour::OnWheelMove(b) => e.chain(c::OnWheelMove(|d, _s| {
                        err = b.0.call(py, (d,), None).err();
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

impl Into<vng_core::event::MouseButton> for MouseButton {
    fn into(self) -> vng_core::event::MouseButton {
        match self {
            MouseButton::Left => vng_core::event::MouseButton::Left,
            MouseButton::Middle => vng_core::event::MouseButton::Middle,
            MouseButton::Right => vng_core::event::MouseButton::Right,
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
        "Key1" => Key1,
        "Key2" => Key2,
        "Key3" => Key3,
        "Key4" => Key4,
        "Key5" => Key5,
        "Key6" => Key6,
        "Key7" => Key7,
        "Key8" => Key8,
        "Key9" => Key9,
        "Key0" => Key0,
        "A" => A,
        "B" => B,
        "C" => C,
        "D" => D,
        "E" => E,
        "F" => F,
        "G" => G,
        "H" => H,
        "I" => I,
        "J" => J,
        "K" => K,
        "L" => L,
        "M" => M,
        "N" => N,
        "O" => O,
        "P" => P,
        "Q" => Q,
        "R" => R,
        "S" => S,
        "T" => T,
        "U" => U,
        "V" => V,
        "W" => W,
        "X" => X,
        "Y" => Y,
        "Z" => Z,
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
        "Snapshot" => Snapshot,
        "Scroll" => Scroll,
        "Pause" => Pause,
        "Insert" => Insert,
        "Home" => Home,
        "Delete" => Delete,
        "End" => End,
        "PageDown" => PageDown,
        "PageUp" => PageUp,
        "Left" => Left,
        "Up" => Up,
        "Right" => Right,
        "Down" => Down,
        "Back" => Back,
        "Return" => Return,
        "Space" => Space,
        "Compose" => Compose,
        "Caret" => Caret,
        "Numlock" => Numlock,
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
        "NumpadEquals" => NumpadEquals,
        "NumpadMultiply" => NumpadMultiply,
        "NumpadSubtract" => NumpadSubtract,
        "AbntC1" => AbntC1,
        "AbntC2" => AbntC2,
        "Apostrophe" => Apostrophe,
        "Apps" => Apps,
        "Asterisk" => Asterisk,
        "At" => At,
        "Ax" => Ax,
        "Backslash" => Backslash,
        "Calculator" => Calculator,
        "Capital" => Capital,
        "Colon" => Colon,
        "Comma" => Comma,
        "Convert" => Convert,
        "Equals" => Equals,
        "Grave" => Grave,
        "Kana" => Kana,
        "Kanji" => Kanji,
        "LAlt" => LAlt,
        "LBracket" => LBracket,
        "LControl" => LControl,
        "LShift" => LShift,
        "LWin" => LWin,
        "Mail" => Mail,
        "MediaSelect" => MediaSelect,
        "MediaStop" => MediaStop,
        "Minus" => Minus,
        "Mute" => Mute,
        "MyComputer" => MyComputer,
        "NavigateForward" => NavigateForward,
        "NavigateBackward" => NavigateBackward,
        "NextTrack" => NextTrack,
        "NoConvert" => NoConvert,
        "OEM102" => OEM102,
        "Period" => Period,
        "PlayPause" => PlayPause,
        "Plus" => Plus,
        "Power" => Power,
        "PrevTrack" => PrevTrack,
        "RAlt" => RAlt,
        "RBracket" => RBracket,
        "RControl" => RControl,
        "RShift" => RShift,
        "RWin" => RWin,
        "Semicolon" => Semicolon,
        "Slash" => Slash,
        "Sleep" => Sleep,
        "Stop" => Stop,
        "Sysrq" => Sysrq,
        "Tab" => Tab,
        "Underline" => Underline,
        "Unlabeled" => Unlabeled,
        "VolumeDown" => VolumeDown,
        "VolumeUp" => VolumeUp,
        "Wake" => Wake,
        "WebBack" => WebBack,
        "WebFavorites" => WebFavorites,
        "WebForward" => WebForward,
        "WebHome" => WebHome,
        "WebRefresh" => WebRefresh,
        "WebSearch" => WebSearch,
        "WebStop" => WebStop,
        "Yen" => Yen,
        "Copy" => Copy,
        "Paste" => Paste,
        "Cut" => Cut,
        _ => return None,
    })
}
