use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, VirtualKeyCode, WindowEvent},
};

use crate::data::Vector;

pub type Change = winit::event::WindowEvent<'static>;

pub trait Behavior {
    fn input(&mut self, event: Event) -> EventChain;
}

impl<F: FnMut(Event) -> EventChain> Behavior for F {
    fn input(&mut self, event: Event) -> EventChain {
        self(event)
    }
}

pub type Key = VirtualKeyCode;
pub struct OnKeyPress<F: FnMut()>(pub Key, pub F);

impl<F: FnMut()> Behavior for OnKeyPress<F> {
    fn input(&mut self, event: Event) -> EventChain {
        if let winit::event::WindowEvent::KeyboardInput { input, .. } = event.change {
            if input.state == ElementState::Pressed {
                if matches!(input.virtual_keycode, Some(v) if v == self.0) {
                    (self.1)();
                    return EventChain::Consumed;
                }
            }
        }
        EventChain::Available(event)
    }
}

pub use winit::event::ModifiersState;
pub use winit::event::MouseButton;

pub type MousePosition = Vector<2, i32>;
pub type MouseDelta = Vector<2, i32>;

impl TryFrom<PhysicalPosition<f64>> for MousePosition {
    type Error = std::num::TryFromIntError;

    fn try_from(position: PhysicalPosition<f64>) -> Result<Self, Self::Error> {
        let x: i32 = (position.x as i64).try_into()?;
        let y: i32 = (position.y as i64).try_into()?;
        Ok(Vector::from([y, x]))
    }
}

pub struct OnMouseClick<F: FnMut(MousePosition)>(pub MouseButton, pub F);

impl<F: FnMut(MousePosition)> Behavior for OnMouseClick<F> {
    fn input(&mut self, event: Event) -> EventChain {
        match &event.change {
            WindowEvent::MouseInput { state, button, .. }
                if *state == ElementState::Pressed && *button == self.0 =>
            {
                if let Some(state) = event.state.mouse_state {
                    (self.1)(state.pos);
                    EventChain::Consumed
                } else {
                    EventChain::Available(event)
                }
            }
            _ => EventChain::Available(event),
        }
    }
}

pub struct OnMouseDrag<F: FnMut(MousePosition, MouseDelta)>(pub MouseButton, pub F);

impl<F: FnMut(MousePosition, MouseDelta)> Behavior for OnMouseDrag<F> {
    fn input(&mut self, event: Event) -> EventChain {
        if event.state.mouse_button(self.0) == ButtonState::Down {
            if let Some(state) = event.state.mouse_state {
                (self.1)(state.pos, state.delta);
                EventChain::Consumed
            } else {
                event.into()
            }
        } else {
            event.into()
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum ButtonState {
    Down,
    Up,
}

#[derive(Clone)]
pub struct MouseState {
    pub pos: MousePosition,
    pub delta: MouseDelta,
}

#[derive(Clone, Default)]
pub struct EventState {
    keys: im_rc::HashMap<Key, ButtonState>,
    mouse_buttons: im_rc::HashMap<MouseButton, ButtonState>,
    pub mouse_state: Option<MouseState>,
}

#[derive(Clone)]
pub struct Event {
    pub change: Change,
    pub state: EventState,
}

impl Event {
    pub fn chain(self, mut b: impl Behavior) -> EventChain {
        b.input(self)
    }

    pub fn transform(mut self, mut t: impl FnMut(MousePosition) -> MousePosition) -> Event {
        if let Some(m) = &mut self.state.mouse_state {
            m.pos = t(m.pos);
        }
        #[allow(deprecated)]
        if let WindowEvent::CursorMoved {
            device_id,
            position,
            modifiers,
        } = self.change
        {
            self.change = WindowEvent::CursorMoved {
                device_id,
                position: {
                    let transformed = t(position.try_into().unwrap());
                    PhysicalPosition {
                        x: transformed.x() as _,
                        y: transformed.y() as _,
                    }
                },
                modifiers,
            };
        }
        self
    }
}

impl EventState {
    pub fn key(&self, key: Key) -> ButtonState {
        self.keys.get(&key).cloned().unwrap_or(ButtonState::Up)
    }
    pub fn mouse_button(&self, button: MouseButton) -> ButtonState {
        self.mouse_buttons
            .get(&button)
            .cloned()
            .unwrap_or(ButtonState::Up)
    }
}

#[derive(Default)]
pub struct EventSource {
    current_state: EventState,
    batch: Vec<Event>,
}

impl EventSource {
    pub fn add(&mut self, diff: Change) {
        match diff {
            WindowEvent::KeyboardInput { input, .. } => {
                if let Some(code) = input.virtual_keycode {
                    let state = match input.state {
                        ElementState::Pressed => ButtonState::Down,
                        ElementState::Released => ButtonState::Up,
                    };
                    self.current_state.keys.insert(code, state);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let state = match state {
                    ElementState::Pressed => ButtonState::Down,
                    ElementState::Released => ButtonState::Up,
                };
                self.current_state.mouse_buttons.insert(button, state);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let old_state = &self.current_state.mouse_state;
                let new_pos = position.try_into().unwrap();
                let delta = if let Some(old_state) = old_state {
                    new_pos - old_state.pos
                } else {
                    [0, 0].into()
                };
                self.current_state.mouse_state = Some(MouseState {
                    pos: new_pos,
                    delta,
                });
            }
            //WindowEvent::CursorEntered { device_id } => todo!(),
            WindowEvent::CursorLeft { .. } => self.current_state.mouse_state = None,
            _ => {}
        }
        self.batch.push(Event {
            change: diff,
            state: self.current_state.clone(),
        });
    }

    pub fn current_batch(&mut self) -> EventStream {
        EventStream(std::mem::take(&mut self.batch))
    }
}

#[derive(Clone)]
pub enum EventChain {
    Consumed,
    Available(Event),
}

impl From<Event> for EventChain {
    fn from(value: Event) -> Self {
        Self::Available(value)
    }
}

impl EventChain {
    pub fn chain(self, mut b: impl Behavior) -> Self {
        if let EventChain::Available(e) = self {
            b.input(e)
        } else {
            EventChain::Consumed
        }
    }
}

#[derive(Default)]
pub struct EventStream(Vec<Event>);

impl EventStream {
    pub fn add(&mut self, event: Event) {
        self.0.push(event);
    }

    pub fn act(&mut self, mut f: impl FnMut(Event) -> EventChain) {
        self.0.retain_mut(|e| {
            let r = e.clone();
            if let EventChain::Available(returned) = f(r) {
                *e = returned;
                true
            } else {
                false
            }
        });
    }
}

pub struct OnWheelMove<F: FnMut(f32)>(pub F);

impl<F: FnMut(f32)> Behavior for OnWheelMove<F> {
    fn input(&mut self, event: Event) -> EventChain {
        match event.change {
            winit::event::WindowEvent::MouseWheel { delta, .. } => {
                let motion = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y.signum(),
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y.signum() as f32,
                };
                (self.0)(motion);
                EventChain::Consumed
            }
            _ => event.into(),
        }
    }
}

pub struct Drag<'a>(pub MouseButton, pub &'a mut Vector<2, i32>);

impl<'a> Behavior for Drag<'a> {
    fn input(&mut self, event: Event) -> EventChain {
        let mut f = OnMouseDrag(self.0, |_, delta| *self.1 = *self.1 + delta);
        f.input(event)
    }
}
