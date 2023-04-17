use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, VirtualKeyCode, WindowEvent},
};

use crate::data::Vector;

pub type Event = winit::event::WindowEvent<'static>;

pub trait Behavior {
    fn input(&mut self, event: Event) -> Option<Event>;
}

pub struct EventChain(pub Option<Event>);

impl EventChain {
    pub fn consumed() -> Self {
        EventChain(None)
    }
    pub fn chain(self, mut b: impl Behavior) -> Self {
        if let Some(e) = self.0 {
            EventChain(b.input(e))
        } else {
            EventChain(None)
        }
    }
}

impl<F: FnMut(Event) -> Option<Event>> Behavior for F {
    fn input(&mut self, event: Event) -> Option<Event> {
        self(event)
    }
}

pub type Key = VirtualKeyCode;
pub struct OnKeyPress<F: FnMut()>(pub Key, pub F);

impl<F: FnMut()> Behavior for OnKeyPress<F> {
    fn input(&mut self, event: Event) -> Option<Event> {
        if let winit::event::WindowEvent::KeyboardInput { input, .. } = event {
            if input.state == ElementState::Pressed {
                if matches!(input.virtual_keycode, Some(v) if v == self.0) {
                    (self.1)();
                    return None;
                }
            }
        }
        Some(event)
    }
}

pub type MouseButton = winit::event::MouseButton;
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

pub struct OnMouseClick<'a, F: FnMut(MousePosition)>(&'a mut MouseState, MouseButton, F);

impl<'a, F: FnMut(MousePosition)> Behavior for OnMouseClick<'a, F> {
    fn input(&mut self, event: Event) -> Option<Event> {
        self.0.update(&event);
        match &event {
            WindowEvent::MouseInput { state, button, .. }
                if *state == ElementState::Pressed && *button == self.1 =>
            {
                if let Some(pos) = self.0.last_pos {
                    (self.2)(pos);
                    None
                } else {
                    Some(event)
                }
            }
            _ => Some(event),
        }
    }
}

pub struct MouseDragState {
    inner: MouseState,
    down: bool,
    button: MouseButton,
}

impl MouseDragState {
    pub fn new(button: MouseButton) -> Self {
        Self {
            inner: Default::default(),
            down: false,
            button,
        }
    }
    pub fn while_pressed<'a, F: FnMut(MousePosition, MouseDelta)>(
        &'a mut self,
        f: F,
    ) -> OnMouseDrag<'a, F> {
        OnMouseDrag(self, f)
    }

    fn update(&mut self, event: &Event) {
        self.inner.update(event);
        match &event {
            WindowEvent::MouseInput { state, button, .. } if self.button == *button => {
                self.down = match *state {
                    ElementState::Pressed => true,
                    ElementState::Released => false,
                }
            }
            WindowEvent::CursorLeft { .. } => {
                self.down = false;
            }
            _ => {}
        }
    }
}

pub struct OnMouseDrag<'a, F: FnMut(MousePosition, MouseDelta)>(&'a mut MouseDragState, F);

impl<'a, F: FnMut(MousePosition, MouseDelta)> Behavior for OnMouseDrag<'a, F> {
    fn input(&mut self, event: Event) -> Option<Event> {
        self.0.update(&event);
        if self.0.down {
            if let (Some(pos), Some(delta)) = (self.0.inner.last_pos, self.0.inner.delta) {
                (self.1)(pos, delta);
                None
            } else {
                Some(event)
            }
        } else {
            Some(event)
        }
    }
}

#[derive(Default)]
pub struct MouseState {
    last_pos: Option<MousePosition>,
    delta: Option<MouseDelta>,
}

impl MouseState {
    pub fn update(&mut self, event: &Event) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                //TODO: check on highdpi screens
                let new_pos = Vector::try_from(*position).unwrap();
                if let Some(prev_pos) = self.last_pos {
                    self.delta = Some(new_pos - prev_pos);
                }
                self.last_pos = Some(new_pos);
            }
            WindowEvent::CursorLeft { .. } => {
                self.delta = None;
                self.last_pos = None;
            }
            _ => {}
        }
    }

    pub fn on_click<'a, F: FnMut(MousePosition)>(
        &'a mut self,
        button: MouseButton,
        f: F,
    ) -> OnMouseClick<'a, F> {
        OnMouseClick(self, button, f)
    }
}

#[derive(Default)]
pub struct EventStream(Vec<Event>);

impl EventStream {
    pub fn add(&mut self, event: Event) {
        self.0.push(event);
    }

    pub fn act(&mut self, mut f: impl FnMut(EventChain) -> EventChain) {
        self.0.retain_mut(|e| {
            let placeholder = Event::Destroyed;
            let r = std::mem::replace(e, placeholder);
            if let EventChain(Some(returned)) = f(EventChain(Some(r))) {
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
    fn input(&mut self, event: Event) -> Option<Event> {
        match event {
            winit::event::WindowEvent::MouseWheel { delta, .. } => {
                let motion = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y.signum(),
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y.signum() as f32,
                };
                (self.0)(motion);
                None
            }
            _ => Some(event),
        }
    }
}
