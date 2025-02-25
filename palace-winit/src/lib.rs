use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use ash::vk;
use palace_core::runtime::Deadline;
use palace_core::storage::DataVersionType;
use palace_core::vulkan::window::Window as PWindow;
use palace_core::vulkan::DeviceId;
use palace_core::{
    event::{EventSource, EventStream},
    runtime::RunTime,
    vulkan::VulkanContext,
};
use raw_window_handle::HasWindowHandle;
use winit::event_loop::EventLoop;
use winit::window::Window as WWindow;
use winit::{event::WindowEvent, event_loop::ActiveEventLoop};

//#[cfg(target_family = "unix")]
//fn create_surface_wayland(
//    entry: &ash::Entry,
//    instance: &ash::Instance,
//    window_handle: &raw_window_handle::WaylandWindowHandle,
//) -> Option<vk::SurfaceKHR> {
//    let loader = WaylandSurface::new(entry, instance);
//
//    let create_info = vk::WaylandSurfaceCreateInfoKHR::builder()
//        .display(window.wayland_display()?)
//        .surface(window_handle.surface.as_ptr() as _)
//        .build();
//
//    Some(unsafe { loader.create_wayland_surface(&create_info, None) }.unwrap())
//}
//
//#[cfg(target_family = "unix")]
//fn create_surface_x11(
//    entry: &ash::Entry,
//    instance: &ash::Instance,
//    window_handle: &raw_window_handle::WaylandWindowHandle,
//) -> Option<vk::SurfaceKHR> {
//    use winit::platform::x11::WindowExtX11;
//
//    let x11_display = window.xlib_display()?;
//    let x11_window = window.xlib_window()?;
//    let create_info = vk::XlibSurfaceCreateInfoKHR::builder()
//        .window(x11_window as vk::Window)
//        .dpy(x11_display as *mut vk::Display);
//
//    let xlib_surface_loader = XlibSurface::new(entry, instance);
//    Some(unsafe {
//        xlib_surface_loader
//            .create_xlib_surface(&create_info, None)
//            .unwrap()
//    })
//}
//#[cfg(target_family = "windows")]
//fn create_surface(
//    entry: &ash::Entry,
//    instance: &ash::Instance,
//    window: &winit::window::Window,
//) -> vk::SurfaceKHR {
//    use std::os::raw::c_void;
//    use winit::platform::windows::WindowExtWindows;
//
//    let hinstance = window.hinstance();
//    let hwnd = window.hwnd();
//    let win32_create_info = vk::Win32SurfaceCreateInfoKHR::builder()
//        .hinstance(hinstance as *const c_void)
//        .hwnd(hwnd as *const c_void);
//
//    let win32_surface_loader = Win32Surface::new(entry, instance);
//    unsafe { win32_surface_loader.create_win32_surface(&win32_create_info, None) }.unwrap()
//}

fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> vk::SurfaceKHR {
    use raw_window_handle::{HasDisplayHandle, RawDisplayHandle, RawWindowHandle};

    match (
        window.window_handle().unwrap().as_raw(),
        window.display_handle().unwrap().as_raw(),
    ) {
        (RawWindowHandle::Wayland(window), RawDisplayHandle::Wayland(display)) => {
            use ash::khr::wayland_surface;
            let loader = wayland_surface::Instance::new(entry, instance);

            let create_info = vk::WaylandSurfaceCreateInfoKHR::default()
                .display(display.display.as_ptr() as _)
                .surface(window.surface.as_ptr() as _);

            unsafe { loader.create_wayland_surface(&create_info, None) }.unwrap()
        }
        (RawWindowHandle::Xlib(window), RawDisplayHandle::Xlib(display)) => {
            use ash::khr::xlib_surface;
            let x11_display = display.display.unwrap().as_ptr();
            let x11_window = window.window;
            let create_info = vk::XlibSurfaceCreateInfoKHR::default()
                .window(x11_window as vk::Window)
                .dpy(x11_display as *mut vk::Display);

            let xlib_surface_loader = xlib_surface::Instance::new(entry, instance);
            unsafe {
                xlib_surface_loader
                    .create_xlib_surface(&create_info, None)
                    .unwrap()
            }
        }
        (RawWindowHandle::Xcb(window), RawDisplayHandle::Xcb(display)) => {
            use ash::khr::xcb_surface;
            let connection = display.connection.unwrap().as_ptr();
            let window: u32 = window.window.into();
            let create_info = vk::XcbSurfaceCreateInfoKHR::default()
                .connection(connection as *mut vk::xcb_connection_t)
                .window(window as vk::xcb_window_t);

            let xlib_surface_loader = xcb_surface::Instance::new(entry, instance);
            unsafe {
                xlib_surface_loader
                    .create_xcb_surface(&create_info, None)
                    .unwrap()
            }
        }
        (RawWindowHandle::Win32(window), RawDisplayHandle::Windows(_display)) => {
            let hinstance: isize = window.hinstance.unwrap().into();
            let hwnd: isize = window.hwnd.into();
            let win32_create_info = vk::Win32SurfaceCreateInfoKHR::default()
                .hinstance(hinstance)
                .hwnd(hwnd);

            let win32_surface_loader = ash::khr::win32_surface::Instance::new(entry, instance);
            unsafe { win32_surface_loader.create_win32_surface(&win32_create_info, None) }.unwrap()
        }
        _ => panic!("Unexpected window handle variant"),
    }
}

pub fn create_window(
    ctx: &VulkanContext,
    target: &ActiveEventLoop,
    on_device: Option<DeviceId>,
) -> Result<(WWindow, PWindow), palace_core::Error> {
    let win_attributes = winit::window::Window::default_attributes().with_title("palace");
    let winit_win = target.create_window(win_attributes).unwrap();
    let surface = create_surface(&ctx.entry, &ctx.instance, &winit_win);
    let size = winit_win.inner_size();
    Ok((
        winit_win,
        palace_core::vulkan::window::Window::new(ctx, surface, size.into(), on_device)?,
    ))
}

struct AppState<'a, R, F, E> {
    //state: State,
    last_frame: Instant,
    timeout_per_frame: Duration,
    runtime: &'a mut R,
    window: Option<(WWindow, PWindow)>,
    events: EventSource,
    draw: F,
    run_result: Result<(), E>,
    display_device: Option<DeviceId>,
}

pub trait MutWrapper<Inner> {
    fn with_mut<R, F: FnOnce(&mut Inner) -> R>(&mut self, f: F) -> R;
}

impl<Inner> MutWrapper<Inner> for &mut Inner {
    fn with_mut<R, F: FnOnce(&mut Inner) -> R>(&mut self, f: F) -> R {
        f(self)
    }
}

impl<Inner> MutWrapper<Inner> for Rc<RefCell<Inner>> {
    fn with_mut<R, F: FnOnce(&mut Inner) -> R>(&mut self, f: F) -> R {
        let mut b = self.borrow_mut();
        f(&mut *b)
    }
}

impl<
        R: MutWrapper<RunTime>,
        E,
        F: FnMut(
            &ActiveEventLoop,
            &mut PWindow,
            &mut R,
            EventStream,
            Deadline,
        ) -> Result<DataVersionType, E>,
    > winit::application::ApplicationHandler for AppState<'_, R, F, E>
{
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        //event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        self.window =
            Some(self.runtime.with_mut(|rt| {
                create_window(&rt.vulkan, &event_loop, self.display_device).unwrap()
            }));
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(new_size) => {
                let window = self.window.as_mut().unwrap();
                self.runtime
                    .with_mut(|rt| window.1.resize(new_size, &rt.vulkan));
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let window = self.window.as_mut().unwrap();
                //std::thread::sleep(dbg!(
                //    next_timeout.saturating_duration_since(std::time::Instant::now())
                //));
                let res = (self.draw)(
                    event_loop,
                    &mut window.1,
                    &mut self.runtime,
                    self.events.current_batch(),
                    Deadline::for_frame_duration(self.last_frame, self.timeout_per_frame),
                );
                match res {
                    Ok(version) => {
                        if version == DataVersionType::Final {
                            event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
                        } else {
                            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
                        }
                        let next_last_frame = Instant::now();
                        //println!("Last frame took {:?}", next_last_frame - self.last_frame);
                        self.last_frame = next_last_frame;

                        window.0.request_redraw();
                    }
                    Err(e) => {
                        self.run_result = Err(e);
                        event_loop.exit();
                    }
                }
            }
            o => self.events.add(o),
        }
    }
}

pub fn run_with_window_wrapper<
    R: MutWrapper<RunTime>,
    E,
    F: FnMut(
        &ActiveEventLoop,
        &mut PWindow,
        &mut R,
        EventStream,
        Deadline,
    ) -> Result<DataVersionType, E>,
>(
    runtime: &mut R,
    timeout_per_frame: Duration,
    display_device: Option<DeviceId>,
    draw: F,
) -> Result<(), E> {
    let event_loop = EventLoop::new().unwrap();
    let mut state = AppState {
        runtime,
        last_frame: Instant::now(),
        timeout_per_frame,
        window: None,
        events: EventSource::default(),
        draw,
        run_result: Ok(()),
        display_device,
    };

    event_loop.run_app(&mut state).unwrap();

    if let Some((_, mut window)) = state.window {
        state
            .runtime
            .with_mut(|rt| unsafe { window.deinitialize(&rt.vulkan) });
    }

    state.run_result
}

pub fn run_with_window_on_device<
    F: FnMut(
        &ActiveEventLoop,
        &mut PWindow,
        &mut &mut RunTime,
        EventStream,
        Deadline,
    ) -> Result<DataVersionType, palace_core::Error>,
>(
    mut runtime: &mut RunTime,
    timeout_per_frame: Duration,
    display_device: Option<DeviceId>,
    draw: F,
) -> Result<(), palace_core::Error> {
    run_with_window_wrapper(&mut runtime, timeout_per_frame, display_device, draw)
}

pub fn run_with_window<
    F: FnMut(
        &ActiveEventLoop,
        &mut PWindow,
        &mut &mut RunTime,
        EventStream,
        Deadline,
    ) -> Result<DataVersionType, palace_core::Error>,
>(
    mut runtime: &mut RunTime,
    timeout_per_frame: Duration,
    draw: F,
) -> Result<(), palace_core::Error> {
    run_with_window_on_device(&mut runtime, timeout_per_frame, None, draw)
}
