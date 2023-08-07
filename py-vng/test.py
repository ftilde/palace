import numpy as np
import time
import vng
import math

ram_size = 8 << 30
vram_size = 10 << 30

print(vram_size)

rt = vng.RunTime(ram_size, vram_size)

window = vng.Window(rt)

v = vng.open_volume("/nosnapshot/test-volumes/walnut_float2.vvd")

k = np.array([1, 2, 1]).astype(np.float32) * 0.25

#v2 = vng.linear_rescale(v1, 2, m1)
v = vng.separable_convolution(v, [k]*3)

selected_slice = 0
offset = np.array([0.0, 0.0])
zoom_level = 1.0

fov = np.array(30.0)
eye = np.array([5.5, 0.5, 0.5])
center = np.array([0.5, 0.5, 0.5])
up = np.array([1.0, 1.0, 0.0])

i = 0

gui_state = vng.GuiState(rt)

def normalize(v):
    l = np.linalg.norm(v)
    return v/l

def render(size, events):
    global i
    i += 1

    gui = gui_state.setup(events, vng.Vertical([
        vng.Label("Look at all these buttons"),
        vng.Horizontal([
            vng.Button("yes?", lambda: print("yes!")),
            vng.Button("no?", lambda: print("no!")),
            vng.Slider(fov, 10, 50),
        ]),
    ]))

    #frame = render_raycast(size, events)
    frame = render_slice(size, events)
    frame = gui.render(frame)

    return frame

def render_raycast(size, events):

    def drag(pos, delta):
        global eye, center, up
        delta = np.array(delta).astype(np.float64)

        look = center - eye;
        look_len = np.linalg.norm(look);
        left = normalize(np.cross(up, look))
        move_factor = 0.005;
        delta *= move_factor

        new_look = normalize(normalize(look) + up * delta[0] + left * -delta[1]) * look_len

        eye = center - new_look;
        left = np.cross(up, new_look)
        up = normalize(np.cross(new_look, left))

    def wheel(delta, pos):
        global eye, center
        look = center - eye;
        new_look = look * (1.0 - delta * 0.1);
        eye = center - new_look;

    events.act([
        vng.OnMouseDrag(vng.MouseButton.Left, drag),
        #vng.OnMouseClick(vng.MouseButton.Left, bar),
        vng.OnWheelMove(wheel),
        #vng.OnKeyPress("A", lambda: print("indeed a key")),
    ]);

    md = vng.ImageMetadata(size, [512]*2)

    look_at = vng.look_at(eye, center, up);
    perspective = vng.perspective(md, fov, 0.01, 100)
    proj = perspective.dot(look_at)

    eep = vng.entry_exit_points(v.metadata, md, proj)
    frame = vng.raycast(v, eep)
    frame = vng.rechunk(frame, [vng.chunk_size_full]*3)

    return frame

def render_slice(size, events):
    def drag_l(pos, delta):
        global offset
        offset += delta

    def drag_r(pos, delta):
        global selected_slice
        selected_slice += delta[0]
        selected_slice = max(selected_slice, 0)

    def wheel(delta, pos):
        global zoom_level, offset

        zoom_change = math.exp(-delta * 0.05)
        zoom_level *= zoom_change;

        pos = np.array(pos)
        offset = (offset - pos) / zoom_change + pos;

    events.act([
        vng.OnMouseDrag(vng.MouseButton.Left, drag_l),
        vng.OnMouseDrag(vng.MouseButton.Right, drag_r),
        vng.OnWheelMove(wheel),
    ]);

    md = vng.ImageMetadata(size, [512]*2)

    proj = vng.slice_projection_mat_z(v.metadata, md, selected_slice, offset, zoom_level)

    frame = vng.render_slice(v, md, proj)
    frame = vng.rechunk(frame, [vng.chunk_size_full]*3)

    return frame

window.run(rt, render, timeout_ms=10)
