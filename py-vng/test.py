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

slice_state0 = vng.SliceviewState(0, [0.0, 0.0], 1.0)
slice_state1 = vng.SliceviewState(0, [0.0, 0.0], 1.0)
slice_state2 = vng.SliceviewState(0, [0.0, 0.0], 1.0)

fov = np.array(30.0)
eye = np.array([5.5, 0.5, 0.5])
center = np.array([0.5, 0.5, 0.5])
up = np.array([1.0, 1.0, 0.0])

i = 0

gui_state = vng.GuiState(rt)

def normalize(v):
    l = np.linalg.norm(v)
    return v/l

#General pattern for renderable components:
# component: size, events -> frame operator
def split(dim, fraction, render_first, render_last):
    def inner(size, events):
        splitter = vng.Splitter(size, 0.5, dim)

        events_l, events_r = splitter.split_events(events)

        frame_l = render_first(splitter.metadata_l().dimensions, events_l)
        frame_r = render_last(splitter.metadata_r().dimensions, events_r)

        return splitter.render(frame_l, frame_r)

    return inner

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

    lower = split(vng.SplitDirection.Horizontal, 0.5, render_slice(0, slice_state0), render_slice(1, slice_state1))
    upper = split(vng.SplitDirection.Horizontal, 0.5, render_raycast, render_slice(2, slice_state2))

    frame = split(vng.SplitDirection.Vertical, 0.5, upper, lower)(size, events)

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

    md = vng.tensor_metadata(size, [512]*2)

    look_at = vng.look_at(eye, center, up);
    perspective = vng.perspective(md, fov, 0.01, 100)
    proj = perspective.dot(look_at)

    eep = vng.entry_exit_points(v.metadata, md, proj)
    frame = vng.raycast(v, eep)
    frame = vng.rechunk(frame, [vng.chunk_size_full]*3)

    return frame

def render_slice(dim, slice_state):
    def inner(size, events):
        events.act([
            vng.OnMouseDrag(vng.MouseButton.Left, lambda pos, delta: slice_state.drag(delta)),
            vng.OnMouseDrag(vng.MouseButton.Right, lambda pos, delta: slice_state.scroll(delta[0])),
            vng.OnWheelMove(lambda delta, pos: slice_state.zoom(delta, pos)),
        ]);

        md = vng.tensor_metadata(size, [512]*2)

        proj = vng.slice_projection_mat(dim, v.metadata, md, slice_state.selected, slice_state.offset, slice_state.zoom_level)

        frame = vng.render_slice(v, md, proj)
        frame = vng.rechunk(frame, [vng.chunk_size_full]*3)

        return frame
    return inner

window.run(render, timeout_ms=10)
