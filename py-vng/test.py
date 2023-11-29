import numpy as np
import time
import vng
import math

ram_size = 8 << 30
vram_size = 10 << 30

rt = vng.RunTime(ram_size, vram_size)

window = vng.Window(rt)

vol = vng.open_volume("/nosnapshot/test-volumes/walnut_float2.vvd")

k = np.array([1, 2, 1]).astype(np.float32) * 0.25

#v2 = vng.linear_rescale(v1, 2, m1)
#vol = vng.separable_convolution(vol, [k]*3)
vol = vol.create_lod(2.0)

#rechunked = vng.rechunk(vol.levels[-1], [4]*3)
#print(rt.resolve(rechunked, [0]*3))
#m = vng.mean(vol.levels[-1])
#print(rt.resolve_scalar(m))
#print(rt.resolve(vol.levels[0], [0]*3))

store = vng.Store()

slice_state0 = vng.SliceviewState(0, [0.0, 0.0], 1.0).store(store)
slice_state1 = vng.SliceviewState(0, [0.0, 0.0], 1.0).store(store)
slice_state2 = vng.SliceviewState(0, [0.0, 0.0], 1.0).store(store)

#slice_state1.zoom_level().link_to(slice_state0.zoom_level())
#slice_state2.zoom_level().link_to(slice_state0.zoom_level())
#slice_state1.offset().link_to(slice_state0.offset())
#slice_state2.offset().link_to(slice_state0.offset())

camera_state = vng.CameraState(
        vng.TrackballState(
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            ),
        30.0
        ).store(store)

camera_state.trackball().eye().map(lambda v: np.array(v) + [1.1, 1.0, 1.0])

gui_state = vng.GuiState(rt)

# General pattern for renderable components:
# component: size, events -> frame operator
#
# This is also the signature required for the first argument of window.run!
def split(dim, fraction, render_first, render_last):
    def inner(size, events):
        splitter = vng.Splitter(size, 0.5, dim)

        events_l, events_r = splitter.split_events(events)

        frame_l = render_first(splitter.metadata_l().dimensions, events_l)
        frame_r = render_last(splitter.metadata_r().dimensions, events_r)

        return splitter.render(frame_l, frame_r)

    return inner

# Raycasting render component
def render_raycast(vol, camera_state):
    def inner(size, events):
        events.act([
            vng.OnMouseDrag(vng.MouseButton.Left, lambda pos, delta: camera_state.trackball().mutate(lambda tb: tb.pan_around(delta))),
            vng.OnWheelMove(lambda delta, pos: camera_state.trackball().mutate(lambda tb: tb.move_inout(delta))),
        ]);

        md = vng.tensor_metadata(size, [512]*2)
        proj = camera_state.load().projection_mat(size)

        eep = vng.entry_exit_points(vol.fine_metadata(), vol.fine_embedding_data(), md, proj)
        frame = vng.raycast(vol, eep)
        frame = vng.rechunk(frame, [vng.chunk_size_full]*2)

        return frame
    return inner

# Slice render component
def render_slice(vol, dim, slice_state):
    def inner(size, events):
        events.act([
            vng.OnMouseDrag(vng.MouseButton.Left, lambda pos, delta: slice_state.mutate(lambda s: s.drag(delta))),
            vng.OnMouseDrag(vng.MouseButton.Right, lambda pos, delta: slice_state.mutate(lambda s: s.scroll(delta[0]))),
            vng.OnWheelMove(lambda delta, pos: slice_state.mutate(lambda s: s.zoom(delta, pos))),
        ]);

        md = vng.tensor_metadata(size, [512]*2)

        proj = slice_state.load().projection_mat(dim, vol.fine_metadata(), vol.fine_embedding_data(), size)

        frame = vng.render_slice(vol, md, proj)
        frame = vng.rechunk(frame, [vng.chunk_size_full]*2)

        return frame
    return inner

# Top-level render component
def render(size, events):
    gui = gui_state.setup(events, vng.Vertical([
        vng.Label("Look at all these buttons"),
        vng.Horizontal([
            vng.Button("yes?", lambda: print("yes!")),
            vng.Button("no?", lambda: print("no!")),
            vng.Slider(camera_state.fov(), 10, 50),
            vng.Slider(camera_state.fov(), 20, 60),
        ]),
    ]))

    lower = split(vng.SplitDirection.Horizontal, 0.5, render_slice(vol, 0, slice_state0), render_slice(vol, 1, slice_state1))
    upper = split(vng.SplitDirection.Horizontal, 0.5, render_raycast(vol, camera_state), render_slice(vol, 2, slice_state2))

    frame = split(vng.SplitDirection.Vertical, 0.5, upper, lower)(size, events)

    frame = gui.render(frame)

    return frame


# So this does not actually work, because Vector<4, u8> cannot be converted into a numpy type. hmm...
# (And also currently only the conversion for f32 is implemented...)
#print(rt.resolve(render([10, 10], vng.Events.none()), [0]*2))

window.run(render, timeout_ms=10)
