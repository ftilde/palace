import numpy as np
import time
import vng
import math

ram_size = 8 << 30
vram_size = 10 << 30
disk_cache_size = 20 << 30

rt = vng.RunTime(ram_size, vram_size, disk_cache_size)

window = vng.Window(rt)

vol = vng.open_volume("/nosnapshot/test-volumes/walnut_float2.vvd")
#vol = vng.open_volume("/nosnapshot/test-volumes/liver_c01.vvd")

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

l0 = vol.levels[0]
l0md = l0.inner.metadata
l0ed = l0.embedding_data

slice_state0 = vng.SliceviewState.for_volume(l0md, l0ed, 0).store(store)
slice_state1 = vng.SliceviewState.for_volume(l0md, l0ed, 1).store(store)
slice_state2 = vng.SliceviewState.for_volume(l0md, l0ed, 2).store(store)
camera_state = vng.CameraState.for_volume(l0md, l0ed, 30.0).store(store)
raycaster_config = vng.RaycasterConfig().store(store)
view = store.store_primitive("raycast")

slice_state0.depth().link_to(camera_state.trackball().center().at(0))
slice_state1.depth().link_to(camera_state.trackball().center().at(1))
slice_state2.depth().link_to(camera_state.trackball().center().at(2))
#slice_state1.zoom_level().link_to(slice_state0.zoom_level())
#slice_state2.zoom_level().link_to(slice_state0.zoom_level())
#slice_state1.offset().link_to(slice_state0.offset())
#slice_state2.offset().link_to(slice_state0.offset())



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

        md = vng.tensor_metadata(size, size)
        proj = camera_state.load().projection_mat(size)

        eep = vng.entry_exit_points(vol.fine_metadata(), vol.fine_embedding_data(), md, proj)
        conf = vng.RaycasterConfig()
        frame = vng.raycast(vol, eep, raycaster_config.load())
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

        md = vng.tensor_metadata(size, size)

        proj = slice_state.load().projection_mat(vol.fine_metadata(), vol.fine_embedding_data(), size)

        frame = vng.render_slice(vol, md, proj)
        frame = vng.rechunk(frame, [vng.chunk_size_full]*2)

        return frame
    return inner

# Top-level render component
def render(size, events):

    def named_slider(name, state, min, max):
        return vng.Horizontal([
            vng.Label(name),
            vng.Slider(state, min, max),
        ])
    gui = gui_state.setup(events, vng.Vertical([
        vng.Vertical([
            named_slider("fov", camera_state.fov(), 10, 50),
            named_slider("LOD coarseness", raycaster_config.lod_coarseness(), 0.1, 10),
            named_slider("Oversampling", raycaster_config.oversampling_factor(), 0.1, 10),
            vng.ComboBox("Options", view, ["quad", "raycast", "x", "y", "z"]),
        ]),
    ]))

    slice0 = render_slice(vol, 0, slice_state0)
    slice1 = render_slice(vol, 1, slice_state1)
    slice2 = render_slice(vol, 2, slice_state2)
    ray = render_raycast(vol, camera_state)

    match view.load():
        case "quad":
            lower = split(vng.SplitDirection.Horizontal, 0.5, slice0, slice1)
            upper = split(vng.SplitDirection.Horizontal, 0.5, ray, slice2)
            frame = split(vng.SplitDirection.Vertical, 0.5, upper, lower)
        case "raycast":
            frame = ray
        case "x":
            frame = slice0
        case "y":
            frame = slice1
        case "z":
            frame = slice2

    frame = gui.render(frame(size, events))

    return frame


# So this does not actually work, because Vector<4, u8> cannot be converted into a numpy type. hmm...
# (And also currently only the conversion for f32 is implemented...)
#print(rt.resolve(render([10, 10], vng.Events.none()), [0]*2))

window.run(render, timeout_ms=10)
