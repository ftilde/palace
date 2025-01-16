import palace as pc
import numpy as np

# General pattern for renderable components:
# component: size, events -> frame operator
#
# This is also the signature required for the first argument of window.run!
def split(dim, fraction, render_first, render_last):
    def inner(size, events):
        splitter = pc.Splitter(size, 0.5, dim)

        events_l, events_r = splitter.split_events(events)

        frame_l = render_first(splitter.metadata_l().dimensions, events_l)
        frame_r = render_last(splitter.metadata_r().dimensions, events_r)

        return splitter.render(frame_l, frame_r)

    return inner

def quad(tl, tr, bl, br):
    upper = split(pc.SplitDirection.Horizontal, 0.5, tl, tr)
    lower = split(pc.SplitDirection.Horizontal, 0.5, bl, br)
    return split(pc.SplitDirection.Vertical, 0.5, upper, lower)

# Raycasting render component
def render_raycast(vol, camera_state, config, tf):
    def inner(size, events):
        events.act([
            pc.OnMouseDrag(pc.MouseButton.Left, lambda pos, delta: camera_state.trackball().mutate(lambda tb: tb.pan_around(delta))),
            pc.OnWheelMove(lambda delta, pos: camera_state.trackball().mutate(lambda tb: tb.move_inout(delta))),
        ]);

        md = pc.TensorMetaData(size, size)
        proj = camera_state.load().projection_mat(size)

        eep = pc.entry_exit_points(vol.fine_metadata(), vol.fine_embedding_data(), md, proj)
        frame = pc.raycast(vol, eep, config.load(), tf)
        frame = frame.rechunk([pc.chunk_size_full]*2)

        return frame
    return inner

# Slice render component
def render_slice(vol, slice_state, tf=None, coarse_lod_factor=1.0):
    def inner(size, events):
        events.act([
            pc.OnMouseDrag(pc.MouseButton.Left, lambda pos, delta: slice_state.mutate(lambda s: s.drag(delta))),
            pc.OnMouseDrag(pc.MouseButton.Right, lambda pos, delta: slice_state.mutate(lambda s: s.scroll(delta[0]))),
            pc.OnWheelMove(lambda delta, pos: slice_state.mutate(lambda s: s.zoom(delta, pos))),
        ]);

        md = pc.TensorMetaData(size, size)

        proj = slice_state.load().projection_mat(vol.fine_metadata(), vol.fine_embedding_data(), size)

        frame = pc.render_slice(vol, md, proj, tf, coarse_lod_factor)
        frame = frame.rechunk([pc.chunk_size_full]*2)

        return frame
    return inner

# Helper to go from slice to volume coordinates (assuming a sliceviewer)
def mouse_to_volume_pos(slice_state, embedded_tensor, pos, frame_size):
    md = embedded_tensor.inner.metadata
    ed = embedded_tensor.embedding_data
    pos = np.array([1.0, 0.0] + pos)
    mat = slice_state.projection_mat(md, ed, frame_size)
    vol_pos = np.dot(mat, pos)[1:]
    vol_pos = np.round(vol_pos)
    vol_pos = np.astype(vol_pos, np.int32)
    if (vol_pos >= 0).all() and (vol_pos < md.dimensions).all():
        return vol_pos

# Helper to go from slice to volume coordinates (assuming a sliceviewer)
def mouse_to_image_pos(image_state: pc.ImageViewerState, embedded_tensor, pos, frame_size):
    md = embedded_tensor.inner.metadata
    ed = embedded_tensor.embedding_data
    pos = np.array([1.0] + pos)
    mat = image_state.projection_mat(md, ed, frame_size)
    img_pos = np.dot(mat, pos)[1:]
    img_pos = np.round(img_pos)
    img_pos = np.astype(img_pos, np.int32)
    if (img_pos >= 0).all() and (img_pos < md.dimensions).all():
        return img_pos

# Helper to extract an element from a tensor
def extract_tensor_value(rt, embedded_tensor, pos):
    md = embedded_tensor.inner.metadata
    chunk_pos = md.chunk_pos(pos)
    chunk = rt.resolve(embedded_tensor, chunk_pos)
    pos_in_chunk = md.pos_in_chunk(chunk_pos, pos)
    return chunk[tuple(pos_in_chunk)]

def extract_slice_value(rt, size, events: pc.Events, slice_state, volume):
    mouse_pos = events.latest_state().mouse_pos()
    vol_pos = mouse_pos and mouse_to_volume_pos(slice_state.load(), volume, mouse_pos, size)
    if vol_pos is not None:
        value = extract_tensor_value(rt, volume, vol_pos)
        return (vol_pos, value)


def inspect_component(component, action):
    def inner(size, events):
        action(size, events)
        return component(size, events)
    return inner

def alpha_blending(render_over, render_under):
    def inner(size, events):
        max_val = pc.jit(255).splat(4)
        over = render_over(size, events).cast(pc.ScalarType.F32.vec(4)) /  max_val
        under = render_under(size, events).cast(pc.ScalarType.F32.vec(4)) / max_val

        alpha = over.index(3).splat(4)
        one_minus_alpha = pc.jit(1.0).splat(4) - alpha
        return ((over * alpha + under * one_minus_alpha) * max_val).cast(pc.ScalarType.U8.vec(4))

    return inner


def fit_tf_range(rt, vol, tf):
    tf.min = rt.resolve_scalar(vol.min_value(10))
    tf.max = rt.resolve_scalar(vol.max_value(10))

def slice_time_4d(ts, v):
    def select_time_slice_4d_lod(ts, base_level, target_level):
        #assert base_level.dim() == 4
        #assert target_level.dim() == 4

        pos = np.array([1, ts, 0, 0, 0], dtype=np.float32)
        pos = target_level.inner.metadata.norm_to_voxel().dot(base_level.inner.metadata.voxel_to_norm().dot(pos));
        ts = round(pos[1])
        return target_level[ts,:,:,:]

    return v.map(lambda l: select_time_slice_4d_lod(ts, v.levels[0], l))


# Gui stuff
def named_slider(name, state, min, max, logarithmic=False):
    return pc.Horizontal([
        pc.Slider(state, min, max, logarithmic),
        pc.Label(name),
    ])

