import palace as pc

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
        frame = pc.rechunk(frame, [pc.chunk_size_full]*2)

        return frame
    return inner

# Slice render component
def render_slice(vol, dim, slice_state):
    def inner(size, events):
        events.act([
            pc.OnMouseDrag(pc.MouseButton.Left, lambda pos, delta: slice_state.mutate(lambda s: s.drag(delta))),
            pc.OnMouseDrag(pc.MouseButton.Right, lambda pos, delta: slice_state.mutate(lambda s: s.scroll(delta[0]))),
            pc.OnWheelMove(lambda delta, pos: slice_state.mutate(lambda s: s.zoom(delta, pos))),
        ]);

        md = pc.TensorMetaData(size, size)

        proj = slice_state.load().projection_mat(vol.fine_metadata(), vol.fine_embedding_data(), size)

        frame = pc.render_slice(vol, md, proj)
        frame = pc.rechunk(frame, [pc.chunk_size_full]*2)

        return frame
    return inner



# Gui stuff
def named_slider(name, state, min, max, logarithmic=False):
    return pc.Horizontal([
        pc.Slider(state, min, max, logarithmic),
        pc.Label(name),
    ])

