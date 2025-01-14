import numpy as np
import palace as pc
import palace_util
import argparse
import time

ram_size = 8 << 30
vram_size = 10 << 30
disk_cache_size = 20 << 30

parser = argparse.ArgumentParser()
parser.add_argument('volume_file')
parser.add_argument('-t', '--transfunc', type=str)

args = parser.parse_args()

rt = pc.RunTime(ram_size, vram_size, disk_cache_size, device=0)

try:
    vol = pc.open_lod(args.volume_file)
except:
    vol = pc.open(args.volume_file)
    steps = list(reversed([2.0 if i < 3 else pc.FixedStep(2.0) for i in range(0, vol.nd())]))
    vol = vol.create_lod(steps)
    #vol = vol.single_level_lod()

vol = vol.map(lambda v: v.cast(pc.ScalarType.F32))
#for l in vol.levels:
#    print(l.inner.metadata.dimensions)
#    print(l.inner.metadata.chunk_size)
#
#print(vol.levels[0].inner.metadata.dimensions)
#print(vol.levels[0].inner.metadata.chunk_size)


if args.transfunc:
    tf = pc.load_tf(args.transfunc)
else:
    tf = pc.grey_ramp_tf(0.0, 1.0)

store = pc.Store()

nd = vol.nd()

def select_vol_from_ts(ts):
    match nd:
        case 3:
            return vol
        case 4:
            return palace_util.slice_time_4d(ts, vol)
        case o:
            raise f"Invalid number of tensor dimensions: {o}"

v0 = select_vol_from_ts(0)
l0md = v0.fine_metadata()
l0ed = v0.fine_embedding_data()

min_scale = l0ed.spacing.min() / 10.0
max_scale = (l0ed.spacing * l0md.dimensions).mean() / 5.0

slice_state0 = pc.SliceviewState.for_volume(l0md, l0ed, 0).store(store)
slice_state1 = pc.SliceviewState.for_volume(l0md, l0ed, 1).store(store)
slice_state2 = pc.SliceviewState.for_volume(l0md, l0ed, 2).store(store)
camera_state = pc.CameraState.for_volume(l0md, l0ed, 30.0).store(store)
raycaster_config = pc.RaycasterConfig().store(store)
view = store.store_primitive("raycast")
processing = store.store_primitive("passthrough")
do_threshold = store.store_primitive("no")
threshold_val = store.store_primitive(0.5)
timestep = store.store_primitive(0)

smoothing_std = store.store_primitive(min_scale * 2.0)
vesselness_rad_min = store.store_primitive(min_scale * 2.0)
vesselness_rad_max = store.store_primitive(max_scale * 0.3)
vesselness_steps = store.store_primitive(3)

slice_state0.depth().link_to(camera_state.trackball().center().at(0))
slice_state1.depth().link_to(camera_state.trackball().center().at(1))
slice_state2.depth().link_to(camera_state.trackball().center().at(2))
#slice_state1.zoom_level().link_to(slice_state0.zoom_level())
#slice_state2.zoom_level().link_to(slice_state0.zoom_level())
#slice_state1.offset().link_to(slice_state0.offset())
#slice_state2.offset().link_to(slice_state0.offset())

gui_state = pc.GuiState(rt)

animate = False
next_update = None

def animate_on():
    global animate, next_update
    animate = True
    next_update = None
def animate_off():
    global animate
    animate = False

def map_timestep(ts):
    global next_update
    if next_update is None:
        next_update = time.time()

    update_time_step = 0.3

    if next_update < time.time():
        next_update += update_time_step
        return (ts + 1) % vol.fine_metadata().dimensions[0]
    else:
        return ts


def fit_tf_to_values(vol):
    palace_util.fit_tf_range(rt, vol.levels[0], tf)

# Top-level render component
def render(size, events):
    v = select_vol_from_ts(timestep.load())
    #print("===")
    #for level in v.levels:
    #    print(level.inner.metadata.dimensions)
    #    print(level.inner.metadata.chunk_size)
    #    print(level.embedding_data.spacing)

    # Volume Processing
    match processing.load():
        case "passthrough":
            #v = v.map(lambda v: pc.add(v, v))
            pass
        case "smooth":
            def smooth(evol, k):
                return evol.separable_convolution([pc.gauss_kernel(k / s) for s in evol.embedding_data.spacing])

            k = smoothing_std.load()
            v = v.map(lambda evol: smooth(evol, k))

        case "vesselness":
            v = v.map(lambda evol: pc.vesselness(evol, vesselness_rad_min.load(), vesselness_rad_max.load(), vesselness_steps.load()))

    match do_threshold.load():
        case "yes":
            v = v.map(lambda evol: (evol >= threshold_val.load()).select(1.0, 0.0))
        case "no":
            pass


    # GUI stuff
    widgets = []

    widgets.append(pc.ComboBox("View", view, ["quad", "raycast", "x", "y", "z"]))
    match view.load():
        case "quad" | "raycast":
            widgets.append(palace_util.named_slider("FOV", camera_state.fov(), 10, 50))
            widgets.append(palace_util.named_slider("LOD coarseness", raycaster_config.lod_coarseness(), 1.0, 10, logarithmic=True))
            widgets.append(palace_util.named_slider("Oversampling", raycaster_config.oversampling_factor(), 0.01, 10, logarithmic=True))
            widgets.append(pc.ComboBox("Compositing", raycaster_config.compositing_mode(), ["MOP", "DVR"]))
            widgets.append(pc.ComboBox("Shading", raycaster_config.shading(), ["None", "Phong"]))
        case "x" | "y" | "z":
            pass

    widgets.append(pc.ComboBox("Processing", processing, ["passthrough", "smooth", "vesselness"]))
    match processing.load():
        case "passthrough":
            pass
        case "smooth":
            widgets.append(palace_util.named_slider("Smoothing std", smoothing_std, min_scale, max_scale, logarithmic=True))
        case "vesselness":
            widgets.append(palace_util.named_slider("Min vessel rad", vesselness_rad_min, min_scale, max_scale, logarithmic=True))
            widgets.append(palace_util.named_slider("Max vessel rad", vesselness_rad_max, min_scale, max_scale, logarithmic=True))
            widgets.append(palace_util.named_slider("Vesselness steps", vesselness_steps, 1, 10))

    widgets.append(pc.ComboBox("Apply Threshold", do_threshold, ["yes", "no"]))
    match do_threshold.load():
        case "yes":
            widgets.append(palace_util.named_slider("Threshold Value", threshold_val, 0.01, 10.0, logarithmic=True))
        case "no":
            pass

    if nd != 3:
        widgets.append(palace_util.named_slider("Timestep", timestep, 0, vol.fine_metadata().dimensions[0]-1))
        if animate:
            timestep.map(map_timestep)
            widgets.append(pc.Button("Stop animation", lambda: animate_off()))
        else:
            widgets.append(pc.Button("Start animation", lambda: animate_on()))

    widgets.append(pc.Button("Fit Transfer Function", lambda: fit_tf_to_values(v)))

    gui = gui_state.setup(events, pc.Vertical(widgets))

    # Actual composition of the rendering
    slice0 = palace_util.render_slice(v, slice_state0, tf)
    slice1 = palace_util.render_slice(v, slice_state1, tf)
    slice2 = palace_util.render_slice(v, slice_state2, tf)
    ray = palace_util.render_raycast(v, camera_state, raycaster_config, tf)

    match view.load():
        case "quad":
            frame = palace_util.quad(ray, slice0, slice1, slice2)
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
#print(rt.resolve(render([10, 10], pc.Events.none()), [0]*2))

rt.run_with_window(render, timeout_ms=10)
