import numpy as np
import palace as pc
import palace_util
import argparse

ram_size = 8 << 30
vram_size = 10 << 30
disk_cache_size = 20 << 30

parser = argparse.ArgumentParser()
parser.add_argument('volume_file')
parser.add_argument('-t', '--transfunc', type=str)

args = parser.parse_args()

rt = pc.RunTime(ram_size, vram_size, disk_cache_size, device=0)

vol = pc.open_volume(args.volume_file)

if args.transfunc:
    tf = pc.load_tf(args.transfunc)
else:
    tf = None

store = pc.Store()

l0md = vol.inner.metadata
l0ed = vol.embedding_data

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

# Top-level render component
def render(size, events):

    # Volume Processing
    v = vol.create_lod(2.0)
    match processing.load():
        case "passthrough":
            #v = v.map(lambda v: pc.add(v, v))
            pass
        case "smooth":
            def smooth(evol, k):
                return pc.separable_convolution(evol, [pc.gauss_kernel(k / s) for s in evol.embedding_data.spacing])

            k = smoothing_std.load()
            v = v.map(lambda evol: smooth(evol, k))

        case "vesselness":
            v = v.map(lambda evol: pc.vesselness(evol, vesselness_rad_min.load(), vesselness_rad_max.load(), vesselness_steps.load()))

    match do_threshold.load():
        case "yes":
            v = v.map(lambda evol: pc.threshold(evol, threshold_val.load()))
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

    gui = gui_state.setup(events, pc.Vertical(widgets))

    # Actual composition of the rendering
    slice0 = palace_util.render_slice(v, 0, slice_state0)
    slice1 = palace_util.render_slice(v, 1, slice_state1)
    slice2 = palace_util.render_slice(v, 2, slice_state2)
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
