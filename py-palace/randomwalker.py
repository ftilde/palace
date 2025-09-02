import numpy as np
import palace as pc
import palace_util
import argparse
import xml.etree.ElementTree as ET
import time
from pathlib import Path

def read_vge(path, l0ed):
    root = ET.parse(path)
    spacing = l0ed.spacing
    def extract_row(i):
        row = root.findall('.//transformationMatrix.row' + str(i))[0]
        return np.array([float(row.attrib['x']),float(row.attrib['y']),float(row.attrib['z']),float(row.attrib['w'])])

    mat = np.array([extract_row(i) for i in range(4)])
    print(mat)

    points = []
    for elem in root.findall('.//item/item'):
        point = np.array([float(elem.attrib['x']), float(elem.attrib['y']), float(elem.attrib['z']), 1.0])
        point_transformed = mat.dot(point)
        point_zyx = np.flip(point_transformed[0:3])
        point_physical = point_zyx * spacing
        #print(point_physical)
        points.append(point_physical)
    return np.array(points, dtype=np.float32)

def read_seeds(path, l0ed):
    match Path(path).suffix:
        case ".vge":
            return read_vge(path, l0ed)
        case ".npy":
            return np.load(path)
        case o:
            raise f"Unknown suffix {o}"

parser = argparse.ArgumentParser()
parser.add_argument('volume_file')
parser.add_argument('-fg', '--foreground_seeds', required=False)
parser.add_argument('-bg', '--background_seeds', required=False)
parser.add_argument('-t', '--transfunc', type=str)
parser.add_argument('--max-iter', type=int, default=1000)
parser.add_argument('--max-residuum-norm', type=float, default=1e-3)
palace_util.add_runtime_args(parser)

args = parser.parse_args()

display_device = None

rt = palace_util.build_runtime_from_args(args)

try:
    vol = pc.open_lod(args.volume_file)
    #vol = vol.map(lambda vol: vol.rechunk([16]*vol.nd()))
except:
    vol = pc.open(args.volume_file)
    #vol = vol.rechunk([32]*vol.nd())
    #vol = vol[:5,:,:,:]
    steps = list(reversed([2.0 if i < 3 else pc.FixedStep(2.0) for i in range(0, vol.nd())]))
    vol = vol.create_lod(steps)
    #vol = vol.single_level_lod()

nd = vol.nd()

# Reduce the LOD hierarchy and compute root on a larger full volume. This is still manageable in terms of computation time, but increases the "range" of labels in smaller structures quite a bit.
max_level = len(vol.levels) - 2
vol.levels = [l if i != max_level else l.rechunk([pc.ChunkSizeFull()] * nd) for i, l in enumerate(vol.levels) if i <= max_level]

for level in vol.levels:
    print("{} {}".format(level.inner.metadata.dimensions, level.inner.metadata.chunk_size))

dim_t = vol.fine_metadata().dimensions[0]

def select_vol_from_ts(v, ts):
    match nd:
        case 3:
            return v
        case 4:
            return palace_util.slice_time_nd(ts, v)
        case o:
            raise f"Invalid number of tensor dimensions: {o}"

v0 = select_vol_from_ts(vol, 0)
l0md: pc.TensorMetaData = v0.fine_metadata()
l0ed = v0.fine_embedding_data()

foreground_seeds = np.empty(shape=[0,nd], dtype=np.float32)
background_seeds = np.empty(shape=[0,nd], dtype=np.float32)

if args.foreground_seeds:
    foreground_seeds = np.concat([foreground_seeds, read_seeds(args.foreground_seeds, l0ed)])
if args.background_seeds:
    background_seeds = np.concat([background_seeds, read_seeds(args.background_seeds, l0ed)])

vol = vol.map(lambda vol: vol.cast(pc.ScalarType.F32))
#vol = vol.map(lambda vol: vol * (1.0/(1 << 16)))

if args.transfunc:
    tf = pc.load_tf(args.transfunc)
else:
    tf = pc.grey_ramp_tf(0.0, 1.0)

def fit_tf_to_values(vol):
    palace_util.fit_tf_range(rt, vol.levels[0], tf)

num_tf_values = 128
def prob_tf_from_values(values):
    def tf_curve(v):
        gamma = 0.25
        return np.pow(v/255, gamma)*255
    values = [list(map(tf_curve, l)) for l in values]
    #tf_table = [[0, 0, 255, 255] for i in range(num_tf_values)] + [[255, 0, 0, 255] for i in range(num_tf_values)]
    tf_table = pc.from_numpy(np.array(values, np.uint8)).fold_into_dtype()
    return pc.TransFuncOperator(0.0, 1.0, tf_table)


alpha_mul = 0.01
tf_prob = prob_tf_from_values([[0, 0, num_tf_values-i-1, alpha_mul*(num_tf_values-i-1)] for i in range(num_tf_values)] + [[i, 0, 0, alpha_mul*i] for i in range(num_tf_values)])
tf_prob3d = prob_tf_from_values([[0, 0, 0, 0] for i in range(num_tf_values)] + [[i, i, 0, i] for i in range(num_tf_values)])

store = pc.Store()

slice_state0 = pc.SliceviewState.for_volume(l0md, l0ed, 0).store(store)
slice_state1 = pc.SliceviewState.for_volume(l0md, l0ed, 1).store(store)
slice_state2 = pc.SliceviewState.for_volume(l0md, l0ed, 2).store(store)
camera_state = pc.CameraState.for_volume(l0md, l0ed, 30.0).store(store)
raycaster_config = pc.RaycasterConfig().store(store)
raycaster_config_rw = pc.RaycasterConfig().store(store)

lod_coarseness_2d = store.store_primitive(1.0)
raycaster_config.lod_coarseness().link_to(lod_coarseness_2d)
raycaster_config_rw.lod_coarseness().link_to(lod_coarseness_2d)
view = store.store_primitive("quad")
timestep = store.store_primitive(0)

raycaster_config_rw.compositing_mode().write("DVR")

mode = store.store_primitive("hierarchical")
#mode = store.store_primitive("normal")
weight_function = store.store_primitive("bhatt_var_gaussian")
min_edge_weight = store.store_primitive(1e-5)

beta = store.store_primitive(128.0)
extent = store.store_primitive(1)

gui_state = pc.GuiState(rt)

mouse_reading_enabled = False
def set_mouse_reading(enabled):
    global mouse_reading_enabled
    mouse_reading_enabled = enabled
mouse_pos_and_value = None

animate = False
next_update = None

def animate_on():
    global animate, next_update
    animate = True
    next_update = None
def animate_off():
    global animate
    animate = False

def ts_next(ts):
    ts.write((ts.load() + 1) % dim_t)

def ts_prev(ts):
    ts.write((ts.load() + dim_t - 1) % dim_t)

def map_timestep(ts):
    global next_update
    if next_update is None:
        next_update = time.time()

    update_time_step = 0.3

    if next_update < time.time():
        next_update += update_time_step
        return (ts + 1) % dim_t
    else:
        return ts

def save_seeds():
    np.save("seeds_foreground", foreground_seeds)
    np.save("seeds_background", background_seeds)

def apply_weight_function(volume):
    #ext = [0] + [extent.load()] * (nd-1)
    ext = [extent.load()] * nd

    match weight_function.load():
        case "grady":
            return pc.randomwalker_weights(volume, min_edge_weight.load(), beta.load())
        case "bian_mean":
            return pc.randomwalker_weights_bian(volume, min_edge_weight.load(), ext)
        case "bhatt_var_gaussian":
            return pc.randomwalker_weights_bhattacharyya_var_gaussian(volume, min_edge_weight.load(), ext)
        case "ttest":
            return pc.randomwalker_weights_ttest(volume, min_edge_weight.load(), ext)

def apply_rw_mode(input):
    fg_seeds_tensor = pc.from_numpy(foreground_seeds).fold_into_dtype()
    bg_seeds_tensor = pc.from_numpy(background_seeds).fold_into_dtype()

    match mode.load():
        case "normal":
            i = input.levels[0].rechunk([pc.ChunkSizeFull()]*nd)
            md: pc.TensorMetaData = i.inner.metadata
            ed = i.embedding_data
            weights = apply_weight_function(i)
            seeds = pc.rasterize_seed_points(fg_seeds_tensor, bg_seeds_tensor, md, ed)
            rw_result = pc.randomwalker(weights, seeds, max_iter=args.max_iter, max_residuum_norm=args.max_residuum_norm)
            return (input, rw_result.single_level_lod())

        case "hierarchical":
            weights = input.map(lambda level: apply_weight_function(level.inner).embedded(pc.TensorEmbeddingData(np.append(level.embedding_data.spacing, [1.0])))).cache_coarse_levels()
            rw_result = pc.hierarchical_randomwalker(weights, fg_seeds_tensor, bg_seeds_tensor, max_iter=args.max_iter, max_residuum_norm=args.max_residuum_norm).cache()
            return (input, rw_result)

def render(size, events: pc.Events):
    global mouse_reading_enabled
    global mouse_pos_and_value

    v, rw_result = apply_rw_mode(vol)

    v = select_vol_from_ts(v, timestep.load())
    rw_result = select_vol_from_ts(rw_result, timestep.load())

    # GUI stuff
    widgets = []

    widgets.append(pc.ComboBox("View", view, ["quad", "raycast", "x", "y", "z"]))
    widgets.append(palace_util.named_slider("LOD coarseness", lod_coarseness_2d, 1.0, 10, logarithmic=True))

    widgets.append(pc.ComboBox("Mode", mode, ["normal", "hierarchical"]))
    widgets.append(pc.ComboBox("Weight Function", weight_function, ["grady", "bian_mean", "bhatt_var_gaussian", "ttest"]))
    match weight_function.load():
        case "grady":
            widgets.append(palace_util.named_slider("beta", beta, 0.01, 10000, logarithmic=True))
        case "bian_mean" | "bhatt_var_gaussian" | "ttest":
            widgets.append(palace_util.named_slider("extent", extent, 1, 5))

    widgets.append(palace_util.named_slider("min_edge_weight", min_edge_weight, 1e-20, 1, logarithmic=True))
    widgets.append(pc.Button("Fit Transfer Function", lambda: fit_tf_to_values(v)))

    if nd != 3:
        widgets.append(palace_util.named_slider("Timestep", timestep, 0, vol.fine_metadata().dimensions[0]-1))
        if animate:
            timestep.map(map_timestep)
            widgets.append(pc.Button("Stop animation", lambda: animate_off()))
        else:
            widgets.append(pc.Button("Start animation", lambda: animate_on()))

    widgets.append(pc.Button("Save seeds", lambda: save_seeds()))

    mouse_widgets = []
    if mouse_reading_enabled:
        mouse_widgets.append(pc.Button("Disable", lambda: set_mouse_reading(False)))
        if mouse_pos_and_value is not None:
            vol_pos, value = mouse_pos_and_value
            mouse_widgets.append(pc.Label(f"Value at {vol_pos} = {value}"))
            mouse_pos_and_value = None
    else:
        mouse_widgets.append(pc.Button("Enable mouse read", lambda: set_mouse_reading(True)))
    widgets.append(pc.Horizontal(mouse_widgets))

    def add_seed_point(slice_state, embedded_tensor, pos, frame_size, foreground):
        global foreground_seeds, background_seeds

        pos3d = palace_util.mouse_to_volume_pos(slice_state.load(), embedded_tensor, pos, frame_size)
        if pos3d is not None:
            match nd:
                case 3:
                    pos_nd = np.array(pos3d, dtype=np.float32)
                case 4:
                    pos_nd = [timestep.load()] + list(pos3d)
                    pos_nd = np.array(pos_nd, dtype=np.float32)

            pos_nd = np.concatenate(([1], pos_nd), dtype = np.float32)
            pos_nd = vol.fine_embedding_data().voxel_to_physical().dot(pos_nd)
            pos_nd = pos_nd[1:]

            pos_nd = pos_nd.reshape((1, nd))

            if foreground:
                foreground_seeds = np.concat([foreground_seeds, pos_nd])
            else:
                background_seeds = np.concat([background_seeds, pos_nd])

    def overlay_slice(state):
        tile_size = 512

        slice = palace_util.render_slice(v, state, tf=tf, coarse_lod_factor=lod_coarseness_2d.load(), tile_size=tile_size)
        slice_rw = palace_util.render_slice(rw_result, state, tf=tf_prob, coarse_lod_factor=lod_coarseness_2d.load(), tile_size=tile_size)

        #slice_edge = palace_util.render_slice(edge_w, 0, slice_state0, tf)

        out = palace_util.alpha_blending(slice_rw, slice)
        #out = slice_rw

        def inspect(size, events):
            global mouse_pos_and_value

            vol = rw_result.levels[0]
            if mouse_reading_enabled:
                mouse_pos_and_value = palace_util.extract_slice_value(rt, size, events, state, vol) or mouse_pos_and_value
            events.act([
                pc.OnMouseClick(pc.MouseButton.Left, lambda x: add_seed_point(state, vol, x, size, True)).when(lambda s: s.is_down("ShiftLeft")),
                pc.OnMouseClick(pc.MouseButton.Right, lambda x: add_seed_point(state, vol, x, size, False)).when(lambda s: s.is_down("ShiftLeft")),
                ])

        return palace_util.inspect_component(out, inspect)

    def overlay_ray(state):
        ray = palace_util.render_raycast(v, state, raycaster_config, tf)
        ray_rw = palace_util.render_raycast(rw_result, state, raycaster_config_rw, tf_prob3d)

        return palace_util.alpha_blending(ray_rw, ray)
        #return ray_rw

    events.act([
        pc.OnKeyPress("N", lambda: ts_next(timestep)),
        pc.OnKeyPress("P", lambda: ts_prev(timestep)),
        ])

    # Actual composition of the rendering
    slice0 = overlay_slice(slice_state0)
    slice1 = overlay_slice(slice_state1)
    slice2 = overlay_slice(slice_state2)
    ray = overlay_ray(camera_state)

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

    gui = gui_state.setup(events, pc.Vertical(widgets))
    frame = gui.render(frame(size, events))

    return frame

rt.run_with_window(render, timeout_ms=10, record_task_stream=False, bench=False, display_device=display_device)
