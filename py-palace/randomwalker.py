import numpy as np
import palace as pc
import palace_util
import argparse
import xml.etree.ElementTree as ET

def read_vge(path):
    root = ET.parse(path)
    points = []
    for elem in root.findall('.//item/item'):
        points.append([float(elem.attrib['z']), float(elem.attrib['y']), float(elem.attrib['x'])])
    #print(points)
    return np.array(points, dtype=np.float32)

ram_size = 8 << 30
vram_size = 10 << 30
disk_cache_size = 20 << 30

parser = argparse.ArgumentParser()
parser.add_argument('volume_file')
parser.add_argument('-fg', '--foreground_seeds', required=False)
parser.add_argument('-bg', '--background_seeds', required=False)
parser.add_argument('-t', '--transfunc', type=str)

args = parser.parse_args()

rt = pc.RunTime(ram_size, vram_size, disk_cache_size, device=0)

try:
    vol = pc.open_lod(args.volume_file)
except:
    vol = pc.open(args.volume_file)
    #vol = vol.rechunk([32]*vol.nd())
    #vol = vol[:5,:,:,:]
    steps = list(reversed([2.0 if i < 3 else pc.FixedStep(2.0) for i in range(0, vol.nd())]))
    vol = vol.create_lod(steps)
    #vol = vol.single_level_lod()

nd = vol.nd()

foreground_seeds = np.empty(shape=[0,nd], dtype=np.float32)
background_seeds = np.empty(shape=[0,nd], dtype=np.float32)

if args.foreground_seeds:
    foreground_seeds = np.concat([foreground_seeds, read_vge(args.foreground_seeds)])
if args.background_seeds:
    background_seeds = np.concat([background_seeds, read_vge(args.background_seeds)])

def select_vol_from_ts(v, ts):
    match nd:
        case 3:
            return v
        case 4:
            return palace_util.slice_time_4d(ts, v)
        case o:
            raise f"Invalid number of tensor dimensions: {o}"

v0 = select_vol_from_ts(vol, 0)
l0md: pc.TensorMetaData = v0.fine_metadata()
l0ed = v0.fine_embedding_data()

vol = vol.map(lambda vol: vol.cast(pc.ScalarType.F32))
vol = vol.map(lambda vol: vol * (1.0/(1 << 16)))

if args.transfunc:
    tf = pc.load_tf(args.transfunc)
else:
    tf = pc.grey_ramp_tf(0.0, 1.0)

def fit_tf_to_values(vol):
    palace_util.fit_tf_range(rt, vol.levels[0], tf)

num_tf_values = 128
def prob_tf_from_values(values):
    #tf_table = [[0, 0, 255, 255] for i in range(num_tf_values)] + [[255, 0, 0, 255] for i in range(num_tf_values)]
    tf_table = pc.from_numpy(np.array(values, np.uint8)).fold_into_dtype()
    return pc.TransFuncOperator(0.0, 1.0, tf_table)

tf_prob = prob_tf_from_values([[0, 0, num_tf_values-i-1, num_tf_values-i-1] for i in range(num_tf_values)] + [[i, 0, 0, i] for i in range(num_tf_values)])
tf_prob3d = prob_tf_from_values([[0, 0, 0, 0] for i in range(num_tf_values)] + [[i, i, 0, i] for i in range(num_tf_values)])

store = pc.Store()

slice_state0 = pc.SliceviewState.for_volume(l0md, l0ed, 0).store(store)
slice_state1 = pc.SliceviewState.for_volume(l0md, l0ed, 1).store(store)
slice_state2 = pc.SliceviewState.for_volume(l0md, l0ed, 2).store(store)
camera_state = pc.CameraState.for_volume(l0md, l0ed, 30.0).store(store)
raycaster_config = pc.RaycasterConfig().store(store)
raycaster_config_rw = pc.RaycasterConfig().store(store)
timestep = store.store_primitive(0)

#raycaster_config_rw.compositing_mode().write("DVR")

mode = store.store_primitive("hierarchical")
#mode = store.store_primitive("normal")
weight_function = store.store_primitive("bian_mean")
min_edge_weight = store.store_primitive(1e-5)

beta = store.store_primitive(128.0)
extent = store.store_primitive(1)

gui_state = pc.GuiState(rt)

mouse_pos_and_value = None

def apply_weight_function(volume):
    match weight_function.load():
        case "grady":
            return pc.randomwalker_weights(volume, min_edge_weight.load(), beta.load())
        case "bian_mean":
            return pc.randomwalker_weights_bian(volume, min_edge_weight.load(), extent.load())

def apply_rw_mode(input):
    fg_seeds_tensor = pc.from_numpy(foreground_seeds).fold_into_dtype()
    bg_seeds_tensor = pc.from_numpy(background_seeds).fold_into_dtype()

    match mode.load():
        case "normal":
            i = input.levels[0].rechunk([pc.chunk_size_full]*nd)
            md: pc.TensorMetaData = i.inner.metadata
            ed = i.embedding_data
            weights = apply_weight_function(i)
            seeds = pc.rasterize_seed_points(fg_seeds_tensor, bg_seeds_tensor, md, ed)
            rw_result = pc.randomwalker(weights, seeds, max_iter=1000, max_residuum_norm=0.001)
            return (input, rw_result.single_level_lod())

        case "hierarchical":
            weights = input.map(lambda level: apply_weight_function(level.inner).embedded(pc.TensorEmbeddingData(np.append(level.embedding_data.spacing, [1.0])))).cache_coarse_levels()
            rw_result = pc.hierarchical_randomwalker(weights, fg_seeds_tensor, bg_seeds_tensor).cache_coarse_levels()
            return (input, rw_result)


def render(size, events: pc.Events):
    global mouse_pos_and_value

    v, rw_result = apply_rw_mode(vol)

    v = select_vol_from_ts(v, timestep.load())
    rw_result = select_vol_from_ts(rw_result, timestep.load())

    # GUI stuff
    widgets = []

    widgets.append(pc.ComboBox("Mode", mode, ["normal", "hierarchical"]))
    widgets.append(pc.ComboBox("Weight Function", weight_function, ["grady", "bian_mean"]))
    match weight_function.load():
        case "grady":
            widgets.append(palace_util.named_slider("beta", beta, 0.01, 10000, logarithmic=True))
        case "bian_mean":
            widgets.append(palace_util.named_slider("extent", extent, 1, 5))

    widgets.append(palace_util.named_slider("min_edge_weight", min_edge_weight, 1e-20, 1, logarithmic=True))
    widgets.append(pc.Button("Fit Transfer Function", lambda: fit_tf_to_values(v)))

    if mouse_pos_and_value is not None:
        vol_pos, value = mouse_pos_and_value
        widgets.append(pc.Label(f"Value at {vol_pos} = {value}"))
        mouse_pos_and_value = None

    if nd != 3:
        widgets.append(palace_util.named_slider("Timestep", timestep, 0, vol.fine_metadata().dimensions[0]-1))

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

        slice = palace_util.render_slice(v, state, tf)
        slice_rw = palace_util.render_slice(rw_result, state, tf_prob)

        #slice_edge = palace_util.render_slice(edge_w, 0, slice_state0, tf)

        out = palace_util.alpha_blending(slice_rw, slice)
        #out = slice_rw

        def inspect(size, events):
            vol = rw_result.levels[0]
            extract_slice_value(size, events, state, vol)
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

    def extract_slice_value(size, events: pc.Events, slice_state, volume):
        global mouse_pos_and_value
        mouse_pos = events.latest_state().mouse_pos()
        vol_pos = mouse_pos and palace_util.mouse_to_volume_pos(slice_state.load(), volume, mouse_pos, size)
        if vol_pos is not None:
            value = palace_util.extract_tensor_value(rt, volume, vol_pos)
            mouse_pos_and_value = (vol_pos, value)

    # Actual composition of the rendering
    slice0 = overlay_slice(slice_state0)
    slice1 = overlay_slice(slice_state1)
    slice2 = overlay_slice(slice_state2)
    ray = overlay_ray(camera_state)

    frame = palace_util.quad(ray, slice0, slice1, slice2)
    #frame = ray
    #frame = slice0
    gui = gui_state.setup(events, pc.Vertical(widgets))

    frame = gui.render(frame(size, events))

    return frame

rt.run_with_window(render, timeout_ms=10, record_task_stream=False, bench=False)
