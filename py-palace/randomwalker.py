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
    return pc.from_numpy(np.array(points, dtype=np.float32)).fold_into_dtype()

ram_size = 8 << 30
vram_size = 10 << 30
disk_cache_size = 20 << 30

parser = argparse.ArgumentParser()
parser.add_argument('volume_file')
parser.add_argument('foreground_seeds')
parser.add_argument('background_seeds')
parser.add_argument('-t', '--transfunc', type=str)

args = parser.parse_args()

foreground_seeds = read_vge(args.foreground_seeds)
background_seeds = read_vge(args.background_seeds)

rt = pc.RunTime(ram_size, vram_size, disk_cache_size, device=0)

vol = pc.open_volume(args.volume_file)
vol = pc.rechunk(vol, [pc.chunk_size_full]*3)
md: pc.TensorMetaData = vol.inner.metadata
ed = vol.embedding_data

vol = pc.cast(vol, pc.ScalarType.F32)
vol = pc.mul(vol, 1.0/(1 << 16))
seeds = pc.rasterize_seed_points(foreground_seeds, background_seeds, md, ed)

if args.transfunc:
    tf = pc.load_tf(args.transfunc)
else:
    tf = pc.grey_ramp_tf(0.0, 1.0)

num_tf_values = 128
def prob_tf_from_values(values):
    #tf_table = [[0, 0, 255, 255] for i in range(num_tf_values)] + [[255, 0, 0, 255] for i in range(num_tf_values)]
    tf_table = pc.from_numpy(np.array(values, np.uint8)).fold_into_dtype()
    return pc.TransFuncOperator(0.0, 1.0, tf_table)

tf_prob = prob_tf_from_values([[0, 0, num_tf_values-i-1, num_tf_values-i-1] for i in range(num_tf_values)] + [[i, 0, 0, i] for i in range(num_tf_values)])
tf_prob3d = prob_tf_from_values([[0, 0, 0, 0] for i in range(num_tf_values)] + [[i, i, 0, i] for i in range(num_tf_values)])

store = pc.Store()

slice_state0 = pc.SliceviewState.for_volume(md, ed, 0).store(store)
slice_state1 = pc.SliceviewState.for_volume(md, ed, 1).store(store)
slice_state2 = pc.SliceviewState.for_volume(md, ed, 2).store(store)
camera_state = pc.CameraState.for_volume(md, ed, 30.0).store(store)
raycaster_config = pc.RaycasterConfig().store(store)
raycaster_config_rw = pc.RaycasterConfig().store(store)

#raycaster_config_rw.compositing_mode().write("DVR")

weight_function = store.store_primitive("bian_mean")
min_edge_weight = store.store_primitive(1e-5)

beta = store.store_primitive(128.0)
extent = store.store_primitive(1)

gui_state = pc.GuiState(rt)

vol_min = rt.resolve_scalar(pc.min_value(vol))
vol_max = rt.resolve_scalar(pc.max_value(vol))

rw_input = pc.div(pc.sub(vol, vol_min), vol_max-vol_min)

#print(tf.max)
#print(tf.min)

mouse_pos_and_value = None

def render(size, events: pc.Events):
    global mouse_pos_and_value

    match weight_function.load():
        case "grady":
            weights = pc.randomwalker_weights(rw_input, min_edge_weight.load(), beta.load())
        case "bian_mean":
            weights = pc.randomwalker_weights_bian(rw_input, min_edge_weight.load(), extent.load())

    edge_w = pc.min(pc.min(
        pc.index(weights.fold_into_dtype(), 0).embedded(ed),
        pc.index(weights.fold_into_dtype(), 1).embedded(ed)),
        pc.index(weights.fold_into_dtype(), 2).embedded(ed)).embedded(ed)

    #print(rt.resolve_scalar(pc.min_value(edge_w)))
    #print(rt.resolve_scalar(pc.max_value(edge_w)))

    rw_result = pc.randomwalker(weights, seeds, max_iter=1000, max_residuum_norm=0.001)


    v = vol.embedded(ed).create_lod(2.0)
    rw_result = rw_result.create_lod(2.0)
    edge_w = edge_w.create_lod(2.0)


    # GUI stuff
    widgets = []

    widgets.append(pc.ComboBox("Weight Function", weight_function, ["grady", "bian_mean"]))
    match weight_function.load():
        case "grady":
            widgets.append(palace_util.named_slider("beta", beta, 0.01, 10000, logarithmic=True))
        case "bian_mean":
            widgets.append(palace_util.named_slider("extent", extent, 1, 5))

    widgets.append(palace_util.named_slider("min_edge_weight", min_edge_weight, 1e-20, 1, logarithmic=True))

    if mouse_pos_and_value is not None:
        vol_pos, value = mouse_pos_and_value
        widgets.append(pc.Label(f"Value at {vol_pos} = {value}"))
        mouse_pos_and_value = None

    def overlay_slice(state):
        slice = palace_util.render_slice(v, state, tf)
        slice_rw = palace_util.render_slice(rw_result, state, tf_prob)

        #slice_edge = palace_util.render_slice(edge_w, 0, slice_state0, tf)

        out = palace_util.alpha_blending(slice_rw, slice)
        #out = slice_rw

        return palace_util.inspect_component(out, lambda size, events: extract_slice_value(size, events, state, rw_result.levels[0]))

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
    gui = gui_state.setup(events, pc.Vertical(widgets))

    frame = gui.render(frame(size, events))

    return frame

rt.run_with_window(render, timeout_ms=10)
