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
md = vol.inner.metadata
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

beta = store.store_primitive(1000.0)
min_edge_weight = store.store_primitive(1e-6)

gui_state = pc.GuiState(rt)

vol_min = rt.resolve_scalar(pc.min_value(vol))
vol_max = rt.resolve_scalar(pc.max_value(vol))

rw_input = pc.div(pc.sub(vol, vol_min), vol_max-vol_min)

#print(tf.max)
#print(tf.min)


def render(size, events):
    weights = pc.randomwalker_weights(rw_input, min_edge_weight.load(), beta.load())
    edge_w = pc.min(pc.min(
        pc.index(weights.fold_into_dtype(), 0).embedded(ed),
        pc.index(weights.fold_into_dtype(), 1).embedded(ed)),
        pc.index(weights.fold_into_dtype(), 2).embedded(ed)).embedded(ed)

    #print(rt.resolve_scalar(pc.min_value(edge_w)))
    #print(rt.resolve_scalar(pc.max_value(edge_w)))

    rw_result = pc.randomwalker(weights, seeds, max_iter=10000, max_residuum_norm=0.001)


    v = vol.embedded(ed).create_lod(2.0)
    rw_result = rw_result.create_lod(2.0)
    edge_w = edge_w.create_lod(2.0)


    # GUI stuff
    widgets = []

    widgets.append(palace_util.named_slider("beta", beta, 0.01, 10000, logarithmic=True))
    widgets.append(palace_util.named_slider("min_edge_weight", min_edge_weight, 1e-20, 1, logarithmic=True))

    gui = gui_state.setup(events, pc.Vertical(widgets))

    def overlay_slice(dim, state):
        slice = palace_util.render_slice(v, dim, state, tf)
        slice_rw = palace_util.render_slice(rw_result, dim, state, tf_prob)

        #slice_edge = palace_util.render_slice(edge_w, 0, slice_state0, tf)

        return palace_util.alpha_blending(slice_rw, slice)
        #return slice_rw

    def overlay_ray(state):
        ray = palace_util.render_raycast(v, state, raycaster_config, tf)
        ray_rw = palace_util.render_raycast(rw_result, state, raycaster_config_rw, tf_prob3d)

        return palace_util.alpha_blending(ray_rw, ray)


    # Actual composition of the rendering
    #slice0 = palace_util.render_slice(v, 0, slice_state0, tf)
    #slice0_rw = palace_util.render_slice(rw_result, 0, slice_state0, tf_prob)

    #frame = palace_util.alpha_blending(slice0_rw, slice0)
    #frame = slice0_rw
    #frame = slice0

    slice0 = overlay_slice(0, slice_state0)
    slice1 = overlay_slice(1, slice_state1)
    slice2 = overlay_slice(2, slice_state2)
    ray = overlay_ray(camera_state)

    frame = palace_util.quad(ray, slice0, slice1, slice2)

    frame = gui.render(frame(size, events))

    return frame

rt.run_with_window(render, timeout_ms=10)
