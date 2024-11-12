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
    return pc.from_numpy(np.array(points, dtype=np.float32)).unfold_into_vec_dtype()

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
#vol = pc.mul(vol, 1.0/(1 << 16))
seeds = pc.rasterize_seed_points(foreground_seeds, background_seeds, md, ed)

if args.transfunc:
    tf = pc.load_tf(args.transfunc)
else:
    tf = None

store = pc.Store()

slice_state0 = pc.SliceviewState.for_volume(md, ed, 0).store(store)
slice_state1 = pc.SliceviewState.for_volume(md, ed, 1).store(store)
slice_state2 = pc.SliceviewState.for_volume(md, ed, 2).store(store)
camera_state = pc.CameraState.for_volume(md, ed, 30.0).store(store)
raycaster_config = pc.RaycasterConfig().store(store)

beta = store.store_primitive(100.0)

gui_state = pc.GuiState(rt)

def render(size, events):
    v = pc.randomwalker(vol, seeds.inner, beta.load())
    v = v.create_lod(2.0)

    # GUI stuff
    widgets = []

    widgets.append(palace_util.named_slider("beta", beta, 0.01, 1000, logarithmic=True))

    gui = gui_state.setup(events, pc.Vertical(widgets))

    # Actual composition of the rendering
    slice0 = palace_util.render_slice(v, 0, slice_state0)
    slice1 = palace_util.render_slice(v, 1, slice_state1)
    slice2 = palace_util.render_slice(v, 2, slice_state2)
    ray = palace_util.render_raycast(v, camera_state, raycaster_config, tf)

    frame = palace_util.quad(ray, slice0, slice1, slice2)

    frame = gui.render(frame(size, events))

    return frame

rt.run_with_window(render, timeout_ms=10)
