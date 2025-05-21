import numpy as np
import palace as pc
import palace_util
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('volume_file')
parser.add_argument('-t', '--transfunc', type=str)
parser.add_argument('--tfmin', type=float, default=0.0)
parser.add_argument('--tfmax', type=float, default=1.0)
palace_util.add_runtime_args(parser)

args = parser.parse_args()

rt = palace_util.build_runtime_from_args(args)

try:
    vol = pc.open_lod(args.volume_file)
except:
    vol = pc.open(args.volume_file)
    steps = list(reversed([2.0 if i < 3 else pc.FixedStep(2.0) for i in range(0, vol.nd())]))
    vol = vol.create_lod(steps)
    #vol = vol.single_level_lod()


#vol = vol.map(lambda v: v.cast(pc.ScalarType.F32))
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

tf.min = args.tfmin
tf.max = args.tfmax

store = pc.Store()

nd = vol.nd()

l0md = vol.fine_metadata()
l0ed = vol.fine_embedding_data()

min_scale = l0ed.spacing.min() / 10.0
max_scale = (l0ed.spacing * l0md.dimensions).mean() / 5.0

camera_state = pc.CameraState.for_volume(l0md, l0ed, 30.0).store(store)
raycaster_config = pc.RaycasterConfig().store(store)

camera_state.trackball().mutate(lambda v: v.move_inout(8.5))

def fit_tf_to_values(vol):
    palace_util.fit_tf_range(rt, vol.levels[0], tf)

# Top-level render component
def render(size, events):
    global mouse_pos_and_value

    v = vol

    devices = rt.all_devices()
    tile_size = 512

    #camera_state.trackball().mutate(lambda v: v.pan_around([0, 10]))

    frame = palace_util.render_raycast(v, camera_state, raycaster_config, tf, tile_size=tile_size, devices=devices)

    return frame(size, events)

rt.run_with_window(render, timeout_ms=10, bench=True, record_task_stream=False)
