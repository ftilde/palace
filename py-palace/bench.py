import numpy as np
import palace as pc
import palace_util
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('volume_file')
parser.add_argument('-t', '--transfunc', type=str)
parser.add_argument('--compositing', type=str, default="MOP")
parser.add_argument('--tfmin', type=float, default=0.0)
parser.add_argument('--tfmax', type=float, default=1.0)
parser.add_argument('--width', type=int, default=800)
parser.add_argument('--height', type=int, default=600)
parser.add_argument('--normalized-size', action='store_true')
parser.add_argument('--save-image', type=str, default=None)
parser.add_argument('--camera-distance', type=float, default=None)
parser.add_argument('--fov', type=float, default=30)
parser.add_argument('--const-chunk-table', type=str, default=None)
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

const_chunk_table = pc.open_lod(args.const_chunk_table) if args.const_chunk_table else None


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

if args.normalized_size:
    levels = []
    for l in vol.levels:
        ed = l.embedding_data
        ed.spacing = np.array([2.0]*3, dtype=np.float32) / np.array(l.metadata.dimensions, dtype=np.float32)
        l.embedding_data = ed
        levels.append(l)
    vol.levels = levels

l0md = vol.fine_metadata()
l0ed = vol.fine_embedding_data()

raycaster_config = pc.RaycasterConfig().store(store)
raycaster_config.compositing_mode().write(args.compositing)
camera_state = pc.CameraState.for_volume(l0md, l0ed, args.fov).store(store)

if args.camera_distance:
    camera_state.trackball().eye().write([-args.camera_distance, 0.0, 0.0])
    #camera_state.trackball().center().write(np.array([1.0]*3))
    camera_state.trackball().up().write(np.array([0.0, -1.0, 0.0]))

def fit_tf_to_values(vol):
    palace_util.fit_tf_range(rt, vol.levels[0], tf)

# Top-level render component
def render(size, events):
    global mouse_pos_and_value

    v = vol

    devices = rt.all_devices()
    tile_size = 512

    #camera_state.trackball().mutate(lambda v: v.pan_around([0, 10]))

    frame = palace_util.render_raycast(v, camera_state, raycaster_config, tf, tile_size=tile_size, devices=devices, const_brick_table=const_chunk_table)
    print(np.linalg.norm(camera_state.trackball().eye().load()))

    return frame(size, events)

image_size = (args.width,args.height)
elapsed = rt.run_with_window(render, timeout_ms=10, bench=True, record_task_stream=False, window_size=image_size)
print("Elapsed time: {}s".format(elapsed))
if args.save_image:
    frame = render(image_size, pc.Events.none()).unfold_dtype()[:,:,:3]
    palace_util.save_screenshot(rt, args.save_image, frame, image_size)
