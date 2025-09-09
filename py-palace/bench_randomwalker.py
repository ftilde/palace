import numpy as np
import palace as pc
import palace_util
import argparse
import xml.etree.ElementTree as ET
import time
import itertools
import functools
import math
from functools import cmp_to_key
from pathlib import Path

def read_vge(path, l0ed):
    root = ET.parse(path)
    spacing = l0ed.spacing
    def extract_row(i):
        row = root.findall('.//transformationMatrix.row' + str(i))[0]
        return np.array([float(row.attrib['x']),float(row.attrib['y']),float(row.attrib['z']),float(row.attrib['w'])])

    mat = np.array([extract_row(i) for i in range(4)])

    points = []
    for elem in root.findall('.//item/item'):
        point = np.array([float(elem.attrib['x']), float(elem.attrib['y']), float(elem.attrib['z']), 1.0])
        point_transformed = mat.dot(point)
        point_zyx = np.flip(point_transformed[0:3])
        point_physical = point_zyx * spacing
        #print(point_physical)
        points.append(point_physical)
    #print(points)
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
parser.add_argument('-b', '--batch', type=int, default=128)
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

l0md: pc.TensorMetaData = vol.fine_metadata()
l0ed = vol.fine_embedding_data()

foreground_seeds = np.empty(shape=[0,nd], dtype=np.float32)
background_seeds = np.empty(shape=[0,nd], dtype=np.float32)

if args.foreground_seeds:
    foreground_seeds = np.concat([foreground_seeds, read_seeds(args.foreground_seeds, l0ed)])
if args.background_seeds:
    background_seeds = np.concat([background_seeds, read_seeds(args.background_seeds, l0ed)])

vol = vol.map(lambda vol: vol.cast(pc.ScalarType.F32))
#vol = vol.map(lambda vol: vol * (1.0/(1 << 16)))

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
weight_function = store.store_primitive("ttest")
min_edge_weight = store.store_primitive(1e-5)

beta = store.store_primitive(128.0)
extent = store.store_primitive(1)

gui_state = pc.GuiState(rt)

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


fg_seeds_tensor = pc.from_numpy(foreground_seeds).fold_into_dtype()
bg_seeds_tensor = pc.from_numpy(background_seeds).fold_into_dtype()
weights = vol.map(lambda level: apply_weight_function(level.inner).embedded(pc.TensorEmbeddingData(np.append(level.embedding_data.spacing, [1.0]))))
rw_result, rw_cct = pc.hierarchical_randomwalker(weights, fg_seeds_tensor, bg_seeds_tensor, max_iter=args.max_iter, max_residuum_norm=args.max_residuum_norm)
rw_result = rw_result.cache_coarse_levels()
rw_cct = rw_cct.cache()

cct_l0md = rw_cct.levels[0].metadata


print("Chunk size is: {}".format(l0md.chunk_size))
print("DType: {}".format(rw_result.levels[0].dtype))

chunk_elements = functools.reduce(lambda a, b: a*b, l0md.chunk_size, 1)
chunk_size_bytes = np.astype(chunk_elements * rw_result.levels[0].dtype.size_in_bytes(), np.uint64)

dim_in_chunks = list(map(lambda l,r: math.ceil(l/r), l0md.dimensions, l0md.chunk_size))
batch_size = args.batch

def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

start = time.time()
begin = start
total_size = 0
total_skipped = 0
total_considered = 0

def sorted_zorder(chunks):
    def less_msb(x: int, y: int) -> bool:
        return x < y and x < (x ^ y)

    def cmp_zorder(lhs, rhs):
        """Compare z-ordering."""
        # Assume lhs and rhs array-like objects of indices.
        assert len(lhs) == len(rhs)
        # Will contain the most significant dimension.
        msd = 0
        # Loop over the other dimensions.
        for dim in range(1, len(lhs)):
            # Check if the current dimension is more significant
            # by comparing the most significant bits.
            if less_msb(lhs[msd] ^ rhs[msd], lhs[dim] ^ rhs[dim]):
                msd = dim
        return lhs[msd] - rhs[msd]

    return sorted(chunks, key=cmp_to_key(cmp_zorder))

all_chunks = list(itertools.product(*map(lambda end: range(0, end), dim_in_chunks)))
all_chunks = sorted_zorder(all_chunks)

for full_batch in itertools.batched(all_chunks, batch_size):
    cct_chunks_positions = list(map(lambda c: list(map(lambda l,r: l//r, c, cct_l0md.chunk_size)), full_batch))
    cct_in_chunks_positions = list(map(lambda c: list(map(lambda l,r: l%r, c, cct_l0md.chunk_size)), full_batch))
    cct_chunks = rt.resolve(rw_cct.levels[0], cct_chunks_positions, record_task_stream=False)
    is_const = map(lambda cct_chunk, in_chunk_pos: not math.isnan(cct_chunk[tuple(in_chunk_pos)]), cct_chunks, cct_in_chunks_positions)
    batch = [pos for pos, is_const in zip(full_batch, is_const) if not is_const]

    chunks = rt.resolve(rw_result.levels[0], batch, record_task_stream=False)
    end = time.time()
    total_considered += len(full_batch)
    total_skipped += len(full_batch) - len(batch)
    io_size = len(full_batch) * chunk_size_bytes
    total_size += io_size

    io_per_s = io_size / (end - begin)
    io_per_s_str = sizeof_fmt(io_per_s, suffix="B/s")

    total_io_per_s = total_size / (end - start)
    total_io_per_s_str = sizeof_fmt(total_io_per_s, suffix="B/s")

    total_io_str = sizeof_fmt(total_size, suffix="B")

    print("Got {} chunks, {} total, {} skipped \t| {} \t| total {} \t| sum {}".format(len(full_batch), total_considered, total_skipped, io_per_s_str, total_io_per_s_str, total_io_str))
    begin = end
