import numpy as np
import palace as pc
import argparse
import math
import itertools
import functools
import time

ram_size = 8 << 30
vram_size = 10 << 30
disk_cache_size = 20 << 30

parser = argparse.ArgumentParser()
parser.add_argument('volume_file')
parser.add_argument('-b', '--batch', type=int, default=128)

args = parser.parse_args()

devices = [1]
rt = pc.RunTime(ram_size, vram_size, disk_cache_size, devices=devices, num_compute_threads=None)

try:
    vol = pc.open_lod(args.volume_file)
except:
    vol = pc.open(args.volume_file)
    steps = list(reversed([2.0 if i < 3 else pc.FixedStep(2.0) for i in range(0, vol.nd())]))
    vol = vol.single_level_lod()


nd = vol.nd()

vol = vol.levels[0]

md = vol.inner.metadata

print("Chunk size is: {}".format(md.chunk_size))
print("DType: {}".format(vol.inner.dtype))

chunk_elements = functools.reduce(lambda a, b: a*b, md.chunk_size, 1)
chunk_size_bytes = np.astype(chunk_elements * vol.inner.dtype.size_in_bytes(), np.uint64)

dim_in_chunks = list(map(lambda x: math.ceil(x[0]/x[1]), zip(md.dimensions, md.chunk_size)))
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
for batch in itertools.batched(itertools.product(*map(lambda end: range(0, end), dim_in_chunks)), batch_size):
    chunks = rt.resolve(vol, batch)
    end = time.time()
    io_size = len(chunks) * chunk_size_bytes
    total_size += io_size

    io_per_s = io_size / (end - begin)
    io_per_s_str = sizeof_fmt(io_per_s, suffix="B/s")

    total_io_per_s = total_size / (end - start)
    total_io_per_s_str = sizeof_fmt(total_io_per_s, suffix="B/s")

    total_io_str = sizeof_fmt(total_size, suffix="B")

    print("Got {} chunks, {} \t| total {} \t| sum {}".format(len(chunks), io_per_s_str, total_io_per_s_str, total_io_str))
    begin = end
