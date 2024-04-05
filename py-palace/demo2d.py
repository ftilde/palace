import numpy as np
import palace as pc
import argparse

ram_size = 8 << 30
vram_size = 10 << 30
disk_cache_size = 20 << 30

parser = argparse.ArgumentParser()
parser.add_argument('img_file')
parser.add_argument('-t', '--transfunc', type=str)

args = parser.parse_args()

rt = pc.RunTime(ram_size, vram_size, disk_cache_size, device=0)

if args.img_file == "mandelbrot":
    s = 1024*1024
    md = pc.tensor_metadata([s, s], [256, 256])
    img = pc.mandelbrot(md)
    tf = pc.load_tf(args.transfunc)
    tf.min = 0.0;
    tf.max = 1.0;
    img = img.map(lambda img: pc.apply_tf(img, tf))
else:
    img = pc.read_png(args.img_file)
    img = img.embedded(pc.TensorEmbeddingData([1.0, 1.0])).single_level_lod()

store = pc.Store()

view_state = pc.ImageViewerState().store(store)

def render(size, events):
    events.act([
        pc.OnMouseDrag(pc.MouseButton.Left, lambda pos, delta: view_state.mutate(lambda s: s.drag(delta))),
        pc.OnWheelMove(lambda delta, pos: view_state.mutate(lambda s: s.zoom(delta, pos))),
    ]);

    md = pc.tensor_metadata(size, size)
    frame = pc.view_image(img, md, view_state.load())

    return frame

window = pc.Window(rt)
window.run(render, timeout_ms=10)
