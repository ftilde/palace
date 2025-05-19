import numpy as np
import palace as pc
import palace_util
import argparse

ram_size = 8 << 30
vram_size = 10 << 30
disk_cache_size = 20 << 30

parser = argparse.ArgumentParser()
parser.add_argument('img_file')
parser.add_argument('-l', '--img_location', type=str)
parser.add_argument('-t', '--transfunc', type=str)
parser.add_argument('-d', '--devices', type=palace_util.list_of_ints, default=[])

args = parser.parse_args()

rt = pc.RunTime(ram_size, vram_size, disk_cache_size, devices=args.devices)

if args.img_file == "mandelbrot":
    b = 1024*2
    s = b*1024*16
    md = pc.TensorMetaData([s, s], [b, b])
    ed = pc.TensorEmbeddingData([1.0, 1.0])
    img = pc.mandelbrot(md, ed)
    tf = pc.load_tf(args.transfunc)
    tf.min = 0.0;
    tf.max = 1.0;


    img = img.map(lambda img: pc.apply_tf(img, tf))

    #def smooth(img):
    #    kernels = [pc.gauss_kernel(2.0)]*2 + [np.array([1], np.float32)]
    #    img.inner = img.inner.unfold_dtype().cast(pc.ScalarType.F32).separable_convolution(kernels).fold_into_dtype().cast(pc.DType(pc.ScalarType.U8, 4))
    #    return img
    #img = img.map(smooth)

elif args.img_file == "circle":
    s = 50
    img = np.zeros((s, s), dtype=np.float32)
    for y in range(s):
        for x in range(s):
            px = x-(s-1)/2
            py = y-(s-1)/2
            d = np.sqrt(px*px + py*py)
            img[y, x] = np.sin(d/2)/2+0.5
    img = pc.from_numpy(img)
    tf = pc.load_tf(args.transfunc)
    tf.min = 0.0;
    tf.max = 1.0;

    #img = pc.separable_convolution(img, [np.array([1,2,1], np.float32)/4]*2)
    #img = pc.separable_convolution(img, [pc.gauss_kernel(25.0)]*2)
    img = pc.apply_tf(img, tf)
    #img = pc.cast(pc.separable_convolution(pc.cast(img.unfold_dtype(), pc.ScalarType.F32), [pc.gauss_kernel(2.0)]*2 + [np.array([1], np.float32)]).fold_into_dtype(), pc.DType(pc.ScalarType.U8, 4))
    img = img.embedded(pc.TensorEmbeddingData([1.0, 1.0])).single_level_lod()
else:
    def unify_img(img):
        match img.nd():
            case 2:
                return img
            case 3:
                chunks = list(img.inner.metadata.chunk_size)
                chunks[-1] = pc.chunk_size_full

                img = img.rechunk(chunks).fold_into_dtype()
                img.inner = palace_util.pad_dtype_channels_to(img.inner, 4, 255)

            case o:
                raise "Cannot handle img of dim {}".format(o)
        return img

    try:
        img = pc.open_lod(args.img_file)
    except:
        img = pc.open(args.img_file, tensor_path_hint=args.img_location)
        steps = list([2.0 if i < 2 else None for i in range(0, img.nd())])
        img = img.create_lod(steps)

    img = img.map(lambda img: unify_img(img))

store = pc.Store()

view_state = pc.ImageViewerState().store(store)

def render(size, events):
    events.act([
        pc.OnMouseDrag(pc.MouseButton.Left, lambda pos, delta: view_state.mutate(lambda s: s.drag(delta))),
        pc.OnWheelMove(lambda delta, pos: view_state.mutate(lambda s: s.zoom(delta, pos))),
    ]);

    md = pc.TensorMetaData(size, size)
    frame = pc.view_image(img, md, view_state.load())

    return frame

rt.run_with_window(render, timeout_ms=10)
