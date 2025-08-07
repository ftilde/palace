import numpy as np
import palace as pc
import palace_util
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('img_file')
parser.add_argument('-t', '--transfunc', type=str)
parser.add_argument('--max-iter', type=int, default=1000)
parser.add_argument('--max-residuum-norm', type=float, default=1e-3)
palace_util.add_runtime_args(parser)

args = parser.parse_args()

img = pc.open(args.img_file)

try:
    img = pc.open_lod(args.img_file)
    if img.levels[0].metadata.dimensions[-1] < 4:
        img = img.map(lambda i: i.fold_into_dtype())

    lod_args = list(reversed([2.0 if i < 3 else pc.FixedStep(2.0) for i in range(0, img.nd())]))

    #img.levels = img.levels[2:]
except:
    img = pc.open(args.img_file)
    if img.metadata.dimensions[-1] < 4:
        img = img.fold_into_dtype()

    chunks = list(reversed([128 if i < 2 else 8 for i in range(0, img.nd())]))
    img = img.rechunk(chunks)
    lod_args = list(reversed([2.0 if i < 3 else pc.FixedStep(2.0) for i in range(0, img.nd())]))
    img = img.create_lod(lod_args)

img = img.map(lambda i: palace_util.pad_dtype_channels_to(i, 4, 255))

md: pc.TensorMetaData = img.fine_metadata()
nd = img.nd()
size_time = md.dimensions[0]

scalar_weight_functions = ["grady", "bian_mean", "bhatt_var_gaussian", "ttest"]
vector_weight_functions = ["grady rgb"]
weight_functions = scalar_weight_functions + vector_weight_functions

for l in img.levels:
    print(l.metadata.dimensions)
    #print(l.metadata.chunk_size)

#img = img.rechunk(chunk_sizes)

rt = palace_util.build_runtime_from_args(args)

ed = img.fine_embedding_data()

foreground_seeds = np.empty(shape=[0,nd], dtype=np.float32)
background_seeds = np.empty(shape=[0,nd], dtype=np.float32)

#if args.transfunc:
#    tf = pc.load_tf(args.transfunc)
#else:
#    tf = pc.grey_ramp_tf(0.0, 1.0)

store = pc.Store()
view_state = pc.ImageViewerState().store(store)

mode = store.store_primitive("hierarchical")
weight_function = store.store_primitive("grady")
min_edge_weight = store.store_primitive(1e-5)

beta = store.store_primitive(128.0)
extent = store.store_primitive(1)

timestep = store.store_primitive(0)

gui_state = pc.GuiState(rt)

num_tf_values = 128
def prob_tf_from_values(values):
    def tf_curve(v):
        gamma = 0.1
        return np.pow(v/255, gamma)*255
    values = [list(map(tf_curve, l)) for l in values]
    #values = [[0, 0, 255, 255] for i in range(num_tf_values)] + [[255, 0, 0, 255] for i in range(num_tf_values)]
    tf_table = pc.from_numpy(np.array(values, np.uint8)).fold_into_dtype()
    return pc.TransFuncOperator(0.0, 1.0, tf_table)

alpha_mul = 0.01
tf_prob = prob_tf_from_values([[0, 0, num_tf_values-i-1, alpha_mul*(num_tf_values-i-1)] for i in range(num_tf_values)] + [[i, 0, 0, alpha_mul*i] for i in range(num_tf_values)])

mouse_pos_and_value = None

def select_from_ts(v, ts):
    match nd:
        case 2:
            return v
        case 3:
            return palace_util.slice_time_nd(ts, v)
            #return v[ts,:,:]
        case o:
            raise f"Invalid number of tensor dimensions: {o}"

def ts_next(ts):
    ts.write((ts.load() + 1) % size_time)

def ts_prev(ts):
    ts.write((ts.load() + size_time - 1) % size_time)

def apply_weight_function(tensor):
    wf = weight_function.load()

    if wf in scalar_weight_functions:
        i = tensor.cast(pc.ScalarType.F32.vec(tensor.dtype.size))
        i = (i*i).hsum().sqrt()
    else:
        i = tensor

    match wf:
        case "grady":
            return pc.randomwalker_weights(i, min_edge_weight.load(), beta.load())
        case "grady rgb":
            pairs = pc.randomwalker_weight_pairs(i)
            n = pairs.dtype.size
            n2 = n//2;
            orig_dtype = pairs.dtype.scalar
            pairs = pairs.cast(pc.ScalarType.F32.vec(n))
            if orig_dtype.is_integer():
                pairs = pairs / pc.jit(orig_dtype.max_value()).splat(n)
            diff = pairs.index_range(0, n2) - pairs.index_range(n2, n)
            w = (-(diff * diff).hsum() * beta.load()).exp()
            return w.max(min_edge_weight.load())

        case "bian_mean":
            return pc.randomwalker_weights_bian(i, min_edge_weight.load(), extent.load())
        case "bhatt_var_gaussian":
            return pc.randomwalker_weights_bhattacharyya_var_gaussian(i, min_edge_weight.load(), extent.load())
        case "ttest":
            return pc.randomwalker_weights_ttest(i, min_edge_weight.load(), extent.load())

def apply_rw_mode(input):

    fg_seeds_tensor = pc.from_numpy(foreground_seeds).fold_into_dtype()
    bg_seeds_tensor = pc.from_numpy(background_seeds).fold_into_dtype()

    match mode.load():
        case "normal":
            i = input.levels[0].rechunk([pc.ChunkSizeFull()]*nd)
            md: pc.TensorMetaData = i.inner.metadata
            weights = apply_weight_function(i)
            seeds = pc.rasterize_seed_points(fg_seeds_tensor, bg_seeds_tensor, md, ed)
            rw_result = pc.randomwalker(weights, seeds, max_iter=args.max_iter, max_residuum_norm=args.max_residuum_norm)
            return (input, rw_result.create_lod(lod_args))

        case "hierarchical":
            weights = input.map(lambda level: apply_weight_function(level.inner).embedded(pc.TensorEmbeddingData(np.append(level.embedding_data.spacing, [1.0])))).cache_coarse_levels()
            rw_result = pc.hierarchical_randomwalker(weights, fg_seeds_tensor, bg_seeds_tensor, max_iter=args.max_iter, max_residuum_norm=args.max_residuum_norm).cache_coarse_levels()
            return (input, rw_result)


def render(size, events: pc.Events):
    global mouse_pos_and_value


    v, rw_result = apply_rw_mode(img)

    v = select_from_ts(v, timestep.load())
    rw_result = select_from_ts(rw_result, timestep.load())

    # GUI stuff
    widgets = []

    widgets.append(pc.ComboBox("Mode", mode, ["normal", "hierarchical"]))
    widgets.append(pc.ComboBox("Weight Function", weight_function, weight_functions))
    match weight_function.load():
        case "grady" | "custom":
            widgets.append(palace_util.named_slider("beta", beta, 0.01, 10000, logarithmic=True))
        case "bian_mean" | "var_gaussian" | "ttest":
            widgets.append(palace_util.named_slider("extent", extent, 1, 5))

    widgets.append(palace_util.named_slider("min_edge_weight", min_edge_weight, 1e-20, 1, logarithmic=True))

    if nd > 2:
        widgets.append(palace_util.named_slider("Timestep", timestep, 0, size_time-1))

    if mouse_pos_and_value is not None:
        img_pos, value = mouse_pos_and_value
        widgets.append(pc.Label(f"Value at {img_pos} = {value}"))
        mouse_pos_and_value = None

    def add_seed_point(slice_state, embedded_tensor, pos, frame_size, foreground):
        global foreground_seeds, background_seeds

        image_pos = palace_util.mouse_to_image_pos(slice_state.load(), embedded_tensor, pos, frame_size)
        if image_pos is not None:
            match nd:
                case 2:
                    pos_nd = np.array(image_pos, dtype=np.float32)
                case 3:
                    pos_nd = [timestep.load()] + list(img_pos)
                    pos_nd = np.array(pos_nd, dtype=np.float32)

            pos_nd = np.concatenate(([1], pos_nd), dtype = np.float32)
            pos_nd = img.fine_embedding_data().voxel_to_physical().dot(pos_nd)
            pos_nd = pos_nd[1:]
            pos_nd = pos_nd.reshape((1, nd))

            if foreground:
                foreground_seeds = np.concat([foreground_seeds, pos_nd])
            else:
                background_seeds = np.concat([background_seeds, pos_nd])

    def view_image(image, state):
        def inner(size, events):
            md = pc.TensorMetaData(size, size)
            return pc.view_image(image, md, state.load())

        return inner

    def overlay(state):

        frame = view_image(v, view_state)

        rw_img = rw_result.map(lambda img: pc.apply_tf(img, tf_prob))
        frame_rw = view_image(rw_img, view_state)

        #slice_edge = palace_util.render_slice(edge_w, 0, slice_state0, tf)

        out = palace_util.alpha_blending(frame_rw, frame)

        def inspect(size, events):
            tensor = rw_result.levels[0]
            extract_slice_value(size, events, state, tensor)
            events.act([
                pc.OnMouseClick(pc.MouseButton.Left, lambda x: add_seed_point(state, tensor, x, size, True)).when(lambda s: s.is_down("ShiftLeft")),
                pc.OnMouseClick(pc.MouseButton.Right, lambda x: add_seed_point(state, tensor, x, size, False)).when(lambda s: s.is_down("ShiftLeft")),
                pc.OnMouseDrag(pc.MouseButton.Left, lambda pos, delta: view_state.mutate(lambda s: s.drag(delta))),
                pc.OnWheelMove(lambda delta, pos: view_state.mutate(lambda s: s.zoom(delta, pos))),
                ])

        return palace_util.inspect_component(out, inspect)

    def extract_slice_value(size, events: pc.Events, slice_state, tensor):
        global mouse_pos_and_value
        mouse_pos = events.latest_state().mouse_pos()
        img_pos = mouse_pos and palace_util.mouse_to_image_pos(slice_state.load(), tensor, mouse_pos, size)
        if img_pos is not None:
            value = palace_util.extract_tensor_value(rt, tensor, img_pos)
            mouse_pos_and_value = (img_pos, value)

    events.act([
        pc.OnKeyPress("N", lambda: ts_next(timestep)),
        pc.OnKeyPress("P", lambda: ts_prev(timestep)),
        ])

    # Actual composition of the rendering
    frame = overlay(view_state)

    gui = gui_state.setup(events, pc.Vertical(widgets))

    frame = gui.render(frame(size, events))

    return frame

rt.run_with_window(render, timeout_ms=10, record_task_stream=False, bench=False)
