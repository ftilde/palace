import numpy as np
import palace as pc
import palace_util
import argparse

ram_size = 8 << 30
vram_size = 10 << 30
disk_cache_size = 20 << 30

parser = argparse.ArgumentParser()
parser.add_argument('img_file')
parser.add_argument('-t', '--transfunc', type=str)

args = parser.parse_args()

img = pc.open(args.img_file)
md: pc.TensorMetaData = img.inner.metadata
ndim = len(md.dimensions)

img = img.rechunk([128]*ndim)
img = img.create_lod([2.0]*2)

devices = []
rt = pc.RunTime(ram_size, vram_size, disk_cache_size, devices=devices)

ed = img.levels[0].embedding_data

foreground_seeds = np.empty(shape=[0,ndim], dtype=np.float32)
background_seeds = np.empty(shape=[0,ndim], dtype=np.float32)

#if args.transfunc:
#    tf = pc.load_tf(args.transfunc)
#else:
#    tf = pc.grey_ramp_tf(0.0, 1.0)

store = pc.Store()
view_state = pc.ImageViewerState().store(store)

mode = store.store_primitive("hierarchical")
weight_function = store.store_primitive("bian_mean")
min_edge_weight = store.store_primitive(1e-5)

beta = store.store_primitive(128.0)
extent = store.store_primitive(1)

gui_state = pc.GuiState(rt)

num_tf_values = 128
def prob_tf_from_values(values):
    #tf_table = [[0, 0, 255, 255] for i in range(num_tf_values)] + [[255, 0, 0, 255] for i in range(num_tf_values)]
    tf_table = pc.from_numpy(np.array(values, np.uint8)).fold_into_dtype()
    return pc.TransFuncOperator(0.0, 1.0, tf_table)

tf_prob = prob_tf_from_values([[0, 0, num_tf_values-i-1, num_tf_values-i-1] for i in range(num_tf_values)] + [[i, 0, 0, i] for i in range(num_tf_values)])

mouse_pos_and_value = None

def apply_weight_function(tensor):
    match weight_function.load():
        case "grady":
            return pc.randomwalker_weights(tensor, min_edge_weight.load(), beta.load())
        case "bian_mean":
            return pc.randomwalker_weights_bian(tensor, min_edge_weight.load(), extent.load())
        case "bhatt_var_gaussian":
            return pc.randomwalker_weights_bhattacharyya_var_gaussian(tensor, min_edge_weight.load(), extent.load())
        case "ttest":
            return pc.randomwalker_weights_ttest(tensor, min_edge_weight.load(), extent.load())

def apply_rw_mode(input):

    fg_seeds_tensor = pc.from_numpy(foreground_seeds).fold_into_dtype()
    bg_seeds_tensor = pc.from_numpy(background_seeds).fold_into_dtype()

    def sq_f32(t):
        t = t.cast(pc.ScalarType.F32)
        return t * t

    def to_scalar(input):
        return input.map(lambda l: sq_f32(l.index(0)) + sq_f32(l.index(1)) + sq_f32(l.index(2)))

    match mode.load():
        case "normal":
            i = to_scalar(input)
            i = i.levels[0].rechunk([pc.chunk_size_full]*ndim)
            md: pc.TensorMetaData = i.inner.metadata
            weights = apply_weight_function(i)
            seeds = pc.rasterize_seed_points(fg_seeds_tensor, bg_seeds_tensor, md, ed)
            rw_result = pc.randomwalker(weights, seeds, max_iter=1000, max_residuum_norm=0.001)
            return (input, rw_result.create_lod([2.0]*2))

        case "hierarchical":
            i = to_scalar(input)
            weights = i.map(lambda level: apply_weight_function(level.inner).embedded(pc.TensorEmbeddingData(np.append(level.embedding_data.spacing, [1.0])))).cache_coarse_levels()
            rw_result = pc.hierarchical_randomwalker(weights, fg_seeds_tensor, bg_seeds_tensor).cache_coarse_levels()
            return (input, rw_result)


def render(size, events: pc.Events):
    global mouse_pos_and_value

    v, rw_result = apply_rw_mode(img)

    # GUI stuff
    widgets = []

    widgets.append(pc.ComboBox("Mode", mode, ["normal", "hierarchical"]))
    widgets.append(pc.ComboBox("Weight Function", weight_function, ["grady", "bian_mean", "bhatt_var_gaussian", "ttest"]))
    match weight_function.load():
        case "grady":
            widgets.append(palace_util.named_slider("beta", beta, 0.01, 10000, logarithmic=True))
        case "bian_mean" | "var_gaussian" | "ttest":
            widgets.append(palace_util.named_slider("extent", extent, 1, 5))

    widgets.append(palace_util.named_slider("min_edge_weight", min_edge_weight, 1e-20, 1, logarithmic=True))

    if mouse_pos_and_value is not None:
        img_pos, value = mouse_pos_and_value
        widgets.append(pc.Label(f"Value at {img_pos} = {value}"))
        mouse_pos_and_value = None

    def add_seed_point(slice_state, embedded_tensor, pos, frame_size, foreground):
        global foreground_seeds, background_seeds

        image_pos = palace_util.mouse_to_image_pos(slice_state.load(), embedded_tensor, pos, frame_size)
        image_pos = np.array(image_pos, dtype=np.float32).reshape((1, ndim))
        if foreground:
            foreground_seeds = np.concat([foreground_seeds, image_pos])
        else:
            background_seeds = np.concat([background_seeds, image_pos])

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
        #out = slice_rw

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

    # Actual composition of the rendering
    frame = overlay(view_state)

    gui = gui_state.setup(events, pc.Vertical(widgets))

    frame = gui.render(frame(size, events))

    return frame

rt.run_with_window(render, timeout_ms=10, record_task_stream=False, bench=False)
