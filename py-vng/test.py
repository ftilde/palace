import numpy as np
import time
import vng
import math

ram_size = 8 << 30
vram_size = 10 << 30

print(vram_size)

rt = vng.RunTime(ram_size, vram_size)

window = vng.Window(rt)

v = vng.open_volume("/nosnapshot/test-volumes/walnut_float2.vvd")

k = np.array([1, 2, 1]).astype(np.float32) * 0.25

#v2 = vng.linear_rescale(v1, 2, m1)
v = vng.separable_convolution(v, [k]*3)

fov = 30.0
eye = [5.5, 0.5, 0.5]
center = [0.5, 0.5, 0.5]
up = [1.0, 1.0, 0.0]

i = 0

def render(size, events):
    global i
    i += 1

    def foo(pos, delta):
        print(pos, delta)
    def bar(pos):
        print(pos)

    events.act([
        vng.OnMouseDrag(vng.MouseButton.Left, foo),
        vng.OnMouseClick(vng.MouseButton.Left, bar),
        vng.OnWheelMove(bar),
        vng.OnKeyPress("A", lambda: print("indeed a key")),
    ]);

    fov_r = fov# * 1 + math.sin(i/20) * 0.6

    md = vng.ImageMetadata(size, [512]*2)

    look_at = vng.look_at(eye, center, up);
    perspective = vng.perspective(md, fov_r, 0.01, 100)
    proj = perspective.dot(look_at)

    eep = vng.entry_exit_points(v.metadata, md, proj)
    frame = vng.raycast(v, eep)
    frame = vng.rechunk(frame, [vng.chunk_size_full]*3)

    return frame

window.run(rt, render, timeout_ms=10)
