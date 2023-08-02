import numpy as np
import time
from threading import Thread
import vng

ram_size = 1 << 32
vram_size = 1 << 32

rt = vng.RunTime(ram_size, vram_size)

v1 = vng.open_volume("/nosnapshot/test-volumes/walnut_float.vvd")
m1 = rt.resolve(vng.mean(v1))

kx = np.array([1, 2, 1.0]).astype(np.float32)
ky = np.array([-1, 2, -1.0]).astype(np.float32)
kz = np.array([1, 1, 1.0]).astype(np.float32)

v2 = vng.linear_rescale(v1, 2, m1)
v2 = vng.separable_convolution(v2, [kx, ky, kz])
m2 = vng.mean(v2)

print(m1)
print(rt.resolve(m2))
