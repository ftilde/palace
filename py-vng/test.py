import numpy as np
import time
from threading import Thread
import vng

ram_size = 1 << 32
vram_size = 1 << 32

rt = vng.RunTime(ram_size, vram_size)

a = vng.constant(32)


print(rt.resolve(a))
