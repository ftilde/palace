<p align="center">
  <img width="800" src="banner.svg" />
</p>

Palace is the **p**rogressive **a**ccelerated **l**arge **a**rray **c**omputing **e**ngine.
In more detail this means:

- **P**rogressive: Designed for interactive use (i.e. featuring progressive rendering for large data sets)
- **A**ccelerated: Using GPU ressources via Vulkan
- **L**arge: Designed for larger-than-ram/vram data sets
- **A**rray: Handles multidimensional (i.e. grid-shaped) array data
- **C**omputing **E**ngine: A library with rust and python bindings

Palace enables rapid iteration during the development of out-of-core processing pipelines via chunked level-of-detail data representations and a pull-based compute architecture.
A user first defines a pipeline by combining input sources with data-transformation and/or rendering operators.
This pipeline is queried to dynamically compute chunks of the result tensor.
Chunks of input or intermediate tensors which may, for example, be outside of the camera view are never loaded/computed.

# Features

- Out-of-core computing via chunking and pull-based architecture
- Volume raycaster
- Large image viewer
- Random walker tensor segmentation
- Array operations such as slicing
- Fused point-wise operations
- Property linking system 
- On-screen gui (via [egui](https://github.com/emilk/egui)).
- Various data formats:
    - [zarr](https://zarr.dev/)
    - [hdf5](https://www.hdfgroup.org/solutions/hdf5/) (via [hidefix](https://github.com/gauteh/hidefix))
    - [vvd](https://voreen.uni-muenster.de)
    - raw
    - png
    - mp4

# Usage

Palace is a library written in rust that can be used in a number of ways.

## Prerequisites
Install [rust](https://www.rust-lang.org) (preferably using your operating system's package manager)

## Run example applications
Build and run an example application:

```sh
cargo run --release --bin <application>`
```

where `<application>` can be one of the following:
    - convert (useful for converting between data formats)
    - demo-large
    - demo-sliceviewer
    - demo-raycaster
    - demo-mean

## Python bindings

Palace can be used from python.
For building instructions and usage see the Readme in the subfolder `py-palace`.

The following demonstrates a non-interactive processing and rendering pipeline:

```python
import palace as pc
import numpy as np

# Input
time_series_4d = pc.open("path.h5")
raw_vol = time_series_4d[27,:,:,:]

# Tensor processing
kernel = np.array([1.,2.,1.], dtype=np.float32)*0.25
raw_vol = raw_vol.cast(pc.ScalarType.I16)
smooth_vol = raw_vol.separable_convolution([kernel]*3)
vol = (smooth_vol - raw_vol).abs()

fov = 30.0
frame_size = [1920, 1200]
tile_size = [512]*2
config = pc.RaycasterConfig()
tf = pc.grey_ramp_tf(min=0.0, max=1.0)
camera_state = pc.CameraState.for_volume(
  vol.metadata, vol.embedding_data, fov)
frame_md = pc.TensorMetaData(frame_size, tile_size)

# Rendering pipeline
proj = camera_state.projection_mat(frame_md.dimensions)
eep = pc.entry_exit_points(vol.metadata,
    vol.embedding_data, frame_md, proj)
lod = vol.single_level_lod()
frame = pc.raycast(lod, eep, config, tf)

# Frame processing
frame = frame.cast(pc.ScalarType.F32.vec(4))
smooth_frame = frame.separable_convolution([kernel]*2)
frame = (smooth_frame - frame).abs()
frame = frame.cast(pc.ScalarType.U8.vec(4))

# Create runtime
rt = pc.RunTime(ram_storage_size=10<<30, vram_storage_size=10<<30)
# Query top left rendered tile
# Only here actual computation happens
top_left_tile = rt.resolve(frame.unfold_dtype(), [[0]*3])
```

# License

Copyright 2025 University of Münster

This software is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
