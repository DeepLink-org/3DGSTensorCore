# tcgs_module
This is a flexible library which **utilize Tensor Cores** for accelerating 3D Gaussian Splatting. You can install the module for any 3D Gaussian Splatting kernels.

# Usage

## 1. Install this library as the directories followed:
```
-cuda_rasterizer
--tcgs
---tcgs_forward.cu
---tcgs_utils.h
---tcgs.h
--auxiliary.h
....
--rasterizer.h
....
-rastierze_points.cu
....
```

```
cd ./cuda_rasterizer
git clone https://github.com/zmliao/tcgs
cd ..
```
## 2. Update the ```setup.py``` and the  ```CMakeLists.txt```

### Examples:

```setup.py```:
```py

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

base_dir = os.path.dirname(os.path.abspath(__file__))
tcgs_dir = tcgs_dir = os.path.join(base_dir, "cuda_rasterizer/tcgs")

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "cuda_rasterizer/tcgs/tcgs_forward.cu",
                "rasterize_points.cu",
                "ext.cpp"],
            include_dirs=[
                os.path.join(base_dir, "cuda_rasterizer"),
                os.path.join(base_dir, "third_party/glm/"),
                tcgs_dir
            ],
            extra_compile_args={
                "nvcc": [
                    "--expt-relaxed-constexpr",
                    "--ptxas-options=-v",
                    "-DTCGS_ENABLED=1",
                ],
                "cxx": ["-DTCGS_ENABLED=1"]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```
```CMakeList.txt```:
```
cmake_minimum_required(VERSION 3.20)

project(DiffRast LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(CudaRasterizer
	cuda_rasterizer/backward.h
	cuda_rasterizer/backward.cu
	cuda_rasterizer/forward.h
	cuda_rasterizer/forward.cu
	cuda_rasterizer/auxiliary.h
	cuda_rasterizer/rasterizer_impl.cu
	cuda_rasterizer/rasterizer_impl.h
	cuda_rasterizer/rasterizer.h
	cuda_rasterizer/tcgs/tcgs.h
	cuda_rasterizer/tcgs/tcgs_utils.h
	cuda_rasterizer/tcgs/tcgs
)

set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86")

target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer/tcgs)
target_include_directories(CudaRasterizer PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
```
## 3. Use TC-GS
### 3.1 Modify ```rasterizer_impl.cu```
1. include tcgs
```cpp
#include "tcgs.h"
```
2. use tcgs
```cpp
#if USE_TCGS
	CHECK_CUDA(TCGS::renderCUDA_Forward(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height, P,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		depth), debug)
#else
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		depth), debug)
#endif
```

### 3.2 Activate TCGS ```config.h```
```cpp
#define USE_TCGS true
```

## 4. Install TCGS
```
python setup.py install
```

# License
This project uses the [MIT License](LICENSE).

This project is based on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
