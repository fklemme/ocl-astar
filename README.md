# ocl-astar
OpenCL A* Implementations

This project uses [Boost Compute](https://github.com/boostorg/compute) to make use of OpenCL. It is a pure template/header library, so no further compiling or linking is required. Compute is part of Boost since version 1.61.

## Linux
If your system repositories don't provide a recent enough version you can run `make boost` to download and place one into the current project.

`make` builds and runs the program.

## Windows
For AMD: Download and install OpenCL SDK from [Github](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases).
For Nvidia: Download and install CUDA Toolkit from [Nvidia.com](https://developer.nvidia.com/cuda-downloads).

Download [boost](http://www.boost.org/) an extract it to a subfolder called `boost` (without version number).

For some reason, Compute wants to make use of `libboost_chrono` so let's just build all the libraries.
Open VS command prompt, switch into the boost directory and run `bootstrap` and `b2`.
