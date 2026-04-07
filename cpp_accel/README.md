# C++ Acceleration Prototype

This directory is an isolated C++ pipeline prototype for `ours_corridor_net`.

## What is included

- `export_corridor_onnx.py`: exports `models/iris_net_best.pth` to ONNX.
- `src/main_ort.cpp`: ONNX Runtime C++ inference + postprocessing timing.
- `src/geometry2d.cpp`: 2D half-space intersection and mask metrics.
- `CMakeLists.txt`: standalone CMake build.

## 1) Export ONNX

```powershell
E:\CondaEnvData\envs\GGMPC\python.exe cpp_accel\export_corridor_onnx.py
```

Expected output model:

- `cpp_accel/models/corridor_ellipse_net.onnx`

## 2) Prepare ONNX Runtime

Download ONNX Runtime for your platform and set `ONNXRUNTIME_ROOT`.

Windows example:

```powershell
$env:ONNXRUNTIME_ROOT = "D:/tools/onnxruntime-win-x64-gpu-1.19.2"
```

If you want GPU acceleration, use the GPU package above and ensure CUDA/cuDNN runtime DLLs are available and compatible with your ONNX Runtime GPU package.

If you are using `cmd.exe`, use:

```cmd
set ONNXRUNTIME_ROOT=D:\ProjectDirectory\GGMPC-dev\CorridorConstraints-running\cpp_accel\onnxruntime-win-x64-gpu-1.19.2
```

Then in the same `cmd.exe` window, build with:

```cmd
cd /d D:\ProjectDirectory\GGMPC-dev\CorridorConstraints-running\cpp_accel
cmake -S . -B build_gpu -G "Visual Studio 17 2022" -DONNXRUNTIME_ROOT=%ONNXRUNTIME_ROOT% -DUSE_CUDA_PROVIDER=ON
cmake --build build_gpu --config Release
```

## 3) Build

```powershell
cd cpp_accel
cmake -S . -B build_gpu -DONNXRUNTIME_ROOT=D:/ProjectDirectory/GGMPC-dev/CorridorConstraints-running/cpp_accel/onnxruntime-win-x64-gpu-1.19.2 -DUSE_CUDA_PROVIDER=ON
cmake --build build_gpu --config Release
```

`cmd.exe` version:

```cmd
cd cpp_accel
cmake -S . -B build_gpu -DONNXRUNTIME_ROOT=D:/ProjectDirectory/GGMPC-dev/CorridorConstraints-running/cpp_accel/onnxruntime-win-x64-gpu-1.19.2 -DUSE_CUDA_PROVIDER=ON
cmake --build build_gpu --config Release
```

## 4) Run

Input mask text format: 128 lines, each line has 128 chars (`0` or `1`).

You can dump one mask from test split:

```powershell
E:\CondaEnvData\envs\GGMPC\python.exe cpp_accel\dump_sample_mask_txt.py 0
```

```powershell
./build_gpu/Release/corridor_cpp_infer --onnx models/corridor_ellipse_net.onnx --mask sample_mask_0.txt --patch 128 --safety 0.5
```

Program output includes timing breakdown and collision ratio (relaxed threshold is left to caller policy).
