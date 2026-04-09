# Neural-IRIS C++ Inference

This directory contains the standalone C++ inference and geometry post-processing implementation for the Neural-IRIS model.

## Contents

- `export_neural_iris_onnx.py`: export `models/neural_iris_net_best.pth` to ONNX
- `src/main_ort.cpp`: ONNX Runtime C++ inference entry
- `src/geometry2d.cpp`: 2D geometry and polygon post-processing
- `src/bridge.cpp`: Python bridge used by `cpp/python/bridge.py`
- `CMakeLists.txt`: build configuration

## 1) Export the ONNX model

Run from the repository root:

```powershell
python cpp\export_neural_iris_onnx.py
```

Expected output:

```text
cpp/models/neural_iris_net.onnx
```

## 2) Prepare ONNX Runtime

If you keep the ONNX Runtime package inside this repository, resolve it to a concrete absolute path first.

GPU package, PowerShell, from the repository root:

```powershell
$ortRoot = (Resolve-Path ".\cpp\onnxruntime-win-x64-gpu-1.19.2").Path
```

CPU package, PowerShell, from the repository root:

```powershell
$ortRoot = (Resolve-Path ".\cpp\onnxruntime-win-x64-1.19.2").Path
```

GPU package, `cmd`, from the repository root:

```cmd
set ORT_ROOT=%CD%\cpp\onnxruntime-win-x64-gpu-1.19.2
```

If you keep ONNX Runtime somewhere else, set `$ortRoot` or `ORT_ROOT` to that directory instead.

## 3) Build with GPU support

PowerShell, from the repository root:

```powershell
$ortRoot = (Resolve-Path ".\cpp\onnxruntime-win-x64-gpu-1.19.2").Path
cmake -S cpp -B cpp/build_gpu -G "Visual Studio 17 2022" "-DONNXRUNTIME_ROOT=$ortRoot" -DUSE_CUDA_PROVIDER=ON
cmake --build cpp/build_gpu --config Release
```

PowerShell, from the `cpp` directory:

```powershell
$ortRoot = (Resolve-Path ".\onnxruntime-win-x64-gpu-1.19.2").Path
cmake -S . -B build_gpu -G "Visual Studio 17 2022" "-DONNXRUNTIME_ROOT=$ortRoot" -DUSE_CUDA_PROVIDER=ON
cmake --build build_gpu --config Release
```

Clean rebuild:

```powershell
Remove-Item -Recurse -Force cpp\build_gpu
$ortRoot = (Resolve-Path ".\cpp\onnxruntime-win-x64-gpu-1.19.2").Path
cmake -S cpp -B cpp/build_gpu -G "Visual Studio 17 2022" "-DONNXRUNTIME_ROOT=$ortRoot" -DUSE_CUDA_PROVIDER=ON
cmake --build cpp/build_gpu --config Release
```

Expected outputs in `build_gpu/Release`:

- `neural_iris_cpp_bridge.dll`
- `neural_iris_cpp_infer.exe`
- `onnxruntime.dll`
- `onnxruntime_providers_cuda.dll`
- `onnxruntime_providers_shared.dll`

## 4) Build with CPU only

```powershell
$ortRoot = (Resolve-Path ".\cpp\onnxruntime-win-x64-1.19.2").Path
Remove-Item -Recurse -Force cpp\build_cpu
cmake -S cpp -B cpp/build_cpu -G "Visual Studio 17 2022" "-DONNXRUNTIME_ROOT=$ortRoot" -DUSE_CUDA_PROVIDER=OFF
cmake --build cpp/build_cpu --config Release
```

Why this quoting matters:

- `-DONNXRUNTIME_ROOT=$env:ONNXRUNTIME_ROOT` can be passed incorrectly by PowerShell in some cases.
- `"-DONNXRUNTIME_ROOT=$ortRoot"` forces CMake to receive the fully expanded path as a single argument.

## 5) Run the standalone executable

First dump a sample mask:

```powershell
python cpp\dump_sample_mask_txt.py 0
```

Then run inference:

```powershell
.\build_gpu\Release\neural_iris_cpp_infer.exe --onnx models/neural_iris_net.onnx --mask sample_mask_0.txt --patch 128 --safety 0.5
```

## 6) Use it from Python

```powershell
$env:NEURAL_IRIS_CPP_BACKEND = "gpu"
python -c "from cpp.python import infer_safe_region_batch; print('cpp bridge ready:', infer_safe_region_batch is not None)"
```

`src/planner/planner.py` will prefer the C++ backend when the bridge DLL and ONNX model are available.



