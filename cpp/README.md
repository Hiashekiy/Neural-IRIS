# C++ 走廊推理模块

本目录是 `ours_corridor_net` 的独立 C++ 推理与几何后处理实现。

## 包含内容

- `export_corridor_onnx.py`：将 `models/iris_net_best.pth` 导出为 ONNX。
- `src/main_ort.cpp`：ONNX Runtime C++ 推理主程序（含计时输出）。
- `src/geometry2d.cpp`：二维几何计算与多边形处理。
- `CMakeLists.txt`：独立 CMake 构建配置。

## 1) 导出 ONNX 模型

在仓库根目录执行：

```powershell
python cpp\export_corridor_onnx.py
```

期望输出：`cpp/models/corridor_ellipse_net.onnx`

## 2) 准备 ONNX Runtime（GPU 版）

当前仓库已包含：`cpp/onnxruntime-win-x64-gpu-1.19.2`

PowerShell：

```powershell
$env:ONNXRUNTIME_ROOT = "D:/ProjectDirectory/GGMPC-dev/CorridorConstraints-running/cpp/onnxruntime-win-x64-gpu-1.19.2"
```

cmd：

```cmd
set ONNXRUNTIME_ROOT=D:\ProjectDirectory\GGMPC-dev\CorridorConstraints-running\cpp\onnxruntime-win-x64-gpu-1.19.2
```

## 3) 编译（GPU）

建议在 `cpp` 目录执行。

首次配置 + 编译：

```powershell
cmake -S . -B build_gpu -G "Visual Studio 17 2022" -DONNXRUNTIME_ROOT=$env:ONNXRUNTIME_ROOT -DUSE_CUDA_PROVIDER=ON
cmake --build build_gpu --config Release
```

若需要彻底重建：

```powershell
Remove-Item -Recurse -Force build_gpu
cmake -S . -B build_gpu -G "Visual Studio 17 2022" -DONNXRUNTIME_ROOT=$env:ONNXRUNTIME_ROOT -DUSE_CUDA_PROVIDER=ON
cmake --build build_gpu --config Release
```

编译成功后，产物位于 `build_gpu/Release`，至少应包含：

- `corridor_cpp_bridge.dll`
- `corridor_cpp_infer.exe`
- `onnxruntime.dll`
- `onnxruntime_providers_cuda.dll`
- `onnxruntime_providers_shared.dll`

## 4) 独立运行（可选）

输入掩码格式：128 行，每行 128 个字符（`0` 或 `1`）。

先导出测试样本：

```powershell
python cpp\dump_sample_mask_txt.py 0
```

再运行推理程序：

```powershell
.\build_gpu\Release\corridor_cpp_infer.exe --onnx models/corridor_ellipse_net.onnx --mask sample_mask_0.txt --patch 128 --safety 0.5
```

## 5) 在 Planner 中启用

```powershell
$env:OURS_CPP_BACKEND = "gpu"
python -c "from experiment.ours_corridor_cpp.method import infer_polygon_batch; print('cpp bridge ready:', infer_polygon_batch is not None)"
```

说明：`planner.py` 会优先走 C++ GPU 路径，只有在 C++ 桥接库不可用时才回退到旧实现。
