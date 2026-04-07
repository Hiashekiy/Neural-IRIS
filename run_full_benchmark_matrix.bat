@echo off
setlocal enabledelayedexpansion

REM One-click full benchmark matrix runner
REM Covers:
REM - WSL-only methods: iris,decomputil,firi
REM - Windows methods: baselines + ours(py/cpp-cpu/cpp-gpu)
REM - batch / non-batch modes
REM - merged summaries per mode/version

set ROOT=%~dp0
cd /d "%ROOT%"

set PY=E:\CondaEnvData\envs\GGMPC\python.exe
set OUT=experiment\metrics_results\matrix
if not exist "%OUT%" mkdir "%OUT%"

echo [1/12] WSL single (iris,decomputil,firi)
wsl -e bash -lc "cd /mnt/d/ProjectDirectory/GGMPC-dev/CorridorConstraints-running; python3 experiment/run_parameter_metrics.py --methods iris,decomputil,firi --no_batch --raw_out experiment/metrics_results/matrix/raw_wsl_single.npz --out_txt experiment/metrics_results/matrix/summary_wsl_single.txt --out_csv experiment/metrics_results/matrix/summary_wsl_single.csv"
if errorlevel 1 goto :fail

echo [2/12] WSL batch (iris,decomputil,firi)
wsl -e bash -lc "cd /mnt/d/ProjectDirectory/GGMPC-dev/CorridorConstraints-running; python3 experiment/run_parameter_metrics.py --methods iris,decomputil,firi --raw_out experiment/metrics_results/matrix/raw_wsl_batch.npz --out_txt experiment/metrics_results/matrix/summary_wsl_batch.txt --out_csv experiment/metrics_results/matrix/summary_wsl_batch.csv"
if errorlevel 1 goto :fail

echo [3/12] Windows non-batch (baselines + ours python)
%PY% experiment\run_parameter_metrics.py --methods ours_corridor_net,largest_empty_circle,rotated_rectangle,heuristic_ellipse_fit,segmentation_polygon_postprocess,direct_polygon_regression --no_batch --raw_out %OUT%\raw_win_py_single.npz --out_txt %OUT%\summary_win_py_single.txt --out_csv %OUT%\summary_win_py_single.csv
if errorlevel 1 goto :fail

echo [4/12] Windows batch (baselines + ours python)
%PY% experiment\run_parameter_metrics.py --methods ours_corridor_net,largest_empty_circle,rotated_rectangle,heuristic_ellipse_fit,segmentation_polygon_postprocess,direct_polygon_regression --batch_size 20 --raw_out %OUT%\raw_win_py_batch.npz --out_txt %OUT%\summary_win_py_batch.txt --out_csv %OUT%\summary_win_py_batch.csv
if errorlevel 1 goto :fail

echo [5/12] Windows non-batch (baselines + ours cpp cpu)
set OURS_CPP_BACKEND=cpu
%PY% experiment\run_parameter_metrics.py --methods ours_corridor_cpp,largest_empty_circle,rotated_rectangle,heuristic_ellipse_fit,segmentation_polygon_postprocess,direct_polygon_regression --no_batch --raw_out %OUT%\raw_win_cpp_cpu_single.npz --out_txt %OUT%\summary_win_cpp_cpu_single.txt --out_csv %OUT%\summary_win_cpp_cpu_single.csv
if errorlevel 1 goto :fail

echo [6/12] Windows batch (baselines + ours cpp cpu)
set OURS_CPP_BACKEND=cpu
%PY% experiment\run_parameter_metrics.py --methods ours_corridor_cpp,largest_empty_circle,rotated_rectangle,heuristic_ellipse_fit,segmentation_polygon_postprocess,direct_polygon_regression --batch_size 20 --raw_out %OUT%\raw_win_cpp_cpu_batch.npz --out_txt %OUT%\summary_win_cpp_cpu_batch.txt --out_csv %OUT%\summary_win_cpp_cpu_batch.csv
if errorlevel 1 goto :fail

echo [7/12] Prep PATH for GPU backend
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;E:\CondaEnvData\envs\GGMPC\Lib\site-packages\torch\lib;%ROOT%cpp_accel\build_gpu\Release;%PATH%

echo [8/12] Windows non-batch (baselines + ours cpp gpu)
set OURS_CPP_BACKEND=gpu
%PY% experiment\run_parameter_metrics.py --methods ours_corridor_cpp,largest_empty_circle,rotated_rectangle,heuristic_ellipse_fit,segmentation_polygon_postprocess,direct_polygon_regression --no_batch --raw_out %OUT%\raw_win_cpp_gpu_single.npz --out_txt %OUT%\summary_win_cpp_gpu_single.txt --out_csv %OUT%\summary_win_cpp_gpu_single.csv
if errorlevel 1 goto :fail

echo [9/12] Windows batch (baselines + ours cpp gpu)
set OURS_CPP_BACKEND=gpu
%PY% experiment\run_parameter_metrics.py --methods ours_corridor_cpp,largest_empty_circle,rotated_rectangle,heuristic_ellipse_fit,segmentation_polygon_postprocess,direct_polygon_regression --batch_size 20 --raw_out %OUT%\raw_win_cpp_gpu_batch.npz --out_txt %OUT%\summary_win_cpp_gpu_batch.txt --out_csv %OUT%\summary_win_cpp_gpu_batch.csv
if errorlevel 1 goto :fail

echo [10/12] Merge: Python single + WSL single
%PY% experiment\merge_parameter_metrics.py --raw_a %OUT%\raw_win_py_single.npz --raw_b %OUT%\raw_wsl_single.npz --out_txt %OUT%\merged_py_single.txt --out_csv %OUT%\merged_py_single.csv
if errorlevel 1 goto :fail

echo [11/12] Merge: C++ CPU batch + WSL batch
%PY% experiment\merge_parameter_metrics.py --raw_a %OUT%\raw_win_cpp_cpu_batch.npz --raw_b %OUT%\raw_wsl_batch.npz --out_txt %OUT%\merged_cpp_cpu_batch.txt --out_csv %OUT%\merged_cpp_cpu_batch.csv
if errorlevel 1 goto :fail

echo [12/12] Merge: C++ GPU batch + WSL batch
%PY% experiment\merge_parameter_metrics.py --raw_a %OUT%\raw_win_cpp_gpu_batch.npz --raw_b %OUT%\raw_wsl_batch.npz --out_txt %OUT%\merged_cpp_gpu_batch.txt --out_csv %OUT%\merged_cpp_gpu_batch.csv
if errorlevel 1 goto :fail

echo.
echo All done. Key outputs in: %OUT%
echo - merged_py_single.txt
echo - merged_cpp_cpu_batch.txt
echo - merged_cpp_gpu_batch.txt
goto :eof

:fail
echo.
echo FAILED with errorlevel %errorlevel%.
echo Check last command output above.
exit /b %errorlevel%
