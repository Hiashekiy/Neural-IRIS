@echo off
setlocal enabledelayedexpansion

REM One-click full benchmark matrix runner
REM Covers:
REM - WSL-only methods: iris,decomputil,firi (single only)
REM - Windows baselines: single only
REM - ours methods: single + batch
REM - merged summaries per mode/version
REM - one total summary across all completed groups

set ROOT=%~dp0
cd /d "%ROOT%"

set PY=E:\CondaEnvData\envs\GGMPC\python.exe
set OUT=experiment\metrics_results\matrix
if not exist "%OUT%" mkdir "%OUT%"

echo [1/12] WSL single (iris,decomputil,firi)
if exist "%OUT%\raw_wsl_single.npz" (
	echo   skipping because %OUT%\raw_wsl_single.npz already exists
) else (
	wsl -e bash -lc "cd /mnt/d/ProjectDirectory/GGMPC-dev/CorridorConstraints-running; python3 experiment/run_parameter_metrics.py --methods iris,decomputil,firi --no_batch --raw_out experiment/metrics_results/matrix/raw_wsl_single.npz --out_txt experiment/metrics_results/matrix/summary_wsl_single.txt --out_csv experiment/metrics_results/matrix/summary_wsl_single.csv"
	if errorlevel 1 goto :fail
)

echo [2/12] Windows non-batch (baselines only)
%PY% experiment\run_parameter_metrics.py --methods largest_empty_circle,rotated_rectangle,heuristic_ellipse_fit,segmentation_polygon_postprocess,direct_polygon_regression --no_batch --raw_out %OUT%\raw_win_baseline_single.npz --out_txt %OUT%\summary_win_baseline_single.txt --out_csv %OUT%\summary_win_baseline_single.csv
if errorlevel 1 goto :fail

echo [3/12] Windows non-batch (ours python)
%PY% experiment\run_parameter_metrics.py --methods ours_corridor_net --no_batch --raw_out %OUT%\raw_win_ours_py_single.npz --out_txt %OUT%\summary_win_ours_py_single.txt --out_csv %OUT%\summary_win_ours_py_single.csv
if errorlevel 1 goto :fail

echo [4/12] Windows batch (ours python, N=20)
%PY% experiment\run_parameter_metrics.py --methods ours_corridor_net --batch_size 20 --raw_out %OUT%\raw_win_ours_py_batch.npz --out_txt %OUT%\summary_win_ours_py_batch.txt --out_csv %OUT%\summary_win_ours_py_batch.csv
if errorlevel 1 goto :fail

echo [5/12] Windows non-batch (ours cpp cpu)
set OURS_CPP_BACKEND=cpu
%PY% experiment\run_parameter_metrics.py --methods ours_corridor_cpp --no_batch --raw_out %OUT%\raw_win_ours_cpp_cpu_single.npz --out_txt %OUT%\summary_win_ours_cpp_cpu_single.txt --out_csv %OUT%\summary_win_ours_cpp_cpu_single.csv
if errorlevel 1 goto :fail

echo [6/12] Windows batch (ours cpp cpu, N=20)
set OURS_CPP_BACKEND=cpu
%PY% experiment\run_parameter_metrics.py --methods ours_corridor_cpp --batch_size 20 --raw_out %OUT%\raw_win_ours_cpp_cpu_batch.npz --out_txt %OUT%\summary_win_ours_cpp_cpu_batch.txt --out_csv %OUT%\summary_win_ours_cpp_cpu_batch.csv
if errorlevel 1 goto :fail

echo [7/12] Prep PATH for GPU backend
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;E:\CondaEnvData\envs\GGMPC\Lib\site-packages\torch\lib;%ROOT%cpp\build_gpu\Release;%PATH%

echo [8/12] Windows non-batch (ours cpp gpu)
set OURS_CPP_BACKEND=gpu
%PY% experiment\run_parameter_metrics.py --methods ours_corridor_cpp --no_batch --raw_out %OUT%\raw_win_ours_cpp_gpu_single.npz --out_txt %OUT%\summary_win_ours_cpp_gpu_single.txt --out_csv %OUT%\summary_win_ours_cpp_gpu_single.csv
if errorlevel 1 goto :fail

echo [9/12] Windows batch (ours cpp gpu, N=20)
set OURS_CPP_BACKEND=gpu
%PY% experiment\run_parameter_metrics.py --methods ours_corridor_cpp --batch_size 20 --raw_out %OUT%\raw_win_ours_cpp_gpu_batch.npz --out_txt %OUT%\summary_win_ours_cpp_gpu_batch.txt --out_csv %OUT%\summary_win_ours_cpp_gpu_batch.csv
if errorlevel 1 goto :fail

echo [10/14] Merge: baselines single + WSL single
%PY% experiment\merge_parameter_metrics.py --raw_a %OUT%\raw_win_baseline_single.npz --raw_b %OUT%\raw_wsl_single.npz --out_txt %OUT%\merged_baseline_single.txt --out_csv %OUT%\merged_baseline_single.csv
if errorlevel 1 goto :fail

echo [11/14] Merge: ours python single + WSL single
%PY% experiment\merge_parameter_metrics.py --raw_a %OUT%\raw_win_ours_py_single.npz --raw_b %OUT%\raw_wsl_single.npz --out_txt %OUT%\merged_ours_py_single.txt --out_csv %OUT%\merged_ours_py_single.csv
if errorlevel 1 goto :fail

echo [12/14] Merge: ours python batch + WSL single
%PY% experiment\merge_parameter_metrics.py --raw_a %OUT%\raw_win_ours_py_batch.npz --raw_b %OUT%\raw_wsl_single.npz --out_txt %OUT%\merged_ours_py_batch.txt --out_csv %OUT%\merged_ours_py_batch.csv
if errorlevel 1 goto :fail

echo [13/14] Merge: ours cpp variants + WSL single
%PY% experiment\merge_parameter_metrics.py --raw_a %OUT%\raw_win_ours_cpp_cpu_single.npz --raw_b %OUT%\raw_wsl_single.npz --out_txt %OUT%\merged_ours_cpp_cpu_single.txt --out_csv %OUT%\merged_ours_cpp_cpu_single.csv
if errorlevel 1 goto :fail
%PY% experiment\merge_parameter_metrics.py --raw_a %OUT%\raw_win_ours_cpp_cpu_batch.npz --raw_b %OUT%\raw_wsl_single.npz --out_txt %OUT%\merged_ours_cpp_cpu_batch.txt --out_csv %OUT%\merged_ours_cpp_cpu_batch.csv
if errorlevel 1 goto :fail
%PY% experiment\merge_parameter_metrics.py --raw_a %OUT%\raw_win_ours_cpp_gpu_single.npz --raw_b %OUT%\raw_wsl_single.npz --out_txt %OUT%\merged_ours_cpp_gpu_single.txt --out_csv %OUT%\merged_ours_cpp_gpu_single.csv
if errorlevel 1 goto :fail
%PY% experiment\merge_parameter_metrics.py --raw_a %OUT%\raw_win_ours_cpp_gpu_batch.npz --raw_b %OUT%\raw_wsl_single.npz --out_txt %OUT%\merged_ours_cpp_gpu_batch.txt --out_csv %OUT%\merged_ours_cpp_gpu_batch.csv
if errorlevel 1 goto :fail

echo [14/14] Build total summary from all completed groups
%PY% experiment\collect_benchmark_total.py --input wsl_single=%OUT%\summary_wsl_single.csv --input baseline_single=%OUT%\summary_win_baseline_single.csv --input ours_py_single=%OUT%\summary_win_ours_py_single.csv --input ours_py_batch=%OUT%\summary_win_ours_py_batch.csv --input ours_cpp_cpu_single=%OUT%\summary_win_ours_cpp_cpu_single.csv --input ours_cpp_cpu_batch=%OUT%\summary_win_ours_cpp_cpu_batch.csv --input ours_cpp_gpu_single=%OUT%\summary_win_ours_cpp_gpu_single.csv --input ours_cpp_gpu_batch=%OUT%\summary_win_ours_cpp_gpu_batch.csv --norm_override %OUT%\merged_baseline_single.csv --norm_override %OUT%\merged_ours_py_single.csv --norm_override %OUT%\merged_ours_py_batch.csv --norm_override %OUT%\merged_ours_cpp_cpu_single.csv --norm_override %OUT%\merged_ours_cpp_cpu_batch.csv --norm_override %OUT%\merged_ours_cpp_gpu_single.csv --norm_override %OUT%\merged_ours_cpp_gpu_batch.csv --out_txt %OUT%\summary_total.txt --out_csv %OUT%\summary_total.csv
if errorlevel 1 goto :fail

echo.
echo All done. Key outputs in: %OUT%
echo - summary_total.txt
echo - summary_total.csv
echo - merged_baseline_single.txt
echo - merged_ours_py_single.txt
echo - merged_ours_py_batch.txt
echo - merged_ours_cpp_cpu_single.txt
echo - merged_ours_cpp_cpu_batch.txt
echo - merged_ours_cpp_gpu_single.txt
echo - merged_ours_cpp_gpu_batch.txt
goto :eof

:fail
echo.
echo FAILED with errorlevel %errorlevel%.
echo Check last command output above.
exit /b %errorlevel%
