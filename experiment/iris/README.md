# IRIS Baseline (Drake Python)

This baseline uses `pydrake.geometry.optimization.Iris` to generate a convex safe region from a local occupancy map.

## Interface

- Input: `obs_mask` (128x128 bool), `center` (default `[64, 64]`)
- Output: convex polygon vertices (Nx2)

## Notes

- The implementation is in `method.py`.
- It requires official Drake bindings (`pydrake.geometry.optimization`).
- If Drake is unavailable, the method raises `NotImplementedError` and will be skipped by `run_all_baselines.py`.

## Environment Hint

- In this workspace's current Windows environment, official Drake wheel may be unavailable from the configured package sources.
- Recommended path: use a Linux/WSL Python environment with Drake installed, then rerun `experiment/run_all_baselines.py`.
