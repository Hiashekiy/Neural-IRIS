import numpy as np
import os

from experiment.common.geometry_utils import (
    fallback_patch_box,
    convex_polygon_from_radial_bounds,
)

_TORCH_READY = False
try:
    import torch
    from experiment.direct_polygon_regression.radial_model import RadialPolygonNet

    _TORCH_READY = True
except Exception:
    _TORCH_READY = False


_MODEL = None
_MODEL_K = 32


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _model_path(k_dirs: int = 32):
    return os.path.join(_project_root(), "models", f"direct_polygon_regression_k{k_dirs}.pth")


def _load_model_if_available(k_dirs: int = 32):
    global _MODEL, _MODEL_K
    if not _TORCH_READY:
        return None

    if _MODEL is not None and _MODEL_K == k_dirs:
        return _MODEL

    path = _model_path(k_dirs)
    if not os.path.isfile(path):
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadialPolygonNet(k_dirs=k_dirs).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    _MODEL = model
    _MODEL_K = k_dirs
    return _MODEL


def _smooth_circular(arr: np.ndarray, k: int = 2):
    n = len(arr)
    out = np.zeros_like(arr)
    for i in range(n):
        idx = [(i + j) % n for j in range(-k, k + 1)]
        out[i] = np.mean(arr[idx])
    return out


def _raycast_distances_vectorized(obs_mask: np.ndarray, center: np.ndarray, angles: np.ndarray, max_dist: float = 90.0, step: float = 0.5) -> np.ndarray:
    """Vectorized radial raycast on a binary obstacle map for all directions."""
    h, w = obs_mask.shape
    dists = np.arange(0.0, max_dist + step, step, dtype=float)

    cos_t = np.cos(angles)[:, None]
    sin_t = np.sin(angles)[:, None]

    xs = center[0] + cos_t * dists[None, :]
    ys = center[1] + sin_t * dists[None, :]

    ix = np.rint(xs).astype(np.int32)
    iy = np.rint(ys).astype(np.int32)

    in_bounds = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix_clip = np.clip(ix, 0, w - 1)
    iy_clip = np.clip(iy, 0, h - 1)

    free = in_bounds & (~obs_mask[iy_clip, ix_clip])
    hit = ~free

    any_hit = np.any(hit, axis=1)
    first_hit = np.argmax(hit, axis=1)
    safe_idx = np.maximum(first_hit - 1, 0)
    safe_dist = dists[safe_idx]

    return np.where(any_hit, safe_dist, max_dist)


def infer_polygon(obs_mask, center=(64.0, 64.0), patch_size=128):
    center = np.asarray(center, dtype=float)
    obs_mask = np.asarray(obs_mask, dtype=bool)

    k_dirs = 32
    angles = np.linspace(0.0, 2.0 * np.pi, k_dirs, endpoint=False)
    radii_raw = _raycast_distances_vectorized(obs_mask, center, angles, max_dist=90.0, step=0.5)
    radii_raw = np.maximum(radii_raw - 0.8, 1.0)

    model = _load_model_if_available(k_dirs=k_dirs)
    if model is not None:
        device = next(model.parameters()).device
        x = torch.from_numpy(obs_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            radii_pred = model(x).squeeze(0).detach().cpu().numpy()
    else:
        # 无模型时退回几何替代版本。
        radii_pred = _smooth_circular(radii_raw, k=2)

    # 平滑可能把半径抬高并穿过障碍，强制不超过观测到的安全上界。
    radii_safe = np.minimum(radii_pred, radii_raw)

    vertices = convex_polygon_from_radial_bounds(center, angles, radii_safe)
    if vertices is None or len(vertices) < 3:
        return fallback_patch_box(patch_size)

    vertices[:, 0] = np.clip(vertices[:, 0], 0.0, patch_size - 1.0)
    vertices[:, 1] = np.clip(vertices[:, 1], 0.0, patch_size - 1.0)

    return vertices
