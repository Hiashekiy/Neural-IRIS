import numpy as np

from experiment.common.geometry_utils import obstacle_boundary, fallback_patch_box


_THETAS = np.linspace(0.0, np.pi, 12, endpoint=False)
_COS = np.cos(_THETAS)
_SIN = np.sin(_THETAS)


def _best_local_rectangle(rel_points: np.ndarray, max_half: float):
    if len(rel_points) == 0:
        return max_half * 0.8, max_half * 0.8

    x_abs = np.clip(np.abs(rel_points[:, 0]), 0.0, max_half)
    y_abs = np.clip(np.abs(rel_points[:, 1]), 0.0, max_half)

    order = np.argsort(x_abs)
    x_sorted = x_abs[order]
    y_sorted = y_abs[order]
    prefix_min_y = np.minimum.accumulate(y_sorted)

    # Uniform + sparse obstacle-derived candidates,减少候选数量同时保持覆盖。
    uniform = np.linspace(1.0, max_half, 48)
    stride = max(1, len(x_sorted) // 64)
    sparse = x_sorted[::stride]
    candidates = np.unique(np.clip(np.concatenate([uniform, sparse, np.array([max_half])]), 1.0, max_half))

    idx = np.searchsorted(x_sorted, candidates, side="left")
    hy = np.where(idx > 0, prefix_min_y[np.maximum(idx - 1, 0)], max_half)
    hy = np.clip(hy - 1e-3, 0.0, max_half)

    valid = hy > 0.5
    if not np.any(valid):
        return 2.0, 2.0

    area = 4.0 * candidates * hy
    area[~valid] = -1.0
    k = int(np.argmax(area))
    return float(candidates[k] * 0.95), float(hy[k] * 0.95)


def infer_polygon(obs_mask, center=(64.0, 64.0), patch_size=128):
    center = np.asarray(center, dtype=float)
    obs_pts = obstacle_boundary(obs_mask)
    if len(obs_pts) == 0:
        return fallback_patch_box(patch_size)

    max_half = patch_size / 2.0
    best = None
    best_area = -1.0

    for c, s in zip(_COS, _SIN):
        r_world_to_local = np.array([[c, s], [-s, c]], dtype=float)
        r_local_to_world = np.array([[c, -s], [s, c]], dtype=float)

        rel = (obs_pts - center[None, :]) @ r_world_to_local.T
        hx, hy = _best_local_rectangle(rel, max_half=max_half)
        if hx <= 0.5 or hy <= 0.5:
            continue

        local_vertices = np.array([
            [-hx, -hy],
            [hx, -hy],
            [hx, hy],
            [-hx, hy],
        ])
        world_vertices = local_vertices @ r_local_to_world.T + center[None, :]

        # 确保矩形完全在 patch 内
        if np.any(world_vertices < 0.0) or np.any(world_vertices > (patch_size - 1)):
            scales = []
            for v in world_vertices:
                dx = v[0] - center[0]
                dy = v[1] - center[1]
                s_max = 1.0
                if abs(dx) > 1e-9:
                    if dx > 0:
                        s_max = min(s_max, ((patch_size - 1) - center[0]) / dx)
                    else:
                        s_max = min(s_max, (0.0 - center[0]) / dx)
                if abs(dy) > 1e-9:
                    if dy > 0:
                        s_max = min(s_max, ((patch_size - 1) - center[1]) / dy)
                    else:
                        s_max = min(s_max, (0.0 - center[1]) / dy)
                scales.append(s_max)
            scale = max(0.0, min(scales))
            world_vertices = (world_vertices - center[None, :]) * (0.98 * scale) + center[None, :]

        area = hx * hy
        if area > best_area:
            best_area = area
            best = world_vertices

    if best is None:
        return fallback_patch_box(patch_size)

    return best
