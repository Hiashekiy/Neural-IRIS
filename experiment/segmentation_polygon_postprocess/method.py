import numpy as np
from scipy import ndimage

from experiment.common.geometry_utils import fallback_patch_box, convex_polygon_from_radial_bounds


def _nearest_free_seed(free_mask: np.ndarray, center):
    cy, cx = int(round(center[1])), int(round(center[0]))
    if 0 <= cy < free_mask.shape[0] and 0 <= cx < free_mask.shape[1] and free_mask[cy, cx]:
        return cy, cx

    ys, xs = np.where(free_mask)
    if len(xs) == 0:
        return None
    d2 = (xs - center[0]) ** 2 + (ys - center[1]) ** 2
    k = int(np.argmin(d2))
    return int(ys[k]), int(xs[k])


def _raycast_in_region(region_mask: np.ndarray, origin, angle: float, max_dist: float = 200.0, step: float = 0.5) -> float:
    h, w = region_mask.shape
    ox, oy = float(origin[0]), float(origin[1])
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)

    d = 0.0
    while d <= max_dist:
        x = ox + d * cos_t
        y = oy + d * sin_t
        ix = int(round(x))
        iy = int(round(y))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return max(0.0, d - step)
        if not region_mask[iy, ix]:
            return max(0.0, d - step)
        d += step
    return max_dist


def infer_polygon(obs_mask, center=(64.0, 64.0), patch_size=128):
    free = ~obs_mask
    seed = _nearest_free_seed(free, center)
    if seed is None:
        return fallback_patch_box(patch_size)

    dist = ndimage.distance_transform_edt(free)
    seed_dist = float(dist[seed[0], seed[1]])
    thr = max(1.0, 0.45 * seed_dist)

    region = dist >= thr
    labels, num = ndimage.label(region)
    if num <= 0:
        return fallback_patch_box(patch_size)

    lbl = labels[seed[0], seed[1]]
    if lbl == 0:
        # 如果阈值后的区域不含 seed，则退回自由空间连通域
        labels2, num2 = ndimage.label(free)
        if num2 <= 0:
            return fallback_patch_box(patch_size)
        lbl = labels2[seed[0], seed[1]]
        if lbl == 0:
            return fallback_patch_box(patch_size)
        ys, xs = np.where(labels2 == lbl)
    else:
        ys, xs = np.where(labels == lbl)

    if len(xs) < 3:
        return fallback_patch_box(patch_size)

    region_mask = np.zeros_like(free, dtype=bool)
    region_mask[ys, xs] = True

    num_dirs = 40
    angles = np.linspace(0.0, 2.0 * np.pi, num_dirs, endpoint=False)
    radii = np.array([
        _raycast_in_region(region_mask, center, ang, max_dist=90.0, step=0.5)
        for ang in angles
    ], dtype=float)
    radii = np.maximum(radii - 0.8, 1.0)

    vertices = convex_polygon_from_radial_bounds(np.asarray(center, dtype=float), angles, radii)
    if vertices is None or len(vertices) < 3:
        return fallback_patch_box(patch_size)

    return vertices
