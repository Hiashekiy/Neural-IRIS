import numpy as np

from experiment.common.geometry_utils import (
    raycast_distance,
    fallback_patch_box,
    convex_polygon_from_radial_bounds,
)


def _pca_theta(points: np.ndarray) -> float:
    if points is None or len(points) < 3:
        return 0.0
    mean = points.mean(axis=0)
    cov = np.cov((points - mean).T)
    vals, vecs = np.linalg.eigh(cov)
    idx = int(np.argmax(vals))
    axis = vecs[:, idx]
    return float(np.arctan2(axis[1], axis[0]))


def infer_polygon(obs_mask, center=(64.0, 64.0), patch_size=128):
    center = np.asarray(center, dtype=float)
    free_yx = np.argwhere(~obs_mask)
    if len(free_yx) < 16:
        return fallback_patch_box(patch_size)

    free_xy = free_yx[:, [1, 0]].astype(float)
    dist2 = np.sum((free_xy - center[None, :]) ** 2, axis=1)
    local_free = free_xy[dist2 < (40.0 ** 2)]
    if len(local_free) < 32:
        local_free = free_xy

    theta = _pca_theta(local_free)
    d1 = raycast_distance(obs_mask, center, theta, max_dist=90.0)
    d2 = raycast_distance(obs_mask, center, theta + np.pi, max_dist=90.0)
    d3 = raycast_distance(obs_mask, center, theta + np.pi / 2.0, max_dist=90.0)
    d4 = raycast_distance(obs_mask, center, theta - np.pi / 2.0, max_dist=90.0)

    a = max(1.0, min(d1, d2) - 0.5)
    b = max(1.0, min(d3, d4) - 0.5)

    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=float)

    num_vertices = 40
    angles = np.linspace(0.0, 2.0 * np.pi, num_vertices, endpoint=False)

    # 椭圆先验半径 + 障碍射线安全半径的逐方向最小值，避免椭圆穿障。
    ray_bounds = np.array([raycast_distance(obs_mask, center, ang, max_dist=90.0) for ang in angles], dtype=float)
    ray_bounds = np.maximum(ray_bounds - 0.8, 1.0)

    cth = np.cos(theta)
    sth = np.sin(theta)
    radii = np.zeros_like(angles)
    for i, ang in enumerate(angles):
        u = np.array([np.cos(ang), np.sin(ang)], dtype=float)
        # world->ellipse-local
        u_local = np.array([cth * u[0] + sth * u[1], -sth * u[0] + cth * u[1]], dtype=float)
        denom = (u_local[0] / max(a, 1e-6)) ** 2 + (u_local[1] / max(b, 1e-6)) ** 2
        r_ell = 1.0 / np.sqrt(max(denom, 1e-12))
        radii[i] = min(r_ell, ray_bounds[i])

    vertices = convex_polygon_from_radial_bounds(center, angles, radii)
    if vertices is None or len(vertices) < 3:
        return fallback_patch_box(patch_size)

    vertices[:, 0] = np.clip(vertices[:, 0], 0.0, patch_size - 1.0)
    vertices[:, 1] = np.clip(vertices[:, 1], 0.0, patch_size - 1.0)
    return vertices
