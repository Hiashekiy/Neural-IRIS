import numpy as np

from experiment.common.geometry_utils import obstacle_boundary, fallback_patch_box


def infer_polygon(obs_mask, center=(64.0, 64.0), patch_size=128):
    center = np.asarray(center, dtype=float)
    obs_pts = obstacle_boundary(obs_mask)

    if len(obs_pts) == 0:
        return fallback_patch_box(patch_size)

    d_obs = np.linalg.norm(obs_pts - center[None, :], axis=1).min()
    d_bound = min(center[0], center[1], (patch_size - 1) - center[0], (patch_size - 1) - center[1])
    r = max(1.0, min(d_obs, d_bound) - 0.5)

    num_vertices = 24
    thetas = np.linspace(0.0, 2.0 * np.pi, num_vertices, endpoint=False)
    circle = np.column_stack((np.cos(thetas), np.sin(thetas)))
    return center[None, :] + r * circle
