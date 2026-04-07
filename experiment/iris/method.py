import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

from experiment.common.geometry_utils import fallback_patch_box, obstacle_boundary


def _import_drake_iris():
    try:
        from pydrake.geometry.optimization import HPolyhedron, Iris, IrisOptions

        return HPolyhedron, Iris, IrisOptions, None
    except Exception as e:
        return None, None, None, e


def _boxes_from_obstacles(HPolyhedron, obs_mask, max_boxes=1000):
    obs_pts = obstacle_boundary(obs_mask)
    if len(obs_pts) == 0:
        return []

    # 控制 IRIS 障碍集合规模，避免每个像素都生成 box 导致过慢。
    if len(obs_pts) > max_boxes:
        idx = np.linspace(0, len(obs_pts) - 1, max_boxes, dtype=int)
        obs_pts = obs_pts[idx]

    boxes = []
    for p in obs_pts:
        x, y = float(p[0]), float(p[1])
        lb = np.array([x - 0.49, y - 0.49], dtype=float)
        ub = np.array([x + 0.49, y + 0.49], dtype=float)
        boxes.append(HPolyhedron.MakeBox(lb, ub))
    return boxes


def _halfspace_to_vertices(A, b, interior_point):
    if A is None or b is None or len(A) < 3:
        return None

    halfspaces = np.hstack((A, -b[:, None]))
    hs = HalfspaceIntersection(halfspaces, interior_point)
    pts = hs.intersections
    if len(pts) < 3:
        return None

    hull = ConvexHull(pts)
    return pts[hull.vertices]


def infer_polygon(obs_mask, center=(64.0, 64.0), patch_size=128):
    HPolyhedron, Iris, IrisOptions, err = _import_drake_iris()
    if HPolyhedron is None:
        raise NotImplementedError(
            "IRIS 需要官方 Drake Python 绑定 (pydrake.geometry.optimization)。"
            " 当前环境未检测到可用 Drake。"
            f" 原始导入错误: {err}"
        )

    center = np.asarray(center, dtype=float)
    center = np.clip(center, 1.0, patch_size - 2.0)

    domain = HPolyhedron.MakeBox(np.array([0.0, 0.0]), np.array([patch_size - 1.0, patch_size - 1.0]))
    obstacles = _boxes_from_obstacles(HPolyhedron, obs_mask, max_boxes=900)

    options = IrisOptions()
    # 这些属性在部分 Drake 版本不存在，故采用 hasattr 兼容写法。
    if hasattr(options, "iteration_limit"):
        options.iteration_limit = 12
    if hasattr(options, "require_sample_point_is_contained"):
        options.require_sample_point_is_contained = True

    sample = np.asarray(center, dtype=float)
    region = Iris(obstacles=obstacles, sample=sample, domain=domain, options=options)
    A = np.asarray(region.A(), dtype=float)
    b = np.asarray(region.b(), dtype=float).reshape(-1)

    # 如果 sample 因数值问题不在 region 内，回退到求一个可行内点。
    if np.any(A @ sample > b + 1e-6):
        sample = np.mean(np.column_stack((np.zeros(2), np.ones(2) * (patch_size - 1))), axis=1)

    vertices = _halfspace_to_vertices(A, b, sample)
    if vertices is None:
        return fallback_patch_box(patch_size)

    vertices[:, 0] = np.clip(vertices[:, 0], 0.0, patch_size - 1.0)
    vertices[:, 1] = np.clip(vertices[:, 1], 0.0, patch_size - 1.0)
    return vertices
