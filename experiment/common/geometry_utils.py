import numpy as np


def ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    if vertices is None or len(vertices) < 3:
        return vertices
    area2 = np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1) - np.roll(vertices[:, 0], -1) * vertices[:, 1])
    if area2 < 0:
        return vertices[::-1].copy()
    return vertices


def polygon_to_halfspaces(vertices: np.ndarray):
    vertices = ensure_ccw(np.asarray(vertices, dtype=float))
    n = len(vertices)
    if n < 3:
        return None, None

    A = np.zeros((n, 2), dtype=float)
    b = np.zeros(n, dtype=float)
    for i in range(n):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % n]
        edge = p1 - p0
        normal = np.array([edge[1], -edge[0]], dtype=float)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            return None, None
        normal /= norm
        A[i] = normal
        b[i] = float(np.dot(normal, p0))

    center = np.mean(vertices, axis=0)
    if np.any(A @ center > b + 1e-6):
        A = -A
        b = -b
    return A, b


def point_in_halfspaces(point: np.ndarray, A: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> bool:
    if A is None or b is None or len(A) == 0:
        return False
    return bool(np.all((A @ point) <= (b + eps)))


def clip_vertices_to_patch(vertices: np.ndarray, patch_size: int = 128) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    vertices[:, 0] = np.clip(vertices[:, 0], 0.0, patch_size - 1.0)
    vertices[:, 1] = np.clip(vertices[:, 1], 0.0, patch_size - 1.0)
    return vertices


def obstacle_boundary(obs_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(obs_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("obs_mask must be a 2D array")

    if not np.any(mask):
        return np.empty((0, 2), dtype=float)

    up = np.zeros_like(mask)
    down = np.zeros_like(mask)
    left = np.zeros_like(mask)
    right = np.zeros_like(mask)

    up[1:, :] = mask[:-1, :]
    down[:-1, :] = mask[1:, :]
    left[:, 1:] = mask[:, :-1]
    right[:, :-1] = mask[:, 1:]

    interior = up & down & left & right
    boundary = mask & (~interior)

    ys, xs = np.where(boundary)
    return np.column_stack((xs, ys)).astype(float)


def raycast_distance(obs_mask: np.ndarray, origin, angle: float, max_dist: float = 200.0, step: float = 0.5) -> float:
    h, w = obs_mask.shape
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
        if obs_mask[iy, ix]:
            return max(0.0, d - step)
        d += step
    return max_dist


def fallback_patch_box(patch_size: int = 128) -> np.ndarray:
    s = float(patch_size - 1)
    return np.array([[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]], dtype=float)


def halfspace_intersection_vertices(A: np.ndarray, b: np.ndarray, interior_point: np.ndarray):
    if A is None or b is None or len(A) < 3:
        return None
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[1] != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
        return None

    # Fast 2D half-space intersection via pairwise line intersections.
    # Each boundary line is n_i^T x = b_i, and feasible points satisfy A x <= b.
    eps_det = 1e-10
    eps_feas = 1e-6
    m = A.shape[0]
    candidates = []

    for i in range(m):
        a1 = A[i]
        c1 = b[i]
        for j in range(i + 1, m):
            a2 = A[j]
            c2 = b[j]
            det = a1[0] * a2[1] - a1[1] * a2[0]
            if abs(det) <= eps_det:
                continue

            x = (c1 * a2[1] - a1[1] * c2) / det
            y = (a1[0] * c2 - c1 * a2[0]) / det
            p = np.array([x, y], dtype=float)

            if np.all((A @ p) <= (b + eps_feas)):
                candidates.append(p)

    if not candidates:
        return None

    pts = np.asarray(candidates, dtype=float)

    # Deduplicate numerically close points.
    uniq = []
    tol2 = 1e-12
    for p in pts:
        if not uniq:
            uniq.append(p)
            continue
        d2 = np.sum((np.asarray(uniq) - p) ** 2, axis=1)
        if np.min(d2) > tol2:
            uniq.append(p)

    if len(uniq) < 3:
        return None

    pts = np.asarray(uniq, dtype=float)
    center = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(ang)
    poly = ensure_ccw(pts[order])

    # Keep the polygon that actually contains the provided interior point.
    # If orientation issues appear, flipping inequalities would otherwise hide errors.
    if interior_point is not None:
        ip = np.asarray(interior_point, dtype=float)
        Ap, bp = polygon_to_halfspaces(poly)
        if Ap is None or not point_in_halfspaces(ip, Ap, bp):
            return None

    return poly


def convex_polygon_from_radial_bounds(center: np.ndarray, angles: np.ndarray, radii: np.ndarray):
    center = np.asarray(center, dtype=float)
    angles = np.asarray(angles, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if len(angles) < 3 or len(radii) != len(angles):
        return None

    u = np.column_stack((np.cos(angles), np.sin(angles)))
    A = u
    b = (u @ center) + radii
    if np.any(A @ center > b - 1e-8):
        margin = float(np.max((A @ center) - b))
        b = b + margin + 1e-3
    return halfspace_intersection_vertices(A, b, center)
