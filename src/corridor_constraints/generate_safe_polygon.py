import numpy as np
# 如果不再需要可视化顶点，以下两个引入实际上可以废弃
from scipy.spatial import HalfspaceIntersection, ConvexHull

def generate_safe_polygon(P, c, obs_points, patch_size=128):
    if obs_points is None or len(obs_points) == 0:
        A_arr = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        b_arr = np.array([patch_size, 0, patch_size, 0], dtype=float)
        # 直接返回 A 和 b，跳过 HalfspaceIntersection
        return A_arr, b_arr

    MAX_FACES = len(obs_points)
    A_arr = np.zeros((4 + MAX_FACES, 2))
    b_arr = np.zeros(4 + MAX_FACES)

    A_arr[0:4] = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    b_arr[0:4] = [patch_size, 0, patch_size, 0]
    face_count = 4

    try:
        Q = P.T @ P
        diffs = obs_points - c
        dists = np.sum((diffs @ Q) * diffs, axis=1)

        sorted_indices = np.argsort(dists)
        sorted_obs = obs_points[sorted_indices]

        active_obs = sorted_obs
        while len(active_obs) > 0:
            obs = active_obs[0]
            n = Q @ (obs - c)
            norm_n = np.linalg.norm(n)

            if norm_n > 1e-6:
                n_norm = n / norm_n
                A_arr[face_count] = n_norm
                b_val = np.dot(n_norm, obs)
                b_arr[face_count] = b_val
                face_count += 1
                
                distances = np.dot(active_obs[1:], n_norm)
                mask = distances <= b_val + 1e-3
                active_obs = active_obs[1:][mask]
            else:
                active_obs = active_obs[1:]

    except np.linalg.LinAlgError:
        pass

    A_final = A_arr[:face_count]
    b_final = b_arr[:face_count]
    
    # 直接返回超平面约束，彻底砍掉 scipy 的计算开销
    return A_final, b_final