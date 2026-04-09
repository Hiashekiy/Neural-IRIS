import numpy as np


def generate_safe_region(P, c, obs_points, patch_size=128, safety_margin=0.5):
    if obs_points is None or len(obs_points) == 0:
        A_arr = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        b_arr = np.array([patch_size - safety_margin, safety_margin, patch_size - safety_margin, safety_margin], dtype=float)
        return A_arr, b_arr

    max_faces = len(obs_points)
    A_arr = np.zeros((4 + max_faces, 2))
    b_arr = np.zeros(4 + max_faces)

    A_arr[0:4] = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    b_arr[0:4] = [patch_size - safety_margin, safety_margin, patch_size - safety_margin, safety_margin]
    face_count = 4

    try:
        q_mat = P.T @ P
        diffs = obs_points - c
        dists = np.sum((diffs @ q_mat) * diffs, axis=1)

        active_obs = obs_points[np.argsort(dists)]
        while len(active_obs) > 0:
            obs = active_obs[0]
            normal = q_mat @ (obs - c)
            norm_n = np.linalg.norm(normal)

            if norm_n > 1e-6:
                normal = normal / norm_n
                A_arr[face_count] = normal
                b_val = np.dot(normal, obs) - safety_margin
                b_arr[face_count] = b_val
                face_count += 1

                distances = np.dot(active_obs[1:], normal)
                mask = distances <= b_val + 1e-3
                active_obs = active_obs[1:][mask]
            else:
                active_obs = active_obs[1:]

    except np.linalg.LinAlgError:
        pass

    return A_arr[:face_count], b_arr[:face_count]

