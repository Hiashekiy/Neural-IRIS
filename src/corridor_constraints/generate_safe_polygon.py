import numpy as np

def generate_safe_polygon(P, c, obs_points, patch_size=128, safety_margin=0.5):
    """
    Generate safe polygon with safety margin to account for rasterization errors.
    
    Args:
        P: 2x2 PSD matrix from network output (ellipse params)
        c: center point
        obs_points: obstacle boundary points
        patch_size: image size (128)
        safety_margin: shrink all constraints inward by this amount (pixel units)
                      Default 0.5 pixels eliminates rasterization edge effects
    """
    if obs_points is None or len(obs_points) == 0:
        A_arr = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
        b_arr = np.array([patch_size - safety_margin, safety_margin, patch_size - safety_margin, safety_margin], dtype=float)
        return A_arr, b_arr

    MAX_FACES = len(obs_points)
    A_arr = np.zeros((4 + MAX_FACES, 2))
    b_arr = np.zeros(4 + MAX_FACES)

    A_arr[0:4] = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    b_arr[0:4] = [patch_size - safety_margin, safety_margin, patch_size - safety_margin, safety_margin]
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
                # Apply safety margin: shrink inward by safety_margin
                b_val = np.dot(n_norm, obs) - safety_margin
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
    
    return A_final, b_final