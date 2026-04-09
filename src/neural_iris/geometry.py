import numpy as np
import torch


def parse_neural_iris_output(pred_data, patch_size=128):
    if isinstance(pred_data, torch.Tensor):
        pred = pred_data.detach().cpu().numpy()
    else:
        pred = np.asarray(pred_data)

    dx, dy, a, b, sin_t, cos_t = pred

    center_x = (patch_size / 2.0) + dx
    center_y = (patch_size / 2.0) + dy
    c = np.array([center_x, center_y])

    r_mat = np.array([
        [cos_t, -sin_t],
        [sin_t, cos_t],
    ])

    lambda_mat = np.array([
        [1.0 / (a + 1e-4), 0],
        [0, 1.0 / (b + 1e-4)],
    ])

    p_mat = r_mat @ lambda_mat @ r_mat.T
    return p_mat, c


def extract_obstacle_boundary_points(obs_mask):
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


def get_ellipse_points(p_mat, c, scale=1.0, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])
    try:
        p_inv = np.linalg.inv(p_mat)
        ellipse_points = ((scale * p_inv) @ unit_circle).T + c
        return ellipse_points
    except np.linalg.LinAlgError:
        return np.zeros((num_points, 2))


def render_soft_ellipse_mask(preds, grid_x, grid_y, size=128, temperature=10.0):
    batch_size = preds.shape[0]

    center_x = (size / 2.0) + preds[:, 0].view(batch_size, 1, 1)
    center_y = (size / 2.0) + preds[:, 1].view(batch_size, 1, 1)

    dx = grid_x[:batch_size] - center_x
    dy = grid_y[:batch_size] - center_y

    a_axis = preds[:, 2].view(batch_size, 1, 1) + 1e-4
    b_axis = preds[:, 3].view(batch_size, 1, 1) + 1e-4
    sin_t = preds[:, 4].view(batch_size, 1, 1)
    cos_t = preds[:, 5].view(batch_size, 1, 1)

    rx = dx * cos_t + dy * sin_t
    ry = -dx * sin_t + dy * cos_t

    eq = (rx / a_axis) ** 2 + (ry / b_axis) ** 2
    return torch.sigmoid(temperature * (1.0 - eq))

