import os

import numpy as np
import torch
from scipy.spatial import ConvexHull, HalfspaceIntersection

from .geometry import extract_obstacle_boundary_points, parse_neural_iris_output
from .model import NeuralIRISNet
from .safe_region import generate_safe_region

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_MODEL_PATH = os.path.join(_ROOT, "models", "neural_iris_net_best.pth")

_MODEL = None
_MODEL_DEVICE = None
_MODEL_PATH = None


def _load_default_model(model_path=None, device=None):
    global _MODEL, _MODEL_DEVICE, _MODEL_PATH

    resolved_model_path = os.path.abspath(model_path or _DEFAULT_MODEL_PATH)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if (
        _MODEL is not None
        and _MODEL_PATH == resolved_model_path
        and _MODEL_DEVICE == str(resolved_device)
    ):
        return _MODEL, resolved_device

    if not os.path.isfile(resolved_model_path):
        raise FileNotFoundError(f"Neural-IRIS model weights not found: {resolved_model_path}")

    model = NeuralIRISNet().to(resolved_device)
    try:
        state_dict = torch.load(resolved_model_path, map_location=resolved_device, weights_only=True)
    except TypeError:
        state_dict = torch.load(resolved_model_path, map_location=resolved_device)
    model.load_state_dict(state_dict)
    model.eval()

    _MODEL = model
    _MODEL_DEVICE = str(resolved_device)
    _MODEL_PATH = resolved_model_path
    return model, resolved_device


def _constraints_to_vertices(a_mat, b_vec, interior_point):
    if a_mat is None or b_vec is None or len(a_mat) < 3:
        return None

    try:
        halfspaces = np.hstack([
            np.asarray(a_mat, dtype=float),
            -np.asarray(b_vec, dtype=float).reshape(-1, 1),
        ])
        hs = HalfspaceIntersection(halfspaces, np.asarray(interior_point, dtype=float))
        if len(hs.intersections) < 3:
            return None
        hull = ConvexHull(hs.intersections)
        return hs.intersections[hull.vertices]
    except Exception:
        return None


def infer_safe_region(obs_mask, center=(64.0, 64.0), patch_size=128, model=None, device=None, model_path=None):
    masks = np.asarray(obs_mask, dtype=np.uint8)
    if masks.ndim != 2:
        return None, None, None

    safe_regions, p_list, c_list = infer_safe_region_batch(
        masks[np.newaxis, ...],
        center=center,
        patch_size=patch_size,
        model=model,
        device=device,
        model_path=model_path,
    )
    return safe_regions[0], p_list[0], c_list[0]


def infer_safe_region_halfspaces(obs_mask, center=(64.0, 64.0), patch_size=128, model=None, device=None, model_path=None):
    masks = np.asarray(obs_mask, dtype=np.uint8)
    if masks.ndim != 2:
        return None, None, None, None

    a_list, b_list, p_list, c_list = infer_safe_region_batch_halfspaces(
        masks[np.newaxis, ...],
        center=center,
        patch_size=patch_size,
        model=model,
        device=device,
        model_path=model_path,
    )
    return a_list[0], b_list[0], p_list[0], c_list[0]


def infer_safe_region_batch_halfspaces(obs_masks, center=(64.0, 64.0), patch_size=128, model=None, device=None, model_path=None):
    del center  # Kept for API symmetry with the C++ bridge.

    masks = np.asarray(obs_masks, dtype=np.uint8)
    if masks.ndim != 3:
        return (
            [None for _ in range(len(obs_masks))],
            [None for _ in range(len(obs_masks))],
            [None for _ in range(len(obs_masks))],
            [None for _ in range(len(obs_masks))],
        )

    batch_size = int(masks.shape[0])
    if batch_size <= 0:
        return [], [], [], []

    if masks.shape[1] != patch_size or masks.shape[2] != patch_size:
        return (
            [None for _ in range(batch_size)],
            [None for _ in range(batch_size)],
            [None for _ in range(batch_size)],
            [None for _ in range(batch_size)],
        )

    if model is None:
        model, runtime_device = _load_default_model(model_path=model_path, device=device)
    else:
        runtime_device = next(model.parameters()).device
        model.eval()

    batch_tensor = torch.from_numpy(masks.astype(np.float32)).unsqueeze(1).to(runtime_device)
    with torch.no_grad():
        pred_batch = model(batch_tensor).detach().cpu().numpy()

    a_list = []
    b_list = []
    p_list = []
    c_list = []
    for i in range(batch_size):
        p_mat, c_vec = parse_neural_iris_output(pred_batch[i], patch_size=patch_size)
        obs_points = extract_obstacle_boundary_points(masks[i] > 0)
        a_mat, b_vec = generate_safe_region(p_mat, c_vec, obs_points, patch_size=patch_size)
        a_list.append(None if a_mat is None else np.asarray(a_mat, dtype=float))
        b_list.append(None if b_vec is None else np.asarray(b_vec, dtype=float))
        p_list.append(np.asarray(p_mat, dtype=float))
        c_list.append(np.asarray(c_vec, dtype=float))

    return a_list, b_list, p_list, c_list


def infer_safe_region_batch(obs_masks, center=(64.0, 64.0), patch_size=128, model=None, device=None, model_path=None):
    a_list, b_list, p_list, c_list = infer_safe_region_batch_halfspaces(
        obs_masks,
        center=center,
        patch_size=patch_size,
        model=model,
        device=device,
        model_path=model_path,
    )

    safe_regions = []
    for a_mat, b_vec, c_vec in zip(a_list, b_list, c_list):
        safe_regions.append(_constraints_to_vertices(a_mat, b_vec, c_vec))

    return safe_regions, p_list, c_list
