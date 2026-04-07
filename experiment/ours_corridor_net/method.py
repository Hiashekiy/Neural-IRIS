import os
import sys

import numpy as np

from experiment.common.geometry_utils import halfspace_intersection_vertices

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_MODEL = None
_DEVICE = None
_TORCH_READY = False
BATCH_SIZE = 64


def _lazy_init():
    global _MODEL, _DEVICE, _TORCH_READY
    if _MODEL is not None:
        return

    try:
        import torch
        from src.corridor_constraints.model import CorridorEllipseNet

        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CorridorEllipseNet().to(_DEVICE)
        model_path = os.path.join(_ROOT, "models", "iris_net_best.pth")
        state = torch.load(model_path, map_location=_DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _MODEL = model
        _TORCH_READY = True
    except Exception as e:
        raise NotImplementedError(f"ours_corridor_net unavailable: {e}")


def infer_polygon(obs_mask, center=(64.0, 64.0), patch_size=128):
    _lazy_init()

    import torch
    from src.corridor_constraints.geometry import parse_network_output, extract_obstacle_boundary_points
    from src.corridor_constraints.generate_safe_polygon import generate_safe_polygon

    x = torch.from_numpy(obs_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        pred = _MODEL(x).squeeze(0).detach().cpu().numpy()

    p_mat, c = parse_network_output(pred, patch_size=patch_size)
    obs_points = extract_obstacle_boundary_points(obs_mask)
    poly = generate_safe_polygon(p_mat, c, obs_points, patch_size=patch_size)

    if poly is None:
        return None

    if isinstance(poly, tuple) and len(poly) == 2:
        A, b = poly
        if A is None or b is None or len(A) < 3:
            return None
        # Prefer center as interior point for comparable metrics.
        center_pt = np.asarray(center, dtype=float)
        v = halfspace_intersection_vertices(np.asarray(A, dtype=float), np.asarray(b, dtype=float), center_pt)
        if v is None or len(v) < 3:
            v = halfspace_intersection_vertices(np.asarray(A, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float))
        return v

    arr = np.asarray(poly, dtype=float)
    return arr if len(arr) >= 3 else None


def infer_polygon_batch(obs_masks, center=(64.0, 64.0), patch_size=128):
    _lazy_init()

    import torch
    from src.corridor_constraints.geometry import parse_network_output, extract_obstacle_boundary_points
    from src.corridor_constraints.generate_safe_polygon import generate_safe_polygon

    masks = np.asarray(obs_masks, dtype=np.uint8)
    if masks.ndim != 3:
        return [None for _ in range(len(obs_masks))]

    bs = int(masks.shape[0])
    if bs <= 0:
        return []

    if masks.shape[1] != patch_size or masks.shape[2] != patch_size:
        return [None for _ in range(bs)]

    x = torch.from_numpy(masks.astype(np.float32)).unsqueeze(1).to(_DEVICE)
    with torch.no_grad():
        pred_batch = _MODEL(x).detach().cpu().numpy()

    ret = []
    for i in range(bs):
        obs_mask = masks[i] > 0
        pred = pred_batch[i]

        p_mat, c = parse_network_output(pred, patch_size=patch_size)
        obs_points = extract_obstacle_boundary_points(obs_mask)
        poly = generate_safe_polygon(p_mat, c, obs_points, patch_size=patch_size)

        if poly is None:
            ret.append(None)
            continue

        if isinstance(poly, tuple) and len(poly) == 2:
            A, b = poly
            if A is None or b is None or len(A) < 3:
                ret.append(None)
                continue
            center_pt = np.asarray(center, dtype=float)
            v = halfspace_intersection_vertices(np.asarray(A, dtype=float), np.asarray(b, dtype=float), center_pt)
            if v is None or len(v) < 3:
                v = halfspace_intersection_vertices(np.asarray(A, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float))
            ret.append(v if (v is not None and len(v) >= 3) else None)
            continue

        arr = np.asarray(poly, dtype=float)
        ret.append(arr if len(arr) >= 3 else None)

    return ret
