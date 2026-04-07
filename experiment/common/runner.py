import importlib
import os
import time
from typing import Dict, Any

import numpy as np

from experiment.common.geometry_utils import polygon_to_halfspaces, point_in_halfspaces, ensure_ccw


TEST_DATA_PATH = os.path.join("data", "iris-dataset", "splits", "test_iris.npz")


def load_test_patches(npz_path: str):
    data = np.load(npz_path)
    return data["patches"]


def evaluate_method(method_name: str, module_path: str, max_samples: int = 300) -> Dict[str, Any]:
    patches = load_test_patches(TEST_DATA_PATH)
    n = min(max_samples, len(patches))
    center = np.array([64.0, 64.0], dtype=float)

    mod = importlib.import_module(module_path)
    infer_polygon = getattr(mod, "infer_polygon")

    valid = 0
    center_in = 0
    total_edges = 0
    total_ms = 0.0
    failures = 0

    for i in range(n):
        obs_mask = patches[i] > 0.5
        t0 = time.perf_counter()
        try:
            vertices = infer_polygon(obs_mask=obs_mask, center=center, patch_size=128)
        except NotImplementedError:
            raise
        except Exception:
            failures += 1
            continue
        total_ms += (time.perf_counter() - t0) * 1000.0

        if vertices is None:
            failures += 1
            continue

        vertices = ensure_ccw(np.asarray(vertices, dtype=float))
        A, b = polygon_to_halfspaces(vertices)
        if A is None:
            failures += 1
            continue

        valid += 1
        total_edges += len(vertices)
        if point_in_halfspaces(center, A, b):
            center_in += 1

    if valid == 0:
        return {
            "method": method_name,
            "samples": n,
            "valid_rate": 0.0,
            "center_in_rate": 0.0,
            "avg_edges": 0.0,
            "avg_ms": 0.0,
            "failures": failures,
        }

    return {
        "method": method_name,
        "samples": n,
        "valid_rate": valid / n,
        "center_in_rate": center_in / valid,
        "avg_edges": total_edges / valid,
        "avg_ms": total_ms / valid,
        "failures": failures,
    }


def format_results(rows):
    header = (
        f"{'method':<36} {'valid_rate':>10} {'center_in':>10} "
        f"{'avg_edges':>10} {'avg_ms':>10} {'failures':>10}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r['method']:<36} "
            f"{(100.0 * r['valid_rate']):>9.2f}% "
            f"{(100.0 * r['center_in_rate']):>9.2f}% "
            f"{r['avg_edges']:>10.2f} "
            f"{r['avg_ms']:>9.3f} "
            f"{r['failures']:>10d}"
        )
    return "\n".join(lines)
