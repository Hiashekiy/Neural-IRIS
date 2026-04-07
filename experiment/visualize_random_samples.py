import importlib
import os
import sys
import traceback
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TEST_DATA_PATH = os.path.join(ROOT, "data", "iris-dataset", "splits", "test_iris.npz")
OUT_ROOT = os.path.join(ROOT, "experiment", "visual_results", "random50")

METHODS: List[Tuple[str, str]] = [
    ("largest_empty_circle", "experiment.largest_empty_circle.method"),
    ("rotated_rectangle", "experiment.rotated_rectangle.method"),
    ("heuristic_ellipse_fit", "experiment.heuristic_ellipse_fit.method"),
    ("segmentation_polygon_postprocess", "experiment.segmentation_polygon_postprocess.method"),
    ("direct_polygon_regression", "experiment.direct_polygon_regression.method"),
    ("iris", "experiment.iris.method"),
    ("decomputil", "experiment.decomputil.method"),
    ("firi", "experiment.firi.method"),
]


def load_test_patches(npz_path: str):
    data = np.load(npz_path)
    return data["patches"]


def ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    if vertices is None or len(vertices) < 3:
        return vertices
    area2 = np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1) - np.roll(vertices[:, 0], -1) * vertices[:, 1])
    if area2 < 0:
        return vertices[::-1].copy()
    return vertices


def estimate_ellipse_from_vertices(vertices: np.ndarray, center: np.ndarray):
    if vertices is None or len(vertices) < 3:
        return None

    X = vertices - center[None, :]
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-6)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Use vertex spread to estimate axis lengths.
    a = float(np.sqrt(vals[0]) * 2.5)
    b = float(np.sqrt(vals[1]) * 2.5)
    theta = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    return a, b, theta


def method_shape_type(method_name: str):
    if method_name in ("largest_empty_circle", "heuristic_ellipse_fit"):
        return "ellipse"
    return "polygon"


def draw_and_save(obs_mask: np.ndarray, center: np.ndarray, vertices: np.ndarray, method_name: str, sample_id: int, out_dir: str):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    ax.imshow(obs_mask.astype(float), cmap="gray_r", origin="upper")

    ax.scatter([center[0]], [center[1]], c="red", s=16, label="center")

    if vertices is not None and len(vertices) >= 3:
        vertices = ensure_ccw(vertices)
        poly = Polygon(vertices, closed=True, fill=False, edgecolor="deepskyblue", linewidth=1.6)
        ax.add_patch(poly)

        if method_shape_type(method_name) == "ellipse":
            est = estimate_ellipse_from_vertices(vertices, center)
            if est is not None:
                a, b, theta = est
                ell = Ellipse(
                    xy=(center[0], center[1]),
                    width=2.0 * max(1.0, a),
                    height=2.0 * max(1.0, b),
                    angle=theta,
                    fill=False,
                    edgecolor="orange",
                    linewidth=1.4,
                    linestyle="--",
                )
                ax.add_patch(ell)

    ax.set_title(f"{method_name} | sample {sample_id}")
    ax.set_xlim(0, 127)
    ax.set_ylim(127, 0)
    ax.set_aspect("equal")
    ax.grid(False)

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"sample_{sample_id:05d}.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main(sample_count: int = 50, seed: int = 42):
    patches = load_test_patches(TEST_DATA_PATH)
    total = len(patches)
    n = min(sample_count, total)

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total, size=n, replace=False))

    os.makedirs(OUT_ROOT, exist_ok=True)

    summary: Dict[str, Dict[str, int]] = {}

    for method_name, module_path in METHODS:
        out_dir = os.path.join(OUT_ROOT, method_name)
        os.makedirs(out_dir, exist_ok=True)

        summary[method_name] = {"ok": 0, "fail": 0, "skip": 0}

        try:
            mod = importlib.import_module(module_path)
            infer_polygon = getattr(mod, "infer_polygon")
        except Exception:
            summary[method_name]["skip"] = len(indices)
            continue

        for idx in indices:
            obs_mask = patches[idx] > 0.5
            center = np.array([64.0, 64.0], dtype=float)

            try:
                vertices = infer_polygon(obs_mask=obs_mask, center=center, patch_size=128)
                if vertices is None or len(vertices) < 3:
                    summary[method_name]["fail"] += 1
                    vertices = None
                else:
                    vertices = np.asarray(vertices, dtype=float)
                    summary[method_name]["ok"] += 1
            except NotImplementedError:
                summary[method_name]["skip"] += 1
                continue
            except Exception:
                summary[method_name]["fail"] += 1
                vertices = None

            draw_and_save(
                obs_mask=obs_mask,
                center=center,
                vertices=vertices,
                method_name=method_name,
                sample_id=int(idx),
                out_dir=out_dir,
            )

    summary_path = os.path.join(OUT_ROOT, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"random_seed: {seed}\n")
        f.write(f"sample_count: {n}\n")
        f.write("sample_indices: " + ",".join(map(str, indices.tolist())) + "\n\n")
        for m, st in summary.items():
            f.write(f"{m}: ok={st['ok']} fail={st['fail']} skip={st['skip']}\n")

    print("Saved visualization to:", OUT_ROOT)
    print("Summary:")
    for m, st in summary.items():
        print(f"- {m}: ok={st['ok']} fail={st['fail']} skip={st['skip']}")


if __name__ == "__main__":
    try:
        main(sample_count=50, seed=42)
    except Exception:
        traceback.print_exc()
        raise
