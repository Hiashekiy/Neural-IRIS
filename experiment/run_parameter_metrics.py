import os
import sys
import time
import traceback
from typing import Dict, List, Tuple
import argparse
import json

import numpy as np
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiment.common.geometry_utils import ensure_ccw, polygon_to_halfspaces, point_in_halfspaces


METHODS: List[Tuple[str, str]] = [
    ("ours_corridor_net", "experiment.ours_corridor_net.method"),
    ("ours_corridor_cpp", "experiment.ours_corridor_cpp.method"),
    ("largest_empty_circle", "experiment.largest_empty_circle.method"),
    ("rotated_rectangle", "experiment.rotated_rectangle.method"),
    ("heuristic_ellipse_fit", "experiment.heuristic_ellipse_fit.method"),
    ("segmentation_polygon_postprocess", "experiment.segmentation_polygon_postprocess.method"),
    ("direct_polygon_regression", "experiment.direct_polygon_regression.method"),
    ("iris", "experiment.iris.method"),
    ("decomputil", "experiment.decomputil.method"),
    ("firi", "experiment.firi.method"),
]

REF_METHODS = ["iris", "firi", "decomputil"]

TEST_DATA_PATH = os.path.join(ROOT, "data", "iris-dataset", "splits", "test_iris.npz")
OUT_DIR = os.path.join(ROOT, "experiment", "metrics_results")
OUT_TXT = os.path.join(OUT_DIR, "parameter_metrics_summary.txt")
OUT_CSV = os.path.join(OUT_DIR, "parameter_metrics_summary.csv")
OUT_RAW = os.path.join(OUT_DIR, "parameter_metrics_raw.npz")


def load_test_patches(path: str):
    d = np.load(path)
    return d["patches"]


def is_convex_polygon(vertices: np.ndarray, eps: float = 1e-8) -> bool:
    v = ensure_ccw(np.asarray(vertices, dtype=float))
    if len(v) < 3:
        return False
    cross_sign = None
    n = len(v)
    for i in range(n):
        a = v[i]
        b = v[(i + 1) % n]
        c = v[(i + 2) % n]
        ab = b - a
        bc = c - b
        z = ab[0] * bc[1] - ab[1] * bc[0]
        if abs(z) <= eps:
            continue
        s = z > 0
        if cross_sign is None:
            cross_sign = s
        elif cross_sign != s:
            return False
    return True


def polygon_area(vertices: np.ndarray) -> float:
    v = np.asarray(vertices, dtype=float)
    if len(v) < 3:
        return 0.0
    x = v[:, 0]
    y = v[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_mask(vertices: np.ndarray, patch_size: int = 128):
    A, b = polygon_to_halfspaces(vertices)
    if A is None:
        return None
    ys, xs = np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing="ij")
    pts = np.column_stack((xs.ravel().astype(float), ys.ravel().astype(float)))
    inside = np.all((A @ pts.T) <= (b[:, None] + 1e-6), axis=0)
    return inside.reshape((patch_size, patch_size))


def _resolve_methods(method_names: List[str] | None):
    if not method_names:
        return METHODS
    available = {n: p for n, p in METHODS}
    resolved = []
    for n in method_names:
        if n not in available:
            raise ValueError(f"unknown method: {n}")
        resolved.append((n, available[n]))
    return resolved


def evaluate(max_samples=None, patch_size: int = 128, method_names: List[str] | None = None, strict_collision: bool = False, use_batch: bool = True, batch_size_override: int | None = None):
    """
    Args:
        strict_collision: if False (default), collision area must be >= 1% of polygon area to count; 
                         if True, any pixel overlap counts
    """
    selected_methods = _resolve_methods(method_names)
    print(f"[metrics] loading test data: {TEST_DATA_PATH}")
    patches = load_test_patches(TEST_DATA_PATH)
    n = len(patches) if max_samples is None else min(int(max_samples), len(patches))
    patches = patches[:n]
    print(f"[metrics] evaluating samples: {n}")
    collision_threshold = "strict (any pixel)" if strict_collision else "relaxed (≥1% area)"
    print(f"[metrics] collision threshold: {collision_threshold}")

    center = np.array([patch_size / 2.0, patch_size / 2.0], dtype=float)

    method_modules = {}
    infer_funcs = {}
    infer_batch_funcs = {}
    batch_sizes = {}
    skipped = {}

    for name, module_path in selected_methods:
        try:
            mod = __import__(module_path, fromlist=["infer_polygon"])
            infer = getattr(mod, "infer_polygon")
            method_modules[name] = mod
            infer_funcs[name] = infer
            if hasattr(mod, "infer_polygon_batch"):
                infer_batch_funcs[name] = getattr(mod, "infer_polygon_batch")
                default_bs = int(getattr(mod, "BATCH_SIZE", 64))
                batch_sizes[name] = int(batch_size_override) if batch_size_override is not None else default_bs
            print(f"[metrics] method loaded: {name}")
        except Exception as e:
            skipped[name] = f"import failed: {e}"
            print(f"[metrics] method skipped at import: {name} | {e}")

    # Store per-method per-sample results
    results = {
        name: {
            "valid": np.zeros(n, dtype=bool),
            "center_in": np.zeros(n, dtype=bool),
            "collision": np.zeros(n, dtype=bool),
            "poly_area_mask": np.zeros(n, dtype=float),
            "poly_area_native": np.zeros(n, dtype=float),
            "coll_area": np.zeros(n, dtype=float),
            "latency_ms": np.zeros(n, dtype=float),
        }
        for name, _ in selected_methods
    }

    def _accumulate_one(name: str, i: int, obs_mask: np.ndarray, vertices):
        if vertices is None:
            return

        v = ensure_ccw(np.asarray(vertices, dtype=float))
        if len(v) < 3:
            return
        if polygon_area(v) <= 1e-8:
            return
        if not is_convex_polygon(v):
            return

        mask = polygon_mask(v, patch_size=patch_size)
        if mask is None:
            return

        results[name]["valid"][i] = True
        results[name]["poly_area_mask"][i] = float(mask.sum())
        results[name]["poly_area_native"][i] = float(polygon_area(v))

        coll_area_pixels = float(np.sum(mask & obs_mask))
        results[name]["coll_area"][i] = coll_area_pixels

        if strict_collision:
            is_collision = coll_area_pixels > 1e-9
        else:
            collision_percent = coll_area_pixels / results[name]["poly_area_mask"][i] if results[name]["poly_area_mask"][i] > 1e-9 else 0.0
            is_collision = collision_percent >= 0.01

        results[name]["collision"][i] = bool(is_collision)

        A, b = polygon_to_halfspaces(v)
        if A is not None and point_in_halfspaces(center, A, b):
            results[name]["center_in"][i] = True

    total_units = n * max(1, len(selected_methods))
    pbar = tqdm(total=total_units, desc="parameter-metrics", leave=True)

    for name, _ in selected_methods:
        if name not in infer_funcs:
            pbar.update(n)
            continue

        if use_batch and (name in infer_batch_funcs):
            infer_batch = infer_batch_funcs[name]
            bs = max(1, int(batch_sizes.get(name, 64)))
            for s in range(0, n, bs):
                e = min(n, s + bs)
                obs_batch = (patches[s:e] > 0.5)
                t0 = time.perf_counter()
                try:
                    vertices_list = infer_batch(obs_masks=obs_batch, center=center, patch_size=patch_size)
                except NotImplementedError as e1:
                    skipped[name] = str(e1)
                    pbar.update(e - s)
                    continue
                except Exception:
                    vertices_list = [None] * (e - s)
                t1 = time.perf_counter()

                per_ms = ((t1 - t0) * 1000.0) / max(1, (e - s))
                for j, i in enumerate(range(s, e)):
                    results[name]["latency_ms"][i] = per_ms
                    _accumulate_one(name, i, obs_batch[j], vertices_list[j] if j < len(vertices_list) else None)
                pbar.update(e - s)
        else:
            infer = infer_funcs[name]
            for i in range(n):
                obs_mask = patches[i] > 0.5
                t0 = time.perf_counter()
                try:
                    vertices = infer(obs_mask=obs_mask, center=center, patch_size=patch_size)
                except NotImplementedError as e1:
                    skipped[name] = str(e1)
                    vertices = None
                except Exception:
                    vertices = None
                t1 = time.perf_counter()
                results[name]["latency_ms"][i] = (t1 - t0) * 1000.0
                _accumulate_one(name, i, obs_mask, vertices)
                pbar.update(1)

        pbar.set_postfix(method=name)

    pbar.close()

    # Build reference area for each sample.
    ref_area = np.zeros(n, dtype=float)
    for i in range(n):
        vals = []
        for rn in REF_METHODS:
            if rn in results and results[rn]["valid"][i]:
                vals.append(results[rn]["poly_area_native"][i])
        ref_area[i] = max(vals) if vals else 0.0

    rows = []
    for name, _ in selected_methods:
        r = results[name]
        valid_cnt = int(r["valid"].sum())

        valid_rate = valid_cnt / n
        center_in_rate = float(r["center_in"].sum()) / n

        # As defined over total samples.
        collision_rate = float(r["collision"].sum()) / n

        poly_area_sum = float(r["poly_area_mask"].sum())
        coll_area_sum = float(r["coll_area"].sum())
        collision_area_ratio = (coll_area_sum / poly_area_sum) if poly_area_sum > 1e-9 else np.nan

        # Normalized area over samples where both method and ref are valid.
        denom_mask = r["valid"] & (ref_area > 1e-9)
        if np.any(denom_mask):
            norm_area = float(np.mean(r["poly_area_native"][denom_mask] / ref_area[denom_mask]))
        else:
            norm_area = np.nan

        mean_latency = float(np.mean(r["latency_ms"]))

        rows.append({
            "method": name,
            "valid_rate": valid_rate,
            "center_in_rate": center_in_rate,
            "collision_rate": collision_rate,
            "collision_area_ratio": collision_area_ratio,
            "normalized_area": norm_area,
            "mean_latency_ms": mean_latency,
            "samples": n,
            "valid_count": valid_cnt,
            "native_area_mean": float(np.mean(r["poly_area_native"])) if n > 0 else np.nan,
        })

    return rows, skipped, results, selected_methods, n


def format_table(rows: List[Dict]):
    header = (
        f"{'method':<34} {'valid':>8} {'center_in':>10} {'coll_rate':>10} "
        f"{'coll_area':>10} {'norm_area':>10} {'lat_ms':>10}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        ca = r["collision_area_ratio"]
        na = r["normalized_area"]
        ca_s = f"{100.0 * ca:9.3f}%" if np.isfinite(ca) else "    nan   "
        na_s = f"{na:10.4f}" if np.isfinite(na) else "    nan   "
        lines.append(
            f"{r['method']:<34} "
            f"{100.0 * r['valid_rate']:7.2f}% "
            f"{100.0 * r['center_in_rate']:9.2f}% "
            f"{100.0 * r['collision_rate']:9.2f}% "
            f"{ca_s:>10} "
            f"{na_s:>10} "
            f"{r['mean_latency_ms']:9.3f}"
        )
    return "\n".join(lines)


def save_csv(rows: List[Dict], path: str):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = [
        "method",
        "samples",
        "valid_count",
        "valid_rate",
        "center_in_rate",
        "collision_rate",
        "collision_area_ratio",
        "normalized_area",
        "mean_latency_ms",
        "native_area_mean",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _save_raw(results: Dict[str, Dict[str, np.ndarray]], selected_methods: List[Tuple[str, str]], n: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "sample_count": np.array([n], dtype=np.int32),
        "method_names": np.array([m for m, _ in selected_methods], dtype=object),
    }
    for name, _ in selected_methods:
        r = results[name]
        payload[f"{name}__valid"] = r["valid"].astype(np.uint8)
        payload[f"{name}__center_in"] = r["center_in"].astype(np.uint8)
        payload[f"{name}__collision"] = r["collision"].astype(np.uint8)
        payload[f"{name}__poly_area_mask"] = r["poly_area_mask"].astype(np.float64)
        payload[f"{name}__poly_area_native"] = r["poly_area_native"].astype(np.float64)
        payload[f"{name}__coll_area"] = r["coll_area"].astype(np.float64)
        payload[f"{name}__latency_ms"] = r["latency_ms"].astype(np.float64)
    np.savez_compressed(path, **payload)


def main(max_samples=None, method_names: List[str] | None = None, out_txt: str = OUT_TXT, out_csv: str = OUT_CSV, raw_out: str | None = OUT_RAW, strict_collision: bool = False, use_batch: bool = True, batch_size_override: int | None = None):
    rows, skipped, results, selected_methods, n = evaluate(max_samples=max_samples, method_names=method_names, strict_collision=strict_collision, use_batch=use_batch, batch_size_override=batch_size_override)

    txt = []
    txt.append("[Parameter Metrics Summary]")
    txt.append(f"samples={n}")
    txt.append("")
    txt.append(format_table(rows))

    if skipped:
        txt.append("\n[Skipped / Import Errors]")
        for k, v in skipped.items():
            txt.append(f"- {k}: {v}")

    out = "\n".join(txt)
    print(out)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(out)
    save_csv(rows, out_csv)
    if raw_out:
        _save_raw(results, selected_methods, n, raw_out)
    print(f"\nSaved: {out_txt}")
    print(f"Saved: {out_csv}")
    if raw_out:
        print(f"Saved: {raw_out}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--max_samples", type=int, default=None)
        parser.add_argument("--methods", type=str, default=None, help="comma-separated method names")
        parser.add_argument("--out_txt", type=str, default=OUT_TXT)
        parser.add_argument("--out_csv", type=str, default=OUT_CSV)
        parser.add_argument("--raw_out", type=str, default=OUT_RAW)
        parser.add_argument("--strict", action="store_true", help="use strict collision (any pixel); default is relaxed (≥1%% area)")
        parser.add_argument("--no_batch", action="store_true", help="disable batch inference path even if method provides infer_polygon_batch")
        parser.add_argument("--batch_size", type=int, default=None, help="override batch size for methods with infer_polygon_batch")
        args = parser.parse_args()
        method_names = None
        if args.methods:
            method_names = [x.strip() for x in args.methods.split(",") if x.strip()]
        main(
            max_samples=args.max_samples,
            method_names=method_names,
            out_txt=args.out_txt,
            out_csv=args.out_csv,
            raw_out=args.raw_out,
            strict_collision=args.strict,
            use_batch=(not args.no_batch),
            batch_size_override=args.batch_size,
        )
    except Exception:
        traceback.print_exc()
        raise
