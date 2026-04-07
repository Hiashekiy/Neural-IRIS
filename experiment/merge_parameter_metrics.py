import argparse
import csv
import os
from typing import Dict, List

import numpy as np

REF_METHODS = ["iris", "firi", "decomputil"]


def load_raw(path: str):
    d = np.load(path, allow_pickle=True)
    methods = [str(x) for x in d["method_names"].tolist()]
    n = int(d["sample_count"][0])
    out = {}
    for m in methods:
        out[m] = {
            "valid": d[f"{m}__valid"].astype(bool),
            "center_in": d[f"{m}__center_in"].astype(bool),
            "collision": d[f"{m}__collision"].astype(bool),
            "poly_area_mask": d[f"{m}__poly_area_mask"].astype(float),
            "poly_area_native": d[f"{m}__poly_area_native"].astype(float),
            "coll_area": d[f"{m}__coll_area"].astype(float),
            "latency_ms": d[f"{m}__latency_ms"].astype(float),
        }
    return n, out


def merge_raw(raws: List[Dict[str, Dict[str, np.ndarray]]], n: int):
    merged: Dict[str, Dict[str, np.ndarray]] = {}
    for rd in raws:
        for m, vals in rd.items():
            if m not in merged:
                merged[m] = vals
            else:
                # same method appears twice -> keep the one with more valid samples.
                if vals["valid"].sum() > merged[m]["valid"].sum():
                    merged[m] = vals
    return merged


def summarize(n: int, results: Dict[str, Dict[str, np.ndarray]]):
    ref = np.zeros(n, dtype=float)
    for i in range(n):
        cands = []
        for rn in REF_METHODS:
            if rn in results and results[rn]["valid"][i]:
                cands.append(results[rn]["poly_area_native"][i])
        ref[i] = max(cands) if cands else 0.0

    rows = []
    for m, r in results.items():
        valid_rate = float(r["valid"].sum()) / n
        center_in_rate = float(r["center_in"].sum()) / n
        collision_rate = float(r["collision"].sum()) / n

        poly_area_sum = float(r["poly_area_mask"].sum())
        coll_area_sum = float(r["coll_area"].sum())
        coll_area_ratio = (coll_area_sum / poly_area_sum) if poly_area_sum > 1e-9 else np.nan

        mask = r["valid"] & (ref > 1e-9)
        norm_area = float(np.mean(r["poly_area_native"][mask] / ref[mask])) if np.any(mask) else np.nan

        rows.append({
            "method": m,
            "samples": n,
            "valid_count": int(r["valid"].sum()),
            "valid_rate": valid_rate,
            "center_in_rate": center_in_rate,
            "collision_rate": collision_rate,
            "collision_area_ratio": coll_area_ratio,
            "normalized_area": norm_area,
            "mean_latency_ms": float(np.mean(r["latency_ms"])),
            "native_area_mean": float(np.mean(r["poly_area_native"])),
        })

    rows.sort(key=lambda x: x["method"])
    return rows


def write_outputs(rows, out_txt, out_csv):
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)

    header = (
        f"{'method':<34} {'valid':>8} {'center_in':>10} {'coll_rate':>10} "
        f"{'coll_area':>10} {'norm_area':>10} {'lat_ms':>10}"
    )
    sep = "-" * len(header)
    lines = ["[Merged Parameter Metrics Summary]", "", header, sep]

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

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

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
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved: {out_txt}")
    print(f"Saved: {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_a", type=str, required=True)
    parser.add_argument("--raw_b", type=str, required=True)
    parser.add_argument("--out_txt", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    args = parser.parse_args()

    n1, d1 = load_raw(args.raw_a)
    n2, d2 = load_raw(args.raw_b)
    if n1 != n2:
        raise ValueError(f"sample_count mismatch: {n1} vs {n2}")

    merged = merge_raw([d1, d2], n1)
    rows = summarize(n1, merged)
    write_outputs(rows, args.out_txt, args.out_csv)


if __name__ == "__main__":
    main()
