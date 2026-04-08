import argparse
import csv
import os
from typing import List, Dict


FIELDS = [
    "group",
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


def load_rows(path: str, group_name: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            item = {key: row.get(key, "") for key in FIELDS if key != "group"}
            item["group"] = group_name
            rows.append(item)
        return rows


def load_norm_overrides(path: str) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row.get("method", "").strip()
            norm_area = row.get("normalized_area", "").strip()
            if method:
                overrides[method] = norm_area
    return overrides


def write_txt(rows: List[Dict[str, str]], out_txt: str) -> None:
    header = (
        f"{'group':<24} {'method':<34} {'valid':>8} {'center_in':>10} {'coll_rate':>10} "
        f"{'coll_area':>10} {'norm_area':>10} {'lat_ms':>10}"
    )
    sep = "-" * len(header)
    lines = ["[Total Benchmark Summary]", "", header, sep]

    for row in rows:
        valid_rate = float(row["valid_rate"])
        center_in_rate = float(row["center_in_rate"])
        collision_rate = float(row["collision_rate"])
        collision_area_ratio = float(row["collision_area_ratio"])
        normalized_area = float(row["normalized_area"])
        mean_latency_ms = float(row["mean_latency_ms"])

        ca_s = f"{100.0 * collision_area_ratio:9.3f}%" if collision_area_ratio == collision_area_ratio else "    nan   "
        na_s = f"{normalized_area:10.4f}" if normalized_area == normalized_area else "    nan   "
        lines.append(
            f"{row['group']:<24} "
            f"{row['method']:<34} "
            f"{100.0 * valid_rate:7.2f}% "
            f"{100.0 * center_in_rate:9.2f}% "
            f"{100.0 * collision_rate:9.2f}% "
            f"{ca_s:>10} "
            f"{na_s:>10} "
            f"{mean_latency_ms:9.3f}"
        )

    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input summary CSV in the form group_name=path/to/file.csv",
    )
    parser.add_argument(
        "--norm_override",
        action="append",
        default=[],
        help="Optional merged CSV used to backfill normalized_area by method name",
    )
    parser.add_argument("--out_txt", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    args = parser.parse_args()

    norm_overrides: Dict[str, str] = {}
    for path in args.norm_override:
        norm_overrides.update(load_norm_overrides(path))

    rows: List[Dict[str, str]] = []
    for spec in args.input:
        if "=" not in spec:
            raise ValueError(f"Invalid --input value: {spec!r}")
        group_name, path = spec.split("=", 1)
        group_name = group_name.strip()
        path = path.strip()
        if not group_name or not path:
            raise ValueError(f"Invalid --input value: {spec!r}")
        for row in load_rows(path, group_name):
            if row.get("normalized_area", "").lower() == "nan" and row.get("method", "") in norm_overrides:
                row["normalized_area"] = norm_overrides[row["method"]]
            rows.append(row)

    rows.sort(key=lambda row: (row["group"], row["method"]))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    write_txt(rows, args.out_txt)
    print(f"Saved: {args.out_txt}")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()