import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_EVAL_ROOT = os.path.join(ROOT, "logs", "neural_iris_eval")
PATCH_SIZE = 128.0
ANGLE_RANGE_DEG = 90.0
DEFAULT_OUTLIER_IQR_MULTIPLIER = 3.0
MAX_DISPLAY_FLIERS_PER_METRIC = 80


def resolve_input_path(input_path):
    if input_path is None:
        run_dirs = sorted(
            [
                os.path.join(DEFAULT_EVAL_ROOT, name)
                for name in os.listdir(DEFAULT_EVAL_ROOT)
                if os.path.isdir(os.path.join(DEFAULT_EVAL_ROOT, name))
            ]
        )
        if not run_dirs:
            raise FileNotFoundError(f"No evaluation runs found under {DEFAULT_EVAL_ROOT}")
        input_path = run_dirs[-1]

    if os.path.isdir(input_path):
        json_path = os.path.join(input_path, "test_metrics.json")
        csv_path = os.path.join(input_path, "test_metrics.csv")
        if os.path.isfile(json_path):
            return os.path.abspath(json_path)
        if os.path.isfile(csv_path):
            return os.path.abspath(csv_path)
        raise FileNotFoundError(f"No test_metrics.json or test_metrics.csv found in {input_path}")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Metrics file not found: {input_path}")

    return os.path.abspath(input_path)


def load_metrics(input_path):
    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(input_path, "r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    metrics = {}
    for key, value in row.items():
        if key == "sample_count":
            metrics[key] = int(float(value))
        elif value.startswith("[") and value.endswith("]"):
            metrics[key] = json.loads(value)
        else:
            metrics[key] = float(value)
    return metrics


def resolve_output_path(input_path, output_path):
    if output_path is not None:
        return os.path.abspath(output_path)

    base_dir = os.path.dirname(input_path)
    return os.path.join(base_dir, "test_metrics_paper.png")


def apply_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 17,
        "axes.titlesize": 20,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.linewidth": 1.15,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def build_normalized_box_stats(metrics):
    spec = [
        ("Center Error\n/ 128 px", "center_error_px", PATCH_SIZE, "#4c78a8", False),
        ("Axis Error\n/ 128 px", "axis_error_px", PATCH_SIZE, "#e45756", False),
        ("Angle Error\n/ 90 deg", "angle_error_deg", ANGLE_RANGE_DEG, "#7b6fd0", False),
        ("Overlap Error\n(1 - IoU)", "iou_percent", 100.0, "#2e8b57", True),
        ("Collision Ratio\n(collided only)", "collision_area_percent_collided", 100.0, "#e08d2d", False),
    ]
    required_suffixes = ["q1", "median", "q3"]
    missing = [
        f"{prefix}_{suffix}"
        for _, prefix, _, _, _ in spec
        for suffix in required_suffixes
        if f"{prefix}_{suffix}" not in metrics
    ]
    if missing:
        raise ValueError(
            "This metrics file does not contain the quartile statistics needed for a box plot. "
            "Please rerun scripts/train/final_evaluate_neural_iris.py to regenerate test_metrics.json/csv."
        )

    stats = []
    for label, prefix, scale, color, invert in spec:
        whislo_key = f"{prefix}_whislo"
        whishi_key = f"{prefix}_whishi"
        if whislo_key in metrics and whishi_key in metrics:
            whislo = metrics[whislo_key] / scale
            whishi = metrics[whishi_key] / scale
        else:
            whislo = metrics.get(f"{prefix}_min", metrics[f"{prefix}_q1"]) / scale
            whishi = metrics.get(f"{prefix}_max", metrics[f"{prefix}_q3"]) / scale
        q1 = metrics[f"{prefix}_q1"] / scale
        med = metrics[f"{prefix}_median"] / scale
        q3 = metrics[f"{prefix}_q3"] / scale
        mean = metrics[prefix] / scale
        fliers = [v / scale for v in metrics.get(f"{prefix}_fliers", [])]
        if invert:
            whislo, q1, med, q3, whishi, mean = (
                1.0 - whishi,
                1.0 - q3,
                1.0 - med,
                1.0 - q1,
                1.0 - whislo,
                1.0 - mean,
            )
            fliers = [1.0 - v for v in fliers]
        stats.append(
            {
                "label": label,
                "color": color,
                "outlier_count": int(metrics.get(f"{prefix}_outlier_count", 0)),
                "stats": {
                    "label": label,
                    "whislo": whislo,
                    "q1": q1,
                    "med": med,
                    "q3": q3,
                    "whishi": whishi,
                    "mean": mean,
                    "fliers": fliers,
                },
            }
        )
    return stats


def save_plot(metrics, output_path):
    apply_plot_style()
    items = build_normalized_box_stats(metrics)
    outlier_iqr_multiplier = float(metrics.get("outlier_iqr_multiplier", DEFAULT_OUTLIER_IQR_MULTIPLIER))

    def style_axis(ax, show_bottom):
        ax.grid(False)
        ax.set_facecolor("white")
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.15)
        ax.spines["top"].set_visible(False)
        if show_bottom:
            ax.spines["bottom"].set_linewidth(1.15)
        else:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.tick_params(axis="both", width=1.0, length=5)

    all_values = []
    for item in items:
        all_values.extend([
            item["stats"]["whislo"],
            item["stats"]["q1"],
            item["stats"]["med"],
            item["stats"]["q3"],
            item["stats"]["whishi"],
            item["stats"]["mean"],
        ])
        all_values.extend(item["stats"].get("fliers", []))

    overall_max = max(all_values) if all_values else 1.0
    core_upper = max(max(item["stats"]["q3"], item["stats"]["mean"]) for item in items)
    lower_upper = 0.25 if overall_max > 0.30 else min(max(core_upper * 1.55, 0.12), overall_max * 0.72)
    upper_candidates = [v for v in all_values if v > lower_upper]
    use_broken_axis = len(upper_candidates) > 0 and overall_max > lower_upper * 1.25

    if use_broken_axis:
        upper_lower = min(upper_candidates) * 0.96
        upper_upper = overall_max * 1.04
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            figsize=(8.5, 6.3),
            facecolor="white",
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 3.2], "hspace": 0.05},
        )
        axes = [ax_top, ax_bottom]
        limits = [(upper_lower, upper_upper), (0.0, lower_upper)]
    else:
        fig, ax_bottom = plt.subplots(figsize=(8.5, 6.1), facecolor="white")
        axes = [ax_bottom]
        limits = [(0.0, max(overall_max * 1.08, 0.12))]

    for axis_idx, (ax, axis_limits) in enumerate(zip(axes, limits)):
        style_axis(ax, show_bottom=(axis_idx == len(axes) - 1))
        ax.set_ylim(*axis_limits)
        artists = ax.bxp(
            [item["stats"] for item in items],
            showmeans=True,
            showfliers=False,
            patch_artist=True,
            widths=0.48,
            boxprops={"edgecolor": "#222222", "linewidth": 1.15},
            whiskerprops={"color": "#222222", "linewidth": 1.1},
            capprops={"color": "#222222", "linewidth": 1.1},
            medianprops={"color": "#111111", "linewidth": 1.4},
            meanprops={"marker": "D", "markerfacecolor": "#111111", "markeredgecolor": "#111111", "markersize": 5},
        )
        for patch, item in zip(artists["boxes"], items):
            patch.set_facecolor(item["color"])
            patch.set_alpha(0.82)

        for idx, item in enumerate(items, start=1):
            fliers = item["stats"].get("fliers", [])
            if not fliers:
                continue
            fliers_arr = np.sort(np.asarray(fliers, dtype=float))
            if fliers_arr.size > MAX_DISPLAY_FLIERS_PER_METRIC:
                sample_idx = np.linspace(0, fliers_arr.size - 1, MAX_DISPLAY_FLIERS_PER_METRIC, dtype=int)
                fliers_arr = fliers_arr[sample_idx]
            rng = np.random.default_rng(1000 + idx)
            offsets = rng.uniform(-0.11, 0.11, size=fliers_arr.size)
            visible_mask = (fliers_arr >= axis_limits[0]) & (fliers_arr <= axis_limits[1])
            if np.any(visible_mask):
                ax.scatter(
                    idx + offsets[visible_mask],
                    fliers_arr[visible_mask],
                    s=9,
                    c="#4a4a4a",
                    alpha=0.18,
                    linewidths=0,
                    zorder=3,
                    rasterized=True,
                )

        if axis_idx == len(axes) - 1:
            upper = axis_limits[1]
            for idx, item in enumerate(items, start=1):
                mean_v = item["stats"]["mean"]
                if axis_limits[0] <= mean_v <= axis_limits[1]:
                    ax.text(idx, min(mean_v + upper * 0.045, upper * 0.98), f"{mean_v:.3f}", ha="center", va="bottom", fontsize=12.5, color="#222222")

    if use_broken_axis:
        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        d = 0.012
        kwargs = dict(transform=ax_top.transAxes, color="#666666", clip_on=False, linewidth=1.0)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bottom.transAxes)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    ax_main = axes[-1]
    ax_main.set_ylabel("Normalized Value")
    ax_main.set_xticks(range(1, len(items) + 1))
    ax_main.set_xticklabels([item["label"] for item in items])

    sample_count = metrics.get("sample_count")
    collision_count = metrics.get("collision_count")
    fig.suptitle("Normalized Final Test Metrics", y=0.985, fontsize=20)
    top_text = (
        f"N={sample_count} | Collision Rate: {metrics['collision_rate']:.2f}% | Collided Samples: {collision_count}/{sample_count}\n"
        f"Overlap Error = 1 - IoU | Collision Ratio = collision area / predicted ellipse area\n"
        f"Whiskers: {outlier_iqr_multiplier:.1f} IQR | Dots: sampled outliers"
    )
    fig.text(0.5, 0.945, top_text, ha="center", va="top", fontsize=14.6, color="#555555")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.84, hspace=0.05)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot final Neural-IRIS test metrics from an existing JSON/CSV file.")
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to test_metrics.json, test_metrics.csv, or an evaluation run directory. Defaults to the latest run.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to <run_dir>/test_metrics_paper.png.",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    metrics = load_metrics(input_path)
    output_path = resolve_output_path(input_path, args.output)
    save_plot(metrics, output_path)

    print(f"Loaded metrics: {input_path}")
    print(f"Saved plot PNG: {output_path}")


if __name__ == "__main__":
    main()
