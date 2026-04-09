import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
DEFAULT_MODEL_PATH = os.path.join(ROOT, "models", "neural_iris_net_best.pth")
DEFAULT_TEST_PATH = os.path.join(ROOT, "data", "iris-dataset", "splits", "test_iris.npz")
DEFAULT_OUTPUT_ROOT = os.path.join(ROOT, "logs", "neural_iris_eval")
PATCH_SIZE = 128.0
ANGLE_RANGE_DEG = 90.0
OUTLIER_IQR_MULTIPLIER = 3.0
MAX_DISPLAY_FLIERS_PER_METRIC = 80


def load_model(model_path, device):
    from src.neural_iris.model import NeuralIRISNet

    model = NeuralIRISNet().to(device)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_dataset(npz_path, num_samples=None):
    data = np.load(npz_path)
    patches = data["patches"]
    labels = data["labels"]

    if num_samples is not None and num_samples > 0:
        patches = patches[:num_samples]
        labels = labels[:num_samples]

    return patches, labels


def build_grids(max_batch_size, device):
    y_idx, x_idx = torch.meshgrid(
        torch.arange(128, device=device),
        torch.arange(128, device=device),
        indexing="ij",
    )
    grid_x = x_idx.float().unsqueeze(0).expand(max_batch_size, -1, -1)
    grid_y = y_idx.float().unsqueeze(0).expand(max_batch_size, -1, -1)
    return grid_x, grid_y


def evaluate_model(model, patches_np, labels_np, batch_size, device):
    from src.neural_iris.geometry import render_soft_ellipse_mask

    total_samples = len(patches_np)
    grid_x, grid_y = build_grids(batch_size, device)

    center_errors = []
    axis_errors = []
    angle_errors_deg = []
    iou_values = []
    collision_flags_percent = []
    collision_area_percent = []
    collision_area_percent_collided = []

    def add_distribution_stats(metrics, prefix, values):
        arr = np.asarray(values, dtype=np.float64)
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        lower_fence = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        upper_fence = q3 + OUTLIER_IQR_MULTIPLIER * iqr
        inlier_mask = (arr >= lower_fence) & (arr <= upper_fence)
        inlier_arr = arr[inlier_mask]
        if inlier_arr.size == 0:
            inlier_arr = arr
        outlier_mask = ~inlier_mask

        metrics[f"{prefix}_min"] = float(np.min(arr))
        metrics[f"{prefix}_q1"] = q1
        metrics[f"{prefix}_median"] = float(np.median(arr))
        metrics[f"{prefix}_q3"] = q3
        metrics[f"{prefix}_max"] = float(np.max(arr))
        metrics[f"{prefix}_whislo"] = float(np.min(inlier_arr))
        metrics[f"{prefix}_whishi"] = float(np.max(inlier_arr))
        metrics[f"{prefix}_outlier_count"] = int(np.sum(outlier_mask))
        metrics[f"{prefix}_fliers"] = arr[outlier_mask].astype(float).tolist()

    with torch.no_grad():
        for start in tqdm(range(0, total_samples, batch_size), desc="Evaluating", leave=False):
            end = min(start + batch_size, total_samples)
            current_bs = end - start

            patches = torch.from_numpy(patches_np[start:end]).unsqueeze(1).float().to(device)
            labels = torch.from_numpy(labels_np[start:end]).float().to(device)
            preds = model(patches)

            center_diff = preds[:, 0:2] - labels[:, 0:2]
            center_errors.extend(torch.norm(center_diff, dim=1).detach().cpu().tolist())

            axis_errors.extend(torch.abs(preds[:, 2:4] - labels[:, 2:4]).mean(dim=1).detach().cpu().tolist())

            pred_theta = torch.atan2(preds[:, 4], preds[:, 5])
            target_theta = torch.atan2(labels[:, 4], labels[:, 5])
            angle_diff = torch.remainder(pred_theta - target_theta + 0.5 * math.pi, math.pi) - 0.5 * math.pi
            angle_errors_deg.extend(torch.rad2deg(torch.abs(angle_diff)).detach().cpu().tolist())

            pred_mask = render_soft_ellipse_mask(
                preds,
                grid_x[:current_bs],
                grid_y[:current_bs],
                size=128,
                temperature=100.0,
            ) > 0.5
            target_mask = render_soft_ellipse_mask(
                labels,
                grid_x[:current_bs],
                grid_y[:current_bs],
                size=128,
                temperature=100.0,
            ) > 0.5

            inter = (pred_mask & target_mask).float().sum(dim=(1, 2))
            uni = (pred_mask | target_mask).float().sum(dim=(1, 2))
            hard_iou = (inter + 1e-6) / (uni + 1e-6)
            iou_values.extend((hard_iou * 100.0).detach().cpu().tolist())

            obs_mask = patches.squeeze(1) > 0.5
            collision_pixels = (pred_mask & obs_mask).float().sum(dim=(1, 2))
            pred_area = pred_mask.float().sum(dim=(1, 2))
            collided_mask = collision_pixels > 0
            collision_flags_percent.extend((collided_mask.float() * 100.0).detach().cpu().tolist())
            collision_ratio = collision_pixels / (pred_area + 1e-6)
            collision_area_percent_batch = (collision_ratio * 100.0).detach().cpu().tolist()
            collision_area_percent.extend(collision_area_percent_batch)
            collision_area_percent_collided.extend((collision_ratio[collided_mask] * 100.0).detach().cpu().tolist())

    metrics = {
        "sample_count": total_samples,
        "outlier_iqr_multiplier": OUTLIER_IQR_MULTIPLIER,
        "center_error_px": float(np.mean(center_errors)),
        "axis_error_px": float(np.mean(axis_errors)),
        "angle_error_deg": float(np.mean(angle_errors_deg)),
        "iou": float(np.mean(iou_values) / 100.0),
        "iou_percent": float(np.mean(iou_values)),
        "collision_rate": float(np.mean(collision_flags_percent)),
        "collision_count": int(np.sum(np.asarray(collision_flags_percent) > 0.0)),
        "collision_area_ratio": float(np.mean(collision_area_percent) / 100.0),
        "collision_area_percent": float(np.mean(collision_area_percent)),
        "collision_area_ratio_collided": float(np.mean(collision_area_percent_collided) / 100.0) if collision_area_percent_collided else 0.0,
        "collision_area_percent_collided": float(np.mean(collision_area_percent_collided)) if collision_area_percent_collided else 0.0,
        "collision_area_collided_sample_count": int(len(collision_area_percent_collided)),
    }
    add_distribution_stats(metrics, "center_error_px", center_errors)
    add_distribution_stats(metrics, "axis_error_px", axis_errors)
    add_distribution_stats(metrics, "angle_error_deg", angle_errors_deg)
    add_distribution_stats(metrics, "iou_percent", iou_values)
    add_distribution_stats(metrics, "collision_rate", collision_flags_percent)
    add_distribution_stats(metrics, "collision_area_percent", collision_area_percent)
    if collision_area_percent_collided:
        add_distribution_stats(metrics, "collision_area_percent_collided", collision_area_percent_collided)
    else:
        for suffix, value in {
            "min": 0.0,
            "q1": 0.0,
            "median": 0.0,
            "q3": 0.0,
            "max": 0.0,
            "whislo": 0.0,
            "whishi": 0.0,
            "outlier_count": 0,
            "fliers": [],
        }.items():
            metrics[f"collision_area_percent_collided_{suffix}"] = value
    return metrics


def make_output_dir(output_dir):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.abspath(output_dir)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved = os.path.join(DEFAULT_OUTPUT_ROOT, run_name)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def save_metrics_csv(metrics, csv_path):
    serialized = {}
    for key, value in metrics.items():
        if isinstance(value, list):
            serialized[key] = json.dumps(value)
        else:
            serialized[key] = value
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(serialized.keys()))
        writer.writeheader()
        writer.writerow(serialized)


def save_metrics_json(metrics, json_path, model_path, data_path):
    payload = dict(metrics)
    payload["model_path"] = os.path.abspath(model_path)
    payload["data_path"] = os.path.abspath(data_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_summary_plot(metrics, plot_path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 17,
        "axes.titlesize": 20,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.linewidth": 1.15,
    })

    spec = [
        ("Center Error\n/ 128 px", "center_error_px", PATCH_SIZE, "#4c78a8", False),
        ("Axis Error\n/ 128 px", "axis_error_px", PATCH_SIZE, "#e45756", False),
        ("Angle Error\n/ 90 deg", "angle_error_deg", ANGLE_RANGE_DEG, "#7b6fd0", False),
        ("Overlap Error\n(1 - IoU)", "iou_percent", 100.0, "#2e8b57", True),
        ("Collision Ratio\n(collided only)", "collision_area_percent_collided", 100.0, "#e08d2d", False),
    ]
    items = []
    for label, prefix, scale, color, invert in spec:
        whislo = metrics[f"{prefix}_whislo"] / scale
        q1 = metrics[f"{prefix}_q1"] / scale
        med = metrics[f"{prefix}_median"] / scale
        q3 = metrics[f"{prefix}_q3"] / scale
        whishi = metrics[f"{prefix}_whishi"] / scale
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
        items.append(
            {
                "label": label,
                "color": color,
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

    fig.suptitle("Normalized Final Test Metrics", y=0.985, fontsize=20)
    top_text = (
        f"N={metrics['sample_count']} | Collision Rate: {metrics['collision_rate']:.2f}% | Collided Samples: {metrics['collision_count']}/{metrics['sample_count']}\n"
        f"Overlap Error = 1 - IoU | Collision Ratio = collision area / predicted ellipse area\n"
        f"Whiskers: {OUTLIER_IQR_MULTIPLIER:.1f} IQR | Dots: outliers"
    )
    fig.text(0.5, 0.945, top_text, ha="center", va="top", fontsize=14.6, color="#555555")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.84, hspace=0.05)
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Final Neural-IRIS evaluation on the held-out test split.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to model weights.")
    parser.add_argument("--data-path", default=DEFAULT_TEST_PATH, help="Path to test_iris.npz.")
    parser.add_argument("--output-dir", default=None, help="Directory to save JSON/CSV/PNG outputs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Evaluation batch size.")
    parser.add_argument("--num-samples", type=int, default=None, help="Optional subset size for a quick dry run.")
    parser.add_argument("--device", default=None, help="Force device, e.g. cuda or cpu.")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Evaluating on device: {device}")

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(f"Test dataset not found: {args.data_path}")

    output_dir = make_output_dir(args.output_dir)
    model = load_model(args.model_path, device)
    patches_np, labels_np = load_dataset(args.data_path, num_samples=args.num_samples)
    metrics = evaluate_model(model, patches_np, labels_np, args.batch_size, device)

    csv_path = os.path.join(output_dir, "test_metrics.csv")
    json_path = os.path.join(output_dir, "test_metrics.json")
    plot_path = os.path.join(output_dir, "test_metrics.png")

    save_metrics_csv(metrics, csv_path)
    save_metrics_json(metrics, json_path, args.model_path, args.data_path)
    save_summary_plot(metrics, plot_path)

    print("Final test metrics:")
    print(f"  Samples        : {metrics['sample_count']}")
    print(f"  Center Error   : {metrics['center_error_px']:.2f} px")
    print(f"  Axis Error     : {metrics['axis_error_px']:.2f} px")
    print(f"  Angle Error    : {metrics['angle_error_deg']:.2f} deg")
    print(f"  IoU            : {metrics['iou_percent']:.2f}%")
    print(f"  Collision Rate : {metrics['collision_rate']:.2f}%")
    print(f"  Collision Area : {metrics['collision_area_percent']:.2f}%")
    print(f"  Collision Area (Collided Only): {metrics['collision_area_percent_collided']:.2f}%")
    print(f"Saved CSV  : {csv_path}")
    print(f"Saved JSON : {json_path}")
    print(f"Saved PNG  : {plot_path}")


if __name__ == "__main__":
    main()
