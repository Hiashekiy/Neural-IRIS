import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_LOG_ROOT = os.path.join(ROOT, "logs", "neural_iris_train")


def resolve_paths(input_path, output_path):
    if input_path is None:
        run_dirs = sorted(
            [
                os.path.join(DEFAULT_LOG_ROOT, name)
                for name in os.listdir(DEFAULT_LOG_ROOT)
                if os.path.isdir(os.path.join(DEFAULT_LOG_ROOT, name))
            ]
        )
        if not run_dirs:
            raise FileNotFoundError(f"No training runs found under {DEFAULT_LOG_ROOT}")
        run_dir = run_dirs[-1]
        csv_path = os.path.join(run_dir, "metrics.csv")
    elif os.path.isdir(input_path):
        run_dir = input_path
        csv_path = os.path.join(run_dir, "metrics.csv")
    else:
        csv_path = input_path
        run_dir = os.path.dirname(csv_path)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    if output_path is None:
        output_path = os.path.join(run_dir, "metrics_replot.png")

    return os.path.abspath(csv_path), os.path.abspath(output_path)


def load_history(csv_path):
    history = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key == "epoch":
                    parsed[key] = int(float(value))
                else:
                    parsed[key] = float(value)
            if "iou_percent" not in parsed:
                parsed["iou_percent"] = parsed["iou"] * 100.0
            history.append(parsed)

    if not history:
        raise ValueError(f"No metric rows found in {csv_path}")
    return history


def save_training_curves(history, plot_path):
    epochs = [row["epoch"] for row in history]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.linewidth": 1.2,
    })

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), facecolor="white")

    def style_axis(ax):
        ax.grid(False)
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.tick_params(axis="both", width=1.1, length=5)

    def set_lr_axis_style(ax, lr_values):
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_linewidth(1.2)
        ax.tick_params(axis="y", width=1.1, length=5)
        ax.set_yscale("log")
        lr_min = max(min(lr_values), 1e-12)
        lr_max = max(lr_values)
        ax.set_ylim(lr_min / 1.8, lr_max * 1.35)

    ax_loss = axes[0, 0]
    loss_lines = []
    loss_lines += ax_loss.plot(epochs, [row["train_loss"] for row in history], label="Train Loss", linewidth=2.6, color="#1f77b4")
    loss_lines += ax_loss.plot(epochs, [row["val_loss"] for row in history], label="Validation Loss", linewidth=2.6, color="#d95f02")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    style_axis(ax_loss)
    ax_loss.legend(
        loss_lines,
        [line.get_label() for line in loss_lines],
        loc="upper right",
        frameon=False,
        fancybox=False,
    )

    axes[0, 1].plot(epochs, [row["iou_percent"] for row in history], label="IoU (%)", linewidth=2.6, color="#1f77b4")
    axes[0, 1].plot(epochs, [row["collision_rate"] for row in history], label="Collision Rate (%)", linewidth=2.6, color="#d62728")
    axes[0, 1].set_title("Overlap and Collision")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Percent")
    style_axis(axes[0, 1])
    axes[0, 1].legend(loc="upper right", frameon=False, fancybox=False)

    axes[1, 0].plot(epochs, [row["center_error_px"] for row in history], label="Center Error", linewidth=2.8, color="#1f77b4")
    axes[1, 0].plot(epochs, [row["axis_error_px"] for row in history], label="Axis Error", linewidth=2.8, color="#e31a1c")
    axes[1, 0].set_title("Geometric Errors")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Pixels")
    style_axis(axes[1, 0])
    axes[1, 0].legend(loc="upper right", frameon=False, fancybox=False)

    ax_angle = axes[1, 1]
    ax_angle_lr = ax_angle.twinx()
    angle_lines = []
    angle_lines += ax_angle.plot(epochs, [row["angle_error_deg"] for row in history], label="Angle Error (deg)", linewidth=2.6, color="#4c78a8")
    angle_lines += ax_angle_lr.plot(
        epochs,
        [row["lr"] for row in history],
        label="Learning Rate",
        linewidth=2.4,
        color="#f58518",
        linestyle="--",
    )
    ax_angle.set_title("Angle Error and Learning Rate")
    ax_angle.set_xlabel("Epoch")
    ax_angle.set_ylabel("Degrees")
    ax_angle_lr.set_ylabel("Learning Rate")
    style_axis(ax_angle)
    set_lr_axis_style(ax_angle_lr, [row["lr"] for row in history])
    ax_angle.legend(
        angle_lines,
        [line.get_label() for line in angle_lines],
        loc="upper right",
        frameon=False,
        fancybox=False,
    )

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Replot Neural-IRIS training metrics from an existing CSV.")
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to metrics.csv or a run directory. Defaults to the latest run under logs/neural_iris_train.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to <run_dir>/metrics_replot.png.",
    )
    args = parser.parse_args()

    csv_path, plot_path = resolve_paths(args.input, args.output)
    history = load_history(csv_path)
    save_training_curves(history, plot_path)

    print(f"Loaded metrics CSV: {csv_path}")
    print(f"Saved plot PNG: {plot_path}")


if __name__ == "__main__":
    main()
