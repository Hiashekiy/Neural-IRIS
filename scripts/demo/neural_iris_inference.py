import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.neural_iris import infer_safe_region as py_infer_safe_region
from src.neural_iris.geometry import get_ellipse_points, parse_neural_iris_output

try:
    from cpp.python import infer_safe_region as cpp_infer_safe_region
except Exception:
    cpp_infer_safe_region = None

DATA_PATH = os.path.join(root_path, "data", "iris-dataset", "splits", "test_iris.npz")


def resolve_backend(backend):
    if backend == "cpp":
        if cpp_infer_safe_region is None:
            raise RuntimeError("C++ Neural-IRIS backend is unavailable.")
        return cpp_infer_safe_region, "cpp"
    return py_infer_safe_region, "python"


def render_sample(patch, target_label, pred_polygon, pred_p, pred_c, idx, save_path=None, show_gui=False, backend_name="python"):
    patch_size = patch.shape[0]
    p_gt, c_gt = parse_neural_iris_output(target_label, patch_size)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, patch_size)
    ax.set_ylim(0, patch_size)
    ax.set_aspect("equal")

    ax.imshow(patch, cmap="Greys", origin="lower", extent=[0, patch_size, 0, patch_size], alpha=0.3)

    py, px = np.where(patch > 0)
    if len(px) > 0:
        obs_points = np.column_stack((px, py)).astype(float)
        ax.scatter(obs_points[:, 0], obs_points[:, 1], c="black", s=10, marker="s", label="Obstacles")

    gt_ellipse = get_ellipse_points(p_gt, c_gt)
    ax.plot(gt_ellipse[:, 0], gt_ellipse[:, 1], color="blue", linestyle="--", linewidth=2, label="GT Ellipse (IRIS)")
    ax.scatter(c_gt[0], c_gt[1], color="blue", marker="+", s=100)

    if pred_p is not None and pred_c is not None:
        pred_ellipse = get_ellipse_points(pred_p, pred_c)
        ax.plot(pred_ellipse[:, 0], pred_ellipse[:, 1], color="red", linestyle="-", linewidth=2, label=f"Predicted Ellipse ({backend_name})")
        ax.scatter(pred_c[0], pred_c[1], color="red", marker="x", s=100)

    if pred_polygon is not None and len(pred_polygon) >= 3:
        poly = plt.Polygon(
            pred_polygon,
            facecolor="lightgreen",
            edgecolor="green",
            alpha=0.5,
            linewidth=2,
            label=f"Safe Region ({backend_name})",
        )
        ax.add_patch(poly)

    ax.set_title(f"Test Sample {idx} - Neural-IRIS Inference", fontweight="bold")
    ax.legend(loc="upper right")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show_gui:
        plt.show()

    plt.close(fig)


def run_inference(mode="headless", num_samples=5, backend="python"):
    infer_safe_region, backend_name = resolve_backend(backend)

    test_data_path = DATA_PATH
    if not os.path.exists(test_data_path):
        print(f"Error: {test_data_path} not found!")
        return

    print(f"Loading test dataset for backend={backend_name}...")
    data = np.load(test_data_path)
    patches = data["patches"]
    labels = data["labels"]
    total_test = len(patches)

    if num_samples == -1 or num_samples >= total_test:
        print(f"Testing all {total_test} samples in the dataset...")
        num_samples = total_test
        sample_indices = np.arange(total_test)
    else:
        sample_indices = np.random.choice(total_test, num_samples, replace=False)

    output_dir = os.path.join(root_path, "inference_results", backend_name)
    if mode == "headless":
        os.makedirs(output_dir, exist_ok=True)
        print(f"Running headless mode. Saving {num_samples} figures to {output_dir}")

    for i, idx in enumerate(sample_indices):
        patch_np = patches[idx]
        label_np = labels[idx]
        pred_polygon, pred_p, pred_c = infer_safe_region(patch_np, patch_size=patch_np.shape[0])

        if mode == "headless":
            save_path = os.path.join(output_dir, f"sample_{i:06d}_idx_{idx}.png")
            render_sample(
                patch_np,
                label_np,
                pred_polygon,
                pred_p,
                pred_c,
                idx,
                save_path=save_path,
                show_gui=False,
                backend_name=backend_name,
            )
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_samples} samples...")
        else:
            print(f"Displaying sample {i + 1}/{num_samples} (index: {idx}, backend={backend_name}).")
            render_sample(
                patch_np,
                label_np,
                pred_polygon,
                pred_p,
                pred_c,
                idx,
                save_path=None,
                show_gui=True,
                backend_name=backend_name,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural-IRIS unified inference demo")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["python", "cpp"],
        default="python",
        help="Select the Neural-IRIS inference backend.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["headless", "gui"],
        default="gui",
        help="Choose 'headless' to save images to disk, or 'gui' to show plot windows.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="Number of samples to infer and visualize. Use -1 for all samples.",
    )
    args = parser.parse_args()

    run_inference(mode=args.mode, num_samples=args.num, backend=args.backend)
