import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np


def ellipse_points(center, a, b, theta, num_points=200):
    angles = np.linspace(0.0, 2.0 * np.pi, num_points)
    circle = np.vstack((np.cos(angles), np.sin(angles)))
    scale = np.array([[a, 0.0], [0.0, b]], dtype=float)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ], dtype=float)
    points = (rotation @ scale @ circle).T + np.asarray(center, dtype=float)
    return points


def draw_sample(ax, patch, label, title_prefix=""):
    patch_size = patch.shape[0]
    dx, dy, a, b, sin_theta, cos_theta = label.astype(float)
    theta = np.arctan2(sin_theta, cos_theta)

    center = np.array([patch_size / 2.0 + dx, patch_size / 2.0 + dy], dtype=float)
    ell = ellipse_points(center, a, b, theta)

    ax.imshow(patch, cmap="gray_r", origin="upper", interpolation="nearest")
    ax.plot(ell[:, 0], ell[:, 1], color="#ff4d4d", linewidth=2)
    ax.scatter([center[0]], [center[1]], c="#00d4ff", s=18, marker="x", linewidths=1.5)
    ax.set_xlim(0, patch_size - 1)
    ax.set_ylim(patch_size - 1, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{title_prefix}dx={dx:.1f}, dy={dy:.1f}, a={a:.2f}, b={b:.2f}",
        fontsize=9,
    )


class DatasetViewer:
    def __init__(self, patches, labels, batch_size=6, cols=3, seed=None):
        self.patches = patches
        self.labels = labels
        self.batch_size = batch_size
        self.cols = max(1, cols)
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.patches))
        self.cursor = 0
        self.shuffle_order()

        self.fig = None
        self.axes = None

    def shuffle_order(self):
        self.rng.shuffle(self.order)
        self.cursor = 0

    def next_batch_indices(self):
        if len(self.order) == 0:
            return np.array([], dtype=int)

        if self.cursor + self.batch_size > len(self.order):
            self.shuffle_order()

        batch = self.order[self.cursor:self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return batch

    def render_batch(self):
        indices = self.next_batch_indices()
        if len(indices) == 0:
            return

        rows = math.ceil(len(indices) / self.cols)
        if self.fig is None or self.axes is None:
            self.fig, self.axes = plt.subplots(
                rows,
                self.cols,
                figsize=(4.2 * self.cols, 4.2 * rows),
                squeeze=False,
            )
            self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        else:
            self.fig.clf()
            self.axes = self.fig.subplots(rows, self.cols, squeeze=False)

        self.fig.suptitle(
            "Random Dataset Viewer - press ENTER for next batch, Q/Esc to quit",
            fontsize=14,
            fontweight="bold",
        )

        flat_axes = self.axes.ravel()
        for ax in flat_axes:
            ax.axis("off")

        for plot_idx, sample_idx in enumerate(indices):
            ax = flat_axes[plot_idx]
            ax.axis("on")
            draw_sample(
                ax,
                self.patches[sample_idx],
                self.labels[sample_idx],
                title_prefix=f"#{sample_idx} ",
            )

        self.fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key == "enter":
            self.render_batch()
        elif event.key in {"q", "escape"}:
            plt.close(self.fig)


def load_dataset(npz_path):
    data = np.load(npz_path)
    if "patches" not in data or "labels" not in data:
        raise ValueError("Dataset must contain 'patches' and 'labels' arrays.")

    patches = data["patches"]
    labels = data["labels"]
    if len(patches) != len(labels):
        raise ValueError("'patches' and 'labels' must have the same length.")

    return patches, labels


def main():
    parser = argparse.ArgumentParser(description="Randomly visualize IRIS dataset samples.")
    parser.add_argument("dataset", nargs="?", default="data\\iris-dataset\\full_iris_dataset.npz", help="Path to the .npz dataset")
    parser.add_argument("--batch-size", type=int, default=6, help="Number of samples shown per batch")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in the grid")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for batch ordering")
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset)
    patches, labels = load_dataset(dataset_path)

    if len(patches) == 0:
        raise ValueError(f"Dataset {dataset_path} is empty.")

    viewer = DatasetViewer(
        patches=patches,
        labels=labels,
        batch_size=args.batch_size,
        cols=args.cols,
        seed=args.seed,
    )
    viewer.render_batch()
    plt.show()


if __name__ == "__main__":
    main()