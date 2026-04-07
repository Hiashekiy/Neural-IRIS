import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiment.common.geometry_utils import raycast_distance


def _radial_to_ellipse_from_point(center_point, ellipse_center, a, b, theta, angles):
    cp = np.asarray(center_point, dtype=float)
    ec = np.asarray(ellipse_center, dtype=float)

    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=float)

    y0 = R.T @ (cp - ec)

    radii = np.zeros(len(angles), dtype=float)
    for i, ang in enumerate(angles):
        u = np.array([np.cos(ang), np.sin(ang)], dtype=float)
        v = R.T @ u

        A = (v[0] / max(a, 1e-6)) ** 2 + (v[1] / max(b, 1e-6)) ** 2
        B = 2.0 * ((y0[0] * v[0]) / max(a * a, 1e-8) + (y0[1] * v[1]) / max(b * b, 1e-8))
        C = (y0[0] / max(a, 1e-6)) ** 2 + (y0[1] / max(b, 1e-6)) ** 2 - 1.0

        disc = B * B - 4.0 * A * C
        if A < 1e-12 or disc < 0.0:
            radii[i] = 1.0
            continue

        sq = np.sqrt(max(disc, 0.0))
        t1 = (-B - sq) / (2.0 * A)
        t2 = (-B + sq) / (2.0 * A)
        candidates = [t for t in (t1, t2) if t > 1e-6]
        radii[i] = min(candidates) if candidates else 1.0

    return np.maximum(radii, 1.0)


def convert_one_split(in_path: str, out_path: str, k_dirs: int = 32, patch_size: int = 128, margin: float = 0.8):
    print(f"[prepare_labels] loading: {in_path}")
    data = np.load(in_path)
    patches = data["patches"]
    labels = data["labels"]

    center = np.array([patch_size / 2.0, patch_size / 2.0], dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, k_dirs, endpoint=False)

    radial_labels = np.zeros((len(patches), k_dirs), dtype=np.float32)

    pbar = tqdm(range(len(patches)), desc=f"convert {os.path.basename(in_path)}", leave=False)
    for i in pbar:
        dx, dy, a, b, sin_t, cos_t = labels[i]
        theta = float(np.arctan2(sin_t, cos_t))
        ellipse_center = np.array([center[0] + dx, center[1] + dy], dtype=float)

        r_ellipse = _radial_to_ellipse_from_point(center, ellipse_center, float(a), float(b), theta, angles)

        obs_mask = patches[i] > 0.5
        r_obs = np.array([raycast_distance(obs_mask, center, ang, max_dist=90.0) for ang in angles], dtype=float)
        r_obs = np.maximum(r_obs - margin, 1.0)

        r_safe = np.minimum(r_ellipse, r_obs)
        radial_labels[i] = r_safe.astype(np.float32)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, patches=patches.astype(np.uint8), radial_labels=radial_labels, angles=angles.astype(np.float32))
    print(f"[prepare_labels] saved: {out_path} | patches={patches.shape} radial_labels={radial_labels.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_dirs", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.8)
    parser.add_argument("--patch_size", type=int, default=128)
    args = parser.parse_args()

    base = os.path.join(ROOT, "data", "iris-dataset", "splits")
    for name in ("train", "val", "test"):
        print(f"\n[prepare_labels] processing split: {name}")
        in_path = os.path.join(base, f"{name}_iris.npz")
        out_path = os.path.join(base, f"{name}_radial_k{args.k_dirs}.npz")
        convert_one_split(in_path, out_path, k_dirs=args.k_dirs, patch_size=args.patch_size, margin=args.margin)

    print("\n[prepare_labels] all splits completed.")


if __name__ == "__main__":
    main()
