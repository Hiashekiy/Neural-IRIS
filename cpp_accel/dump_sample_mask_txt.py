import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main(idx=0):
    npz_path = os.path.join(ROOT, "data", "iris-dataset", "splits", "test_iris.npz")
    d = np.load(npz_path)
    patches = d["patches"]

    idx = int(np.clip(idx, 0, len(patches) - 1))
    obs = (patches[idx] > 0.5).astype(np.uint8)

    out_dir = os.path.join(ROOT, "cpp_accel")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sample_mask_{idx}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        for y in range(obs.shape[0]):
            line = "".join("1" if obs[y, x] else "0" for x in range(obs.shape[1]))
            f.write(line + "\n")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    idx = 0
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
    main(idx)
