# -*- coding: utf-8 -*-
"""Download pretrained weights / dataset / ONNX model from GitHub Releases.

Before publishing, create a GitHub Release and replace the RELEASE_BASE
placeholder below with the real download base URL, then run:

    python scripts/download_assets.py             # model weights (.pth)
    python scripts/download_assets.py --onnx      # ONNX model for the C++ backend
    python scripts/download_assets.py --dataset   # train/val/test splits (zip)

The same commands are documented in README.md.
"""

import argparse
import os
import sys
import tempfile
import urllib.request
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# TODO: 创建 GitHub Release 后，把占位地址替换为真实地址，例如：
#   https://github.com/<用户名>/<仓库名>/releases/download/v1.0.0
RELEASE_BASE = "https://github.com/Hiashekiy/Neural-IRIS/releases/download/v1.0.0"

ASSETS = {
    "weights": {
        "url": f"{RELEASE_BASE}/neural_iris_net_best.pth",
        "path": os.path.join("models", "neural_iris_net_best.pth"),
    },
    "onnx": {
        "url": f"{RELEASE_BASE}/neural_iris_net.onnx",
        "path": os.path.join("cpp", "models", "neural_iris_net.onnx"),
    },
    "dataset": {
        "url": f"{RELEASE_BASE}/iris_dataset_splits.zip",
        "path": os.path.join("data", "iris-dataset", "splits"),
        "archive": True,
    },
}


def _download(url, dest):
    print(f"Downloading {url}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".part")
    os.close(tmp_fd)
    try:
        urllib.request.urlretrieve(url, tmp_path)  # noqa: S310
        os.replace(tmp_path, dest)
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Download failed: {exc}") from exc
    print(f"Saved to {os.path.relpath(dest, ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Download Neural-IRIS release assets.")
    parser.add_argument("--weights", action="store_true", help="Download model weights (.pth)")
    parser.add_argument("--onnx", action="store_true", help="Download ONNX model for C++ backend")
    parser.add_argument("--dataset", action="store_true", help="Download dataset splits (zip)")
    args = parser.parse_args()

    if not (args.weights or args.onnx or args.dataset):
        args.weights = True

    if "<YOUR_USER>" in RELEASE_BASE:
        print("Release download URL is not configured yet.")
        print("Please create a GitHub Release and replace RELEASE_BASE at the top of")
        print("scripts/download_assets.py with the real URL, e.g.:")
        print("  https://github.com/<username>/<repo>/releases/download/v1.0.0")
        sys.exit(1)

    for key in ("weights", "onnx", "dataset"):
        if not getattr(args, key):
            continue
        asset = ASSETS[key]
        if asset.get("archive"):
            zip_path = os.path.join(tempfile.gettempdir(), "neural_iris_dataset_splits.zip")
            _download(asset["url"], zip_path)
            os.makedirs(asset["path"], exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(asset["path"])
            # 兼容 zip 内带顶层目录的情况：把散落的 npz 统一提升到目标目录
            for root, _, files in os.walk(asset["path"]):
                for fn in files:
                    if fn.endswith(".npz"):
                        src = os.path.join(root, fn)
                        dst = os.path.join(asset["path"], fn)
                        if os.path.abspath(src) != os.path.abspath(dst):
                            os.replace(src, dst)
            print(f"Extracted dataset to {os.path.relpath(asset['path'], ROOT)}")
            os.remove(zip_path)
        else:
            _download(asset["url"], os.path.join(ROOT, asset["path"]))


if __name__ == "__main__":
    main()
