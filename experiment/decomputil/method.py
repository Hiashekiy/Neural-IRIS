import os
import shutil
import subprocess

import numpy as np
from scipy.spatial import ConvexHull

from experiment.common.geometry_utils import fallback_patch_box, obstacle_boundary


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CPP_DIR = os.path.join(_THIS_DIR, "cpp")
_THIRD_PARTY_DECOMP = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "third_party", "DecompUtil"))


def _is_wsl() -> bool:
    if os.name != "posix":
        return False
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _build_paths():
    # 在 WSL 中不要把 CMake 构建目录放在 /mnt/*，否则常见 Operation not permitted。
    if _is_wsl() and _THIS_DIR.startswith("/mnt/"):
        cache_root = os.path.join(os.path.expanduser("~"), ".cache", "ggmpc", "decomputil")
        build_dir = os.path.join(cache_root, "build")
    else:
        build_dir = os.path.join(_THIS_DIR, "build")
    bin_name = "decomp_cli.exe" if os.name == "nt" else "decomp_cli"
    return build_dir, os.path.join(build_dir, bin_name)


def _ensure_built():
    build_dir, bin_path = _build_paths()

    if os.path.isfile(bin_path):
        return

    if shutil.which("cmake") is None:
        raise NotImplementedError(
            "DecompUtil 需要 cmake。请在 WSL 执行: sudo apt update && sudo apt install -y cmake build-essential libeigen3-dev"
        )

    if shutil.which("g++") is None:
        raise NotImplementedError(
            "DecompUtil 需要 g++。请在 WSL 执行: sudo apt update && sudo apt install -y build-essential"
        )

    if not os.path.isdir(_THIRD_PARTY_DECOMP):
        raise NotImplementedError(
            "DecompUtil 未找到，请先克隆到 third_party/DecompUtil。"
        )

    os.makedirs(build_dir, exist_ok=True)

    try:
        cfg = subprocess.run(
            ["cmake", "-S", _CPP_DIR, "-B", build_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise NotImplementedError(
            "DecompUtil 需要 cmake。请在 WSL 执行: sudo apt update && sudo apt install -y cmake build-essential libeigen3-dev"
        )
    if cfg.returncode != 0:
        raise NotImplementedError(
            "DecompUtil C++桥接配置失败，请检查 CMake/g++/Eigen。输出:\n"
            + cfg.stdout
        )

    try:
        build = subprocess.run(
            ["cmake", "--build", build_dir, "-j"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise NotImplementedError(
            "DecompUtil 需要 cmake。请在 WSL 执行: sudo apt update && sudo apt install -y cmake build-essential libeigen3-dev"
        )
    if build.returncode != 0 or (not os.path.isfile(bin_path)):
        raise NotImplementedError(
            "DecompUtil C++桥接编译失败。输出:\n" + build.stdout
        )


def infer_polygon(obs_mask, center=(64.0, 64.0), patch_size=128):
    _ensure_built()
    _, bin_path = _build_paths()

    center = np.asarray(center, dtype=float)
    obs_pts = obstacle_boundary(obs_mask)

    # 限制点数，避免边界极密时 DecompUtil 过慢。
    if len(obs_pts) > 1200:
        idx = np.linspace(0, len(obs_pts) - 1, 1200, dtype=int)
        obs_pts = obs_pts[idx]

    lines = [
        f"{float(center[0])} {float(center[1])}",
        str(int(patch_size)),
        str(int(len(obs_pts))),
    ]
    lines.extend([f"{float(p[0])} {float(p[1])}" for p in obs_pts])
    payload = "\n".join(lines) + "\n"

    run = subprocess.run(
        [bin_path],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if run.returncode != 0:
        raise RuntimeError("decomp_cli failed: " + run.stdout.strip())

    out = [ln.strip() for ln in run.stdout.splitlines() if ln.strip()]
    if not out:
        return fallback_patch_box(patch_size)

    head = out[0].split()
    if len(head) < 2 or head[0] != "OK":
        return fallback_patch_box(patch_size)

    n = int(head[1])
    if n < 3 or len(out) < (n + 1):
        return fallback_patch_box(patch_size)

    vertices = []
    for i in range(1, n + 1):
        parts = out[i].split()
        if len(parts) < 2:
            return fallback_patch_box(patch_size)
        vertices.append([float(parts[0]), float(parts[1])])

    vertices = np.asarray(vertices, dtype=float)
    vertices[:, 0] = np.clip(vertices[:, 0], 0.0, patch_size - 1.0)
    vertices[:, 1] = np.clip(vertices[:, 1], 0.0, patch_size - 1.0)

    # 去重+凸包，避免 DecompUtil 偶发输出重复/近共线点导致后续半空间转换失败。
    uniq = np.unique(np.round(vertices, 6), axis=0)
    if len(uniq) < 3:
        return fallback_patch_box(patch_size)
    try:
        hull = ConvexHull(uniq)
        vertices = uniq[hull.vertices]
    except Exception:
        return fallback_patch_box(patch_size)

    return vertices
