import os
import shutil
import subprocess

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

from experiment.common.geometry_utils import fallback_patch_box, obstacle_boundary


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CPP_DIR = os.path.join(_THIS_DIR, "cpp")
_THIRD_PARTY_GCOPTER = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "third_party", "GCOPTER"))


def _is_wsl() -> bool:
    if os.name != "posix":
        return False
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _build_paths():
    if _is_wsl() and _THIS_DIR.startswith("/mnt/"):
        cache_root = os.path.join(os.path.expanduser("~"), ".cache", "ggmpc", "firi")
        build_dir = os.path.join(cache_root, "build")
    else:
        build_dir = os.path.join(_THIS_DIR, "build")
    bin_name = "firi_cli.exe" if os.name == "nt" else "firi_cli"
    return build_dir, os.path.join(build_dir, bin_name)


def _ensure_built():
    build_dir, bin_path = _build_paths()
    if os.path.isfile(bin_path):
        return

    if shutil.which("cmake") is None or shutil.which("g++") is None:
        raise NotImplementedError(
            "FIRI 需要 cmake 与 g++。请在 WSL 执行: sudo apt update && sudo apt install -y cmake build-essential libeigen3-dev"
        )

    if not os.path.isdir(_THIRD_PARTY_GCOPTER):
        raise NotImplementedError("GCOPTER 未找到，请先克隆到 third_party/GCOPTER。")

    os.makedirs(build_dir, exist_ok=True)

    cfg = subprocess.run(
        ["cmake", "-S", _CPP_DIR, "-B", build_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if cfg.returncode != 0:
        raise NotImplementedError("FIRI C++桥接配置失败。输出:\n" + cfg.stdout)

    build = subprocess.run(
        ["cmake", "--build", build_dir, "-j"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if build.returncode != 0 or (not os.path.isfile(bin_path)):
        raise NotImplementedError("FIRI C++桥接编译失败。输出:\n" + build.stdout)


def _halfspaces_to_vertices(A: np.ndarray, b: np.ndarray, interior: np.ndarray):
    hs = np.hstack((A, -b[:, None]))
    inter = HalfspaceIntersection(hs, interior)
    pts = inter.intersections
    if len(pts) < 3:
        return None
    hull = ConvexHull(pts)
    return pts[hull.vertices]


def infer_polygon(obs_mask, center=(64.0, 64.0), patch_size=128):
    _ensure_built()
    _, bin_path = _build_paths()

    center = np.asarray(center, dtype=float)
    obs_pts = obstacle_boundary(obs_mask)

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
        raise RuntimeError("firi_cli failed: " + run.stdout.strip())

    out = [ln.strip() for ln in run.stdout.splitlines() if ln.strip()]
    if not out:
        return fallback_patch_box(patch_size)

    head = out[0].split()
    if len(head) < 2 or head[0] != "OK":
        return fallback_patch_box(patch_size)

    n = int(head[1])
    if n < 3 or len(out) < (n + 1):
        return fallback_patch_box(patch_size)

    A = np.zeros((n, 2), dtype=float)
    b = np.zeros(n, dtype=float)
    for i in range(n):
        parts = out[i + 1].split()
        if len(parts) < 3:
            return fallback_patch_box(patch_size)
        nx, ny, d = float(parts[0]), float(parts[1]), float(parts[2])
        A[i, 0] = nx
        A[i, 1] = ny
        b[i] = -d

    if np.any(A @ center > b - 1e-6):
        margin = (A @ center - b).max()
        b = b + margin + 1.0

    try:
        vertices = _halfspaces_to_vertices(A, b, center)
    except Exception:
        return fallback_patch_box(patch_size)

    if vertices is None or len(vertices) < 3:
        return fallback_patch_box(patch_size)

    vertices[:, 0] = np.clip(vertices[:, 0], 0.0, patch_size - 1.0)
    vertices[:, 1] = np.clip(vertices[:, 1], 0.0, patch_size - 1.0)
    return vertices
