import atexit
import ctypes
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_LIB = None
_READY = False
_HAS_BATCH_WITH_ELLIPSE = False
_HAS_BATCH_HALFSPACES = False
_MAX_VERTS = 128
_MAX_HALFSPACES = 256
BATCH_SIZE = 64


def _candidate_paths():
    base = os.path.join(_ROOT, "cpp")
    mode = os.environ.get("NEURAL_IRIS_CPP_BACKEND", "").strip().lower()
    if mode == "cpu":
        return [
            os.path.join(base, "build_cpu", "Release", "neural_iris_cpp_bridge.dll"),
            os.path.join(base, "build", "Release", "neural_iris_cpp_bridge.dll"),
        ]
    if mode == "gpu":
        return [
            os.path.join(base, "build_gpu", "Release", "neural_iris_cpp_bridge.dll"),
            os.path.join(base, "build", "Release", "neural_iris_cpp_bridge.dll"),
        ]
    return [
        os.path.join(base, "build_gpu", "Release", "neural_iris_cpp_bridge.dll"),
        os.path.join(base, "build_cpu", "Release", "neural_iris_cpp_bridge.dll"),
        os.path.join(base, "build", "Release", "neural_iris_cpp_bridge.dll"),
    ]


def _find_lib():
    for path in _candidate_paths():
        if os.path.isfile(path):
            return path
    return None


def _lazy_init():
    global _LIB, _READY, _HAS_BATCH_WITH_ELLIPSE, _HAS_BATCH_HALFSPACES
    if _READY:
        return

    lib_path = _find_lib()
    if lib_path is None:
        raise NotImplementedError("Neural-IRIS C++ backend unavailable: neural_iris_cpp_bridge.dll not found")

    lib_dir = os.path.dirname(lib_path)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(lib_dir)

    _LIB = ctypes.CDLL(lib_path)

    _LIB.neural_iris_init.argtypes = [ctypes.c_char_p]
    _LIB.neural_iris_init.restype = ctypes.c_int

    _LIB.neural_iris_infer_safe_region.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    _LIB.neural_iris_infer_safe_region.restype = ctypes.c_int

    _LIB.neural_iris_infer_safe_region_batch.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    _LIB.neural_iris_infer_safe_region_batch.restype = ctypes.c_int

    if hasattr(_LIB, "neural_iris_infer_safe_region_batch_with_ellipse"):
        _LIB.neural_iris_infer_safe_region_batch_with_ellipse.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        _LIB.neural_iris_infer_safe_region_batch_with_ellipse.restype = ctypes.c_int
        _HAS_BATCH_WITH_ELLIPSE = True

    if hasattr(_LIB, "neural_iris_infer_safe_region_batch_halfspaces"):
        _LIB.neural_iris_infer_safe_region_batch_halfspaces.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        _LIB.neural_iris_infer_safe_region_batch_halfspaces.restype = ctypes.c_int
        _HAS_BATCH_HALFSPACES = True

    _LIB.neural_iris_shutdown.argtypes = []
    _LIB.neural_iris_shutdown.restype = None

    onnx_path = os.path.join(_ROOT, "cpp", "models", "neural_iris_net.onnx")
    if not os.path.isfile(onnx_path):
        raise NotImplementedError("Neural-IRIS C++ backend unavailable: ONNX model missing in cpp/models")

    rc = int(_LIB.neural_iris_init(onnx_path.encode("utf-8")))
    if rc != 0:
        raise NotImplementedError(f"Neural-IRIS C++ backend unavailable: neural_iris_init failed with code {rc}")

    _READY = True


def shutdown():
    global _READY, _LIB
    if not _READY or _LIB is None:
        return

    try:
        _LIB.neural_iris_shutdown()
    except Exception:
        pass
    finally:
        _READY = False


atexit.register(shutdown)


def infer_safe_region(obs_mask, center=(64.0, 64.0), patch_size=128):
    masks = np.asarray(obs_mask, dtype=np.uint8)
    if masks.ndim != 2:
        return None, None, None

    safe_regions, p_list, c_list = infer_safe_region_batch(
        masks[np.newaxis, ...],
        center=center,
        patch_size=patch_size,
    )
    return safe_regions[0], p_list[0], c_list[0]


def _polygon_to_halfspaces(poly_points):
    if poly_points is None:
        return None, None

    poly_points = np.asarray(poly_points, dtype=float)
    if poly_points.ndim != 2 or poly_points.shape[0] < 3:
        return None, None

    signed_area = 0.5 * float(
        np.sum(
            poly_points[:, 0] * np.roll(poly_points[:, 1], -1)
            - poly_points[:, 1] * np.roll(poly_points[:, 0], -1)
        )
    )
    if signed_area < 0.0:
        poly_points = poly_points[::-1]

    a_rows = []
    b_vals = []
    for i in range(len(poly_points)):
        p1 = poly_points[i]
        p2 = poly_points[(i + 1) % len(poly_points)]
        edge = p2 - p1
        length = float(np.linalg.norm(edge))
        if length < 1e-8:
            continue
        normal = np.array([edge[1], -edge[0]], dtype=float) / length
        a_rows.append(normal)
        b_vals.append(float(np.dot(normal, p1)))

    if len(a_rows) < 3:
        return None, None

    return np.asarray(a_rows, dtype=float), np.asarray(b_vals, dtype=float)


def infer_safe_region_halfspaces(obs_mask, center=(64.0, 64.0), patch_size=128):
    masks = np.asarray(obs_mask, dtype=np.uint8)
    if masks.ndim != 2:
        return None, None, None, None

    a_list, b_list, p_list, c_list = infer_safe_region_batch_halfspaces(
        masks[np.newaxis, ...],
        center=center,
        patch_size=patch_size,
    )
    return a_list[0], b_list[0], p_list[0], c_list[0]


def infer_safe_region_batch_halfspaces(obs_masks, center=(64.0, 64.0), patch_size=128):
    _lazy_init()

    masks = np.asarray(obs_masks, dtype=np.uint8)
    if masks.ndim != 3:
        return (
            [None for _ in range(len(obs_masks))],
            [None for _ in range(len(obs_masks))],
            [None for _ in range(len(obs_masks))],
            [None for _ in range(len(obs_masks))],
        )

    bs = int(masks.shape[0])
    if bs <= 0:
        return [], [], [], []

    if masks.shape[1] != patch_size or masks.shape[2] != patch_size:
        return (
            [None for _ in range(bs)],
            [None for _ in range(bs)],
            [None for _ in range(bs)],
            [None for _ in range(bs)],
        )

    if _HAS_BATCH_HALFSPACES:
        flat = np.ascontiguousarray(masks.reshape(-1))
        out_a = np.zeros((bs * _MAX_HALFSPACES * 2,), dtype=np.float64)
        out_b = np.zeros((bs * _MAX_HALFSPACES,), dtype=np.float64)
        out_counts = np.zeros((bs,), dtype=np.int32)
        out_p = np.zeros((bs * 4,), dtype=np.float64)
        out_c = np.zeros((bs * 2,), dtype=np.float64)

        rc = int(
            _LIB.neural_iris_infer_safe_region_batch_halfspaces(
                flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.c_int(bs),
                ctypes.c_int(patch_size),
                ctypes.c_double(0.5),
                out_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(_MAX_HALFSPACES),
                out_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                out_p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        )

        if rc == 0:
            a_list = []
            b_list = []
            p_list = []
            c_list = []
            for i in range(bs):
                n = int(out_counts[i])
                if n < 1:
                    a_list.append(None)
                    b_list.append(None)
                else:
                    a_off = i * _MAX_HALFSPACES * 2
                    b_off = i * _MAX_HALFSPACES
                    a_list.append(out_a[a_off : a_off + 2 * n].reshape(n, 2))
                    b_list.append(out_b[b_off : b_off + n].copy())

                p_off = i * 4
                p_list.append(out_p[p_off : p_off + 4].reshape(2, 2))

                c_off = i * 2
                c_list.append(out_c[c_off : c_off + 2])

            return a_list, b_list, p_list, c_list

    safe_regions, p_list, c_list = infer_safe_region_batch(
        masks,
        center=center,
        patch_size=patch_size,
    )

    a_list = []
    b_list = []
    for poly_points in safe_regions:
        a_mat, b_vec = _polygon_to_halfspaces(poly_points)
        a_list.append(a_mat)
        b_list.append(b_vec)

    return a_list, b_list, p_list, c_list


def infer_safe_region_batch(obs_masks, center=(64.0, 64.0), patch_size=128):
    _lazy_init()

    masks = np.asarray(obs_masks, dtype=np.uint8)
    if masks.ndim != 3:
        return (
            [None for _ in range(len(obs_masks))],
            [None for _ in range(len(obs_masks))],
            [None for _ in range(len(obs_masks))],
        )

    bs = int(masks.shape[0])
    if bs <= 0:
        return [], [], []

    if masks.shape[1] != patch_size or masks.shape[2] != patch_size:
        return [None for _ in range(bs)], [None for _ in range(bs)], [None for _ in range(bs)]

    if not _HAS_BATCH_WITH_ELLIPSE:
        raise NotImplementedError("Neural-IRIS C++ backend unavailable: batch ellipse outputs are required")

    flat = np.ascontiguousarray(masks.reshape(-1))
    out = np.zeros((bs * _MAX_VERTS * 2,), dtype=np.float64)
    out_counts = np.zeros((bs,), dtype=np.int32)
    out_p = np.zeros((bs * 4,), dtype=np.float64)
    out_c = np.zeros((bs * 2,), dtype=np.float64)

    rc = int(
        _LIB.neural_iris_infer_safe_region_batch_with_ellipse(
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int(bs),
            ctypes.c_int(patch_size),
            ctypes.c_double(0.5),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(_MAX_VERTS),
            out_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    )

    if rc != 0:
        return [None for _ in range(bs)], [None for _ in range(bs)], [None for _ in range(bs)]

    safe_regions = []
    p_list = []
    c_list = []
    for i in range(bs):
        n = int(out_counts[i])
        if n < 3:
            safe_regions.append(None)
        else:
            offset = i * _MAX_VERTS * 2
            safe_regions.append(out[offset : offset + 2 * n].reshape(n, 2))

        p_offset = i * 4
        p_list.append(out_p[p_offset : p_offset + 4].reshape(2, 2))

        c_offset = i * 2
        c_list.append(out_c[c_offset : c_offset + 2])

    return safe_regions, p_list, c_list
