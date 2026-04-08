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
_MAX_VERTS = 128
BATCH_SIZE = 64


def _candidate_paths():
    base = os.path.join(_ROOT, "cpp")
    mode = os.environ.get("OURS_CPP_BACKEND", "").strip().lower()
    if mode == "cpu":
        return [
            os.path.join(base, "build_cpu", "Release", "corridor_cpp_bridge.dll"),
            os.path.join(base, "build", "Release", "corridor_cpp_bridge.dll"),
        ]
    if mode == "gpu":
        return [
            os.path.join(base, "build_gpu", "Release", "corridor_cpp_bridge.dll"),
            os.path.join(base, "build", "Release", "corridor_cpp_bridge.dll"),
        ]
    return [
        os.path.join(base, "build_gpu", "Release", "corridor_cpp_bridge.dll"),
        os.path.join(base, "build_cpu", "Release", "corridor_cpp_bridge.dll"),
        os.path.join(base, "build", "Release", "corridor_cpp_bridge.dll"),
    ]


def _find_lib():
    for p in _candidate_paths():
        if os.path.isfile(p):
            return p
    return None


def _lazy_init():
    global _LIB, _READY, _HAS_BATCH_WITH_ELLIPSE
    if _READY:
        return

    lib_path = _find_lib()
    if lib_path is None:
        raise NotImplementedError("ours_corridor_cpp unavailable: corridor_cpp_bridge.dll not found")

    lib_dir = os.path.dirname(lib_path)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(lib_dir)

    _LIB = ctypes.CDLL(lib_path)

    _LIB.corridor_init.argtypes = [ctypes.c_char_p]
    _LIB.corridor_init.restype = ctypes.c_int

    _LIB.corridor_infer.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    _LIB.corridor_infer.restype = ctypes.c_int

    _LIB.corridor_infer_batch.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    _LIB.corridor_infer_batch.restype = ctypes.c_int

    if hasattr(_LIB, "corridor_infer_batch_with_ellipse"):
        _LIB.corridor_infer_batch_with_ellipse.argtypes = [
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
        _LIB.corridor_infer_batch_with_ellipse.restype = ctypes.c_int
        _HAS_BATCH_WITH_ELLIPSE = True

    _LIB.corridor_shutdown.argtypes = []
    _LIB.corridor_shutdown.restype = None

    onnx_path = os.path.join(_ROOT, "cpp", "models", "corridor_ellipse_net.onnx")
    if not os.path.isfile(onnx_path):
        raise NotImplementedError("ours_corridor_cpp unavailable: ONNX model missing in cpp/models")

    rc = int(_LIB.corridor_init(onnx_path.encode("utf-8")))
    if rc != 0:
        raise NotImplementedError(f"ours_corridor_cpp unavailable: corridor_init failed with code {rc}")

    _READY = True


def infer_polygon(obs_mask, center=(64.0, 64.0), patch_size=128):
    _lazy_init()

    mask_u8 = np.asarray(obs_mask, dtype=np.uint8)
    if mask_u8.ndim != 2:
        return None

    if mask_u8.shape[0] != patch_size or mask_u8.shape[1] != patch_size:
        return None

    flat = np.ascontiguousarray(mask_u8.reshape(-1))
    out = np.zeros((_MAX_VERTS * 2,), dtype=np.float64)
    out_n = ctypes.c_int(0)

    rc = int(
        _LIB.corridor_infer(
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            int(patch_size),
            ctypes.c_double(0.5),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            int(_MAX_VERTS),
            ctypes.byref(out_n),
        )
    )

    if rc != 0:
        return None

    n = int(out_n.value)
    if n < 3:
        return None

    verts = out[: 2 * n].reshape(n, 2)
    return verts


def infer_polygon_batch(obs_masks, center=(64.0, 64.0), patch_size=128):
    _lazy_init()

    masks = np.asarray(obs_masks, dtype=np.uint8)
    if masks.ndim != 3:
        return [None for _ in range(len(obs_masks))]

    bs = int(masks.shape[0])
    if bs <= 0:
        return []

    if masks.shape[1] != patch_size or masks.shape[2] != patch_size:
        return [None for _ in range(bs)]

    flat = np.ascontiguousarray(masks.reshape(-1))
    out = np.zeros((bs * _MAX_VERTS * 2,), dtype=np.float64)
    out_counts = np.zeros((bs,), dtype=np.int32)

    rc = int(
        _LIB.corridor_infer_batch(
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int(bs),
            ctypes.c_int(patch_size),
            ctypes.c_double(0.5),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(_MAX_VERTS),
            out_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
    )

    if rc != 0:
        return [None for _ in range(bs)]

    ret = []
    for i in range(bs):
        n = int(out_counts[i])
        if n < 3:
            ret.append(None)
            continue
        off = i * _MAX_VERTS * 2
        verts = out[off : off + 2 * n].reshape(n, 2)
        ret.append(verts)
    return ret


def infer_polygon_batch_with_ellipse(obs_masks, center=(64.0, 64.0), patch_size=128):
    _lazy_init()

    masks = np.asarray(obs_masks, dtype=np.uint8)
    if masks.ndim != 3:
        return [None for _ in range(len(obs_masks))], [None for _ in range(len(obs_masks))], [None for _ in range(len(obs_masks))]

    bs = int(masks.shape[0])
    if bs <= 0:
        return [], [], []

    if masks.shape[1] != patch_size or masks.shape[2] != patch_size:
        return [None for _ in range(bs)], [None for _ in range(bs)], [None for _ in range(bs)]

    if not _HAS_BATCH_WITH_ELLIPSE:
        polys = infer_polygon_batch(masks, center=center, patch_size=patch_size)
        return polys, [None for _ in range(bs)], [None for _ in range(bs)]

    flat = np.ascontiguousarray(masks.reshape(-1))
    out = np.zeros((bs * _MAX_VERTS * 2,), dtype=np.float64)
    out_counts = np.zeros((bs,), dtype=np.int32)
    out_p = np.zeros((bs * 4,), dtype=np.float64)
    out_c = np.zeros((bs * 2,), dtype=np.float64)

    rc = int(
        _LIB.corridor_infer_batch_with_ellipse(
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

    polys = []
    p_list = []
    c_list = []
    for i in range(bs):
        n = int(out_counts[i])
        if n < 3:
            polys.append(None)
        else:
            off = i * _MAX_VERTS * 2
            polys.append(out[off : off + 2 * n].reshape(n, 2))

        p_off = i * 4
        p_mat = out_p[p_off : p_off + 4].reshape(2, 2)
        p_list.append(p_mat)

        c_off = i * 2
        c_vec = out_c[c_off : c_off + 2]
        c_list.append(c_vec)

    return polys, p_list, c_list
