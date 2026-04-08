"""调试脚本：分析为什么ours_corridor_net会有碰撞"""

import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiment.common.geometry_utils import halfspace_intersection_vertices

TEST_DATA_PATH = os.path.join(ROOT, "data", "iris-dataset", "splits", "test_iris.npz")

def load_test_patches():
    d = np.load(TEST_DATA_PATH)
    return d["patches"]

def analyze_collisions(n_analyze=50):
    """Find and analyze samples with collisions"""
    print(f"[debug] Loading test patches...")
    patches = load_test_patches()
    
    import torch
    from src.corridor_constraints.model import CorridorEllipseNet
    from src.corridor_constraints.geometry import parse_network_output, extract_obstacle_boundary_points
    from src.corridor_constraints.generate_safe_polygon import generate_safe_polygon
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CorridorEllipseNet().to(device)
    model_path = os.path.join(ROOT, "models", "iris_net_best.pth")
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    
    center = np.array([64.0, 64.0], dtype=float)
    patch_size = 128
    
    collision_samples = []
    collision_info = []
    
    print(f"[debug] Scanning first 1000 samples for collisions...")
    with torch.no_grad():
        for i in range(min(1000, len(patches))):
            obs_mask = patches[i] > 0.5
            
            x = torch.from_numpy(obs_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(x).squeeze(0).detach().cpu().numpy()
            
            p_mat, c = parse_network_output(pred, patch_size=patch_size)
            obs_points = extract_obstacle_boundary_points(obs_mask)
            poly = generate_safe_polygon(p_mat, c, obs_points, patch_size=patch_size)
            
            if isinstance(poly, tuple) and len(poly) == 2:
                A, b = poly
                if A is not None and b is not None and len(A) >= 3:
                    # Get vertices
                    v = halfspace_intersection_vertices(np.asarray(A, dtype=float), np.asarray(b, dtype=float), center)
                    if v is not None and len(v) >= 3:
                        # Check collision: rasterize polygon and check overlap with obstacles
                        from experiment.run_parameter_metrics import collision_overlap_pixels, polygon_mask
                        mask = polygon_mask(v, patch_size=patch_size)
                        if mask is not None:
                            coll_pixels = collision_overlap_pixels(mask, obs_mask, obstacle_margin_px=1)
                            coll_percent = coll_pixels / mask.sum()
                            
                            # Relaxed collision: >= 1% area
                            if coll_percent >= 0.01:
                                collision_samples.append(i)
                                collision_info.append({
                                    'idx': i,
                                    'coll_pixels': coll_pixels,
                                    'coll_percent': coll_percent,
                                    'poly_area': mask.sum(),
                                    'obs_area': obs_mask.sum(),
                                    'obs_boundary_points': len(obs_points) if obs_points is not None else 0,
                                    'A_shape': A.shape,
                                    'vertices_count': len(v),
                                })
                                
                                if len(collision_samples) <= n_analyze:
                                    print(f"\n[collision sample {len(collision_samples)}] idx={i}")
                                    print(f"  collision: {coll_percent*100:.2f}% ({coll_pixels:.0f} pixels out of {mask.sum():.0f})")
                                    print(f"  obstacle area: {obs_mask.sum()} pixels")
                                    print(f"  obstacle boundary points: {len(obs_points) if obs_points is not None else 0}")
                                    print(f"  polygon: {A.shape[0]} halfspaces, {len(v)} vertices")
                                    print(f"  center c: ({c[0]:.1f}, {c[1]:.1f})")
                                    print(f"  P.shape: {p_mat.shape}")
                                    
                                    # 检查点是否都在halfspace内侧
                                    # 对于 Ax <= b，点应满足 A @ point <= b
                                    inside_check = np.all((A @ v.T) <= (b[:, None] + 1e-6), axis=0)
                                    print(f"  all vertices satisfy A*v <= b: {np.all(inside_check)}")
    
    print(f"\n{'='*70}")
    print(f"[debug] Total collisions in first 1000 samples: {len(collision_samples)}")
    print(f"[debug] Collision rate: {100*len(collision_samples)/min(1000, len(patches)):.2f}%")
    
    if collision_info:
        print(f"\n{'Sample':<8} {'Idx':<6} {'Coll%':<8} {'CollPix':<10} {'PolyArea':<10} {'ObsArea':<8} {'ObsPts':<8} {'Halfspaces':<12} {'Vertices':<10}")
        print("-" * 90)
        for info in collision_info[:10]:
            print(f"{collision_info.index(info)+1:<8} {info['idx']:<6} {info['coll_percent']*100:>7.2f} {info['coll_pixels']:>9.0f} {info['poly_area']:>9.0f} {info['obs_area']:>7.0f} {info['obs_boundary_points']:>7.0f} {info['A_shape'][0]:>11.0f} {info['vertices_count']:>9.0f}")

if __name__ == "__main__":
    analyze_collisions(n_analyze=10)
