"""性能分析脚本：测量ours_corridor_net各个步骤的耗时"""

import os
import sys
import time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiment.common.geometry_utils import halfspace_intersection_vertices, polygon_to_halfspaces

TEST_DATA_PATH = os.path.join(ROOT, "data", "iris-dataset", "splits", "test_iris.npz")

def load_test_patches():
    d = np.load(TEST_DATA_PATH)
    return d["patches"]

def profile_ours_corridor_net(n_samples=100):
    """Breakdown of ours_corridor_net inference time"""
    print(f"[profile] Loading {n_samples} test patches...")
    patches = load_test_patches()[:n_samples]
    
    # Import model components
    import torch
    from src.corridor_constraints.model import CorridorEllipseNet
    from src.corridor_constraints.geometry import parse_network_output, extract_obstacle_boundary_points
    from src.corridor_constraints.generate_safe_polygon import generate_safe_polygon
    
    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CorridorEllipseNet().to(device)
    model_path = os.path.join(ROOT, "models", "iris_net_best.pth")
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    
    print(f"[profile] Device: {device}")
    print(f"[profile] Running profiling on {n_samples} samples...\n")
    
    center = np.array([64.0, 64.0], dtype=float)
    patch_size = 128
    
    # Timing accumulators
    times = {
        "patch_to_tensor": 0.0,
        "model_forward": 0.0,
        "parse_output": 0.0,
        "extract_boundary": 0.0,
        "generate_polygon": 0.0,
        "halfspace_intersection": 0.0,
        "total": 0.0,
    }
    
    valid_count = 0
    
    with torch.no_grad():
        for i in range(n_samples):
            obs_mask = patches[i] > 0.5
            
            t0 = time.perf_counter()
            
            # Step 1: Convert to tensor
            t1 = time.perf_counter()
            x = torch.from_numpy(obs_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            t2 = time.perf_counter()
            times["patch_to_tensor"] += (t2 - t1)
            
            # Step 2: Model forward
            t1 = time.perf_counter()
            pred = model(x).squeeze(0).detach().cpu().numpy()
            t2 = time.perf_counter()
            times["model_forward"] += (t2 - t1)
            
            # Step 3: Parse output
            t1 = time.perf_counter()
            p_mat, c = parse_network_output(pred, patch_size=patch_size)
            t2 = time.perf_counter()
            times["parse_output"] += (t2 - t1)
            
            # Step 4: Extract obstacle boundary
            t1 = time.perf_counter()
            obs_points = extract_obstacle_boundary_points(obs_mask)
            t2 = time.perf_counter()
            times["extract_boundary"] += (t2 - t1)
            
            # Step 5: Generate safe polygon
            t1 = time.perf_counter()
            poly = generate_safe_polygon(p_mat, c, obs_points, patch_size=patch_size)
            t2 = time.perf_counter()
            times["generate_polygon"] += (t2 - t1)
            
            # Step 6: Halfspace intersection to get vertices
            t1 = time.perf_counter()
            if isinstance(poly, tuple) and len(poly) == 2:
                A, b = poly
                if A is not None and b is not None and len(A) >= 3:
                    v = halfspace_intersection_vertices(np.asarray(A, dtype=float), np.asarray(b, dtype=float), center)
                    if v is not None and len(v) >= 3:
                        valid_count += 1
            t2 = time.perf_counter()
            times["halfspace_intersection"] += (t2 - t1)
            
            t_total = time.perf_counter() - t0
            times["total"] += t_total
    
    # Print results
    print("=" * 70)
    print(f"{'Component':<30} {'Time (ms)':<15} {'% of Total':<15}")
    print("=" * 70)
    
    total_ms = times["total"] * 1000.0
    for key in ["patch_to_tensor", "model_forward", "parse_output", "extract_boundary", 
                "generate_polygon", "halfspace_intersection"]:
        time_ms = times[key] * 1000.0
        pct = (times[key] / times["total"]) * 100.0 if times["total"] > 0 else 0.0
        print(f"{key:<30} {time_ms:>12.3f}   {pct:>12.1f}%")
    
    print("-" * 70)
    avg_ms = total_ms / n_samples
    print(f"{'TOTAL (average per sample)':<30} {avg_ms:>12.3f}   {100.0:>12.1f}%")
    print("=" * 70)
    print(f"[profile] Valid polygons generated: {valid_count}/{n_samples} ({100*valid_count/n_samples:.1f}%)")
    print(f"[profile] Total time: {total_ms/1000:.2f}s")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    
    profile_ours_corridor_net(n_samples=args.samples)
