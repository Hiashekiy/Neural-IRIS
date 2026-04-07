import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from scipy.spatial import ConvexHull, HalfspaceIntersection

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.corridor_constraints.generate_safe_polygon import generate_safe_polygon
from src.corridor_constraints.model import CorridorEllipseNet
from src.corridor_constraints.geometry import get_ellipse_points, parse_network_output

MODEL_PATH = os.path.join(root_path, "models", "iris_net_best.pth")
DATA_PATH = os.path.join(root_path, "data", "iris-dataset", "splits", "test_iris.npz")


def constraints_to_vertices(a_mat, b_vec, interior_point):
    """将半空间约束 A x <= b 转为可绘制顶点集合。"""
    if a_mat is None or b_vec is None or len(a_mat) < 3:
        return None

    try:
        halfspaces = np.hstack([a_mat, -b_vec.reshape(-1, 1)])
        hs = HalfspaceIntersection(halfspaces, interior_point)
        if len(hs.intersections) < 3:
            return None
        hull = ConvexHull(hs.intersections)
        return hs.intersections[hull.vertices]
    except Exception:
        return None

def render_sample(patch, target_label, pred_label, idx, save_path=None, show_gui=False):
    patch_size = patch.shape[0]
    
    py, px = np.where(patch == 1)
    obs_points = np.column_stack((px, py)).astype(float)
    
    P_gt, c_gt = parse_network_output(target_label, patch_size)
    P_pred, c_pred = parse_network_output(pred_label, patch_size)
    
    poly_result = generate_safe_polygon(P_pred, c_pred, obs_points, patch_size)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, patch_size)
    ax.set_ylim(0, patch_size)
    ax.set_aspect('equal')
    
    ax.imshow(patch, cmap='Greys', origin='lower', extent=[0, patch_size, 0, patch_size], alpha=0.3)
    if len(obs_points) > 0:
        ax.scatter(obs_points[:, 0], obs_points[:, 1], c='black', s=10, marker='s', label="Obstacles")
    
    gt_ellipse = get_ellipse_points(P_gt, c_gt)
    ax.plot(gt_ellipse[:, 0], gt_ellipse[:, 1], color='blue', linestyle='--', linewidth=2, label="GT Ellipse (IRIS)")
    ax.scatter(c_gt[0], c_gt[1], color='blue', marker='+', s=100)
    
    pred_ellipse = get_ellipse_points(P_pred, c_pred)
    ax.plot(pred_ellipse[:, 0], pred_ellipse[:, 1], color='red', linestyle='-', linewidth=2, label="Predicted Ellipse")
    ax.scatter(c_pred[0], c_pred[1], color='red', marker='x', s=100)
    
    poly_points = None
    if isinstance(poly_result, tuple) and len(poly_result) == 2:
        a_mat, b_vec = poly_result
        poly_points = constraints_to_vertices(a_mat, b_vec, c_pred)
    else:
        poly_points = poly_result

    if poly_points is not None and len(poly_points) >= 3:
        poly = plt.Polygon(poly_points, facecolor='lightgreen', edgecolor='green', alpha=0.5, linewidth=2, label="Safe Polygon SFC")
        ax.add_patch(poly)
    
    ax.set_title(f"Test Sample {idx} - Neural IRIS vs Ground Truth", fontweight='bold')
    ax.legend(loc='upper right')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
    if show_gui:
        plt.show()
    
    plt.close(fig)

def run_inference(mode="headless", num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing inference on {device}...")
    
    model = CorridorEllipseNet().to(device)
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model weights loaded successfully.")
    
    test_data_path = DATA_PATH
    if not os.path.exists(test_data_path):
        print(f"Error: {test_data_path} not found!")
        return
        
    print("Loading test dataset...")
    data = np.load(test_data_path)
    patches = data['patches']
    labels = data['labels']
    total_test = len(patches)
    
    if num_samples == -1 or num_samples >= total_test:
        print(f"Testing ALL {total_test} samples in the dataset...")
        num_samples = total_test
        sample_indices = np.arange(total_test)
    else:
        sample_indices = np.random.choice(total_test, num_samples, replace=False)
    
    if mode == "headless":
        os.makedirs("inference_results", exist_ok=True)
        print(f"Running HEADLESS mode. Saving {num_samples} figures to ./inference_results/ ...")
        
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            patch_np = patches[idx]
            label_np = labels[idx]
            
            patch_tensor = torch.from_numpy(patch_np).float().unsqueeze(0).unsqueeze(0).to(device)
            pred_tensor = model(patch_tensor).squeeze(0)
            
            if mode == "headless":
                save_path = f"inference_results/sample_{i:06d}_idx_{idx}.png"
                render_sample(patch_np, label_np, pred_tensor, idx, save_path=save_path, show_gui=False)
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{num_samples} samples...")
            else:
                print(f"Displaying Sample {i+1}/{num_samples} (Index: {idx}). Close window to see the next one.")
                render_sample(patch_np, label_np, pred_tensor, idx, save_path=None, show_gui=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CorridorEllipseNet Inference Script")
    parser.add_argument("--mode", type=str, choices=["headless", "gui"], default="gui", 
                        help="Choose 'headless' to save images to disk, or 'gui' to show plot windows.")
    parser.add_argument("--num", type=int, default=5, 
                        help="Number of samples to infer and visualize. Use -1 for ALL samples.")
    args = parser.parse_args()
    
    run_inference(mode=args.mode, num_samples=args.num)