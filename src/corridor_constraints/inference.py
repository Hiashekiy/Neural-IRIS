import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.corridor_constraints.generate_safe_polygon import generate_safe_polygon

class IRISNet(nn.Module):
    def __init__(self):
        super(IRISNet, self).__init__()
        resnet = models.resnet18(weights=None)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        self.fc = nn.Linear(resnet.fc.in_features, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        out = self.fc(x)
        
        dx_dy = out[:, 0:2]
        a_b = F.softplus(out[:, 2:4])
        angle_raw = out[:, 4:6]
        angle_norm = F.normalize(angle_raw, p=2, dim=1)
        
        return torch.cat([dx_dy, a_b, angle_norm], dim=1)

def parse_network_output(pred_data, patch_size=128):
    """将网络输出张量或 Numpy 数组转换为物理几何参数 c 和 P"""
    if isinstance(pred_data, torch.Tensor):
        pred = pred_data.detach().cpu().numpy()
    else:
        pred = np.asarray(pred_data)
        
    dx, dy, a, b, sin_t, cos_t = pred
    
    center_x = (patch_size / 2.0) + dx
    center_y = (patch_size / 2.0) + dy
    c = np.array([center_x, center_y])
    
    R = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ])
    
    Lambda = np.array([
        [1.0 / (a + 1e-4), 0],
        [0, 1.0 / (b + 1e-4)]
    ])
    
    P = R @ Lambda @ R.T
    return P, c

def get_ellipse_points(P, c, scale=1.0, num_points=100):
    theta = np.linspace(0, 2*np.pi, num_points)
    u = np.array([np.cos(theta), np.sin(theta)])
    try:
        P_inv = np.linalg.inv(P)
        ellipse_points = ((scale * P_inv) @ u).T + c
        return ellipse_points
    except np.linalg.LinAlgError:
        return np.zeros((num_points, 2))

def render_sample(patch, target_label, pred_label, idx, save_path=None, show_gui=False):
    patch_size = patch.shape[0]
    
    py, px = np.where(patch == 1)
    obs_points = np.column_stack((px, py)).astype(float)
    
    P_gt, c_gt = parse_network_output(target_label, patch_size)
    P_pred, c_pred = parse_network_output(pred_label, patch_size)
    
    poly_points = generate_safe_polygon(P_pred, c_pred, obs_points, patch_size)
    
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
    
    if poly_points is not None:
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
    
    model = IRISNet().to(device)
    model_path = "iris_net_best.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model weights loaded successfully.")
    
    test_data_path = r"data\splits\test_iris.npz"
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
    parser = argparse.ArgumentParser(description="IRISNet Inference Script")
    parser.add_argument("--mode", type=str, choices=["headless", "gui"], default="gui", 
                        help="Choose 'headless' to save images to disk, or 'gui' to show plot windows.")
    parser.add_argument("--num", type=int, default=5, 
                        help="Number of samples to infer and visualize. Use -1 for ALL samples.")
    args = parser.parse_args()
    
    run_inference(mode=args.mode, num_samples=args.num)