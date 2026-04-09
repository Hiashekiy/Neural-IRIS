import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import os
import csv
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.neural_iris.geometry import render_soft_ellipse_mask
from src.neural_iris.model import NeuralIRISNet

torch.backends.cudnn.benchmark = True

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_DATA_PATH = os.path.join(ROOT, "data", "iris-dataset", "splits", "train_iris.npz")
VAL_DATA_PATH = os.path.join(ROOT, "data", "iris-dataset", "splits", "val_iris.npz")
MODEL_OUTPUT_PATH = os.path.join(ROOT, "models", "neural_iris_net_best.pth")
LOG_ROOT = os.path.join(ROOT, "logs", "neural_iris_train")


def create_run_paths():
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(LOG_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
        "csv_path": os.path.join(run_dir, "metrics.csv"),
        "plot_path": os.path.join(run_dir, "metrics.png"),
    }


def append_metrics_row(csv_path, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_training_curves(history, plot_path):
    epochs = [row["epoch"] for row in history]
    iou_percent = [row["iou"] * 100.0 for row in history]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.linewidth": 1.2,
    })

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), facecolor="white")

    def style_axis(ax):
        ax.grid(False)
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.tick_params(axis="both", width=1.1, length=5)

    def set_lr_axis_style(ax, lr_values):
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_linewidth(1.2)
        ax.tick_params(axis="y", width=1.1, length=5)
        ax.set_yscale("log")
        lr_min = max(min(lr_values), 1e-12)
        lr_max = max(lr_values)
        ax.set_ylim(lr_min / 1.8, lr_max * 1.35)

    ax_loss = axes[0, 0]
    loss_lines = []
    loss_lines += ax_loss.plot(epochs, [row["train_loss"] for row in history], label="Train Loss", linewidth=2.6, color="#1f77b4")
    loss_lines += ax_loss.plot(epochs, [row["val_loss"] for row in history], label="Validation Loss", linewidth=2.6, color="#d95f02")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    style_axis(ax_loss)
    ax_loss.legend(
        loss_lines,
        [line.get_label() for line in loss_lines],
        loc="upper right",
        frameon=False,
        fancybox=False,
    )

    axes[0, 1].plot(epochs, iou_percent, label="IoU (%)", linewidth=2.6, color="#1f77b4")
    axes[0, 1].plot(epochs, [row["collision_rate"] for row in history], label="Collision Rate (%)", linewidth=2.6, color="#d62728")
    axes[0, 1].set_title("Overlap and Collision")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Percent")
    style_axis(axes[0, 1])
    axes[0, 1].legend(loc="upper right", frameon=False, fancybox=False)

    axes[1, 0].plot(epochs, [row["center_error_px"] for row in history], label="Center Error", linewidth=2.8, color="#1f77b4")
    axes[1, 0].plot(epochs, [row["axis_error_px"] for row in history], label="Axis Error", linewidth=2.8, color="#e31a1c")
    axes[1, 0].set_title("Geometric Errors")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Pixels")
    style_axis(axes[1, 0])
    axes[1, 0].legend(loc="upper right", frameon=False, fancybox=False)

    ax_angle = axes[1, 1]
    ax_angle_lr = ax_angle.twinx()
    angle_lines = []
    angle_lines += ax_angle.plot(epochs, [row["angle_error_deg"] for row in history], label="Angle Error (deg)", linewidth=2.6, color="#4c78a8")
    angle_lines += ax_angle_lr.plot(
        epochs,
        [row["lr"] for row in history],
        label="Learning Rate",
        linewidth=2.4,
        color="#f58518",
        linestyle="--",
    )
    ax_angle.set_title("Angle Error and Learning Rate")
    ax_angle.set_xlabel("Epoch")
    ax_angle.set_ylabel("Degrees")
    ax_angle_lr.set_ylabel("Learning Rate")
    style_axis(ax_angle)
    set_lr_axis_style(ax_angle_lr, [row["lr"] for row in history])
    ax_angle.legend(
        angle_lines,
        [line.get_label() for line in angle_lines],
        loc="upper right",
        frameon=False,
        fancybox=False,
    )

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def iris_loss(preds, targets, patches, grid_x, grid_y, lambda_param=10.0, lambda_iou=5.0, lambda_coll=2.0, lambda_center_constraint=5.0):
    """
    Loss terms used for Neural-IRIS training.
    1. A larger `lambda_param` keeps center/shape/orientation regression stable early in training.
    2. A larger `lambda_iou` encourages better overlap between predicted and target ellipses.
    3. `lambda_center_constraint` penalizes ellipses that fail to include the patch center (64, 64).
    """
    pred_center, target_center = preds[:, 0:2], targets[:, 0:2]
    pred_shape, target_shape = preds[:, 2:4], targets[:, 2:4]
    pred_angle, target_angle = preds[:, 4:6], targets[:, 4:6]
    
    loss_center = F.smooth_l1_loss(pred_center, target_center, reduction='mean')
    loss_shape = F.mse_loss(pred_shape, target_shape, reduction='mean')
    cos_sim = F.cosine_similarity(pred_angle, target_angle, dim=1)
    loss_angle = (1.0 - cos_sim).mean()
    loss_param = loss_center + loss_shape + loss_angle

    # Enforce that the image center (64, 64) stays inside the predicted ellipse.
    # Vector from ellipse center to the image center.
    dx = -preds[:, 0]
    dy = -preds[:, 1]
    a = preds[:, 2] + 1e-4
    b = preds[:, 3] + 1e-4
    sin_theta = preds[:, 4]
    cos_theta = preds[:, 5]
    
    # Rotate the vector back into the ellipse local frame.
    v_rot_x = dx * cos_theta - dy * sin_theta
    v_rot_y = dx * sin_theta + dy * cos_theta
    
    # Distance metric D = (x/a)^2 + (y/b)^2.
    # Clamp it to avoid exploding values when the predicted axes become too small.
    dist_metric = torch.clamp((v_rot_x / a)**2 + (v_rot_y / b)**2, max=100.0)
    
    # If D > 1, the image center lies outside the ellipse and receives a penalty.
    center_constraint_penalty = F.relu(dist_metric - 1.0)
    loss_center_constraint = center_constraint_penalty.mean()
    # =======================================================
    
    pred_mask = render_soft_ellipse_mask(preds, grid_x, grid_y, size=128)
    with torch.no_grad():
        target_mask = render_soft_ellipse_mask(targets, grid_x, grid_y, size=128)
        
    intersection = (pred_mask * target_mask).sum(dim=(1,2))
    union = pred_mask.sum(dim=(1,2)) + target_mask.sum(dim=(1,2)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    loss_iou = 1.0 - iou.mean()
    
    obs_mask = patches.squeeze(1) 
    collision = pred_mask * obs_mask
    obs_count = collision.sum(dim=(1,2))
    
    # Keep collision loss linear to avoid unstable gradients from a squared penalty.
    loss_coll = obs_count.mean()
    
    total_loss = lambda_param * loss_param + lambda_iou * loss_iou + lambda_coll * loss_coll + lambda_center_constraint * loss_center_constraint
    return total_loss

def calculate_metrics(preds, targets, patches, grid_x, grid_y):
    metrics = {}
    
    center_diff = preds[:, 0:2] - targets[:, 0:2]
    metrics['center_error_px'] = torch.norm(center_diff, dim=1).mean().item()
    metrics['axis_error_px'] = torch.abs(preds[:, 2:4] - targets[:, 2:4]).mean().item()
    
    pred_theta = torch.atan2(preds[:, 4], preds[:, 5])
    target_theta = torch.atan2(targets[:, 4], targets[:, 5])
    angle_diff = torch.remainder(pred_theta - target_theta + 0.5 * math.pi, math.pi) - 0.5 * math.pi
    angle_diff = torch.abs(angle_diff)
    metrics['angle_error_deg'] = torch.rad2deg(angle_diff).mean().item()
    
    with torch.no_grad():
        pred_mask = render_soft_ellipse_mask(preds, grid_x, grid_y, size=128, temperature=100.0) > 0.5
        target_mask = render_soft_ellipse_mask(targets, grid_x, grid_y, size=128, temperature=100.0) > 0.5
        
        inter = (pred_mask & target_mask).float().sum(dim=(1,2))
        uni = (pred_mask | target_mask).float().sum(dim=(1,2))
        hard_iou = (inter + 1e-6) / (uni + 1e-6)
        metrics['iou'] = hard_iou.mean().item()
        
        obs_mask = patches.squeeze(1) > 0.5
        collision_pixels = (pred_mask & obs_mask).float().sum(dim=(1,2))
        collision_rate = (collision_pixels > 0).float().mean().item()
        metrics['collision_rate'] = collision_rate * 100.0
    
    return metrics

def load_data_to_gpu(npz_path, device):
    print(f"Loading {npz_path} directly to GPU...")
    data = np.load(npz_path)
    patches = torch.from_numpy(data['patches']).unsqueeze(1).to(device)
    labels = torch.from_numpy(data['labels']).to(device).float()
    print(f"Loaded {len(patches)} samples.")
    return patches, labels

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    run_paths = create_run_paths()
    history = []
    print(f"Training log CSV: {run_paths['csv_path']}")
    print(f"Training plot PNG: {run_paths['plot_path']}")
    
    max_batch_size = 512
    y_idx, x_idx = torch.meshgrid(torch.arange(128, device=device), torch.arange(128, device=device), indexing='ij')
    grid_x = x_idx.float().unsqueeze(0).expand(max_batch_size, -1, -1)
    grid_y = y_idx.float().unsqueeze(0).expand(max_batch_size, -1, -1)
    
    train_patches, train_labels = load_data_to_gpu(TRAIN_DATA_PATH, device)
    val_patches, val_labels = load_data_to_gpu(VAL_DATA_PATH, device)
    
    num_train_samples = len(train_patches)
    num_val_samples = len(val_patches)
    
    model = NeuralIRISNet().to(device)
    
    # A conservative learning rate works better with the current composite loss.
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # Reduce the learning rate when validation loss stops improving.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    scaler = torch.amp.GradScaler('cuda') 
    
    num_epochs = 100
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        indices = torch.randperm(num_train_samples, device=device)
        num_batches = (num_train_samples + max_batch_size - 1) // max_batch_size
        pbar_train = tqdm(range(num_batches), desc=f"Epoch {epoch+1:03d}/{num_epochs} [Train]", leave=False)
        
        for i in pbar_train:
            batch_idx = indices[i * max_batch_size : (i + 1) * max_batch_size]
            
            patches = train_patches[batch_idx].float()
            labels = train_labels[batch_idx]
            
            optimizer.zero_grad(set_to_none=True)
            
            # Use the current torch AMP API to avoid deprecated warnings.
            with torch.amp.autocast('cuda'):
                preds = model(patches)
                loss = iris_loss(preds, labels, patches, grid_x, grid_y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * patches.size(0)
            pbar_train.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= num_train_samples
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        
        num_val_batches = (num_val_samples + max_batch_size - 1) // max_batch_size
        pbar_val = tqdm(range(num_val_batches), desc=f"Epoch {epoch+1:03d}/{num_epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for i in pbar_val:
                batch_idx = torch.arange(i * max_batch_size, min((i + 1) * max_batch_size, num_val_samples), device=device)
                
                patches = val_patches[batch_idx].float()
                labels = val_labels[batch_idx]
                
                with torch.amp.autocast('cuda'):
                    preds = model(patches)
                    loss = iris_loss(preds, labels, patches, grid_x, grid_y)
                    
                val_loss += loss.item() * patches.size(0)
                all_preds.append(preds)
                pbar_val.set_postfix({'loss': f"{loss.item():.4f}"})
                
        val_loss /= num_val_samples
        scheduler.step(val_loss)
        
        cat_preds = torch.cat(all_preds, dim=0)
        
        val_grid_x = x_idx.float().unsqueeze(0).expand(num_val_samples, -1, -1)
        val_grid_y = y_idx.float().unsqueeze(0).expand(num_val_samples, -1, -1)
        
        val_metrics = calculate_metrics(cat_preds, val_labels, val_patches.float(), val_grid_x, val_grid_y)
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "center_error_px": val_metrics["center_error_px"],
            "axis_error_px": val_metrics["axis_error_px"],
            "angle_error_deg": val_metrics["angle_error_deg"],
            "iou": val_metrics["iou"],
            "iou_percent": val_metrics["iou"] * 100.0,
            "collision_rate": val_metrics["collision_rate"],
        }
        history.append(row)
        append_metrics_row(run_paths["csv_path"], row)
        
        print(f"Epoch {epoch+1:03d}/{num_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Metrics -> Center: {val_metrics['center_error_px']:.2f} px | Axis: {val_metrics['axis_error_px']:.2f} px | Angle: {val_metrics['angle_error_deg']:.2f} deg")
        print(f"  Physics -> IoU: {val_metrics['iou'] * 100.0:.2f}% | Collision Rate: {val_metrics['collision_rate']:.2f}%")
        print("-" * 70)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    
    save_training_curves(history, run_paths["plot_path"])
    print(f"Training complete. Best model saved to {MODEL_OUTPUT_PATH}")
    print(f"Training metrics CSV saved to {run_paths['csv_path']}")
    print(f"Training curves PNG saved to {run_paths['plot_path']}")

if __name__ == "__main__":
    train()


