import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm
from src.corridor_constraints.geometry import render_soft_ellipse
from src.corridor_constraints.model import CorridorEllipseNet

torch.backends.cudnn.benchmark = True

def iris_loss(preds, targets, patches, grid_x, grid_y, lambda_param=10.0, lambda_iou=5.0, lambda_coll=2.0, lambda_center_constraint=5.0):
    """
    更新说明：
    1. lambda_param 提高到 10.0，强迫网络早期必须优先把中心点对齐。
    2. lambda_iou 提高到 5.0，鼓励面积重叠。
    3. 加入了 lambda_center_constraint = 5.0 的硬性约束项：惩罚不包含图片中心（64, 64）的预测椭圆。
    """
    pred_center, target_center = preds[:, 0:2], targets[:, 0:2]
    pred_shape, target_shape = preds[:, 2:4], targets[:, 2:4]
    pred_angle, target_angle = preds[:, 4:6], targets[:, 4:6]
    
    loss_center = F.smooth_l1_loss(pred_center, target_center, reduction='mean')
    loss_shape = F.mse_loss(pred_shape, target_shape, reduction='mean')
    cos_sim = F.cosine_similarity(pred_angle, target_angle, dim=1)
    loss_angle = (1.0 - cos_sim).mean()
    loss_param = loss_center + loss_shape + loss_angle

    # === 新增约束：保证图片中心点 (64, 64) 位于预测的椭圆内部 ===
    # 图片中心点相对椭圆中心的向量 v = (dx, dy)
    dx = -preds[:, 0]
    dy = -preds[:, 1]
    a = preds[:, 2] + 1e-4
    b = preds[:, 3] + 1e-4
    sin_theta = preds[:, 4]
    cos_theta = preds[:, 5]
    
    # 将向量旋转回椭圆的局部坐标系
    v_rot_x = dx * cos_theta - dy * sin_theta
    v_rot_y = dx * sin_theta + dy * cos_theta
    
    # 计算距离度量 D = (x/a)^2 + (y/b)^2
    # 【修复爆炸问题】加入数值截断 torch.clamp，防止网络预测 a/b 过小时除以极小值导致距离度量上亿爆炸
    dist_metric = torch.clamp((v_rot_x / a)**2 + (v_rot_y / b)**2, max=100.0)
    
    # 如果 D > 1，说明中心点在椭圆外，施加惩罚（如果 <= 1，惩罚为 0）
    center_constraint_penalty = F.relu(dist_metric - 1.0)
    loss_center_constraint = center_constraint_penalty.mean()
    # =======================================================
    
    pred_mask = render_soft_ellipse(preds, grid_x, grid_y, size=128)
    with torch.no_grad():
        target_mask = render_soft_ellipse(targets, grid_x, grid_y, size=128)
        
    intersection = (pred_mask * target_mask).sum(dim=(1,2))
    union = pred_mask.sum(dim=(1,2)) + target_mask.sum(dim=(1,2)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    loss_iou = 1.0 - iou.mean()
    
    obs_mask = patches.squeeze(1) 
    collision = pred_mask * obs_mask
    obs_count = collision.sum(dim=(1,2))
    
    # 彻底移除 **2 二次方惩罚，防止梯度把网络吓跑
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
    angle_diff = torch.abs(pred_theta - target_theta)
    angle_diff = torch.min(angle_diff, math.pi - angle_diff)
    metrics['angle_error_deg'] = torch.rad2deg(angle_diff).mean().item()
    
    with torch.no_grad():
        pred_mask = render_soft_ellipse(preds, grid_x, grid_y, size=128, temperature=100.0) > 0.5
        target_mask = render_soft_ellipse(targets, grid_x, grid_y, size=128, temperature=100.0) > 0.5
        
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
    
    max_batch_size = 512
    y_idx, x_idx = torch.meshgrid(torch.arange(128, device=device), torch.arange(128, device=device), indexing='ij')
    grid_x = x_idx.float().unsqueeze(0).expand(max_batch_size, -1, -1)
    grid_y = y_idx.float().unsqueeze(0).expand(max_batch_size, -1, -1)
    
    train_patches, train_labels = load_data_to_gpu(r"data\splits\train_iris.npz", device)
    val_patches, val_labels = load_data_to_gpu(r"data\splits\val_iris.npz", device)
    
    num_train_samples = len(train_patches)
    num_val_samples = len(val_patches)
    
    model = CorridorEllipseNet().to(device)
    
    # 恢复相对稳妥的学习率 3e-4（对于现在的复杂损失地形更加安全）
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # 衰减策略：耐心值(patience)保持为 5，每次衰减比例(factor)改为 0.5
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
            
            # 使用官方推荐的新版 API 修复废弃警告
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
        
        print(f"Epoch {epoch+1:03d}/{num_epochs} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Metrics -> Center: {val_metrics['center_error_px']:.2f} px | Axis: {val_metrics['axis_error_px']:.2f} px | Angle: {val_metrics['angle_error_deg']:.2f}°")
        print(f"  Physics -> IoU: {val_metrics['iou']:.4f} | Collision Rate: {val_metrics['collision_rate']:.2f}%")
        print("-" * 70)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "iris_net_best.pth")
            
    print("Training complete. Best model saved to iris_net_best.pth")

if __name__ == "__main__":
    train()