import torch
import numpy as np
import time
import math
import multiprocessing
from tqdm import tqdm
from train import IRISNet, render_soft_ellipse
from generate_safe_polygon import generate_safe_polygon
from scipy.spatial import Delaunay
import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_path not in sys.path:
    sys.path.append(root_path)



def point_in_convex_poly(poly, pt):
    """使用叉乘快速判断点是否在凸多边形内，极大降低 Delaunay 开销"""
    p1 = poly
    p2 = np.roll(poly, -1, axis=0)
    edges = p2 - p1
    pt_vec = pt - p1
    cross = edges[:, 0] * pt_vec[:, 1] - edges[:, 1] * pt_vec[:, 0]
    return np.all(cross >= -1e-5) or np.all(cross <= 1e-5)

def check_poly_worker(args):
    dx, dy, a, b, sin_t, cos_t, obs_mask = args
    c = np.array([64.0 + dx, 64.0 + dy])
    R = np.array([[cos_t, -sin_t], [sin_t,  cos_t]])
    P = R @ np.diag([a, b]) @ R.T
    
    obs_y, obs_x = np.where(obs_mask)
    if len(obs_x) > 0:
        obs_points = np.column_stack((obs_x, obs_y)).astype(float)
    else:
        obs_points = np.array([])
        
    poly = generate_safe_polygon(P, c, obs_points, patch_size=128)
    
    is_in = False
    if poly is not None and len(poly) >= 3:
        is_in = point_in_convex_poly(poly, np.array([64.0, 64.0]))
    return 1.0 if is_in else 0.0

def load_test_data(path, device):
    print(f"Loading test data from {path}...")
    try:
        data = np.load(path)
        patches = torch.from_numpy(data['patches']).unsqueeze(1).to(device)
        labels = torch.from_numpy(data['labels']).to(device).float()
        return patches, labels
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None

def shift_patches_batch(patches, dx, dy):
    """根据给定的像素偏差(dx, dy)，平移图像批次，超界部分用障碍物(1.0)填充"""
    B, C, H, W = patches.shape
    new_patches = torch.ones_like(patches)  # 默认全被障碍填充
    
    # 转换为整数像素偏差
    dx_int = torch.round(dx).long()
    dy_int = torch.round(dy).long()
    
    for b in range(B):
        ox, oy = dx_int[b].item(), dy_int[b].item()
        
        # 计算原图与目标图的有效切片范围
        src_x1, src_x2 = max(0, ox), min(W, W + ox)
        src_y1, src_y2 = max(0, oy), min(H, H + oy)
        
        dst_x1, dst_x2 = max(0, -ox), min(W, W - ox)
        dst_y1, dst_y2 = max(0, -oy), min(H, H - oy)
        
        if src_x1 < src_x2 and src_y1 < src_y2:
            new_patches[b, :, dst_y1:dst_y2, dst_x1:dst_x2] = patches[b, :, src_y1:src_y2, src_x1:src_x2]
            
    return new_patches

def calc_inclusion_dist(preds, rel_x, rel_y):
    """计算相对点 (rel_x, rel_y) 在预测椭圆内的距离度量 D = (x/a)^2 + (y/b)^2"""
    a = preds[:, 2] + 1e-4
    b = preds[:, 3] + 1e-4
    sin_t = preds[:, 4]
    cos_t = preds[:, 5]
    
    v_rot_x = rel_x * cos_t - rel_y * sin_t
    v_rot_y = rel_x * sin_t + rel_y * cos_t
    
    return (v_rot_x / a)**2 + (v_rot_y / b)**2

def test(model_path="iris_net_best.pth", data_path="data/splits/test_iris.npz"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 挂载模型
    model = IRISNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Successfully loaded '{model_path}'")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Please ensure you have finished training and 'iris_net_best.pth' exists.")
        return
    
    model.eval()

    # 2. 挂载测试集
    patches, labels = load_test_data(data_path, device)
    if patches is None:
        return
        
    total_samples = len(patches)
    print(f"Total test samples: {total_samples}")

    max_batch_size = 512
    y_idx, x_idx = torch.meshgrid(torch.arange(128, device=device), torch.arange(128, device=device), indexing='ij')
    grid_x = x_idx.float().unsqueeze(0).expand(max_batch_size, -1, -1)
    grid_y = y_idx.float().unsqueeze(0).expand(max_batch_size, -1, -1)

    # 3. 测量纯推理时间（消除第一帧启动误差）
    print("\nMeasuring inference time...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            _ = model(patches[:10].float())
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for i in tqdm(range(0, total_samples, max_batch_size), desc="Inference"):
                batch_patches = patches[i:i+max_batch_size].float()
                _ = model(batch_patches)
                
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    
    total_inf_time = end_time - start_time
    ms_per_sample = (total_inf_time / total_samples) * 1000
    fps = total_samples / total_inf_time
    
    # 4. 计算精细化指标
    print("Calculating detailed metrics (IoU, Physics, Constraints)...")
    
    all_iou = []
    all_collision_rate = []
    all_collision_ratio = []
    
    all_center_err = []
    all_axis_err = []
    all_angle_err = []
    
    MAX_ITERS = 3
    all_in_full_iters = [[] for _ in range(MAX_ITERS + 1)]
    
    all_in_2_3 = []
    all_in_1_3 = []
    
    all_poly_in = []
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, max_batch_size), desc="Calculating detailed metrics"):
            end_idx = min(i + max_batch_size, total_samples)
            batch_size = end_idx - i
            
            batch_patches = patches[i:end_idx].float()
            batch_labels = labels[i:end_idx]
            
            b_grid_x = grid_x[:batch_size]
            b_grid_y = grid_y[:batch_size]
            
            with torch.amp.autocast('cuda'):
                preds = model(batch_patches)
                
                # --- A. 基础物理指标 (IoU, 碰撞) ---
                pred_mask = render_soft_ellipse(preds, b_grid_x, b_grid_y, size=128, temperature=100.0) > 0.5
                target_mask = render_soft_ellipse(batch_labels, b_grid_x, b_grid_y, size=128, temperature=100.0) > 0.5
                
                # IoU
                inter = (pred_mask & target_mask).float().sum(dim=(1,2))
                uni = (pred_mask | target_mask).float().sum(dim=(1,2))
                iou = (inter + 1e-6) / (uni + 1e-6)
                all_iou.append(iou)
                
                # 碰撞率 与 碰撞像素占比
                obs_mask = batch_patches.squeeze(1) > 0.5
                overlap = (pred_mask & obs_mask).float()
                collision_pixels = overlap.sum(dim=(1,2))
                
                pred_area = pred_mask.float().sum(dim=(1,2))
                # 计算碰撞面积占整个预测椭圆面积的比例
                coll_ratio = collision_pixels / (pred_area + 1e-6)
                
                all_collision_rate.append((collision_pixels > 0).float())
                all_collision_ratio.append(coll_ratio)
                
                # --- B. 传统几何误差 ---
                c_err = torch.norm(preds[:, 0:2] - batch_labels[:, 0:2], dim=1)
                all_center_err.append(c_err)
                
                a_err = torch.abs(preds[:, 2:4] - batch_labels[:, 2:4]).mean(dim=1)
                all_axis_err.append(a_err)
                
                pred_theta = torch.atan2(preds[:, 4], preds[:, 5])
                target_theta = torch.atan2(batch_labels[:, 4], batch_labels[:, 5])
                angle_diff = torch.abs(pred_theta - target_theta)
                angle_diff = torch.min(angle_diff, math.pi - angle_diff)
                all_angle_err.append(torch.rad2deg(angle_diff))
                
                # --- C. 图片中心点保护范围测试 (含补偿迭代/自回归视界) ---
                # 1. 第 0 阶预测 (One-Shot)
                dx_0, dy_0 = -preds[:, 0], -preds[:, 1]
                dist_0 = calc_inclusion_dist(preds, dx_0, dy_0)
                
                in_full_any = (dist_0 <= 1.0)
                all_in_full_iters[0].append(in_full_any.float())
                all_in_2_3.append((dist_0 <= (2.0/3.0)**2).float())
                all_in_1_3.append((dist_0 <= (1.0/3.0)**2).float())
                
                # 开始最高 MAX_ITERS 补偿迭代
                cum_dx = preds[:, 0]
                cum_dy = preds[:, 1]
                
                for k in range(1, MAX_ITERS + 1):
                    # 以生成的椭圆中心为锚点，平移环境视界
                    shifted_patches = shift_patches_batch(batch_patches, cum_dx, cum_dy)
                    preds_k = model(shifted_patches)
                    
                    # 新的预测相对于原始绝对点位置 P0 的总累计偏移
                    total_dx = cum_dx + preds_k[:, 0]
                    total_dy = cum_dy + preds_k[:, 1]
                    
                    # 检查原本的 P0(-total_dx, -total_dy) 是否已被新扩出的椭圆包裹
                    dist_k = calc_inclusion_dist(preds_k, -total_dx, -total_dy)
                    in_full_any = in_full_any | (dist_k <= 1.0)
                    all_in_full_iters[k].append(in_full_any.float())
                    
                    cum_dx = total_dx
                    cum_dy = total_dy
                
                # --- D. 约束多边形生成与中点包含测试 ---
                preds_np = preds.cpu().float().numpy()
                obs_np = (batch_patches.squeeze(1) > 0.5).cpu().numpy()
                
                # 准备送入多进程的参数
                poly_args = [
                    (
                        preds_np[b_idx, 0],
                        preds_np[b_idx, 1],
                        preds_np[b_idx, 2] + 1e-4,
                        preds_np[b_idx, 3] + 1e-4,
                        preds_np[b_idx, 4],
                        preds_np[b_idx, 5],
                        obs_np[b_idx]
                    )
                    for b_idx in range(batch_size)
                ]
                
                # 多进程并行计算生成及检查约束多边形
                batch_poly_in = pool.map(check_poly_worker, poly_args)
                    
                all_poly_in.append(torch.tensor(batch_poly_in))
    
    pool.close()
    pool.join()
    
    # 汇总计算平均值
    agg_iou = torch.cat(all_iou).mean().item()
    agg_coll_rate = torch.cat(all_collision_rate).mean().item() * 100.0
    agg_coll_ratio = torch.cat(all_collision_ratio).mean().item() * 100.0
    
    agg_c_err = torch.cat(all_center_err).mean().item()
    agg_a_err = torch.cat(all_axis_err).mean().item()
    agg_ang_err = torch.cat(all_angle_err).mean().item()
    
    agg_in_full_iters = [torch.cat(all_in_full_iters[k]).mean().item() * 100.0 for k in range(MAX_ITERS + 1)]
    
    agg_in_2_3 = torch.cat(all_in_2_3).mean().item() * 100.0
    agg_in_1_3 = torch.cat(all_in_1_3).mean().item() * 100.0
    
    agg_poly_in = torch.cat(all_poly_in).mean().item() * 100.0
    
    print("\n" + "="*55)
    print("                [ IRISNET TEST REPORT ] ")
    print("="*55)
    print(f" Total Samples Tested : {total_samples}")
    print(f" Inference Speed      : {ms_per_sample:.4f} ms/sample  |  ({fps:.1f} FPS)")
    print("-" * 55)
    print(" [ Geometrical Precision (几何精度) ]")
    print(f" Center Offset Error  : {agg_c_err:.2f} px")
    print(f" Axis Length Error    : {agg_a_err:.2f} px")
    print(f" Rotation Angle Error : {agg_ang_err:.2f} °")
    print("-" * 55)
    print(" [ Physical Constraints (物理碰撞指标) ]")
    print(f" Model-Target IoU     : {agg_iou:.4f}")
    print(f" Collision Freq (Rate): {agg_coll_rate:.2f} %  (样本整体触碰障碍比例)")
    print(f" Collision Area Ratio : {agg_coll_ratio:.4f} %  (穿模面积占椭圆总面积比例)")
    print("-" * 55)
    print(" [ Center Protection/Inclusion (中心点回环防撞指标) ]")
    print(f" (Iter  0) In 100% Ellipse Area : {agg_in_full_iters[0]:.2f} % (单次网络裸推理)")
    for k in range(1, MAX_ITERS + 1):
        print(f" (Iter {k:2d}) 追踪外扩累计包含率     : {agg_in_full_iters[k]:.2f} % (+{k} Frame)")
    print(f" -----------------------------")
    print(f" In  66% Ellipse Area : {agg_in_2_3:.2f} %  (位于核心范围 2/3 内)")
    print(f" In  33% Ellipse Area : {agg_in_1_3:.2f} %  (极度紧凑的核心 1/3)")
    print("-" * 55)
    print(" [ Constraint Polygon (多边形约束指标) ]")
    print(f" Center in Safe Polygon : {agg_poly_in:.2f} %  (中点在生成的约束多边形内)")
    print("="*55)

if __name__ == "__main__":
    model_path = os.path.join(root_path, "models/iris_net_best.pth")
    data_path = os.path.join(root_path, "data/iris/splits/test_iris.npz")
    test(model_path=model_path, data_path=data_path)