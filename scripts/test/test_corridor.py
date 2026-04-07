import torch
import numpy as np
import time
import math
import multiprocessing
import argparse
from tqdm import tqdm
import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.corridor_constraints.geometry import (
    render_soft_ellipse,
    parse_network_output,
    extract_obstacle_boundary_points,
)
from src.corridor_constraints.model import CorridorEllipseNet
from src.corridor_constraints.generate_safe_polygon import generate_safe_polygon

MODEL_PATH = os.path.join(root_path, "models/iris_net_best.pth")
DATA_PATH = os.path.join(root_path, "data/iris-dataset/splits/test_iris.npz")

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
    pred_data = np.array([dx, dy, a, b, sin_t, cos_t], dtype=float)
    p_mat, c = parse_network_output(pred_data, patch_size=128)
    obs_points = extract_obstacle_boundary_points(obs_mask)
        
    poly = generate_safe_polygon(p_mat, c, obs_points, patch_size=128)

    is_in = False
    center_pt = np.array([64.0, 64.0], dtype=float)
    if poly is not None:
        # 新版 generate_safe_polygon 返回 (A, b)，其中约束为 A x <= b
        if isinstance(poly, tuple) and len(poly) == 2:
            A, b_vec = poly
            if A is not None and b_vec is not None and len(A) > 0:
                is_in = np.all((A @ center_pt) <= (b_vec + 1e-5))
        # 兼容旧版：直接返回顶点
        elif len(poly) >= 3:
            is_in = point_in_convex_poly(poly, center_pt)
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

def test(model_path, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 挂载模型
    model = CorridorEllipseNet().to(device)
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
                # 先做 2pi 周期归一化，再折叠到 [0, pi]
                angle_diff = torch.remainder(pred_theta - target_theta + math.pi, 2 * math.pi) - math.pi
                angle_diff = torch.abs(angle_diff)
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
    print("                [ CorridorEllipseNet TEST REPORT ] ")
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


def _safe_pearson(x, y):
    if len(x) < 2:
        return float("nan")
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a):
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    return ranks


def _safe_spearman(x, y):
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _safe_pearson(rx, ry)


def benchmark_constraint_polygon(model_path, data_path, max_samples=None, warmup=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CorridorEllipseNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Successfully loaded '{model_path}'")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.eval()

    patches, _ = load_test_data(data_path, device)
    if patches is None:
        return

    if max_samples is not None:
        patches = patches[:max_samples]
    total_samples = len(patches)
    print(f"Total samples for benchmark: {total_samples}")

    max_batch_size = 512
    all_preds = []
    all_obs = []

    print("Running model inference for benchmark inputs...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            _ = model(patches[: min(10, total_samples)].float())

        for i in tqdm(range(0, total_samples, max_batch_size), desc="Inference (benchmark)"):
            batch_patches = patches[i:i + max_batch_size].float()
            with torch.amp.autocast('cuda'):
                preds = model(batch_patches)
            all_preds.append(preds.cpu().float().numpy())
            all_obs.append((batch_patches.squeeze(1) > 0.5).cpu().numpy())

    preds_np = np.concatenate(all_preds, axis=0)
    obs_np = np.concatenate(all_obs, axis=0)

    print("Benchmarking generate_safe_polygon latency...")
    times_ms = []
    obs_counts = []
    valid_poly = 0

    warmup = max(0, min(warmup, total_samples))
    for i in tqdm(range(total_samples), desc="Polygon timing"):
        dx, dy, a, b, sin_t, cos_t = preds_np[i]
        pred_data = np.array([dx, dy, a, b, sin_t, cos_t], dtype=float)
        p_mat, c = parse_network_output(pred_data, patch_size=128)
        obs_points = extract_obstacle_boundary_points(obs_np[i])

        t0 = time.perf_counter()
        poly = generate_safe_polygon(p_mat, c, obs_points, patch_size=128)
        t1 = time.perf_counter()

        if i >= warmup:
            times_ms.append((t1 - t0) * 1000.0)
            obs_counts.append(float(obs_points.shape[0]))

            if poly is not None:
                if isinstance(poly, tuple) and len(poly) == 2:
                    A, b_vec = poly
                    if A is not None and b_vec is not None and len(A) > 0:
                        valid_poly += 1
                elif len(poly) >= 3:
                    valid_poly += 1

    times_ms = np.array(times_ms, dtype=float)
    obs_counts = np.array(obs_counts, dtype=float)

    if len(times_ms) == 0:
        print("No benchmark samples remained after warmup.")
        return

    pearson_r = _safe_pearson(obs_counts, times_ms)
    spearman_r = _safe_spearman(obs_counts, times_ms)
    slope = np.polyfit(obs_counts, times_ms, deg=1)[0] if len(times_ms) >= 2 else float("nan")

    p50 = np.percentile(times_ms, 50)
    p90 = np.percentile(times_ms, 90)
    p95 = np.percentile(times_ms, 95)
    p99 = np.percentile(times_ms, 99)

    # 5 个分位桶，观察障碍点数量与耗时变化趋势
    edges = np.quantile(obs_counts, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    print("\n" + "=" * 70)
    print("      [ Constraint Polygon Latency vs Obstacle Count Benchmark ]")
    print("=" * 70)
    print(f"Samples (after warmup): {len(times_ms)}")
    print(f"Valid polygon generated : {valid_poly}/{len(times_ms)}")
    print("-" * 70)
    print(f"Mean latency            : {times_ms.mean():.4f} ms")
    print(f"Median latency (P50)    : {p50:.4f} ms")
    print(f"P90 / P95 / P99         : {p90:.4f} / {p95:.4f} / {p99:.4f} ms")
    print("-" * 70)
    print(f"Obstacle Count -> Time Pearson r  : {pearson_r:.4f}")
    print(f"Obstacle Count -> Time Spearman r : {spearman_r:.4f}")
    print(f"Linear slope (ms per obstacle)    : {slope:.8f}")
    print("-" * 70)
    print("Bucket stats by obstacle-count quantiles:")

    for bi in range(5):
        lo = edges[bi]
        hi = edges[bi + 1]
        if bi < 4:
            mask = (obs_counts >= lo) & (obs_counts < hi)
        else:
            mask = (obs_counts >= lo) & (obs_counts <= hi)
        if np.any(mask):
            cnt = int(mask.sum())
            avg_obs = float(obs_counts[mask].mean())
            avg_t = float(times_ms[mask].mean())
            print(f" Q{bi+1}: n={cnt:5d}, avg_obs={avg_obs:8.2f}, avg_time={avg_t:8.4f} ms")
        else:
            print(f" Q{bi+1}: n=    0")

    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CorridorEllipseNet and benchmark polygon generation.")
    parser.add_argument("--mode", choices=["test", "benchmark", "both"], default="test")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    if args.mode in ["test", "both"]:
        test(model_path=args.model_path, data_path=args.data_path)
    if args.mode in ["benchmark", "both"]:
        benchmark_constraint_polygon(
            model_path=args.model_path,
            data_path=args.data_path,
            max_samples=args.max_samples,
            warmup=args.warmup,
        )