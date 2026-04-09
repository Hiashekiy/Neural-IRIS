import os
import glob
import numpy as np
import cvxpy as cp
from tqdm import tqdm
import warnings
import multiprocessing
import scipy.ndimage as ndimage
import logging
from queue import Empty
import argparse
import time
import threading

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MAP_DIR = os.path.join(PROJECT_ROOT, "data", "street-map")

try:
    import cvxopt
    from cvxopt import matrix, solvers
    CVXOPT_AVAILABLE = True
except ImportError:
    CVXOPT_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 禁用 cvxopt 求解器的冗长输出
if CVXOPT_AVAILABLE:
    solvers.options['show_progress'] = False

def load_moving_ai_map(filepath):
    """解析 Moving AI .map 文件"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    grid = np.zeros((height, width), dtype=np.uint8)
    
    for i, line in enumerate(lines[4:]):
        for j, char in enumerate(line.strip()):
            if char in ['@', 'O', 'T', 'W']:
                grid[i, j] = 1  # 障碍物
    return grid

def get_padded_patch(grid, cx, cy, size=128):
    """提取局部切片，超界部分用 1 (障碍物) 填充"""
    half = size // 2
    h, w = grid.shape
    patch = np.ones((size, size), dtype=np.uint8)
    
    x_min = max(0, cx - half)
    x_max = min(w, cx + half)
    y_min = max(0, cy - half)
    y_max = min(h, cy + half)
    
    px_min = half - (cx - x_min)
    px_max = half + (x_max - cx)
    py_min = half - (cy - y_min)
    py_max = half + (y_max - cy)
    
    patch[py_min:py_max, px_min:px_max] = grid[y_min:y_max, x_min:x_max]
    return patch

def extract_obstacle_constraints(patch, dilation_iters=1, boundary_jitter=1):
    """对障碍物做膨胀，并将边界点扩展成更密的约束点云。"""
    struct = np.ones((3, 3), dtype=bool)
    obstacle_mask = patch == 1

    if dilation_iters > 0:
        obstacle_mask = ndimage.binary_dilation(obstacle_mask, structure=struct, iterations=dilation_iters)

    boundary_mask = obstacle_mask ^ ndimage.binary_erosion(obstacle_mask, structure=struct)
    py, px = np.where(boundary_mask)
    boundary_points = np.column_stack((px, py)).astype(float)

    if boundary_jitter > 0 and len(boundary_points) > 0:
        offsets = []
        for dy in range(-boundary_jitter, boundary_jitter + 1):
            for dx in range(-boundary_jitter, boundary_jitter + 1):
                if dx == 0 and dy == 0:
                    continue
                offsets.append((dx, dy))

        dense_points = [boundary_points]
        for dx, dy in offsets:
            shifted = boundary_points + np.array([dx, dy], dtype=float)
            shifted[:, 0] = np.clip(shifted[:, 0], 0, patch.shape[1] - 1)
            shifted[:, 1] = np.clip(shifted[:, 1], 0, patch.shape[0] - 1)
            dense_points.append(shifted)

        boundary_points = np.unique(np.vstack(dense_points), axis=0)

    return obstacle_mask, boundary_points

def solve_iris_offline(obs_points, bounds=[0, 128, 0, 128], seed=[64.0, 64.0], max_iters=15, K_bins=32, **kwargs):
    """
    高效 IRIS 求解器 (基于射线投影和凸多面体切面)
    1. 计算障碍物相对于当前椭圆在各方向（K_bins 个扇区）上的最近点，避免引入海量冗余约束。
    2. 计算分离超平面（由于只取最近点，保证切割后内部区域绝对安全且约束极少，仅有 K_bins + 4 个）。
    3. 利用 CVXPY (CLARABEL) 求解最大内接椭圆(MVIE)。
    """
    seed_c = np.array(seed, dtype=float)
    xmin, xmax, ymin, ymax = bounds
    
    if len(obs_points) == 0:
        radius = min(xmax - xmin, ymax - ymin) / 2.0
        return np.eye(2) * radius, seed_c
        
    # 初始化：寻找距离 seed 最近的障碍物，确定初始安全半径
    dists = np.linalg.norm(obs_points - seed_c, axis=1)
    min_dist = np.min(dists)
    
    # 给定一个合理的初始半径，至少 1.0 以上，避免初期梯度完全消失
    init_radius = max(min_dist * 0.5, 1.0)
    P_val = np.eye(2) * init_radius
    c_val = seed_c
    
    for i in range(max_iters):
        try:
            P_inv = np.linalg.inv(P_val)
            # P_inv2 = P^{-T} P^{-1} 用于把变换空间的法向量转回原空间
            P_inv2 = P_inv.T @ P_inv
        except np.linalg.LinAlgError:
            break
            
        # 1. 坐标变换：把所有障碍物转换到以当前椭圆 c_val 为中心的“单位球空间”
        obs_shifted = obs_points - c_val
        obs_trans = obs_shifted @ P_inv.T
        
        # 2. 计算在转换球形空间下的极坐标（角度和距离）
        dists_trans = np.linalg.norm(obs_trans, axis=1)
        angles = np.arctan2(obs_trans[:, 1], obs_trans[:, 0])
        
        # 3. 将 360 度空间均匀划分为 K_bins 个扇区
        bin_indices = np.floor((angles + np.pi) / (2 * np.pi) * K_bins).astype(int)
        bin_indices = np.clip(bin_indices, 0, K_bins - 1)
        
        A, b = [], []
        
        # 加入物理地图边界的 4 面墙
        A.extend([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        b.extend([xmax, -xmin, ymax, -ymin])
        
        # 4. 在每个扇区中，提取距离椭圆中心“在空间比例下最近”的障碍物作为关键约束
        for k in range(K_bins):
            mask = bin_indices == k
            if not np.any(mask):
                continue
                
            sector_dists = dists_trans[mask]
            min_idx = np.argmin(sector_dists)
            
            # 获取原坐标系下导致该扇形瓶颈的障碍点
            active_obs = obs_points[mask][min_idx]
            
            # 求解该点的分离超平面法向量 (基于当前椭圆形状的梯度方向)
            n = P_inv2 @ (active_obs - c_val)
            norm_n = np.linalg.norm(n)
            
            if norm_n > 1e-6:
                n_norm = n / norm_n
                A.append(n_norm)
                b.append(np.dot(n_norm, active_obs))
                
        A_mat = np.array(A)
        b_vec = np.array(b)
        
        # 5. 建立凸多边形内的最大内接椭圆模型并求解
        P = cp.Variable((2, 2), PSD=True)
        c = cp.Variable(2)
        
        constraints = [cp.norm(P @ A_mat[j]) + A_mat[j] @ c <= b_vec[j] for j in range(len(A_mat))]
        objective = cp.Maximize(cp.log_det(P))
        prob = cp.Problem(objective, constraints)
        
        try:
            # 优先使用 CLARABEL (现代、稳定，比 ECOS / SCS 在对数行列式上更好)
            prob.solve(solver=cp.CLARABEL, verbose=False)
            
            if prob.status not in ('optimal', 'optimal_inaccurate') or P.value is None:
                # 若失败，启用具有较高容差的 SCS 备用
                prob.solve(solver=cp.SCS, max_iters=500, eps=1e-3, verbose=False)
        except Exception:
            break
            
        if prob.status not in ('optimal', 'optimal_inaccurate') or P.value is None or c.value is None:
            break
            
        vol_old = np.linalg.det(P_val)
        vol_new = np.linalg.det(P.value)
        
        P_val = P.value
        c_val = c.value
        
        # 6. 收敛判定：若体积增长不足 5%，则结束迭代
        if vol_old > 1e-8 and (vol_new - vol_old) / vol_old < 0.05:
            break
            
    return P_val, c_val

def convert_labels(P, c, patch_size=128):
    """将 P, c 转换为网络可学习的物理标签: dx, dy, a, b, sin_theta, cos_theta"""
    seed_center = patch_size / 2.0
    dx = c[0] - seed_center
    dy = c[1] - seed_center
    
    P = (P + P.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(P)
    
    a, b = np.abs(eigvals[0]), np.abs(eigvals[1])
    
    v = eigvecs[:, 0]
    theta = np.arctan2(v[1], v[0])
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    return np.array([dx, dy, a, b, sin_theta, cos_theta], dtype=np.float32)


def is_ellipse_safe(P, c, obstacle_mask, interior_samples=48, boundary_samples=48):
    """用边界点 + 随机内部点做快速验收，避免把明显穿过障碍的椭圆写入数据集。"""
    if P is None or c is None:
        return False

    try:
        P = (P + P.T) / 2.0
        if np.linalg.det(P) <= 1e-10:
            return False
    except np.linalg.LinAlgError:
        return False

    angles = np.linspace(0.0, 2.0 * np.pi, boundary_samples, endpoint=False)
    unit_boundary = np.vstack((np.cos(angles), np.sin(angles)))
    boundary_points = (P @ unit_boundary).T + c

    rng = np.random.default_rng()
    radii = np.sqrt(rng.random(interior_samples))
    theta = rng.uniform(0.0, 2.0 * np.pi, interior_samples)
    unit_interior = np.vstack((radii * np.cos(theta), radii * np.sin(theta)))
    interior_points = (P @ unit_interior).T + c

    points = np.vstack((boundary_points, interior_points))
    xs = np.rint(points[:, 0]).astype(int)
    ys = np.rint(points[:, 1]).astype(int)

    valid = (
        (xs >= 0) & (xs < obstacle_mask.shape[1]) &
        (ys >= 0) & (ys < obstacle_mask.shape[0])
    )
    if not np.all(valid):
        return False

    return not np.any(obstacle_mask[ys, xs])


def is_trivial_ellipse(P, c, patch_size, area_threshold=16.0, center_threshold=2.0):
    """判定是否仍接近初始椭圆或过小，通常意味着求解器没有真正发散或遇到狭窄空间。"""
    if P is None or c is None:
        return True

    P = np.asarray(P, dtype=float)
    c = np.asarray(c, dtype=float)
    if P.shape != (2, 2) or c.shape != (2,):
        return True

    det_P = np.linalg.det(P)
    if det_P <= 0:
        return True

    # 检查面积：如果椭圆的行列式 (等价于 a*b) 小于给定阈值 (例如对应半径小于 4 的纯圆)
    if det_P <= area_threshold:
        return True

    center = np.array([patch_size / 2.0, patch_size / 2.0], dtype=float)
    if np.linalg.norm(c - center) > center_threshold:
        return False

    return False

def process_batch(args):
    """子进程函数：处理一个 batch 的采样坐标点（轻量级，无进度条冲突）"""
    map_file, points_batch, patch_size, progress_queue, obstacle_dilation_iters, boundary_jitter = args
    grid = load_moving_ai_map(map_file)
    
    batch_patches = []
    batch_labels = []
    map_name = os.path.basename(map_file)
    batch_start = time.time()

    progress_queue.put((map_name, "batch_start", len(points_batch)))
    
    # 子进程轻量级处理，每10个点汇报一次进度
    for idx, (cx, cy) in enumerate(points_batch):
        patch = get_padded_patch(grid, cx, cy, size=patch_size)
        obstacle_mask, obs_points = extract_obstacle_constraints(
            patch,
            dilation_iters=obstacle_dilation_iters,
            boundary_jitter=boundary_jitter,
        )
        
        free_y, free_x = np.where(patch == 0)
        candidate_seeds = [np.array([patch_size / 2, patch_size / 2], dtype=float)]
        if len(free_x) > 0:
            free_points = np.column_stack((free_x, free_y)).astype(float)
            pick_count = min(3, len(free_points))
            picked = free_points[np.random.choice(len(free_points), size=pick_count, replace=False)]
            candidate_seeds.extend([pt for pt in picked])

        P = None
        c = None
        for seed_point in candidate_seeds:
            P_try, c_try = solve_iris_offline(
                obs_points,
                bounds=[0, patch_size, 0, patch_size],
                seed=seed_point,
                max_iters=20,
            )
            if P_try is None or c_try is None:
                continue
            if not is_trivial_ellipse(P_try, c_try, patch_size):
                P, c = P_try, c_try
                break

        if P is None or c is None:
            P, c = solve_iris_offline(
                obs_points,
                bounds=[0, patch_size, 0, patch_size],
                seed=[patch_size / 2, patch_size / 2],
                max_iters=20,
            )
        
        if P is not None and c is not None and is_ellipse_safe(P, c, obstacle_mask):
            labels = convert_labels(P, c, patch_size)
            batch_patches.append(patch)
            batch_labels.append(labels)
        
        # 每处理10个点，汇报一次进度到主进程
        if (idx + 1) % 10 == 0:
            progress_queue.put((map_name, idx + 1, len(points_batch)))
    
    # 最后汇报完成
    progress_queue.put((map_name, len(points_batch), len(points_batch)))
    progress_queue.put((map_name, "batch_done", len(points_batch), time.time() - batch_start))
    return batch_patches, batch_labels, len(points_batch)

def generate_dataset(map_dir, output_file, density=0.02, patch_size=128, batch_size=200, max_maps=None, rng_seed=None,
                     obstacle_dilation_iters=1, boundary_jitter=1):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    map_files = glob.glob(os.path.join(map_dir, "*.map"))
    map_files = sorted(map_files)
    if max_maps is not None:
        map_files = map_files[:max_maps]
    all_patches = []
    all_labels = []
    
    tasks = []
    total_points_to_process = 0
    
    print("Pre-computing sample coordinates and creating batches...")
    for map_file in tqdm(map_files, desc="Parsing Maps"):
        grid = load_moving_ai_map(map_file)
        h, w = grid.shape
        
        free_y, free_x = np.where(grid == 0)
        obs_y, obs_x = np.where(grid == 1)
        
        if len(free_x) == 0: continue
        
        free_points = np.column_stack((free_x, free_y))
        num_samples = int(len(free_points) * density)
        num_samples = max(100, min(15000, num_samples))
        
        # 使用 numpy 向量化一次性生成所有有效采样点
        map_target_points = []
        
        # 1. 均匀采样点
        num_uniform = num_samples // 2
        idx_u = np.random.randint(len(free_points), size=num_uniform)
        map_target_points.extend(free_points[idx_u].tolist())
        
        # 2. 边界噪声采样点
        num_boundary = num_samples - num_uniform
        if len(obs_x) > 0:
            idx_b = np.random.randint(len(obs_x), size=num_boundary)
            base_pts = np.column_stack((obs_x[idx_b], obs_y[idx_b]))
            noise = np.random.randint(-5, 6, size=(num_boundary, 2))
            noisy_pts = base_pts + noise
            
            # 裁剪边界
            noisy_pts[:, 0] = np.clip(noisy_pts[:, 0], 0, w - 1)
            noisy_pts[:, 1] = np.clip(noisy_pts[:, 1], 0, h - 1)
            
            # 过滤掉依然落在障碍物上的点
            for pt in noisy_pts:
                if grid[pt[1], pt[0]] == 0:
                    map_target_points.append(pt.tolist())
                    
        # 将当前地图的点集划分为多个小 Batch
        for i in range(0, len(map_target_points), batch_size):
            points_batch = map_target_points[i:i+batch_size]
            tasks.append((map_file, points_batch, patch_size, obstacle_dilation_iters, boundary_jitter))
            total_points_to_process += len(points_batch)

    print(f"Total points to process: {total_points_to_process} in {len(tasks)} batches.")
    print(f"Starting multiprocessing pool with {multiprocessing.cpu_count()} workers...")
    print("(Real-time progress below)\n")
    
    # 使用 Manager 创建进度队列
    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()
        # 更新任务列表，加入进度队列
        tasks_with_queue = [(map_file, points_batch, patch_size, progress_queue, obstacle_dilation_iters, boundary_jitter) 
                   for map_file, points_batch, patch_size, obstacle_dilation_iters, boundary_jitter in tasks]

        stop_event = threading.Event()

        def progress_monitor():
            while not stop_event.is_set() or not progress_queue.empty():
                try:
                    msg = progress_queue.get(timeout=0.5)
                except Empty:
                    continue

                if len(msg) == 3 and msg[1] == "batch_start":
                    map_name, _, batch_count = msg
                    print(f"[start] {map_name}: {batch_count} points queued", flush=True)
                elif len(msg) == 4 and msg[1] == "batch_done":
                    map_name, _, total_in_batch, elapsed = msg
                    print(f"[done ] {map_name}: {total_in_batch} points in {elapsed:.1f}s", flush=True)

        monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
        monitor_thread.start()
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            # 异步提交所有任务
            async_results = pool.imap_unordered(process_batch, tasks_with_queue, chunksize=2)
            
            # 主进程监听进度队列和结果
            completed_points = 0
            pending_results = len(tasks_with_queue)
            
            with tqdm(total=total_points_to_process, desc="Overall Progress", unit="point") as pbar:
                while pending_results > 0:
                    # 尝试读取一个完成的结果
                    try:
                        patch_list, label_list, processed_count = next(async_results, None)
                        if patch_list is None:
                            break
                        all_patches.extend(patch_list)
                        all_labels.extend(label_list)
                        pbar.update(processed_count)
                        pending_results -= 1
                    except Exception:
                        break

        stop_event.set()
        monitor_thread.join(timeout=2.0)
    
    if len(all_patches) == 0:
        print("Warning: No valid samples generated!")
        return
            
    all_patches = np.array(all_patches, dtype=np.uint8)
    all_labels = np.array(all_labels, dtype=np.float32)
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    np.savez_compressed(output_file, patches=all_patches, labels=all_labels)
    print(f"\n\nDataset successfully generated and saved to {output_file}.")
    print(f"Final valid samples: {len(all_patches)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate IRIS training dataset")
    parser.add_argument("--map-dir", type=str, default=DEFAULT_MAP_DIR)
    parser.add_argument("--output", type=str, default="iris_dataset.npz")
    parser.add_argument("--density", type=float, default=0.02)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-maps", type=int, default=None, help="只处理前 N 张地图，适合小批量预览")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，保证预览结果可复现")
    parser.add_argument("--obstacle-dilation", type=int, default=1, help="障碍膨胀迭代次数，越大越保守")
    parser.add_argument("--boundary-jitter", type=int, default=1, help="边界点扩展半径，越大边界点越密")
    args = parser.parse_args()

    if args.max_maps is not None:
        print(f"Preview mode enabled: only processing first {args.max_maps} maps.")

    generate_dataset(
        args.map_dir,
        args.output,
        density=args.density,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        max_maps=args.max_maps,
        rng_seed=args.seed,
        obstacle_dilation_iters=args.obstacle_dilation,
        boundary_jitter=args.boundary_jitter,
    )
