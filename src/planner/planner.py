import math
import numpy as np
import os
import sys
import time
import torch
import torch.nn.functional as F
import cv2
from scipy.spatial import cKDTree


# --- 路径处理 ---
# 将父目录加入系统路径，确保能够找到 planner 模块
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.planner.car_model import DynamicModel
from src.corridor_constraints.generate_safe_polygon import generate_safe_polygon
from src.corridor_constraints.geometry import parse_network_output
from src.corridor_constraints.model import CorridorEllipseNet
try:
    from experiment.ours_corridor_cpp.method import infer_polygon_batch as cpp_infer_polygon_batch
    from experiment.ours_corridor_cpp.method import infer_polygon_batch_with_ellipse as cpp_infer_polygon_batch_with_ellipse
except Exception:
    cpp_infer_polygon_batch = None
    cpp_infer_polygon_batch_with_ellipse = None
from hpipm_python import (hpipm_ocp_qp_dim, hpipm_ocp_qp,
                          hpipm_ocp_qp_sol, hpipm_ocp_qp_solver_arg, hpipm_ocp_qp_solver)

# =========================================================
# 2. MPC 规划器主体
# =========================================================
class Planner:
    def __init__(self, sample_time: float, horizon_steps: int, veh_wheelbase=2.0):
        # ==========================================
        # 1. 基础系统与维度参数 (System & Dimensions)
        # ==========================================
        self.sample_time = sample_time       # 控制周期/采样时间 (dt)
        self.horizon_steps = horizon_steps   # 预测时域长度 (N，即向前预测多少步)
        self.model = DynamicModel(sample_time, wheelbase=veh_wheelbase) # 车辆动力学/运动学模型

        # --- 状态维度对齐 ---
        # 状态量 x = [x坐标, y坐标, 偏航角phi, 速度v, 转向角psi]
        self.nx = self.model.nx  # 5 维
        
        # --- 控制维度扩充 (软约束支持) ---
        self.N_faces = 12 # 多边形约束面数 (如 12边形，适配更复杂通道)
        # 真实物理控制量 u_orig = [加速度a, 转向角速度omega]
        self.nu_orig = self.model.nu  # 2 维
        # QP优化求解控制量 u = [加速度a, 转向角速度omega, s1..sN]
        self.nu = self.nu_orig + self.N_faces

        # ==========================================
        # 2. 状态追踪惩罚权重 (State Penalty - Q矩阵对应项)
        # ==========================================
        self.base_Q_lat = 3.0       # 横向误差惩罚：最核心权重，迫使车辆紧贴参考路径
        self.base_Q_lon = 10.0      # 纵向误差惩罚：控制车辆在路径上的相对进度误差
        self.base_Q_heading = 10.0  # 偏航角误差惩罚：保证车头朝向与路径切线方向一致
        self.base_Q_vel = 5.0       # 速度误差惩罚：迫使车辆达到设定的目标巡航速度
        self.base_R_steer = 10.0    # 转向角状态惩罚：限制过大的绝对打轮角度，抑制车辆“画龙”

        # ==========================================
        # 3. 控制输出惩罚权重 (Control Effort Penalty - R矩阵对应项)
        # ==========================================
        self.base_R_accel = 1.0       # 加速度控制成本：惩罚急踩油门/急刹车，保证纵向乘坐舒适性
        self.base_R_rate_steer = 10.0 # 转向角速度控制成本：惩罚猛打方向盘，保证横向平顺性

        # ==========================================
        # 4. 避障、奖励与软约束机制权重 (Safety & Reward)
        # ==========================================
        # 近邻障碍物斥力权重：用于放大泰勒展开计算出的障碍物二次排斥力
        self.base_weight_obs = 50.0   
        # 松弛因子惩罚权重：提供巨大的越界代价，迫使车辆严格待在走廊内 (避免权重过大导致 Bang-Bang 控制震荡)
        self.base_weight_slack = 50000.0 
        # 基础速度奖励常数：在目标代价函数中提供线性的速度激励梯度
        self.base_speed_reward_c = 5.0
        # 动态读取安全边界内缩距离（根据 config.json 中的车辆宽度与缓冲）
        import json, os
        cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.json'))
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)['vehicle_parameters']
                # 让走廊至少往内收缩 车宽/2 的距离，外加一个极小的固定安全余量比如 0.1m
                # 或者直接使用一半的车宽 + cfg 设置里面的 safety_clearance_m
                self.r_safe = (cfg.get('car_width_m', 1.4) / 2.0) + cfg.get('safety_clearance_m', 0.1)
        except Exception as e:
            print(f"Failed to read config for safety radius, using default: {e}")   
            self.r_safe = 0.8

        # ==========================================
        # 5. 求解稳定性与数值缩放 (Solver Numerical Scaling)
        # ==========================================
        # 全局代价缩放因子：等比例放大目标函数，防止矩阵数值过小导致 HPIPM 内点法误判收敛或精度不足
        self.cost_scaling = 10.0     
        # 终端代价放大倍率：专门针对预测时域最后一个点 (Terminal State) 的额外惩罚，保证 MPC 算法的闭环渐近稳定性
        self.terminal_scale = 10.0

        self.pos_ub = np.array([100.0, 100.0])
        
        # 优先使用 C++ GPU 批量通道生成。
        os.environ.setdefault("OURS_CPP_BACKEND", "gpu")
        self.use_cpp_corridor = cpp_infer_polygon_batch is not None
        if not self.use_cpp_corridor:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.iris_net = CorridorEllipseNet().to(self.device)
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/iris_net_best.pth'))
            if os.path.exists(model_path):
                try:
                    self.iris_net.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.iris_net.eval()
                except Exception as e:
                    print(f"Failed to load CorridorEllipseNet weights: {e}")
            else:
                print(f"Warning: CorridorEllipseNet model path unreached: {model_path}")
        self.patch_size = 128
        
        # 初始化 HPIPM (高性能内点法) QP 求解器
        self.qp_problem, self.qp_solution, self.qp_solver = self._create_hpipm_solver()

    def _create_hpipm_solver(self):
        """
        初始化 HPIPM 求解器的内存维度和算法参数。
        """
        dims = hpipm_ocp_qp_dim(self.horizon_steps)
        dims.set('nx', self.nx, 0, self.horizon_steps)
        dims.set('nu', self.nu, 0, self.horizon_steps - 1)
        dims.set('nbx', self.nx, 0, self.horizon_steps) 
        dims.set('nbu', self.nu, 0, self.horizon_steps - 1) 

        # 使用一般线性约束 ng = self.N_faces 来限制安全多边形
        dims.set('ng', self.N_faces, 0, self.horizon_steps - 1) 
        
        qp_problem = hpipm_ocp_qp(dims)
        qp_solution = hpipm_ocp_qp_sol(dims)
        
        # speed_abs 模式: 牺牲部分极端精度换取极速求解
        solver_arg = hpipm_ocp_qp_solver_arg(dims, 'speed_abs')
        solver_arg.set('mu0', 1e1)
        solver_arg.set('iter_max', 50) 
        solver_arg.set('tol_stat', 1e-3)
        solver_arg.set('tol_eq', 1e-3)
        solver_arg.set('tol_ineq', 1e-3)
        solver_arg.set('tol_comp', 1e-3)

        qp_solver = hpipm_ocp_qp_solver(dims, solver_arg)
        return qp_problem, qp_solution, qp_solver

    def set_bounds(self, pos_lb=None, pos_ub=None, 
                   v_min=-10.0, v_max=10.0, 
                   steer_min=-0.6, steer_max=0.6, 
                   accel_min=-5.0, accel_max=5.0, 
                   steer_rate_min=-0.15, steer_rate_max=0.15):
        """
        将物理限制 (场地大小、速度限制、机械限制) 写入 QP 求解器中
        """
        if pos_lb is None: pos_lb = np.array([-10.0, -10.0])
        if pos_ub is None: pos_ub = self.pos_ub

        # 构造各阶段的状态维度边界向量
        state_lower_bound = np.array([pos_lb[0], pos_lb[1], -1e5, v_min, steer_min]).reshape(-1, 1)
        state_upper_bound = np.array([pos_ub[0], pos_ub[1],  1e5, v_max, steer_max]).reshape(-1, 1)
        
        # 构造控制输入边界 (为 s1..sN 设置下限 0，严禁负松弛)
        input_lower_bound = np.zeros((self.nu, 1))
        input_upper_bound = np.zeros((self.nu, 1))
        
        input_lower_bound[0, 0] = accel_min
        input_lower_bound[1, 0] = steer_rate_min
        input_upper_bound[0, 0] = accel_max
        input_upper_bound[1, 0] = steer_rate_max
        
        for i in range(self.nu_orig, self.nu):
            input_lower_bound[i, 0] = 0.0
            input_upper_bound[i, 0] = 1e5

        # 遍历预测域应用边界约束
        for k in range(self.horizon_steps + 1):
            self.qp_problem.set('Jbx', np.eye(self.nx), k) 
            if k > 0:
                self.qp_problem.set('lbx', state_lower_bound, k)
                self.qp_problem.set('ubx', state_upper_bound, k)
            if k < self.horizon_steps:
                self.qp_problem.set('Jbu', np.eye(self.nu), k) 
                self.qp_problem.set('lbu', input_lower_bound, k)
                self.qp_problem.set('ubu', input_upper_bound, k)

    def _build_local_kdtree(self, occupancy_map, ego_state, map_resolution, radius=30.0):
        """
        直接对局部栅格地图提取坐标点并建立 cKDTree。
        """
        cx = int(ego_state[0] / map_resolution)
        cy = int(ego_state[1] / map_resolution)
        r_grid = int(radius / map_resolution)

        h, w = occupancy_map.shape
        x_min = max(0, cx - r_grid)
        x_max = min(w, cx + r_grid + 1)
        y_min = max(0, cy - r_grid)
        y_max = min(h, cy + r_grid + 1)
        
        local_map = occupancy_map[y_min:y_max, x_min:x_max]
        obs_indices = np.argwhere(local_map == 1) 
        
        if len(obs_indices) == 0:
            return None, np.empty((0, 2))

        # 栅格索引转真实物理坐标系
        obs_x = (obs_indices[:, 1] + x_min + 0.5) * map_resolution
        obs_y = (obs_indices[:, 0] + y_min + 0.5) * map_resolution
        local_obs = np.column_stack((obs_x, obs_y))
        
        return cKDTree(local_obs), local_obs

    def _predict_ellipse_corridors_batched(self, occupancy_map, states_xy_list, map_resolution, max_bound=10.0):
        # 批量神经网络推理
        batch_size = len(states_xy_list)

        patch_np_list = []
        obs_points_list = []
        c_x_ints = []
        c_y_ints = []
        
        local_radius_m = 10.0
        r_pixel_f = local_radius_m / map_resolution
        r_pixel = int(np.round(r_pixel_f))
        side_len = int(2 * r_pixel)
        
        h, w = occupancy_map.shape
        
        for state_xy in states_xy_list:
            cx_f = state_xy[0] / map_resolution
            cy_f = state_xy[1] / map_resolution
            c_x_int = int(np.round(cx_f))
            c_y_int = int(np.round(cy_f))
            c_x_ints.append(c_x_int)
            c_y_ints.append(c_y_int)
            
            raw_patch = np.zeros((side_len, side_len), dtype=np.uint8)
            
            x_min_map = c_x_int - r_pixel
            x_max_map = c_x_int + r_pixel
            y_min_map = c_y_int - r_pixel
            y_max_map = c_y_int + r_pixel
            
            valid_x_min = max(0, x_min_map)
            valid_x_max = min(w, x_max_map)
            valid_y_min = max(0, y_min_map)
            valid_y_max = min(h, y_max_map)
            
            p_x_min = valid_x_min - x_min_map
            p_x_max = side_len - (x_max_map - valid_x_max)
            p_y_min = valid_y_min - y_min_map
            p_y_max = side_len - (y_max_map - valid_y_max)
            
            if valid_y_max > valid_y_min and valid_x_max > valid_x_min:
                raw_patch[p_y_min:p_y_max, p_x_min:p_x_max] = occupancy_map[valid_y_min:valid_y_max, valid_x_min:valid_x_max]
            
            patch_np = cv2.resize(raw_patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            patch_np_list.append(patch_np)
            
            obs_y, obs_x = np.where(patch_np >= 0.5)
            if len(obs_x) > 0:
                # 加入轻量降采样，步长为 2 可以在不损失包络精度的前提下直接砍掉 75% 的二次型测距计算量
                obs_y = obs_y[::2]
                obs_x = obs_x[::2]
                obs_points_list.append(np.column_stack((obs_x, obs_y)).astype(float))
            else:
                obs_points_list.append(np.array([]))
                
        if self.use_cpp_corridor and cpp_infer_polygon_batch is not None:
            def _polygon_to_halfspaces(poly_world: np.ndarray, center_world: np.ndarray):
                A_poly = np.zeros((self.N_faces, 2))
                b_poly = np.zeros(self.N_faces)

                if poly_world is None or len(poly_world) < 3:
                    A_poly[:4, :] = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
                    b_poly[0] = center_world[0] + 1.0
                    b_poly[1] = -(center_world[0] - 1.0)
                    b_poly[2] = center_world[1] + 1.0
                    b_poly[3] = -(center_world[1] - 1.0)
                    b_poly[4:] = 1e5
                    return A_poly, b_poly, center_world.copy()

                poly_world = np.asarray(poly_world, dtype=float)
                signed_area = 0.5 * float(np.sum(poly_world[:, 0] * np.roll(poly_world[:, 1], -1) - poly_world[:, 1] * np.roll(poly_world[:, 0], -1)))
                if signed_area < 0.0:
                    poly_world = poly_world[::-1]

                valid_num = 0
                centroid = poly_world.mean(axis=0)
                for i in range(len(poly_world)):
                    p1 = poly_world[i]
                    p2 = poly_world[(i + 1) % len(poly_world)]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    length = math.hypot(dx, dy)
                    if length < 1e-8:
                        continue
                    nx = dy / length
                    ny = -dx / length
                    A_poly[valid_num] = [nx, ny]
                    b_poly[valid_num] = nx * p1[0] + ny * p1[1] - self.r_safe
                    valid_num += 1
                    if valid_num >= self.N_faces:
                        break

                if valid_num < self.N_faces:
                    b_poly[valid_num:] = 1e5

                return A_poly, b_poly, centroid

            def _safe_visual_shape(A_poly: np.ndarray, b_poly: np.ndarray, c_world: np.ndarray):
                # 为可视化构造一个保证在 A x <= b 内部的保守内接圆。
                # 绘制端使用 c + P_inv @ unit_circle，因此这里返回 r I。
                valid = (np.linalg.norm(A_poly, axis=1) > 1e-8) & (b_poly < 1e4)
                if not np.any(valid):
                    return np.eye(2)
                A_v = A_poly[valid]
                b_v = b_poly[valid]
                margins = b_v - (A_v @ c_world)
                if margins.size == 0:
                    return np.eye(2)
                r = float(np.min(margins))
                if not np.isfinite(r) or r <= 1e-3:
                    r = 1e-3
                return r * np.eye(2)

            t_start_nn = time.time()
            try:
                if cpp_infer_polygon_batch_with_ellipse is not None:
                    poly_batch, p_batch, c_batch = cpp_infer_polygon_batch_with_ellipse(
                        np.asarray(patch_np_list, dtype=np.float32), patch_size=self.patch_size
                    )
                else:
                    poly_batch = cpp_infer_polygon_batch(np.asarray(patch_np_list, dtype=np.float32), patch_size=self.patch_size)
                    p_batch = [None for _ in range(batch_size)]
                    c_batch = [None for _ in range(batch_size)]
            except Exception as e:
                print(f"[Planner] C++ corridor batch failed, fallback to empty corridors: {e}")
                poly_batch = [None for _ in range(batch_size)]
                p_batch = [None for _ in range(batch_size)]
                c_batch = [None for _ in range(batch_size)]
            t_end_nn = time.time()
            nn_time = t_end_nn - t_start_nn

            results = []
            for k in range(batch_size):
                physical_center_x = c_x_ints[k] * map_resolution
                physical_center_y = c_y_ints[k] * map_resolution
                center_world = np.array([physical_center_x, physical_center_y], dtype=float)

                t1 = time.time()
                poly_points = poly_batch[k] if k < len(poly_batch) else None
                if poly_points is not None:
                    poly_points = np.asarray(poly_points, dtype=float)
                if poly_points is not None and len(poly_points) >= 3:
                    poly_world = (poly_points - r_center) * mapped_resolution + center_world[None, :]
                    A_poly, b_poly, c_world_poly = _polygon_to_halfspaces(poly_world, center_world)
                    p_pixel = p_batch[k] if k < len(p_batch) else None
                    c_pixel = c_batch[k] if k < len(c_batch) else None
                    if p_pixel is not None and c_pixel is not None:
                        c_world = np.array([
                            (float(c_pixel[0]) - r_center) * mapped_resolution + physical_center_x,
                            (float(c_pixel[1]) - r_center) * mapped_resolution + physical_center_y,
                        ])
                        try:
                            P_inv_world = np.linalg.inv(np.asarray(p_pixel, dtype=float)) * mapped_resolution
                        except np.linalg.LinAlgError:
                            P_inv_world = _safe_visual_shape(A_poly, b_poly, c_world)
                    else:
                        c_world = c_world_poly
                        P_inv_world = _safe_visual_shape(A_poly, b_poly, c_world)
                else:
                    A_poly = np.zeros((self.N_faces, 2))
                    b_poly = np.zeros(self.N_faces)
                    A_poly[:4, :] = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
                    b_poly[0] = center_world[0] + 1.0
                    b_poly[1] = -(center_world[0] - 1.0)
                    b_poly[2] = center_world[1] + 1.0
                    b_poly[3] = -(center_world[1] - 1.0)
                    b_poly[4:] = 1e5
                    c_world = center_world
                    P_inv_world = np.eye(2)

                poly_time = time.time() - t1
                results.append((A_poly, b_poly, P_inv_world, c_world, poly_time, nn_time / max(1, batch_size)))

            return results

        t_start_nn = time.time()
        
        # 直接使用 non_blocking=True 加速 H2D 拷贝，并激活 AMP
        batch_tensor = torch.from_numpy(np.array(patch_np_list)).float().unsqueeze(1).to(self.device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            pred_batch_tensor = self.iris_net(batch_tensor)
            
        # 显式同步 GPU 确保计时测量的是真实的纯网络耗时
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        t_end_nn = time.time()
        nn_time = t_end_nn - t_start_nn
        
        mapped_resolution = (2.0 * local_radius_m) / self.patch_size
        r_center = self.patch_size / 2.0
        
        results = []
        for k in range(batch_size):
            pred_tensor = pred_batch_tensor[k]
            P_pixel, c_pixel = parse_network_output(pred_tensor, self.patch_size)
            
            physical_center_x = c_x_ints[k] * map_resolution
            physical_center_y = c_y_ints[k] * map_resolution
            
            c_world = np.array([
                (c_pixel[0] - r_center) * mapped_resolution + physical_center_x,
                (c_pixel[1] - r_center) * mapped_resolution + physical_center_y
            ])
            
            try:
                P_inv_world = np.linalg.inv(P_pixel) * mapped_resolution
            except np.linalg.LinAlgError:
                P_inv_world = np.eye(2) * (r_pixel_f * map_resolution)
                
            t1 = time.time()
            # 配合底层修改，这里直接接收不等式矩阵 A_pix 和 b_pix，不再计算顶点
            A_pix, b_pix = generate_safe_polygon(P_pixel, c_pixel, obs_points_list[k], patch_size=self.patch_size)
            poly_time = time.time() - t1
            
            A_poly = np.zeros((self.N_faces, 2))
            b_poly = np.zeros(self.N_faces)
            
            if A_pix is not None and len(A_pix) > 0:
                # --- O(1) 向量化坐标系仿射映射 ---
                # 1. 转换 A_pix 到世界坐标系
                A_world = A_pix / mapped_resolution
                
                # 过滤并归一化法向量
                norms = np.linalg.norm(A_world, axis=1)
                valid_mask = norms > 1e-5
                
                A_world = A_world[valid_mask]
                b_pix = b_pix[valid_mask]
                A_pix = A_pix[valid_mask]
                norms = norms[valid_mask]
                
                A_world_norm = A_world / norms[:, None]
                
                # 2. 转换 b_pix 到世界坐标系
                center_phys = np.array([physical_center_x, physical_center_y])
                offset_pix = np.dot(A_pix, np.array([r_center, r_center]))
                offset_phys = np.dot(A_world, center_phys)
                b_world_raw = b_pix - offset_pix + offset_phys
                
                b_world_norm = b_world_raw / norms
                
                # 3. 引入车辆物理安全半径 (向内收缩)
                b_world_safe = b_world_norm - self.r_safe
                
                # 4. 计算到局部视野中心的垂直距离进行重要性排序
                dists = np.abs(np.dot(A_world_norm, c_world) - b_world_norm)
                
                valid_num = len(b_world_safe)
                if valid_num > self.N_faces:
                    idx = np.argsort(dists)[:self.N_faces]
                    A_world_norm = A_world_norm[idx]
                    b_world_safe = b_world_safe[idx]
                    valid_num = self.N_faces
                    
                A_poly[:valid_num] = A_world_norm
                b_poly[:valid_num] = b_world_safe
                
                # 不足 N_faces 用无效边界填满
                if valid_num < self.N_faces:
                    b_poly[valid_num:] = 1e5
            else:
                # 极速降级包络：防退化保护
                A_poly[:4, :] = np.array([[1,0], [-1,0], [0,1], [0,-1]])
                b_poly[0] = c_world[0] + 1.0
                b_poly[1] = -(c_world[0] - 1.0)
                b_poly[2] = c_world[1] + 1.0
                b_poly[3] = -(c_world[1] - 1.0)
                b_poly[4:] = 1e5
                
            results.append((A_poly, b_poly, P_inv_world, c_world, poly_time, nn_time / batch_size))
            
        return results

    def _predict_ellipse_corridors_batched(self, occupancy_map, states_xy_list, map_resolution, max_bound=10.0):
        # 批量神经网络推理
        batch_size = len(states_xy_list)
        import cv2
        import torch
        import time
        import math
        
        patch_np_list = []
        obs_points_list = []
        c_x_ints = []
        c_y_ints = []
        
        local_radius_m = 10.0
        r_pixel_f = local_radius_m / map_resolution
        r_pixel = int(np.round(r_pixel_f))
        side_len = int(2 * r_pixel)
        
        h, w = occupancy_map.shape
        
        for state_xy in states_xy_list:
            cx_f = state_xy[0] / map_resolution
            cy_f = state_xy[1] / map_resolution
            c_x_int = int(np.round(cx_f))
            c_y_int = int(np.round(cy_f))
            c_x_ints.append(c_x_int)
            c_y_ints.append(c_y_int)
            
            raw_patch = np.zeros((side_len, side_len), dtype=np.uint8)
            
            x_min_map = c_x_int - r_pixel
            x_max_map = c_x_int + r_pixel
            y_min_map = c_y_int - r_pixel
            y_max_map = c_y_int + r_pixel
            
            valid_x_min = max(0, x_min_map)
            valid_x_max = min(w, x_max_map)
            valid_y_min = max(0, y_min_map)
            valid_y_max = min(h, y_max_map)
            
            p_x_min = valid_x_min - x_min_map
            p_x_max = side_len - (x_max_map - valid_x_max)
            p_y_min = valid_y_min - y_min_map
            p_y_max = side_len - (y_max_map - valid_y_max)
            
            if valid_y_max > valid_y_min and valid_x_max > valid_x_min:
                raw_patch[p_y_min:p_y_max, p_x_min:p_x_max] = occupancy_map[valid_y_min:valid_y_max, valid_x_min:valid_x_max]
            
            patch_np = cv2.resize(raw_patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            patch_np_list.append(patch_np)
            
            obs_y, obs_x = np.where(patch_np >= 0.5)
            if len(obs_x) > 0:
                obs_points_list.append(np.column_stack((obs_x, obs_y)).astype(float))
            else:
                obs_points_list.append(np.array([]))
                
        t_start_nn = time.time()
        
        batch_tensor = torch.from_numpy(np.array(patch_np_list)).float().unsqueeze(1).to(self.device)
        with torch.no_grad():
            pred_batch_tensor = self.iris_net(batch_tensor)
            
        t_end_nn = time.time()
        nn_time = t_end_nn - t_start_nn
        
        mapped_resolution = (2.0 * local_radius_m) / self.patch_size
        r_center = self.patch_size / 2.0
        
        results = []
        for k in range(batch_size):
            pred_tensor = pred_batch_tensor[k]
            P_pixel, c_pixel = parse_network_output(pred_tensor, self.patch_size)
            
            physical_center_x = c_x_ints[k] * map_resolution
            physical_center_y = c_y_ints[k] * map_resolution
            
            c_world = np.array([
                (c_pixel[0] - r_center) * mapped_resolution + physical_center_x,
                (c_pixel[1] - r_center) * mapped_resolution + physical_center_y
            ])
            
            try:
                P_inv_world = np.linalg.inv(P_pixel) * mapped_resolution
            except np.linalg.LinAlgError:
                P_inv_world = np.eye(2) * (r_pixel_f * map_resolution)
                
            t1 = time.time()
            poly_points = generate_safe_polygon(P_pixel, c_pixel, obs_points_list[k], patch_size=self.patch_size)
            poly_time = time.time() - t1
            
            A_poly = np.zeros((self.N_faces, 2))
            b_poly = np.zeros(self.N_faces)
            
            if poly_points is not None and len(poly_points) >= 3:
                poly_world = (poly_points - r_center) * mapped_resolution + np.array([physical_center_x, physical_center_y])
                M = len(poly_world)
                normals = []
                bs = []
                dists = []
                for i in range(M):
                    p1 = poly_world[i]
                    p2 = poly_world[(i + 1) % M]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    length = math.hypot(dx, dy)
                    if length < 1e-5:
                        continue
                    nx = dy / length
                    ny = -dx / length
                    b_val = (nx * p1[0] + ny * p1[1]) - self.r_safe
                    dist_to_center = abs(nx * c_world[0] + ny * c_world[1] - (b_val + self.r_safe))
                    normals.append([nx, ny])
                    bs.append(b_val)
                    dists.append(dist_to_center)
                    
                if normals:
                    normals = np.array(normals)
                    bs = np.array(bs)
                    dists = np.array(dists)
                    valid_num = len(bs)
                    if valid_num > self.N_faces:
                        idx = np.argsort(dists)[:self.N_faces]
                        normals = normals[idx]
                        bs = bs[idx]
                        valid_num = self.N_faces
                    A_poly[:valid_num] = normals
                    b_poly[:valid_num] = bs
            results.append((A_poly, b_poly, P_inv_world, c_world, poly_time, nn_time / batch_size))
            
        return results
    
    def _predict_ellipse_corridors_batched(self, occupancy_map, states_xy_list, map_resolution, max_bound=10.0):
        # 批量神经网络推理
        batch_size = len(states_xy_list)
        
        patch_np_list = []
        obs_points_list = []
        c_x_ints = []
        c_y_ints = []
        
        local_radius_m = 10.0
        r_pixel_f = local_radius_m / map_resolution
        r_pixel = int(np.round(r_pixel_f))
        side_len = int(2 * r_pixel)
        
        h, w = occupancy_map.shape

        # === 新增优化：计算整个 Batch 的联合 ROI 包围盒 ===
        if batch_size > 0:
            # 预先计算所有状态点的像素坐标
            cx_ints_temp = [int(np.round(st[0] / map_resolution)) for st in states_xy_list]
            cy_ints_temp = [int(np.round(st[1] / map_resolution)) for st in states_xy_list]
            
            # 划定包含所有预测点和感受野的最小外接矩形 (包含边界保护)
            roi_min_x = max(0, min(cx_ints_temp) - r_pixel)
            roi_max_x = min(w, max(cx_ints_temp) + r_pixel)
            roi_min_y = max(0, min(cy_ints_temp) - r_pixel)
            roi_max_y = min(h, max(cy_ints_temp) + r_pixel)
            
            # 仅截取包含当前批次轨迹的局部小图进行形态学边缘提取
            if roi_max_x > roi_min_x and roi_max_y > roi_min_y:
                roi_occupancy = occupancy_map[roi_min_y:roi_max_y, roi_min_x:roi_max_x]
                kernel = np.array([[0, 1, 0], 
                                [1, 1, 1], 
                                [0, 1, 0]], dtype=np.uint8)
                roi_boundary = cv2.morphologyEx(roi_occupancy, cv2.MORPH_GRADIENT, kernel)
            else:
                roi_boundary = np.zeros((0, 0), dtype=np.uint8)
        # ===================================================

        for state_xy in states_xy_list:
            cx_f = state_xy[0] / map_resolution
            cy_f = state_xy[1] / map_resolution
            c_x_int = int(np.round(cx_f))
            c_y_int = int(np.round(cy_f))
            c_x_ints.append(c_x_int)
            c_y_ints.append(c_y_int)
            
            raw_patch = np.zeros((side_len, side_len), dtype=np.uint8)
            bound_patch = np.zeros((side_len, side_len), dtype=np.uint8)
            
            x_min_map = c_x_int - r_pixel
            x_max_map = c_x_int + r_pixel
            y_min_map = c_y_int - r_pixel
            y_max_map = c_y_int + r_pixel
            
            valid_x_min = max(0, x_min_map)
            valid_x_max = min(w, x_max_map)
            valid_y_min = max(0, y_min_map)
            valid_y_max = min(h, y_max_map)
            
            p_x_min = valid_x_min - x_min_map
            p_x_max = side_len - (x_max_map - valid_x_max)
            p_y_min = valid_y_min - y_min_map
            p_y_max = side_len - (y_max_map - valid_y_max)
            
            if valid_y_max > valid_y_min and valid_x_max > valid_x_min:
                # 原始占据地图用于神经网络推理
                raw_patch[p_y_min:p_y_max, p_x_min:p_x_max] = occupancy_map[valid_y_min:valid_y_max, valid_x_min:valid_x_max]
                
                # 从联合 ROI 边界图中截取对应区域 (使用相对坐标计算)
                roi_rel_x_min = valid_x_min - roi_min_x
                roi_rel_x_max = valid_x_max - roi_min_x
                roi_rel_y_min = valid_y_min - roi_min_y
                roi_rel_y_max = valid_y_max - roi_min_y
                bound_patch[p_y_min:p_y_max, p_x_min:p_x_max] = roi_boundary[roi_rel_y_min:roi_rel_y_max, roi_rel_x_min:roi_rel_x_max]
            
            patch_np = cv2.resize(raw_patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            patch_np_list.append(patch_np)
            
            # 寻找障碍物边界坐标
            bound_patch_resized = cv2.resize(bound_patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
            obs_y, obs_x = np.where(bound_patch_resized >= 0.5)
            
            if len(obs_x) > 0:
                # 直接输入所有真实的边界点，抛弃危险的步长降采样
                obs_points_list.append(np.column_stack((obs_x, obs_y)).astype(float))
            else:
                obs_points_list.append(np.array([]))

        mapped_resolution = (2.0 * local_radius_m) / self.patch_size
        r_center = self.patch_size / 2.0

        if self.use_cpp_corridor and cpp_infer_polygon_batch is not None:
            def _polygon_to_halfspaces(poly_world: np.ndarray, center_world: np.ndarray):
                A_poly = np.zeros((self.N_faces, 2))
                b_poly = np.zeros(self.N_faces)

                if poly_world is None or len(poly_world) < 3:
                    A_poly[:4, :] = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
                    b_poly[0] = center_world[0] + 1.0
                    b_poly[1] = -(center_world[0] - 1.0)
                    b_poly[2] = center_world[1] + 1.0
                    b_poly[3] = -(center_world[1] - 1.0)
                    b_poly[4:] = 1e5
                    return A_poly, b_poly, center_world.copy()

                poly_world = np.asarray(poly_world, dtype=float)
                signed_area = 0.5 * float(np.sum(poly_world[:, 0] * np.roll(poly_world[:, 1], -1) - poly_world[:, 1] * np.roll(poly_world[:, 0], -1)))
                if signed_area < 0.0:
                    poly_world = poly_world[::-1]

                valid_num = 0
                centroid = poly_world.mean(axis=0)
                for i in range(len(poly_world)):
                    p1 = poly_world[i]
                    p2 = poly_world[(i + 1) % len(poly_world)]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    length = math.hypot(dx, dy)
                    if length < 1e-8:
                        continue
                    nx = dy / length
                    ny = -dx / length
                    A_poly[valid_num] = [nx, ny]
                    b_poly[valid_num] = nx * p1[0] + ny * p1[1] - self.r_safe
                    valid_num += 1
                    if valid_num >= self.N_faces:
                        break

                if valid_num < self.N_faces:
                    b_poly[valid_num:] = 1e5

                return A_poly, b_poly, centroid

            def _safe_visual_shape(A_poly: np.ndarray, b_poly: np.ndarray, c_world: np.ndarray):
                # 为可视化构造一个保证在 A x <= b 内部的保守内接圆。
                # 绘制端使用 c + P_inv @ unit_circle，因此这里返回 r I。
                valid = (np.linalg.norm(A_poly, axis=1) > 1e-8) & (b_poly < 1e4)
                if not np.any(valid):
                    return np.eye(2)
                A_v = A_poly[valid]
                b_v = b_poly[valid]
                margins = b_v - (A_v @ c_world)
                if margins.size == 0:
                    return np.eye(2)
                r = float(np.min(margins))
                if not np.isfinite(r) or r <= 1e-3:
                    r = 1e-3
                return r * np.eye(2)

            t_start_nn = time.time()
            try:
                if cpp_infer_polygon_batch_with_ellipse is not None:
                    poly_batch, p_batch, c_batch = cpp_infer_polygon_batch_with_ellipse(
                        np.asarray(patch_np_list, dtype=np.float32), patch_size=self.patch_size
                    )
                else:
                    poly_batch = cpp_infer_polygon_batch(np.asarray(patch_np_list, dtype=np.float32), patch_size=self.patch_size)
                    p_batch = [None for _ in range(batch_size)]
                    c_batch = [None for _ in range(batch_size)]
            except Exception as e:
                print(f"[Planner] C++ corridor batch failed, fallback to empty corridors: {e}")
                poly_batch = [None for _ in range(batch_size)]
                p_batch = [None for _ in range(batch_size)]
                c_batch = [None for _ in range(batch_size)]
            batch_call_time = time.time() - t_start_nn

            results = []
            for k in range(batch_size):
                physical_center_x = c_x_ints[k] * map_resolution
                physical_center_y = c_y_ints[k] * map_resolution
                center_world = np.array([physical_center_x, physical_center_y], dtype=float)

                t1 = time.time()
                poly_points = poly_batch[k] if k < len(poly_batch) else None
                if poly_points is not None:
                    poly_points = np.asarray(poly_points, dtype=float)
                if poly_points is not None and len(poly_points) >= 3:
                    poly_world = (poly_points - r_center) * mapped_resolution + center_world[None, :]
                    A_poly, b_poly, c_world_poly = _polygon_to_halfspaces(poly_world, center_world)
                    p_pixel = p_batch[k] if k < len(p_batch) else None
                    c_pixel = c_batch[k] if k < len(c_batch) else None
                    if p_pixel is not None and c_pixel is not None:
                        c_world = np.array([
                            (float(c_pixel[0]) - r_center) * mapped_resolution + physical_center_x,
                            (float(c_pixel[1]) - r_center) * mapped_resolution + physical_center_y,
                        ])
                        try:
                            P_inv_world = np.linalg.inv(np.asarray(p_pixel, dtype=float)) * mapped_resolution
                        except np.linalg.LinAlgError:
                            P_inv_world = _safe_visual_shape(A_poly, b_poly, c_world)
                    else:
                        c_world = c_world_poly
                        P_inv_world = _safe_visual_shape(A_poly, b_poly, c_world)
                else:
                    A_poly = np.zeros((self.N_faces, 2))
                    b_poly = np.zeros(self.N_faces)
                    A_poly[:4, :] = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
                    b_poly[0] = center_world[0] + 1.0
                    b_poly[1] = -(center_world[0] - 1.0)
                    b_poly[2] = center_world[1] + 1.0
                    b_poly[3] = -(center_world[1] - 1.0)
                    b_poly[4:] = 1e5
                    c_world = center_world
                    P_inv_world = np.eye(2)

                python_postprocess_time = time.time() - t1
                per_item_batch_call_time = batch_call_time / max(1, batch_size)
                total_corridor_time = per_item_batch_call_time + python_postprocess_time
                results.append((A_poly, b_poly, P_inv_world, c_world, per_item_batch_call_time, python_postprocess_time, total_corridor_time))

            return results
                
        t_start_nn = time.time()
        
        batch_tensor = torch.from_numpy(np.array(patch_np_list)).float().unsqueeze(1).to(self.device)
        with torch.no_grad():
            pred_batch_tensor = self.iris_net(batch_tensor)
            
        t_end_nn = time.time()
        batch_call_time = t_end_nn - t_start_nn
        
        mapped_resolution = (2.0 * local_radius_m) / self.patch_size
        r_center = self.patch_size / 2.0
        
        results = []
        for k in range(batch_size):
            pred_tensor = pred_batch_tensor[k]
            P_pixel, c_pixel = parse_network_output(pred_tensor, self.patch_size)
            
            physical_center_x = c_x_ints[k] * map_resolution
            physical_center_y = c_y_ints[k] * map_resolution
            
            c_world = np.array([
                (c_pixel[0] - r_center) * mapped_resolution + physical_center_x,
                (c_pixel[1] - r_center) * mapped_resolution + physical_center_y
            ])
            
            try:
                P_inv_world = np.linalg.inv(P_pixel) * mapped_resolution
            except np.linalg.LinAlgError:
                P_inv_world = np.eye(2) * (r_pixel_f * map_resolution)
                
            t1 = time.time()
            # 配合底层修改，这里直接接收不等式矩阵 A_pix 和 b_pix，不再计算顶点
            A_pix, b_pix = generate_safe_polygon(P_pixel, c_pixel, obs_points_list[k], patch_size=self.patch_size)
            python_postprocess_time = time.time() - t1
            
            A_poly = np.zeros((self.N_faces, 2))
            b_poly = np.zeros(self.N_faces)
            
            if A_pix is not None and len(A_pix) > 0:
                # --- O(1) 向量化坐标系仿射映射 ---
                # 1. 转换 A_pix 到世界坐标系
                A_world = A_pix / mapped_resolution
                
                # 过滤并归一化法向量
                norms = np.linalg.norm(A_world, axis=1)
                valid_mask = norms > 1e-5
                
                A_world = A_world[valid_mask]
                b_pix = b_pix[valid_mask]
                A_pix = A_pix[valid_mask]
                norms = norms[valid_mask]
                
                A_world_norm = A_world / norms[:, None]
                
                # 2. 转换 b_pix 到世界坐标系
                center_phys = np.array([physical_center_x, physical_center_y])
                offset_pix = np.dot(A_pix, np.array([r_center, r_center]))
                offset_phys = np.dot(A_world, center_phys)
                b_world_raw = b_pix - offset_pix + offset_phys
                
                b_world_norm = b_world_raw / norms
                
                # 3. 引入车辆物理安全半径 (向内收缩)
                b_world_safe = b_world_norm - self.r_safe
                
                # 4. 计算到局部视野中心的垂直距离进行重要性排序
                dists = np.abs(np.dot(A_world_norm, c_world) - b_world_norm)
                
                valid_num = len(b_world_safe)
                if valid_num > self.N_faces:
                    idx = np.argsort(dists)[:self.N_faces]
                    A_world_norm = A_world_norm[idx]
                    b_world_safe = b_world_safe[idx]
                    valid_num = self.N_faces
                    
                A_poly[:valid_num] = A_world_norm
                b_poly[:valid_num] = b_world_safe
                
                # 不足 N_faces 用无效边界填满
                if valid_num < self.N_faces:
                    b_poly[valid_num:] = 1e5
            else:
                # 极速降级包络：防退化保护
                A_poly[:4, :] = np.array([[1,0], [-1,0], [0,1], [0,-1]])
                b_poly[0] = c_world[0] + 1.0
                b_poly[1] = -(c_world[0] - 1.0)
                b_poly[2] = c_world[1] + 1.0
                b_poly[3] = -(c_world[1] - 1.0)
                b_poly[4:] = 1e5
                
            per_item_batch_call_time = batch_call_time / max(1, batch_size)
            total_corridor_time = per_item_batch_call_time + python_postprocess_time
            results.append((A_poly, b_poly, P_inv_world, c_world, per_item_batch_call_time, python_postprocess_time, total_corridor_time))
            
        return results

    # def _predict_ellipse_corridors_batched(self, occupancy_map, states_xy_list, map_resolution, max_bound=10.0):
    #     # 批量神经网络推理
    #     batch_size = len(states_xy_list)

    #     patch_np_list = []
    #     obs_points_list = []
    #     c_x_ints = []
    #     c_y_ints = []
        
    #     local_radius_m = 10.0
    #     r_pixel_f = local_radius_m / map_resolution
    #     r_pixel = int(np.round(r_pixel_f))
    #     side_len = int(2 * r_pixel)
        
    #     h, w = occupancy_map.shape
        
    #     for state_xy in states_xy_list:
    #         cx_f = state_xy[0] / map_resolution
    #         cy_f = state_xy[1] / map_resolution
    #         c_x_int = int(np.round(cx_f))
    #         c_y_int = int(np.round(cy_f))
    #         c_x_ints.append(c_x_int)
    #         c_y_ints.append(c_y_int)
            
    #         raw_patch = np.zeros((side_len, side_len), dtype=np.uint8)
            
    #         x_min_map = c_x_int - r_pixel
    #         x_max_map = c_x_int + r_pixel
    #         y_min_map = c_y_int - r_pixel
    #         y_max_map = c_y_int + r_pixel
            
    #         valid_x_min = max(0, x_min_map)
    #         valid_x_max = min(w, x_max_map)
    #         valid_y_min = max(0, y_min_map)
    #         valid_y_max = min(h, y_max_map)
            
    #         p_x_min = valid_x_min - x_min_map
    #         p_x_max = side_len - (x_max_map - valid_x_max)
    #         p_y_min = valid_y_min - y_min_map
    #         p_y_max = side_len - (y_max_map - valid_y_max)
            
    #         if valid_y_max > valid_y_min and valid_x_max > valid_x_min:
    #             raw_patch[p_y_min:p_y_max, p_x_min:p_x_max] = occupancy_map[valid_y_min:valid_y_max, valid_x_min:valid_x_max]
            
    #         patch_np = cv2.resize(raw_patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    #         patch_np_list.append(patch_np)
            
    #         obs_y, obs_x = np.where(patch_np >= 0.5)
    #         if len(obs_x) > 0:
    #             # 加入轻量降采样，步长为 2 可以在不损失包络精度的前提下直接砍掉 75% 的二次型测距计算量
    #             obs_y = obs_y[::2]
    #             obs_x = obs_x[::2]
    #             obs_points_list.append(np.column_stack((obs_x, obs_y)).astype(float))
    #         else:
    #             obs_points_list.append(np.array([]))
                
    #     t_start_nn = time.time()
        
    #     batch_tensor = torch.from_numpy(np.array(patch_np_list)).float().unsqueeze(1).to(self.device)
    #     with torch.no_grad():
    #         pred_batch_tensor = self.iris_net(batch_tensor)
            
    #     t_end_nn = time.time()
    #     nn_time = t_end_nn - t_start_nn
        
    #     mapped_resolution = (2.0 * local_radius_m) / self.patch_size
    #     r_center = self.patch_size / 2.0
        
    #     results = []
    #     for k in range(batch_size):
    #         pred_tensor = pred_batch_tensor[k]
    #         P_pixel, c_pixel = parse_network_output(pred_tensor, self.patch_size)
            
    #         physical_center_x = c_x_ints[k] * map_resolution
    #         physical_center_y = c_y_ints[k] * map_resolution
            
    #         c_world = np.array([
    #             (c_pixel[0] - r_center) * mapped_resolution + physical_center_x,
    #             (c_pixel[1] - r_center) * mapped_resolution + physical_center_y
    #         ])
            
    #         try:
    #             P_inv_world = np.linalg.inv(P_pixel) * mapped_resolution
    #         except np.linalg.LinAlgError:
    #             P_inv_world = np.eye(2) * (r_pixel_f * map_resolution)
                
    #         t1 = time.time()
    #         # 配合底层修改，这里直接接收不等式矩阵 A_pix 和 b_pix，不再计算顶点
    #         A_pix, b_pix = generate_safe_polygon(P_pixel, c_pixel, obs_points_list[k], patch_size=self.patch_size)
    #         poly_time = time.time() - t1
            
    #         A_poly = np.zeros((self.N_faces, 2))
    #         b_poly = np.zeros(self.N_faces)
            
    #         if A_pix is not None and len(A_pix) > 0:
    #             # --- O(1) 向量化坐标系仿射映射 ---
    #             # 1. 转换 A_pix 到世界坐标系
    #             A_world = A_pix / mapped_resolution
                
    #             # 过滤并归一化法向量
    #             norms = np.linalg.norm(A_world, axis=1)
    #             valid_mask = norms > 1e-5
                
    #             A_world = A_world[valid_mask]
    #             b_pix = b_pix[valid_mask]
    #             A_pix = A_pix[valid_mask]
    #             norms = norms[valid_mask]
                
    #             A_world_norm = A_world / norms[:, None]
                
    #             # 2. 转换 b_pix 到世界坐标系
    #             center_phys = np.array([physical_center_x, physical_center_y])
    #             offset_pix = np.dot(A_pix, np.array([r_center, r_center]))
    #             offset_phys = np.dot(A_world, center_phys)
    #             b_world_raw = b_pix - offset_pix + offset_phys
                
    #             b_world_norm = b_world_raw / norms
                
    #             # 3. 引入车辆物理安全半径 (向内收缩)
    #             b_world_safe = b_world_norm - self.r_safe
                
    #             # 4. 计算到局部视野中心的垂直距离进行重要性排序
    #             dists = np.abs(np.dot(A_world_norm, c_world) - b_world_norm)
                
    #             valid_num = len(b_world_safe)
    #             if valid_num > self.N_faces:
    #                 idx = np.argsort(dists)[:self.N_faces]
    #                 A_world_norm = A_world_norm[idx]
    #                 b_world_safe = b_world_safe[idx]
    #                 valid_num = self.N_faces
                    
    #             A_poly[:valid_num] = A_world_norm
    #             b_poly[:valid_num] = b_world_safe
                
    #             # 不足 N_faces 用无效边界填满
    #             if valid_num < self.N_faces:
    #                 b_poly[valid_num:] = 1e5
    #         else:
    #             # 极速降级包络：防退化保护
    #             A_poly[:4, :] = np.array([[1,0], [-1,0], [0,1], [0,-1]])
    #             b_poly[0] = c_world[0] + 1.0
    #             b_poly[1] = -(c_world[0] - 1.0)
    #             b_poly[2] = c_world[1] + 1.0
    #             b_poly[3] = -(c_world[1] - 1.0)
    #             b_poly[4:] = 1e5
                
    #         results.append((A_poly, b_poly, P_inv_world, c_world, poly_time, nn_time / batch_size))
            
    #     return results


    def get_mpc_matrix_obstacle_force(self, state_vector: np.ndarray, obstacles: np.ndarray,
                                    step_index: int, H_scalar: float, R_influence=1.5) -> tuple:
        """
        核心升级：基于单一最近障碍点距离的一阶泰勒展开二次惩罚 (Nearest Obstacle Quadratic Penalty)
        [性能优化版]：使用平方距离查找最小索引，避免 N 次高昂的 np.hypot 开销
        """
        nx = self.nx
        num_obstacles = obstacles.shape[0]

        if num_obstacles == 0:
            return np.zeros((nx, nx)), np.zeros((nx, 1)), np.zeros((0, 1))

        px = state_vector[0, 0]
        py = state_vector[1, 0]

        # 1. 寻找预测点距离最近的一个障碍物 (使用平方和极速找最小值)
        diffs = np.array([px, py]) - obstacles
        sq_dists = diffs[:, 0]**2 + diffs[:, 1]**2
        min_idx = np.argmin(sq_dists)
        min_sq_dist = sq_dists[min_idx]

        Q_obs = np.zeros((nx, nx))
        f_obs = np.zeros((nx, 1))
        cost_vec = np.zeros((num_obstacles, 1))

        # 2. 如果侵入影响半径，产生排斥力
        if min_sq_dist < R_influence**2 and min_sq_dist > 1e-10:
            min_dist = np.sqrt(min_sq_dist)
            # 穿透误差
            E = R_influence - min_dist
            
            # 远离障碍物的单位法向量 n_vec
            dx = diffs[min_idx, 0]
            dy = diffs[min_idx, 1]
            nx_vec = dx / min_dist
            ny_vec = dy / min_dist
            
            # 将二维法向量扩充为全状态维度
            n_vec = np.array([nx_vec, ny_vec, 0, 0, 0])
            P_bar = np.array([px, py, 0, 0, 0])
            
            # 权重随预测步衰减或终端放大
            scale = self.cost_scaling if step_index == self.horizon_steps else 1.0
            W_obs = H_scalar * scale
            
            # 3. 构建局部二次规划矩阵 Q 和 f
            Q_obs = W_obs * np.outer(n_vec, n_vec)
            f_obs = -W_obs * (np.dot(n_vec, P_bar) + E) * n_vec.reshape(-1, 1)
            
            # 记录当前步的代价以供可视化
            cost_vec[min_idx, 0] = 0.5 * W_obs * (E ** 2)

        return Q_obs, f_obs, cost_vec


    # def get_mpc_matrix(self, predicted_states: np.ndarray, occupancy_map: np.ndarray, 
    #                    guided_points: dict, 
    #                    target_velocity: float,
    #                    weight_lat_scale: float = 1.0, 
    #                    weight_lon_scale: float = 1.0,
    #                    weight_heading_scale: float = 1.0,
    #                    weight_vel_scale: float = 1.0,
    #                    weight_obs_scale: float = 1.0,
    #                    weight_accel_scale: float = 1.0,
    #                    weight_steer_scale: float = 1.0,
    #                    weight_rate_steer_scale: float = 1.0,
    #                    speed_reward_c_scale: float = 0.0,        
    #                    weight_slack_scale: float = 1.0,          
    #                    R_influence=1.5,
    #                    map_resolution=1.0) -> dict:

    #     mpc_stage_dict = {}
    #     pos_refs = guided_points['posi']
    #     angle_refs = guided_points['angle']
    #     sum_nn_time = 0.0
    #     sum_poly_time = 0.0
    #     Q_lat = self.base_Q_lat * weight_lat_scale
    #     Q_lon = self.base_Q_lon * weight_lon_scale
    #     Q_heading = self.base_Q_heading * weight_heading_scale
    #     Q_vel = self.base_Q_vel * weight_vel_scale
        
    #     R_accel = self.base_R_accel * weight_accel_scale
    #     R_steer = self.base_R_steer * weight_steer_scale
    #     R_rate_steer = self.base_R_rate_steer * weight_rate_steer_scale
        
    #     H_obs_scalar = self.base_weight_obs * weight_obs_scale
    #     actual_speed_reward_c = self.base_speed_reward_c * speed_reward_c_scale
        
    #     # 提取松弛因子的 L1 和 L2 惩罚
    #     actual_slack_weight = self.base_weight_slack * weight_slack_scale
    #     Z_slack_sq = actual_slack_weight
    #     z_slack_lin = actual_slack_weight / 10.0

    #     # R(二次控制惩罚) 和 r(一次控制惩罚) 矩阵扩充包含所有松弛变量
    #     current_R_rate = np.zeros((self.nu, self.nu))
    #     current_R_rate[0, 0] = R_accel
    #     current_R_rate[1, 1] = R_rate_steer
    #     for i in range(self.nu_orig, self.nu):
    #         current_R_rate[i, i] = Z_slack_sq
            
    #     current_r_rate = np.zeros((self.nu, 1))
    #     for i in range(self.nu_orig, self.nu):
    #         current_r_rate[i, 0] = z_slack_lin
        
    #     current_ego_state = predicted_states[:, 0]
        
    #     # =========================================================================
    #     # >>> 单次构建全局 KDTree <<<
    #     kdtree, local_obs = self._build_local_kdtree(occupancy_map, current_ego_state, map_resolution, radius=30.0)
    #     # =========================================================================
        
    #     # >>> 单次利用 CorridorEllipseNet 批量生成预测几何多边形边界 <<<
    #     states_xy_list = [(predicted_states[0, i], predicted_states[1, i]) for i in range(self.horizon_steps + 1)]
    #     batched_results = self._predict_ellipse_corridors_batched(occupancy_map, states_xy_list, map_resolution)

    #     for k in range(self.horizon_steps + 1):
    #         raw_ref_heading = angle_refs[k]
    #         ref_pos = pos_refs[k]
            
    #         # 使用真实物理预测状态，平滑旋转投影追踪偏差
    #         current_theta = predicted_states[2, k]
    #         ref_heading = raw_ref_heading + round((current_theta - raw_ref_heading) / (2 * math.pi)) * (2 * math.pi)

    #         # 这里的 Frenet 投影是提供给【追踪参考路径的目标代价 J】使用的，因此必须用 ref_heading
    #         cos_h = math.cos(ref_heading)
    #         sin_h = math.sin(ref_heading)
    #         M = np.array([
    #             [-sin_h, cos_h, 0, 0], 
    #             [ cos_h, sin_h, 0, 0], 
    #             [     0,     0, 1, 0], 
    #             [     0,     0, 0, 1]  
    #         ])
    #         Q_diag_transformed = np.diag([Q_lat, Q_lon, Q_heading, Q_vel])
    #         Q_state = M.T @ Q_diag_transformed @ M

    #         Qk = np.zeros((self.nx, self.nx))
    #         Qk[:4, :4] = Q_state
    #         Qk[4, 4] = R_steer

    #         if k == self.horizon_steps:
    #             Qk *= self.terminal_scale 

    #         # 取用上一个控制周期的暖启动预期轨迹状态，它是物理上平滑过渡的
    #         current_state = predicted_states[:, k]
    #         smooth_x = current_state[0]
    #         smooth_y = current_state[1]
    #         smooth_theta = current_state[2]
            
    #         # --- 局部感知近邻提取 ---
    #         if kdtree is not None:
    #             # 严格围绕预测平滑点构建检索框
    #             dists, idxs = kdtree.query((smooth_x, smooth_y), k=60, distance_upper_bound=10.0)
    #             valid_mask = dists < 10.0
    #             valid_idxs = idxs[valid_mask]
    #             bubble_obs = local_obs[valid_idxs]
    #         else:
    #             bubble_obs = np.empty((0, 2))
            
    #         # 引入全新的预测点最近障碍排斥代价
    #         Qk_obs, fk_obs, cost_vec = self.get_mpc_matrix_obstacle_force(
    #             current_state.reshape(-1, 1), bubble_obs, k, H_scalar=H_obs_scalar, R_influence=R_influence
    #         )

    #         x_ref = np.array([ref_pos[0], ref_pos[1], ref_heading, target_velocity, 0.0])
    #         fk = -Qk @ x_ref
    #         fk[3] -= actual_speed_reward_c

    #         # 加上的二次代价
    #         Qk += Qk_obs
    #         fk += fk_obs.flatten()

    #         # 【包络线构造】利用批量生成的安全椭圆边界结果
    #         A_poly, b_poly, P_inv, c_ell, poly_time, nn_time = batched_results[k]
    #         sum_poly_time += poly_time
    #         sum_nn_time += nn_time

    #         stage = {
    #             'Qk': Qk * self.cost_scaling, 
    #             'fk': fk * self.cost_scaling,
    #             'obstacles': bubble_obs, 
    #             'obs_costs': cost_vec,
    #             'A_poly': A_poly,
    #             'b_poly': b_poly,
    #             'P_inv': P_inv,
    #             'c_ell': c_ell,
    #             'smooth_pos': (smooth_x, smooth_y),
    #             'smooth_heading': smooth_theta,
    #             'ref_pos': ref_pos,
    #             'ref_heading': ref_heading
    #         }
            
    #         if k < self.horizon_steps:
    #             stage['Rk'] = current_R_rate * self.cost_scaling
    #             stage['rk'] = current_r_rate * self.cost_scaling
                
    #             # 建立多边形约束 C_mat (nx维作用于 x, y)
    #             C_mat = np.zeros((self.N_faces, self.nx))
    #             C_mat[:, 0:2] = A_poly
                
    #             # 建立控制依赖的松弛变量作用矩阵 D_mat
    #             D_mat = np.zeros((self.N_faces, self.nu))
    #             for i in range(self.N_faces):
    #                 D_mat[i, self.nu_orig + i] = -1.0
                
    #             # 设定上下界：采用 D_mat 使得实际有效约束为 A_poly * x - s <= b_poly
    #             lg = np.full((self.N_faces, 1), -1e5)
    #             ug = b_poly.reshape(self.N_faces, 1)
                
    #             # 设定掩码屏蔽无穷边界
    #             lg_mask = np.zeros((self.N_faces, 1))
    #             ug_mask = np.ones((self.N_faces, 1))
                
    #             stage['C_mat'] = C_mat
    #             stage['D_mat'] = D_mat
    #             stage['lg'] = lg
    #             stage['ug'] = ug
    #             stage['lg_mask'] = lg_mask
    #             stage['ug_mask'] = ug_mask
            
    #         mpc_stage_dict[k] = stage

    #     return mpc_stage_dict, sum_poly_time, sum_nn_time

    def get_mpc_matrix(self, predicted_states: np.ndarray, occupancy_map: np.ndarray, 
                       guided_points: dict, 
                       target_velocity: float,
                       weight_lat_scale: float = 1.0, 
                       weight_lon_scale: float = 1.0,
                       weight_heading_scale: float = 1.0,
                       weight_vel_scale: float = 1.0,
                       weight_obs_scale: float = 1.0,
                       weight_accel_scale: float = 1.0,
                       weight_steer_scale: float = 1.0,
                       weight_rate_steer_scale: float = 1.0,
                       speed_reward_c_scale: float = 0.0,        
                       weight_slack_scale: float = 1.0,          
                       R_influence=1.5,
                       map_resolution=1.0) -> dict:

        mpc_stage_dict = {}
        pos_refs = np.array(guided_points['posi'])
        angle_refs = np.array(guided_points['angle'])
        sum_batch_call_time = 0.0
        sum_python_postprocess_time = 0.0
        sum_total_corridor_time = 0.0
        
        Q_lat = self.base_Q_lat * weight_lat_scale
        Q_lon = self.base_Q_lon * weight_lon_scale
        Q_heading = self.base_Q_heading * weight_heading_scale
        Q_vel = self.base_Q_vel * weight_vel_scale
        
        R_accel = self.base_R_accel * weight_accel_scale
        R_steer = self.base_R_steer * weight_steer_scale
        R_rate_steer = self.base_R_rate_steer * weight_rate_steer_scale
        
        H_obs_scalar = self.base_weight_obs * weight_obs_scale
        actual_speed_reward_c = self.base_speed_reward_c * speed_reward_c_scale
        
        actual_slack_weight = self.base_weight_slack * weight_slack_scale
        Z_slack_sq = actual_slack_weight
        z_slack_lin = actual_slack_weight / 10.0

        current_R_rate = np.zeros((self.nu, self.nu))
        current_R_rate[0, 0] = R_accel
        current_R_rate[1, 1] = R_rate_steer
        for i in range(self.nu_orig, self.nu):
            current_R_rate[i, i] = Z_slack_sq
            
        current_r_rate = np.zeros((self.nu, 1))
        for i in range(self.nu_orig, self.nu):
            current_r_rate[i, 0] = z_slack_lin
        
        current_ego_state = predicted_states[:, 0]
        
        # =========================================================================
        # >>> 单次构建全局 KDTree <<<
        kdtree, local_obs = self._build_local_kdtree(occupancy_map, current_ego_state, map_resolution, radius=30.0)
        # =========================================================================
        
        # >>> 单次利用 C++ GPU 批量生成预测几何多边形边界 <<<
        states_xy_list = [(predicted_states[0, i], predicted_states[1, i]) for i in range(self.horizon_steps + 1)]
        batched_results = self._predict_ellipse_corridors_batched(occupancy_map, states_xy_list, map_resolution)

        N_steps = self.horizon_steps + 1
        
        # =========================================================================
        # >>> 矩阵计算全向量化 (Vectorization) 消除 Python 循环开销 <<<
        # 1. 批量计算参考航向与偏差角
        current_theta_all = predicted_states[2, :]
        ref_heading_all = angle_refs + np.round((current_theta_all - angle_refs) / (2 * math.pi)) * (2 * math.pi)
        
        cos_h = np.cos(ref_heading_all)
        sin_h = np.sin(ref_heading_all)
        
        # 2. 批量代数展开 M.T @ Q_diag @ M，直接规避高昂的张量乘法
        q00 = Q_lat * sin_h**2 + Q_lon * cos_h**2
        q01 = (Q_lon - Q_lat) * sin_h * cos_h
        q11 = Q_lat * cos_h**2 + Q_lon * sin_h**2
        
        Qk_base_all = np.zeros((N_steps, self.nx, self.nx))
        Qk_base_all[:, 0, 0] = q00
        Qk_base_all[:, 0, 1] = q01
        Qk_base_all[:, 1, 0] = q01
        Qk_base_all[:, 1, 1] = q11
        Qk_base_all[:, 2, 2] = Q_heading
        Qk_base_all[:, 3, 3] = Q_vel
        Qk_base_all[:, 4, 4] = R_steer
        Qk_base_all[-1] *= self.terminal_scale 
        
        x_ref_all = np.zeros((N_steps, self.nx))
        x_ref_all[:, 0] = pos_refs[:, 0]
        x_ref_all[:, 1] = pos_refs[:, 1]
        x_ref_all[:, 2] = ref_heading_all
        x_ref_all[:, 3] = target_velocity
        
        # 批量求取基础状态 fk = -Qk @ x_ref
        fk_base_all = -np.einsum('nij,nj->ni', Qk_base_all, x_ref_all)
        fk_base_all[:, 3] -= actual_speed_reward_c
        
        # 3. 批量执行 KDTree 搜索 (将循环 N 次底层调用合并为 1 次)
        smooth_pts_batch = predicted_states[:2, :].T 
        if kdtree is not None:
            dists_batch, idxs_batch = kdtree.query(smooth_pts_batch, k=60, distance_upper_bound=10.0)
        # =========================================================================

        # 仅保留字典装配的薄层循环
        for k in range(N_steps):
            Qk = Qk_base_all[k].copy()
            fk = fk_base_all[k].copy()
            
            current_state = predicted_states[:, k]
            smooth_x = current_state[0]
            smooth_y = current_state[1]
            smooth_theta = current_state[2]
            
            if kdtree is not None:
                valid_mask = dists_batch[k] < 10.0
                valid_idxs = idxs_batch[k][valid_mask]
                bubble_obs = local_obs[valid_idxs]
            else:
                bubble_obs = np.empty((0, 2))
            
            Qk_obs, fk_obs, cost_vec = self.get_mpc_matrix_obstacle_force(
                current_state.reshape(-1, 1), bubble_obs, k, H_scalar=H_obs_scalar, R_influence=R_influence
            )

            Qk += Qk_obs
            fk += fk_obs.flatten()

            A_poly, b_poly, P_inv, c_ell, batch_call_time, python_postprocess_time, total_corridor_time = batched_results[k]
            sum_batch_call_time += batch_call_time
            sum_python_postprocess_time += python_postprocess_time
            sum_total_corridor_time += total_corridor_time

            stage = {
                'Qk': Qk * self.cost_scaling, 
                'fk': fk * self.cost_scaling,
                'obstacles': bubble_obs, 
                'obs_costs': cost_vec,
                'A_poly': A_poly,
                'b_poly': b_poly,
                'P_inv': P_inv,
                'c_ell': c_ell,
                'smooth_pos': (smooth_x, smooth_y),
                'smooth_heading': smooth_theta,
                'ref_pos': pos_refs[k],
                'ref_heading': ref_heading_all[k]
            }
            
            if k < self.horizon_steps:
                stage['Rk'] = current_R_rate * self.cost_scaling
                stage['rk'] = current_r_rate * self.cost_scaling
                
                C_mat = np.zeros((self.N_faces, self.nx))
                C_mat[:, 0:2] = A_poly
                
                D_mat = np.zeros((self.N_faces, self.nu))
                for i in range(self.N_faces):
                    D_mat[i, self.nu_orig + i] = -1.0
                
                lg = np.full((self.N_faces, 1), -1e5)
                ug = b_poly.reshape(self.N_faces, 1)
                
                lg_mask = np.zeros((self.N_faces, 1))
                ug_mask = np.ones((self.N_faces, 1))
                
                stage['C_mat'] = C_mat
                stage['D_mat'] = D_mat
                stage['lg'] = lg
                stage['ug'] = ug
                stage['lg_mask'] = lg_mask
                stage['ug_mask'] = ug_mask
            
            mpc_stage_dict[k] = stage

        return mpc_stage_dict, sum_batch_call_time, sum_python_postprocess_time, sum_total_corridor_time
    
    def step_once(self, current_state: np.ndarray, last_ctrl: np.ndarray, guesses: tuple,
                  map_obstacle: np.ndarray, guided_points: dict,
                  target_velocity: float, 
                  weight_lat_scale: float = 1.0, 
                  weight_lon_scale: float = 1.0,
                  weight_heading_scale: float = 1.0,
                  weight_vel_scale: float = 1.0,
                  weight_obs_scale: float = 1.0,
                  weight_accel_scale: float = 1.0,
                  weight_steer_scale: float = 1.0,
                  weight_rate_steer_scale: float = 1.0,
                  speed_reward_c_scale: float = 0.0,        
                  weight_slack_scale: float = 1.0,     
                  R_influence=1.5,   
                  map_resolution=1.0,
                  debug_render=False) -> tuple:
        
        X_guess, U_guess = guesses

        if X_guess.shape[0] != self.nx:
            temp = np.zeros((self.nx, X_guess.shape[1]))
            min_dim = min(X_guess.shape[0], self.nx)
            temp[:min_dim, :] = X_guess[:min_dim, :]
            X_guess = temp
        
        s0 = current_state
        pos_guided = guided_points['posi']
        phi_guided = guided_points['angle']
        
        import time
        # 核心：使用上一帧滚动传来的 Guess 轨迹作为包络线提取锚点，使其随时间前进而非原地切片
        X_guess[:, 0] = s0 
        state_horizon = X_guess 

        t_start_matrix = time.time()
        mpc_stage_dict, sum_batch_call_time, sum_python_postprocess_time, sum_total_corridor_time = self.get_mpc_matrix(
            state_horizon, map_obstacle, guided_points,
            target_velocity=target_velocity,
            weight_lat_scale=weight_lat_scale,
            weight_lon_scale=weight_lon_scale,
            weight_heading_scale=weight_heading_scale,
            weight_vel_scale=weight_vel_scale,
            weight_obs_scale=weight_obs_scale,
            weight_accel_scale=weight_accel_scale,
            weight_steer_scale=weight_steer_scale,
            weight_rate_steer_scale=weight_rate_steer_scale,
            speed_reward_c_scale=speed_reward_c_scale,
            weight_slack_scale=weight_slack_scale,
            R_influence=R_influence,
            map_resolution=map_resolution
        )
        t_end_matrix = time.time()
        
        t_start_solve = time.time()
        U_opt, X_opt, info = self.solve_with_hpipm(
            s0.reshape(-1, 1), mpc_stage_dict, x_guess=X_guess, u_guess=U_guess
        )
        t_end_solve = time.time()
        
        time_stats = {
            'matrix_build_time': t_end_matrix - t_start_matrix,
            'solve_time': t_end_solve - t_start_solve,
            'total_time': (t_end_solve - t_start_solve) + (t_end_matrix - t_start_matrix),
            'cpp_batch_call_time': sum_batch_call_time,
            'python_postprocess_time': sum_python_postprocess_time,
            'total_corridor_time': sum_total_corridor_time,
            # 兼容旧日志字段
            'nn_inference_time': sum_batch_call_time,
            'polygon_generation_time': sum_python_postprocess_time
        }

        info['time_stats'] = time_stats

        next_ctrl_nom = U_opt[:, 0]
        next_ctrl_safe = next_ctrl_nom.copy()
        
        info['u_nom'] = next_ctrl_nom.copy()
        info['u_safe'] = next_ctrl_safe.copy()
        
        mpc_corridors = []
        for k in range(self.horizon_steps):
            if k in mpc_stage_dict and 'A_poly' in mpc_stage_dict[k]:
                mpc_corridors.append({
                    'A': mpc_stage_dict[k]['A_poly'],
                    'b': mpc_stage_dict[k]['b_poly'],
                    'P_inv': mpc_stage_dict[k].get('P_inv', None),
                    'c_ell': mpc_stage_dict[k].get('c_ell', None)
                })
        info['mpc_corridors'] = mpc_corridors

        if debug_render:
            self._render_debug(current_state, X_opt, U_opt, mpc_stage_dict, pos_guided, phi_guided, target_velocity)

        next_state = self.model.sim_forward_step(current_state, next_ctrl_safe)
        X_next_guess, U_next_guess = self.horizon_forward_step(X_opt, U_opt, next_state)
        
        return next_state, next_ctrl_safe, X_next_guess, U_next_guess, info

    def solve_with_hpipm(self, initial_state, mpc_stage_dict, x_guess=None, u_guess=None):
        self.qp_problem.set('lbx', initial_state, 0)
        self.qp_problem.set('ubx', initial_state, 0)
         
        if x_guess is None:
            x_guess = np.tile(initial_state, (1, self.horizon_steps + 1))
        
        # 将传入的原生 Guess(2D) 转换为扩充后的求解器所需 Guess(4D)
        u_guess_padded = np.zeros((self.nu, self.horizon_steps))
        if u_guess is not None:
            u_guess_padded[:self.nu_orig, :] = u_guess[:self.nu_orig, :]
            
        for k in range(self.horizon_steps + 1):
            x_ref_k = x_guess[:, k]
            
            if k < self.horizon_steps:
                # 只有原始控制量传入给动力学系统
                u_ref_k = u_guess_padded[:self.nu_orig, k]
                Ad, Bd_orig, cd = self.model.get_linearized_matrices(x_ref_k, u_ref_k)

                # 将动力学 Bd 矩阵补充至 4 列，确保求解器改动 s1 s2 时完全不影响下一帧状态转移
                Bd = np.zeros((self.nx, self.nu))
                Bd[:, :self.nu_orig] = Bd_orig

                self.qp_problem.set('A', Ad, k)
                self.qp_problem.set('B', Bd, k)
                self.qp_problem.set('b', cd.reshape(-1, 1), k) 
                
                self.qp_problem.set('R', mpc_stage_dict[k]['Rk'], k)
                self.qp_problem.set('r', mpc_stage_dict[k]['rk'], k)
                self.qp_solution.set('u', u_guess_padded[:, k], k)
                
                if 'C_mat' in mpc_stage_dict[k]:
                    self.qp_problem.set('C', mpc_stage_dict[k]['C_mat'], k)
                    self.qp_problem.set('D', mpc_stage_dict[k]['D_mat'], k)
                    self.qp_problem.set('lg', mpc_stage_dict[k]['lg'], k)
                    self.qp_problem.set('ug', mpc_stage_dict[k]['ug'], k)
                    self.qp_problem.set('lg_mask', mpc_stage_dict[k]['lg_mask'], k)
                    self.qp_problem.set('ug_mask', mpc_stage_dict[k]['ug_mask'], k)
            
            self.qp_problem.set('Q', mpc_stage_dict[k]['Qk'], k)
            self.qp_problem.set('q', mpc_stage_dict[k]['fk'].reshape(-1, 1), k)
            self.qp_solution.set('x', x_guess[:, k], k)

        try:
            self.qp_solver.solve(self.qp_problem, self.qp_solution)
            status = self.qp_solver.get('status')
            if status != 0:
                print(f"[MPC Warn] HPIPM solver returned status = {status}")
        except Exception as e:
            print(f"[MPC Error] HPIPM solve crash: {e}")
            
        # 从求解器分离返回的最优序列，抛弃优化用的松弛变量，仅保留机械指令
        u_opt = np.zeros((self.nu_orig, self.horizon_steps))
        x_opt = np.zeros((self.nx, self.horizon_steps + 1))
        
        for k in range(self.horizon_steps):
            full_u = self.qp_solution.get('u', k).flatten()
            u_opt[:, k] = full_u[:self.nu_orig]
            
        for k in range(self.horizon_steps + 1):
            x_opt[:, k] = self.qp_solution.get('x', k).flatten()
        
        info = {'status': self.qp_solver.get('status')}
        return u_opt, x_opt, info

    def horizon_forward_step(self, state_traj, control_traj, current_state):
        nx, nu, N = self.nx, self.nu_orig, self.horizon_steps
        new_x = np.zeros((nx, N + 1))
        new_u = np.zeros((nu, N))
        
        new_x[:, :N] = state_traj[:, 1:N + 1]
        new_x[:, N] = state_traj[:, N] 
        new_u[:, :N - 1] = control_traj[:, 1:N]
        new_u[:, N - 1] = control_traj[:, N - 1]
        return new_x, new_u

    def _render_debug(self, current_state, X_opt, U_opt, mpc_stage_dict, pos_guided, phi_guided, target_velocity):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        
        if not hasattr(self, 'debug_fig') or self.debug_fig is None or not plt.fignum_exists(self.debug_fig.number):
            plt.ion()
            self.debug_fig, self.debug_axs = plt.subplots(1, 2, figsize=(14, 5))
            self.debug_fig.canvas.manager.set_window_title("Planner Internal Debug")

        ax_map, ax_cost = self.debug_axs
        ax_map.cla()
        ax_cost.cla()

        cx, cy, heading = current_state[0], current_state[1], current_state[2]
        
        ref_x = [p[0] for p in pos_guided]
        ref_y = [p[1] for p in pos_guided]
        ax_map.plot(ref_x, ref_y, 'g--', linewidth=1.5, label='Reference Path')

        left_bounds_x, left_bounds_y = [], []
        right_bounds_x, right_bounds_y = [], []
        
        for k in range(self.horizon_steps + 1):
            if k in mpc_stage_dict and 'smooth_pos' in mpc_stage_dict[k]:
                rp = mpc_stage_dict[k]['smooth_pos']
                rh = mpc_stage_dict[k]['smooth_heading']
                
                # 绘制多边形边界的法线投影作为可视化提示
                if 'A_poly' in mpc_stage_dict[k] and k % 5 == 0:
                    import scipy.linalg
                    A_poly = mpc_stage_dict[k]['A_poly']
                    b_poly = mpc_stage_dict[k]['b_poly']
                    if 'P_inv' in mpc_stage_dict[k]:
                        P_inv = mpc_stage_dict[k]['P_inv']
                        c_ell = mpc_stage_dict[k]['c_ell']
                        
                        # ==========================================
                        # 1. 绘制提取的安全椭圆 (黄虚线)
                        # ==========================================
                        try:
                            theta_vals = np.linspace(0, 2*np.pi, 50)
                            ell_pts = np.vstack([np.cos(theta_vals), np.sin(theta_vals)])
                            ell_world = c_ell[:, None] + P_inv @ ell_pts
                            ax_map.plot(ell_world[0, :], ell_world[1, :], color='orange', alpha=0.5, linewidth=1.5, linestyle='--')
                            ax_map.plot(c_ell[0], c_ell[1], 'x', color='orange', markersize=4)
                        except Exception:
                            pass
                            
                        # ==========================================
                        # 2. 绘制包络多边形线段 (青色实线)
                        # ==========================================
                        for i in range(len(b_poly)):
                            n_vec = A_poly[i]
                            d = b_poly[i]
                            # 估算切点位置
                            radius = d + self.r_safe - np.dot(n_vec, c_ell)
                            proj_pt = c_ell + n_vec * radius
                            # 回退安全半径还原真实约束界限
                            wall_center = proj_pt - n_vec * self.r_safe
                            
                            tangent_vec = np.array([-n_vec[1], n_vec[0]])
                            # 获取更长一点的相交边线让它看起来像多边形
                            length = radius * math.tan(math.pi / self.N_faces) * 1.5 if self.N_faces > 0 else 2.0
                            p1 = wall_center + tangent_vec * length
                            p2 = wall_center - tangent_vec * length
                            
                            ax_map.plot([p1[0], p2[0]], [p1[1], p2[1]], color='cyan', alpha=0.8, linewidth=1.5)
                
        if X_opt is not None and np.any(X_opt):
            ax_map.plot(X_opt[0, :], X_opt[1, :], 'm-', linewidth=2.5, alpha=0.8, label='MPC Prediction')

        local_obs = mpc_stage_dict[0].get('obstacles', np.empty((0, 2)))
        obs_costs = mpc_stage_dict[0].get('obs_costs', np.empty((0, 1))).flatten()
        
        if len(local_obs) > 0:
            sc = ax_map.scatter(local_obs[:, 0], local_obs[:, 1], c=obs_costs, cmap='autumn_r', 
                                vmin=0, vmax=2.0, s=40, marker='x', label='Obstacles')

        ax_map.plot(cx, cy, 'ko', markersize=8, label='Ego Vehicle')
        ax_map.arrow(cx, cy, math.cos(heading)*3, math.sin(heading)*3, head_width=0.6, color='k')

        ax_map.set_aspect('equal')
        ax_map.set_xlim(cx - 15, cx + 15)
        ax_map.set_ylim(cy - 15, cy + 15)
        ax_map.set_title("Local View & Safe Corridor Boundaries")
        ax_map.legend(loc='upper right', fontsize=8)
        ax_map.grid(True, linestyle=':', alpha=0.6)

        dx = cx - ref_x[0]
        dy = cy - ref_y[0]
        ref_h = phi_guided[0]
        
        lat_err = abs(-math.sin(ref_h) * dx + math.cos(ref_h) * dy)
        lon_err = abs(math.cos(ref_h) * dx + math.sin(ref_h) * dy)
        head_err = abs(heading - ref_h)
        vel_err = abs(current_state[3] - target_velocity)
        
        max_obs_cost = np.max(obs_costs) if len(obs_costs) > 0 else 0.0
        accel_cmd = abs(U_opt[0, 0]) if U_opt is not None else 0.0
        steer_rate_cmd = abs(U_opt[1, 0]) if U_opt is not None else 0.0

        labels = ['Lat Err', 'Lon Err', 'Head Err', 'Vel Err', 'Obs Dist', 'Accel Cmd', 'StrRate Cmd']
        vals = [lat_err, lon_err, head_err, vel_err, max_obs_cost, accel_cmd, steer_rate_cmd]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        bars = ax_cost.bar(labels, vals, color=colors, alpha=0.85)
        ax_cost.set_ylim(0, max(2.5, max(vals) * 1.2))
        ax_cost.set_title("Current Tracking Errors & Control Efforts")
        ax_cost.tick_params(axis='x', rotation=15, labelsize=9)
        
        for bar, val in zip(bars, vals):
            ax_cost.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        plt.pause(0.001)