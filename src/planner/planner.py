import math
import numpy as np
import os
import sys
import time
import cv2
from scipy.spatial import cKDTree


# --- 路径处理 ---
# 将父目录加入系统路径，确保能够找到 planner 模块
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.planner.car_model import DynamicModel
from src.neural_iris import infer_safe_region_batch_halfspaces as py_infer_safe_region_batch_halfspaces
try:
    from cpp.python import infer_safe_region_batch_halfspaces as cpp_infer_safe_region_batch_halfspaces
except Exception:
    cpp_infer_safe_region_batch_halfspaces = None
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
        os.environ.setdefault("NEURAL_IRIS_CPP_BACKEND", "gpu")
        self.use_cpp_neural_iris = cpp_infer_safe_region_batch_halfspaces is not None
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

    def _predict_safe_regions_batched(self, occupancy_map, states_xy_list, map_resolution, max_bound=10.0):
        batch_size = len(states_xy_list)

        patch_np_list = []
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

            patch_np = cv2.resize(
                raw_patch,
                (self.patch_size, self.patch_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)
            patch_np_list.append(patch_np)

        mapped_resolution = (2.0 * local_radius_m) / self.patch_size
        r_center = self.patch_size / 2.0

        def _empty_world_region(center_world: np.ndarray):
            A_poly = np.zeros((self.N_faces, 2))
            b_poly = np.zeros(self.N_faces)
            A_poly[:4, :] = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
            b_poly[0] = center_world[0] + 1.0
            b_poly[1] = -(center_world[0] - 1.0)
            b_poly[2] = center_world[1] + 1.0
            b_poly[3] = -(center_world[1] - 1.0)
            b_poly[4:] = 1e5
            return A_poly, b_poly, np.eye(2), center_world.copy()

        def _safe_visual_shape(A_poly: np.ndarray, b_poly: np.ndarray, c_world: np.ndarray):
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

        def _project_halfspaces_to_world(a_pix, b_pix, p_pixel, c_pixel, center_world: np.ndarray):
            if c_pixel is not None:
                c_world = np.array([
                    (float(c_pixel[0]) - r_center) * mapped_resolution + center_world[0],
                    (float(c_pixel[1]) - r_center) * mapped_resolution + center_world[1],
                ])
            else:
                c_world = center_world.copy()

            A_poly = np.zeros((self.N_faces, 2))
            b_poly = np.zeros(self.N_faces)
            if a_pix is not None and b_pix is not None and len(a_pix) > 0:
                A_pix = np.asarray(a_pix, dtype=float)
                b_pix = np.asarray(b_pix, dtype=float)

                A_world = A_pix / mapped_resolution
                norms = np.linalg.norm(A_world, axis=1)
                valid_mask = norms > 1e-5

                A_world = A_world[valid_mask]
                b_pix = b_pix[valid_mask]
                A_pix = A_pix[valid_mask]
                norms = norms[valid_mask]

                if len(norms) == 0:
                    return _empty_world_region(center_world)

                A_world_norm = A_world / norms[:, None]
                offset_pix = np.dot(A_pix, np.array([r_center, r_center]))
                offset_phys = np.dot(A_world, center_world)
                b_world_raw = b_pix - offset_pix + offset_phys
                b_world_norm = b_world_raw / norms
                b_world_safe = b_world_norm - self.r_safe
                dists = np.abs(np.dot(A_world_norm, c_world) - b_world_norm)

                valid_num = len(b_world_safe)
                if valid_num > self.N_faces:
                    idx = np.argsort(dists)[:self.N_faces]
                    A_world_norm = A_world_norm[idx]
                    b_world_safe = b_world_safe[idx]
                    valid_num = self.N_faces

                A_poly[:valid_num] = A_world_norm
                b_poly[:valid_num] = b_world_safe
                if valid_num < self.N_faces:
                    b_poly[valid_num:] = 1e5
            else:
                return _empty_world_region(center_world)

            if p_pixel is not None:
                try:
                    P_inv_world = np.linalg.inv(np.asarray(p_pixel, dtype=float)) * mapped_resolution
                except np.linalg.LinAlgError:
                    P_inv_world = _safe_visual_shape(A_poly, b_poly, c_world)
            else:
                P_inv_world = _safe_visual_shape(A_poly, b_poly, c_world)

            return A_poly, b_poly, P_inv_world, c_world

        if self.use_cpp_neural_iris and cpp_infer_safe_region_batch_halfspaces is not None:
            t_start_nn = time.time()
            try:
                a_batch, b_batch, p_batch, c_batch = cpp_infer_safe_region_batch_halfspaces(
                    np.asarray(patch_np_list, dtype=np.float32),
                    patch_size=self.patch_size,
                )
            except Exception as e:
                print(f"[Planner] Neural-IRIS C++ batch failed, fallback to empty safe regions: {e}")
                a_batch = [None for _ in range(batch_size)]
                b_batch = [None for _ in range(batch_size)]
                p_batch = [None for _ in range(batch_size)]
                c_batch = [None for _ in range(batch_size)]
            batch_call_time = time.time() - t_start_nn
        else:
            t_start_nn = time.time()
            try:
                a_batch, b_batch, p_batch, c_batch = py_infer_safe_region_batch_halfspaces(
                    np.asarray(patch_np_list, dtype=np.float32),
                    patch_size=self.patch_size,
                )
            except Exception as e:
                print(f"[Planner] Neural-IRIS Python batch failed, fallback to empty safe regions: {e}")
                a_batch = [None for _ in range(batch_size)]
                b_batch = [None for _ in range(batch_size)]
                p_batch = [None for _ in range(batch_size)]
                c_batch = [None for _ in range(batch_size)]
            batch_call_time = time.time() - t_start_nn

        results = []
        per_item_batch_call_time = batch_call_time / max(1, batch_size)
        for k in range(batch_size):
            center_world = np.array([
                c_x_ints[k] * map_resolution,
                c_y_ints[k] * map_resolution,
            ], dtype=float)

            t_unpack = time.time()
            a_pix = a_batch[k] if k < len(a_batch) else None
            b_pix = b_batch[k] if k < len(b_batch) else None
            p_pixel = p_batch[k] if k < len(p_batch) else None
            c_pixel = c_batch[k] if k < len(c_batch) else None
            A_poly, b_poly, P_inv_world, c_world = _project_halfspaces_to_world(
                a_pix, b_pix, p_pixel, c_pixel, center_world
            )
            python_postprocess_time = time.time() - t_unpack
            total_corridor_time = per_item_batch_call_time + python_postprocess_time
            results.append((
                A_poly,
                b_poly,
                P_inv_world,
                c_world,
                per_item_batch_call_time,
                python_postprocess_time,
                total_corridor_time,
            ))

        return results

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
        batched_results = self._predict_safe_regions_batched(occupancy_map, states_xy_list, map_resolution)

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
