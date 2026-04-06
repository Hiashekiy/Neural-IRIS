import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os
import sys
import networkx as nx

# --- 路径处理 ---
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

# 导入新的 BFS 搜索算法
from src.map_utils.bfs_path import find_random_path_bfs

class CommonEnv(gym.Env):
    """
    【通用基类 CommonEnv】
    
    职责：
    1. 统一物理尺度：锁定最大边长 100m，分辨率 0.5m。
    2. 统一观测空间：34维向量 + 64x64 局部地图。
    3. 统一感知逻辑：Lidar, CNN, Path Error, 以及强制障碍物预检测。
    4. 统一基础设施：NavigationGraph, 坐标转换。
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, map_obstacle, max_steps=1000, input_resolution=1.0, path_mode='train'):
        super(CommonEnv, self).__init__()
        
        self.path_mode = path_mode  # 路径模式: 'train' (NavGraphTrain) 或 'deploy' (NavGraphDeploy)
        self.lidar_path_shortcut = None  # 用于存储 LiDAR 裁剪相关信息 (pos, heading, cached_dists)
        # ==========================================
        # 1. 物理世界铁律 (Physics Constants)
        # ==========================================
        # self.MAX_PHYSICAL_SIDE = 100.0    # 物理世界最大边长：100米
        # self.INTERNAL_RESOLUTION = 0.5    # 内部精度：0.5米/格
        self.input_resolution = float(input_resolution)
        self.max_steps = max_steps
        self.dt = 0.1  # 仿真步长
        
        # ==========================================
        # 2. 算法关键超参数 (Algorithm Hyperparameters)
        #    [集中管理设计，方便调参]
        # ==========================================
        # A. 路径投影相关
        self.PROJ_WINDOW_SIZE = 10    # 投影搜索窗口大小 (前后n段)
        self.PROJ_MAX_JUMP = 5.0     # 允许 s 突变的最大距离
        
        # B. 障碍物与观测相关
        self.OBS_SAFE_MARGIN = 4.0   # 障碍物后方多少米视为回归安全点
        self.GEO_NORM_SCALE = 25.0   # 几何特征归一化分母 (对应 detection_len)
        self.COL_CHECK_LEN = 20.0    # 碰撞检测前瞻距离 (米)
        
        # C. 感知相关
        self.LIDAR_MAX_DIST = 25.0   # 雷达最大探测距离
        self.CNN_FOV = 25.0          # CNN 局部地图视野边长
        
        # ==========================================
        # 3. 地图标准化 (Map Normalization)
        # ==========================================
        self.original_map = map_obstacle
        orig_h, orig_w = map_obstacle.shape


        self.MAX_PHYSICAL_SIDE = max(orig_h, orig_w) * self.input_resolution    # 物理世界最大边长：100米
        self.INTERNAL_RESOLUTION = self.input_resolution    # 内部精度：0.5米/格
        self.COL_STEP_SIZE = self.INTERNAL_RESOLUTION     # 碰撞检测采样步长 (米)
        
        self.target_grid_max = int(self.MAX_PHYSICAL_SIDE / self.INTERNAL_RESOLUTION)
        
        if orig_w >= orig_h:
            self.internal_w = self.target_grid_max
            self.scale = self.internal_w / orig_w
            self.internal_h = int(orig_h * self.scale)
        else:
            self.internal_h = self.target_grid_max
            self.scale = self.internal_h / orig_h
            self.internal_w = int(orig_w * self.scale)

        self.internal_w = max(self.internal_w, 20)
        self.internal_h = max(self.internal_h, 20)
        
        resized_map = cv2.resize(self.original_map, (self.internal_w, self.internal_h), interpolation=cv2.INTER_NEAREST)
        _, self.map_obstacle = cv2.threshold(resized_map, 0.5, 1, cv2.THRESH_BINARY)
        
        self.map_width_m = self.internal_w * self.INTERNAL_RESOLUTION
        self.map_height_m = self.internal_h * self.INTERNAL_RESOLUTION
        
        # ==========================================
        # 4. 导航图 (Navigation Graph) - 根据 path_mode 选择
        # ==========================================
        # 旧的 nav_graph 已经被移除，现在这部分留空或者只需保留原本的接口结构。
        self.nav_graph = None # no longer needed
        self.cells = None
        
        self.original_poly_points_np = None  # 原始路径点 (米)，在 reset 时生成
        self.global_path_cids = None  # 记录经过的 cell ID (仅 deploy 模式有效)
        
        # ==========================================
        # 5. 观测空间 (Observation Space)
        # ==========================================
        self.cnn_map_size = 64
        self.cnn_res = self.CNN_FOV / self.cnn_map_size
        self.lidar_num_rays = 9
        self.lidar_fov = np.pi
        
        self.observation_space = spaces.Dict({
            "vec": spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32),
            "img": spaces.Box(low=0, high=255, shape=(1, self.cnn_map_size, self.cnn_map_size), dtype=np.uint8)
        })

        # ==========================================
        # 6. 运行时状态
        # ==========================================
        self.poly_points_np = None          
        self.poly_segment_lens = None       
        self.poly_cumulative_dists = None   
        self.total_poly_len = 0.0
        
        self.state = np.zeros(4) 
        self.virtual_s = 0.0     
        self.last_s = 0.0
        self.current_step = 0
        self.last_heading = 0.0
        self.history_trajectory = []
        
        self.geo_features = {'obs_start': 1.0, 'obs_len': 0.0, 'safe_end_dist': 1.0, 'safe_end_xy': [1.0, 0.0]}
        self.action_space = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.history_trajectory = []
        self.current_step = 0
        self.last_heading = 0.0
        self.geo_features = {'obs_start': 1.0, 'obs_len': 0.0, 'safe_end_dist': 1.0, 'safe_end_xy': [1.0, 0.0]}
        
        self._generate_random_path()
        
        if self.poly_points_np is not None and len(self.poly_points_np) > 0:
            self.start_pos = self.poly_points_np[0]
            self.target_pos = self.poly_points_np[-1]
            init_heading = self._calculate_path_heading(0.0)
            self.state = np.array([self.start_pos[0], self.start_pos[1], 0.0, 0.0])
            self.last_heading = init_heading
            self.original_poly_points_np = self.poly_points_np.copy()
        else:
            self.state = np.zeros(4)
        
        self.virtual_s = 0.0
        self.last_s = 0.0 # Reset last_s explicitly
        self.history_trajectory.append(self.state[:2])
        
        self._update_geo_features_observation()
        
        return self._get_obs(), {}

    def step(self, action):
        raise NotImplementedError("CommonEnv is abstract. Use GeometryEnv, TuningEnv, or HybridEnv.")

    # ==========================================
    # 核心功能实现
    # ==========================================

    def _calculate_progress_on_poly(self, pos):
        """
        [优化合并版] 计算当前位置在路径上的进度 s。
        直接使用类成员变量，结合局部窗口搜索 + 向量化投影，高效且鲁棒。
        """
        # 0. 基础数据检查
        if self.poly_points_np is None: 
            return 0.0
        
        points = self.poly_points_np
        seg_lens = self.poly_segment_lens
        cum_dists = self.poly_cumulative_dists
        prev_s = getattr(self, 'last_s', None) # 获取上一帧进度
        
        num_segments = len(points) - 1
        if num_segments < 1: return 0.0

        # === 参数配置 (从 __init__ 获取) ===
        window_size = self.PROJ_WINDOW_SIZE  # 向前向后搜索的线段数量
        max_jump = self.PROJ_MAX_JUMP        # 允许 s 突变的最大距离

        # 1. 确定搜索窗口 (Search Window)
        if prev_s is None:
            # Case A: 无历史记录 (如 Reset 后第一帧)，全图搜索
            search_start, search_end = 0, num_segments
        else:
            # Case B: 有历史记录，只搜附近的线段 (局部搜索)
            # np.searchsorted 找到 prev_s 所在的线段索引
            idx_approx = np.clip(np.searchsorted(cum_dists, prev_s) - 1, 0, num_segments - 1)
            search_start = max(0, idx_approx - window_size)
            search_end = min(num_segments, idx_approx + window_size + 1) # +1 确保覆盖前方

        # 2. 截取数据片段 (Slicing)
        p_sub = points[search_start : search_end + 1] 
        lens_sub = seg_lens[search_start : search_end]
        cum_dists_start = cum_dists[search_start : search_end] 
        
        if len(lens_sub) == 0: 
            return prev_s if prev_s is not None else 0.0

        # 3. 向量化投影计算 (Vectorized Projection)
        p_a = p_sub[:-1] 
        p_b = p_sub[1:]  
        
        v_seg = p_b - p_a       
        v_pt = pos - p_a        
        
        lens_sub_sq = lens_sub ** 2
        lens_sub_sq = np.where(lens_sub_sq < 1e-8, 1e-8, lens_sub_sq)
        
        dot_prod = np.sum(v_pt * v_seg, axis=1)
        t = np.clip(dot_prod / lens_sub_sq, 0.0, 1.0) 

        # 4. 寻找最近线段
        projections = p_a + v_seg * t[:, np.newaxis]
        dists_sq = np.sum((pos - projections)**2, axis=1)
        
        best_local_idx = np.argmin(dists_sq)
        
        # 5. 计算全局 s
        best_s = cum_dists_start[best_local_idx] + t[best_local_idx] * lens_sub[best_local_idx]

        # 6. 突变保护 (Anti-Jumping Filter)
        if prev_s is not None:
            diff = best_s - prev_s
            if diff > max_jump: 
                best_s = prev_s + max_jump 
            elif diff < -max_jump: 
                best_s = prev_s - max_jump * 0.1 
                
        return best_s

    def _update_geo_features_observation(self):
        """
        【关键修复】沿当前参考路径探测障碍物，更新 self.geo_features。
        使用了统一的 OBS_SAFE_MARGIN 和 GEO_NORM_SCALE 参数。
        """
        if self.poly_points_np is None: return

        # 探测碰撞 (使用统一参数)
        is_col, start_col, end_col = self._detect_collision_on_ref_path(self.virtual_s)
        
        if is_col:
            path_heading = self._calculate_path_heading(self.virtual_s)
            
            obs_start_dist = start_col - self.virtual_s
            obs_len = end_col - start_col
            
            # 使用类参数计算安全回归点
            obs_safe_s = min(end_col + self.OBS_SAFE_MARGIN, self.total_poly_len)
            obs_safe_pt = self._get_poly_point_at_s(obs_safe_s)
            current_pt = self._get_poly_point_at_s(self.virtual_s)
            
            # 转局部坐标
            c, s_ang = np.cos(path_heading), np.sin(path_heading)
            rot = np.array([[c, s_ang], [-s_ang, c]])
            local_obs_end = rot @ (obs_safe_pt - current_pt)
            
            # 使用类参数进行归一化
            norm = self.GEO_NORM_SCALE
            self.geo_features = {
                'obs_start': np.clip(obs_start_dist / norm, 0, 1),
                'obs_len': np.clip(obs_len / norm, 0, 1),
                'safe_end_dist': np.clip(np.linalg.norm(local_obs_end) / norm, 0, 1),
                'safe_end_xy': local_obs_end / norm
            }
        else:
            self.geo_features = {'obs_start': 1.0, 'obs_len': 0.0, 'safe_end_dist': 1.0, 'safe_end_xy': [1.0, 0.0]}

    def _get_obs(self):
        """获取统一观测"""
        self._update_geo_features_observation()
        
        pos = self.state[:2]
        vel = self.state[2:]
        speed = np.linalg.norm(vel)
        
        if speed > 0.1:
            heading = np.arctan2(vel[1], vel[0])
            self.last_heading = heading
        else:
            heading = self.last_heading
            
        ref_pt = self._get_poly_point_at_s(self.virtual_s)
        path_heading = self._calculate_path_heading(self.virtual_s)
        
        heading_err = (heading - path_heading + np.pi) % (2*np.pi) - np.pi
        heading_err /= np.pi 
        cte = np.linalg.norm(pos - ref_pt)
        
        lookahead = self._get_local_lookahead_points(pos, heading, self.virtual_s)
        
        geo = np.array([
            self.geo_features['obs_start'], 
            self.geo_features['obs_len'],
            self.geo_features['safe_end_dist'], 
            self.geo_features['safe_end_xy'][0], 
            self.geo_features['safe_end_xy'][1]
        ], dtype=np.float32)
        
        lidar = self._get_lidar_obs(pos, heading)
        self.lidar_path_shortcut = (pos, heading, lidar * self.LIDAR_MAX_DIST)
        
        vec_obs = np.concatenate(([speed/10.0, self.virtual_s/self.total_poly_len, cte/5.0, heading_err], lookahead, geo, lidar))
        img_obs = self._get_local_map_cnn(pos, heading)
        
        return {
            "vec": vec_obs.astype(np.float32), 
            "img": img_obs
        }

    # ==========================================
    # 工具函数 (Collision, Path, Lidar, CNN)
    # ==========================================

    def _detect_collision_on_ref_path(self, start_s):
        """【性能优化】向量化碰撞检测，避免逐点循环"""
        is_col, start_col, end_col = False, -1.0, -1.0
        max_s = min(start_s + self.COL_CHECK_LEN, self.total_poly_len)
        
        if self.total_poly_len <= 0:
            return False, -1, -1

        # 【优化】一次性生成所有采样点
        s_samples = np.arange(start_s, max_s, self.COL_STEP_SIZE)
        if len(s_samples) == 0:
            return False, -1, -1
        
        # 【优化】批量获取路径点（向量化）
        points = self._get_poly_points_at_s_batch(s_samples)
        
        # 【优化】批量检查碰撞
        collisions = self._is_safe_points_batch(points / self.INTERNAL_RESOLUTION, margin=0)
        
        # 找到碰撞区间
        collision_mask = ~collisions  # True表示碰撞
        if not np.any(collision_mask):
            return False, -1, -1
        
        # 找第一个碰撞点
        first_col_idx = np.argmax(collision_mask)
        start_col = s_samples[first_col_idx]
        is_col = True
        
        # 找碰撞结束点（第一个碰撞后的安全点）
        remaining_mask = collision_mask[first_col_idx:]
        if not np.all(remaining_mask):
            # 找到第一个安全点
            end_col_idx = first_col_idx + np.argmin(remaining_mask)
            end_col = s_samples[end_col_idx]
        else:
            end_col = max_s
        
        return is_col, start_col, end_col
    
    def _get_poly_points_at_s_batch(self, s_vals):
        """【性能优化】批量获取路径点，避免循环调用"""
        if self.poly_points_np is None:
            return np.zeros((len(s_vals), 2))
        
        s_vals = np.clip(s_vals, 0.0, max(self.total_poly_len, 1e-6))
        
        # 批量二分查找
        indices = np.clip(
            np.searchsorted(self.poly_cumulative_dists, s_vals) - 1, 
            0, 
            len(self.poly_points_np) - 2
        )
        
        # 批量插值
        s_starts = self.poly_cumulative_dists[indices]
        s_ends = self.poly_cumulative_dists[indices + 1]
        ratios = np.divide(
            s_vals - s_starts, 
            s_ends - s_starts, 
            out=np.zeros_like(s_vals), 
            where=(s_ends - s_starts) > 1e-6
        )
        
        # 向量化插值计算
        p_starts = self.poly_points_np[indices]
        p_ends = self.poly_points_np[indices + 1]
        points = p_starts + (p_ends - p_starts) * ratios[:, np.newaxis]
        
        return points
    
    def _is_safe_points_batch(self, points, margin=0.0):
        """【性能优化】批量检查点是否安全"""
        points_int = points.astype(int)
        
        # 边界检查
        valid_mask = (
            (points_int[:, 0] >= 0) & (points_int[:, 0] < self.internal_w) &
            (points_int[:, 1] >= 0) & (points_int[:, 1] < self.internal_h)
        )
        
        safe = np.zeros(len(points), dtype=bool)
        
        if margin == 0:
            # 无margin时直接查询
            valid_points = points_int[valid_mask]
            if len(valid_points) > 0:
                obstacles = self.map_obstacle[valid_points[:, 1], valid_points[:, 0]]
                safe[valid_mask] = (obstacles == 0)
        else:
            # 有margin时逐点检查（保持原逻辑）
            for i, (x, y) in enumerate(points_int):
                if valid_mask[i]:
                    x0 = max(0, int(x - margin))
                    x1 = min(self.internal_w, int(x + margin + 1))
                    y0 = max(0, int(y - margin))
                    y1 = min(self.internal_h, int(y + margin + 1))
                    safe[i] = (np.sum(self.map_obstacle[y0:y1, x0:x1]) == 0)
        
        return safe

    def _generate_random_path(self, min_dist_m=20.0):
        """
        基于常规的网格搜索生成随机全局路径。
        直接使用 BFS 算法，不再进行空闲区域的划分。
        """
        # 将物理距离要求转换为 Grid 距离
        min_dist_grid = min_dist_m / self.INTERNAL_RESOLUTION
        margin = max(1, int(2.0 / self.INTERNAL_RESOLUTION))
        
        # 使用基础 BFS 搜索安全起点和目标点的路径
        result = find_random_path_bfs(self.map_obstacle, min_dist_grid, margin=margin)
        
        if result is not None:
            s_grid, t_grid, path_grid = result
            self.start_pos = np.array(s_grid) * self.INTERNAL_RESOLUTION
            self.target_pos = np.array(t_grid) * self.INTERNAL_RESOLUTION
            self.global_path_cids = None # 兼容旧接口
            
            # 使用 RDP (Ramer-Douglas-Peucker) 抽稀或者固定间隔下采样路径
            raw_path_meters = np.array(path_grid) * self.INTERNAL_RESOLUTION
            
            # 简单的平滑下采样：取关键点即可
            # 比如每相隔 1m 取一个点，或者转向点
            if len(raw_path_meters) > 3:
                # 只保留距离上一个点大于 1.0m 的点，以及起点终点
                sampled_path = [raw_path_meters[0]]
                for pt in raw_path_meters[1:-1]:
                    if np.linalg.norm(pt - sampled_path[-1]) >= 1.0:
                        sampled_path.append(pt)
                sampled_path.append(raw_path_meters[-1])
                self.poly_points_np = np.array(sampled_path)
            else:
                self.poly_points_np = raw_path_meters
            
            self._update_path_metrics()
        else:
            # 路径为空，使用备用路径
            cx, cy = self.map_width_m/2, self.map_height_m/2
            self.poly_points_np = np.array([[cx-10, cy-10], [cx+10, cy+10]])
            self._update_path_metrics()

    def _update_path_metrics(self):
        if self.poly_points_np is None or len(self.poly_points_np) < 2:
            self.total_poly_len = 0.0
            return
        p_deltas = np.diff(self.poly_points_np, axis=0)
        self.poly_segment_lens = np.linalg.norm(p_deltas, axis=1)
        self.poly_cumulative_dists = np.concatenate(([0], np.cumsum(self.poly_segment_lens)))
        self.total_poly_len = self.poly_cumulative_dists[-1]

    def _get_poly_point_at_s(self, s_val):
        if self.poly_points_np is None: return np.zeros(2)
        s_val = np.clip(s_val, 0.0, max(self.total_poly_len, 1e-6))
        idx = np.clip(np.searchsorted(self.poly_cumulative_dists, s_val) - 1, 0, len(self.poly_points_np) - 2)
        s_start, s_end = self.poly_cumulative_dists[idx], self.poly_cumulative_dists[idx+1]
        ratio = (s_val - s_start) / (s_end - s_start) if s_end - s_start > 1e-6 else 0.0
        return self.poly_points_np[idx] + (self.poly_points_np[idx+1] - self.poly_points_np[idx]) * ratio

    def _calculate_path_heading(self, s_val, delta=0.5):
        if self.total_poly_len <= 0: return 0.0
        p_next = self._get_poly_point_at_s(min(self.total_poly_len, s_val + delta))
        p_prev = self._get_poly_point_at_s(max(0, s_val - delta))
        vec = p_next - p_prev
        return np.arctan2(vec[1], vec[0]) if np.linalg.norm(vec) > 1e-6 else 0.0

    def _get_local_lookahead_points(self, pos, heading, s):
        pts = []
        c, s_a = np.cos(heading), np.sin(heading)
        rot = np.array([[c, s_a], [-s_a, c]])
        for i in range(-2, 6):
            gp = self._get_poly_point_at_s(s + i * 3.0)
            lp = rot @ (gp - pos)
            pts.extend([lp[0]/self.GEO_NORM_SCALE, lp[1]/self.GEO_NORM_SCALE]) # 使用统一归一化
        return np.array(pts, dtype=np.float32)

    def _get_lidar_obs(self, pos, heading):
        num_rays = self.lidar_num_rays
        max_dist = self.LIDAR_MAX_DIST # 使用统一参数
        step = self.INTERNAL_RESOLUTION * 0.8
        angles = np.linspace(heading - self.lidar_fov / 2, heading + self.lidar_fov / 2, num_rays)
        dists = np.full(num_rays, max_dist)
        
        for i, ang in enumerate(angles):
            dx, dy = np.cos(ang) * step, np.sin(ang) * step
            cx, cy = pos[0], pos[1]
            for d in range(int(max_dist / step)):
                cx += dx; cy += dy
                gx, gy = int(cx/self.INTERNAL_RESOLUTION), int(cy/self.INTERNAL_RESOLUTION)
                if not (0<=gx<self.internal_w and 0<=gy<self.internal_h) or self.map_obstacle[gy, gx] == 1:
                    dists[i] = (d + 1) * step
                    break
        return dists / max_dist

    def _get_local_map_cnn(self, pos, heading):
        cx, cy = pos / self.INTERNAL_RESOLUTION
        M = cv2.getRotationMatrix2D((cx, cy), np.degrees(heading) + 90, self.INTERNAL_RESOLUTION / self.cnn_res)
        M[0, 2] += self.cnn_map_size / 2 - cx
        M[1, 2] += self.cnn_map_size * 0.8 - cy
        src = (self.map_obstacle * 255).astype(np.uint8)
        img = cv2.warpAffine(src, M, (self.cnn_map_size, self.cnn_map_size), 
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img[np.newaxis, :, :]

    def _is_safe_point_grid(self, pt, margin=2.0):
        x, y = int(pt[0]), int(pt[1])
        if not (0 <= x < self.internal_w and 0 <= y < self.internal_h): return False
        x0, x1 = max(0, int(x-margin)), min(self.internal_w, int(x+margin+1))
        y0, y1 = max(0, int(y-margin)), min(self.internal_h, int(y+margin+1))
        return np.sum(self.map_obstacle[y0:y1, x0:x1]) == 0

    def _sample_point_in_cell(self, cell, margin=2.0):
        for _ in range(10):
            col = cell.columns[np.random.randint(len(cell.columns))]
            x, y0, y1 = col
            ymin, ymax = y0 + margin, y1 - margin
            y = (y0+y1)/2.0 if ymax <= ymin else np.random.uniform(ymin, ymax)
            pt = np.array([float(x), float(y)])
            if self._is_safe_point_grid(pt, margin): return pt
        return np.array(cell.get_center(), dtype=np.float64)

    def _check_termination(self):
        """通用终止条件检查"""
        pos = self.state[:2]
        dist_to_goal = np.linalg.norm(pos - self.target_pos)
        collided = not self._is_safe_point_grid(pos / self.INTERNAL_RESOLUTION, margin=0)
        
        terminated = False
        truncated = False
        is_success = False
        
        if collided:
            terminated = True
        elif dist_to_goal < 3.0: 
            is_success = True
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True
            
        return terminated, truncated, is_success
    
    def _update_dist_map(self):
        """[新增] 手动更新距离场缓存"""
        # 反转地图：障碍物=0(黑), 空地=1(白)。DT 计算到最近黑点的距离
        # 结果单位是像素，乘以分辨率转为米
        dist_px = cv2.distanceTransform(
            (1 - self.map_obstacle).astype(np.uint8), 
            cv2.DIST_L2, 
            5
        )
        self.cached_dist_map = dist_px * self.INTERNAL_RESOLUTION