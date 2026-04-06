# -*- coding: utf-8 -*-
import gymnasium as gym
import numpy as np
import cv2

class DynamicObstacleWrapper(gym.Wrapper):
    """
    【Stage 1 专用】动态障碍物 + 路径回滚
    每次 Reset 生成新障碍，并瞬移车辆，用于高频训练几何修路能力。
    
    关键特性：
    1. 在路径上每隔10米采样生成障碍物队列
    2. 每次 reset 时瞬移车辆到障碍前30米
    3. 将终点设置为障碍后20米（只需学会绕过障碍即可，无需走完全程）
    4. 队列中有多个障碍时，逐个训练
    """
    def __init__(self, env, density=0.8, min_obs_radius=1.5, max_obs_radius=2.0, passage_width=2.5):
        super().__init__(env)
        self.density = density
        self.min_r = min_obs_radius
        self.max_r = max_obs_radius
        self.passage_w = passage_width 
        
        # 备份干净地图
        self.clean_map = self.env.unwrapped.map_obstacle.copy()
        self.res = self.env.unwrapped.INTERNAL_RESOLUTION
        self.scenario_queue = []

    def reset(self, **kwargs):
        if len(self.scenario_queue) > 0:
            return self._teleport_to_next_scenario()

        # 生成新回合
        self.env.unwrapped.map_obstacle = self.clean_map.copy()
        obs, info = self.env.reset(**kwargs)
        
        # 预计算距离场
        dist_input = (self.clean_map == 0).astype(np.uint8)
        self.dist_map = cv2.distanceTransform(dist_input, cv2.DIST_L2, 5)
        
        # 沿路径采样生成障碍物队列
        self.scenario_queue = []
        path_len = self.env.unwrapped.total_poly_len
        candidate_s = np.arange(30.0, path_len - 10.0, 10.0) # 每10米采样
        np.random.shuffle(candidate_s)
        
        for s in candidate_s:
            if np.random.random() < self.density:
                pt = self.env.unwrapped._get_poly_point_at_s(s)
                success, c, r = self._calc_obs(pt)
                if success:
                    self.scenario_queue.append((s, c, r))
        
        if self.scenario_queue:
            return self._teleport_to_next_scenario()
        return obs, info

    def _teleport_to_next_scenario(self):
        target_s, center_m, radius_m = self.scenario_queue.pop(0)
        
        # 1. 路径回滚
        self.env.unwrapped.poly_points_np = self.env.unwrapped.original_poly_points_np.copy()
        self.env.unwrapped._update_path_metrics()
        
        # 2. 注入障碍
        self.env.unwrapped.map_obstacle = self.clean_map.copy()
        self._draw_obs(center_m, radius_m)
        self.env.unwrapped._update_dist_map() # 刷新距离场缓存
        
        # 3. 瞬移车辆到障碍前 30m
        start_s = max(0, target_s - 30.0)
        pos = self.env.unwrapped._get_poly_point_at_s(start_s)
        self.env.unwrapped.state[:2] = pos
        self.env.unwrapped.state[2:] = 0.0 # 速度归零
        self.env.unwrapped.virtual_s = start_s
        self.env.unwrapped.last_s = start_s
        self.env.unwrapped.current_step = 0
        
        # 4. 【关键修正】将终点设置为障碍物后方的安全点（20米后）
        path_len = self.env.unwrapped.total_poly_len
        end_s = min(target_s + 20.0, path_len)  # 障碍后20米作为新终点
        self.env.unwrapped.target_pos = self.env.unwrapped._get_poly_point_at_s(end_s)
        
        return self.env.unwrapped._get_obs(), {}

    def _calc_obs(self, pt_m):
        px, py = int(pt_m[0]/self.res), int(pt_m[1]/self.res)
        h, w = self.dist_map.shape
        if not (0<=px<w and 0<=py<h): return False, None, None
        
        dist = self.dist_map[py, px]
        max_r_px = min(dist - self.passage_w/self.res/2, self.max_r/self.res)
        if max_r_px < self.min_r/self.res: return False, None, None
        
        r_px = np.random.uniform(self.min_r/self.res, max_r_px)
        # 随机偏移
        shift = np.random.uniform(0, r_px*0.2)
        ang = np.random.uniform(0, 6.28)
        cx = px + shift*np.cos(ang)
        cy = py + shift*np.sin(ang)
        
        return True, np.array([cx, cy])*self.res, r_px*self.res

    def _draw_obs(self, center_m, radius_m):
        # 绘制不规则多边形
        num = np.random.randint(5, 8)
        angs = np.linspace(0, 6.28, num, endpoint=False) + np.random.uniform(-0.2, 0.2, num)
        rs = radius_m * np.random.uniform(0.8, 1.2, num)
        pts = np.column_stack((center_m[0]+rs*np.cos(angs), center_m[1]+rs*np.sin(angs)))
        pts_px = (pts / self.res).astype(np.int32)
        cv2.fillPoly(self.env.unwrapped.map_obstacle, [pts_px], 1)


class StaticObstacleWrapper(gym.Wrapper):
    """
    [测试专用] 动态障碍物注入层
    每次 Reset 时，在生成的全局路径周围随机撒下静态障碍物。
    """
    def __init__(self, env, density=0.4, min_obs_radius=1.0, max_obs_radius=2.5, passage_width=3.5, verbose=False):
        super().__init__(env)
        self.density = density            # 撒点密度 (概率)
        self.min_obs_radius = min_obs_radius
        self.max_obs_radius = max_obs_radius
        self.passage_width = passage_width # 保证留出的通道宽度 (米)
        self.verbose = verbose            # 是否输出调试信息
        
        # 备份原始纯净地图
        self.clean_map = self.env.unwrapped.map_obstacle.copy()
        # 获取分辨率
        self.internal_res = self.env.unwrapped.INTERNAL_RESOLUTION
        
        # 预计算像素参数
        self.passage_width_px = int(self.passage_width / self.internal_res)
        self.min_obs_radius_px = int(self.min_obs_radius / self.internal_res)
        self.max_obs_radius_px = int(self.max_obs_radius / self.internal_res)
        
        # 记录已放置的障碍物（用于可视化）
        self.placed_obstacles = []

    def reset(self, **kwargs):
        # 1. 恢复纯净地图
        self.env.unwrapped.map_obstacle = self.clean_map.copy()
        
        # 2. 清空障碍物记录
        self.placed_obstacles = []
        
        # 3. 调用原始 Reset
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple):
            obs, info = ret
        else:
            obs, info = ret, {}
        
        # 4. 计算距离场
        dist_input = (self.clean_map == 0).astype(np.uint8) 
        dist_map = cv2.distanceTransform(dist_input, cv2.DIST_L2, 5)
        
        # 5. 沿路径撒点
        if hasattr(self.env.unwrapped, 'total_poly_len') and self.env.unwrapped.poly_points_np is not None:
            total_len = self.env.unwrapped.total_poly_len
            
            if self.verbose:
                print(f"  [Obstacle Wrapper] 路径总长度: {total_len:.1f}m")
            
            if total_len > 20.0:
                sampling_interval = 20.0  # 每隔20米尝试撒一个（降低密度）
                start_safe_m = 10.0       # 起点保护区
                end_safe_m = 10.0         # 终点保护区
                
                candidate_s = np.arange(start_safe_m, total_len - end_safe_m, sampling_interval)
                if self.verbose:
                    print(f"  [Obstacle Wrapper] 采样点数: {len(candidate_s)}, density={self.density}")
                
                attempts = 0
                successes = 0
                for s in candidate_s:
                    if np.random.random() < self.density:
                        attempts += 1
                        pt = self.env.unwrapped._get_poly_point_at_s(s)
                        success, center_m, radius_m = self._calc_obstacle_params(pt, dist_map)
                        if success:
                            self._draw_obstacle(center_m, radius_m)
                            # 记录障碍物位置
                            self.placed_obstacles.append(center_m)
                            successes += 1
                
                if self.verbose:
                    print(f"  [Obstacle Wrapper] 尝试生成: {attempts}, 成功: {successes}")
            elif self.verbose:
                print(f"  [Obstacle Wrapper] 路径太短 (<30m)，跳过障碍物生成")
        elif self.verbose:
            print(f"  [Obstacle Wrapper] 未找到路径信息，跳过障碍物生成")
        
        # 6. 更新环境缓存并重新获取观测
        if hasattr(self.env.unwrapped, '_update_dist_map'): # 如果有缓存更新函数则调用
             self.env.unwrapped._update_dist_map()

        if hasattr(self.env.unwrapped, '_get_obs'):
            obs = self.env.unwrapped._get_obs()
            
        return obs, info

    def _calc_obstacle_params(self, path_pt_m, dist_map):
        px, py = int(path_pt_m[0] / self.internal_res), int(path_pt_m[1] / self.internal_res)
        h, w = dist_map.shape
        if not (0 <= px < w and 0 <= py < h): 
            # print(f"    跳过: 点超出边界 ({px}, {py})")
            return False, None, None
        
        dist_at_path = dist_map[py, px]
        # 放宽条件：使用更小的除数使得障碍物更容易生成
        max_allowed_r = dist_at_path - (self.passage_width_px / 2.0)  # 从 1.5 改为 2.0
        if max_allowed_r < self.min_obs_radius_px: 
            # print(f"    跳过: 空间不足 (dist={dist_at_path*self.internal_res:.1f}m, need>{self.min_obs_radius}m)")
            return False, None, None

        final_max_px = min(max_allowed_r, self.max_obs_radius_px)
        obs_r_px = np.random.uniform(self.min_obs_radius_px, final_max_px)
        
        shift_dist = np.random.uniform(0, obs_r_px * 0.8)
        shift_angle = np.random.uniform(0, 2 * np.pi)
        center_x = px + shift_dist * np.cos(shift_angle)
        center_y = py + shift_dist * np.sin(shift_angle)

        if not (0 <= int(center_x) < w and 0 <= int(center_y) < h): 
            # print(f"    跳过: 中心超出边界")
            return False, None, None
        if dist_map[int(center_y), int(center_x)] < obs_r_px: 
            # print(f"    跳过: 中心位置空间不足")
            return False, None, None

        center_m = np.array([center_x, center_y]) * self.internal_res
        radius_m = obs_r_px * self.internal_res
        return True, center_m, radius_m

    def _draw_obstacle(self, center_m, radius_m):
        num_verts = np.random.randint(5, 9)
        angles = np.linspace(0, 2 * np.pi, num_verts, endpoint=False)
        angles += np.random.uniform(-0.2, 0.2, size=num_verts)
        radii = radius_m * np.random.uniform(0.7, 1.3, size=num_verts)
        
        x = center_m[0] + radii * np.cos(angles)
        y = center_m[1] + radii * np.sin(angles)
        poly_px = (np.column_stack((x, y)) / self.internal_res).astype(np.int32)
        
        cv2.fillPoly(self.env.unwrapped.map_obstacle, [poly_px], 1)


class PopUpObstacleWrapper(gym.Wrapper):
    """
    【动态验证专用】突发障碍物注入层 (Pop-up Obstacles)
    Reset时预先沿路计算好障碍物，但不写入地图。
    Step时实时监测车距，当车辆逼近（如<15米）时，瞬间将障碍物写入地图，强制触发动态形变。
    """
    def __init__(self, env, density=0.4, min_obs_radius=1.0, max_obs_radius=2.5, passage_width=3.5, trigger_dist=15.0, verbose=False):
        super().__init__(env)
        self.density = density
        self.min_obs_radius = min_obs_radius
        self.max_obs_radius = max_obs_radius
        self.passage_width = passage_width
        self.trigger_dist = trigger_dist  # 触发突发障碍物的距离阈值 (米)
        self.verbose = verbose
        
        self.clean_map = self.env.unwrapped.map_obstacle.copy()
        self.internal_res = self.env.unwrapped.INTERNAL_RESOLUTION
        
        self.passage_width_px = int(self.passage_width / self.internal_res)
        self.min_obs_radius_px = int(self.min_obs_radius / self.internal_res)
        self.max_obs_radius_px = int(self.max_obs_radius / self.internal_res)
        
        self.pending_obstacles = []  # 尚未触发的隐藏障碍物队列
        self.placed_obstacles = []   # 已经触发并显示在地图上的障碍物（供渲染调用）

    def reset(self, **kwargs):
        # 1. 恢复纯净地图并清空队列
        self.env.unwrapped.map_obstacle = self.clean_map.copy()
        self.pending_obstacles = []
        self.placed_obstacles = []
        
        # 2. 调用原始 Reset
        ret = self.env.reset(**kwargs)
        obs, info = ret if isinstance(ret, tuple) else (ret, {})
        
        # 3. 计算距离场，用于合法性校验
        dist_input = (self.clean_map == 0).astype(np.uint8) 
        dist_map = cv2.distanceTransform(dist_input, cv2.DIST_L2, 5)
        
        # 4. 沿路径【预计算】障碍物，但【不画到地图上】
        if hasattr(self.env.unwrapped, 'total_poly_len') and self.env.unwrapped.poly_points_np is not None:
            total_len = self.env.unwrapped.total_poly_len
            if total_len > 20.0:
                sampling_interval = 20.0
                candidate_s = np.arange(10.0, total_len - 10.0, sampling_interval)
                
                for s in candidate_s:
                    if np.random.random() < self.density:
                        pt = self.env.unwrapped._get_poly_point_at_s(s)
                        success, center_m, radius_m = self._calc_obstacle_params(pt, dist_map)
                        if success:
                            # 提前生成多边形的像素点坐标
                            poly_px = self._generate_polygon_px(center_m, radius_m)
                            # 压入隐藏队列
                            self.pending_obstacles.append({
                                'center_m': center_m,
                                'poly_px': poly_px
                            })
                            
        # 恢复环境内部状态
        if hasattr(self.env.unwrapped, '_update_dist_map'):
             self.env.unwrapped._update_dist_map()
        if hasattr(self.env.unwrapped, '_get_obs'):
            obs = self.env.unwrapped._get_obs()
            
        return obs, info

    def step(self, action):
        # 1. 先执行底层的动力学推演
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 2. 获取当前车辆物理坐标
        car_pos = self.env.unwrapped.state[:2]
        map_changed = False
        remaining_obstacles = []
        
        # 3. 检查是否有隐藏障碍物进入了触发距离
        for obs_data in self.pending_obstacles:
            dist_to_car = np.linalg.norm(car_pos - obs_data['center_m'])
            
            if dist_to_car <= self.trigger_dist:
                # 【核心】：距离达到阈值，瞬间将多边形写入底层占据栅格地图
                cv2.fillPoly(self.env.unwrapped.map_obstacle, [obs_data['poly_px']], 1)
                self.placed_obstacles.append(obs_data['center_m']) # 供 Matplotlib 实时画出红叉
                map_changed = True
                if self.verbose:
                    print(f"\n⚠️ [Pop-up] 距离 {dist_to_car:.1f}m，突发障碍物已注入！")
            else:
                # 还没到触发距离，保留在隐藏队列中
                remaining_obstacles.append(obs_data)
                
        self.pending_obstacles = remaining_obstacles
        
        # 4. 如果地图发生了改变，必须立刻刷新观测值（防止智能体瞎开一帧）
        if map_changed:
            if hasattr(self.env.unwrapped, '_update_dist_map'):
                 self.env.unwrapped._update_dist_map()
            if hasattr(self.env.unwrapped, '_get_obs'):
                obs = self.env.unwrapped._get_obs() # 重新获取带新障碍的雷达和图像
                
        return obs, reward, terminated, truncated, info

    def _calc_obstacle_params(self, path_pt_m, dist_map):
        px, py = int(path_pt_m[0] / self.internal_res), int(path_pt_m[1] / self.internal_res)
        h, w = dist_map.shape
        if not (0 <= px < w and 0 <= py < h): return False, None, None
        
        dist_at_path = dist_map[py, px]
        max_allowed_r = dist_at_path - (self.passage_width_px / 2.0)
        if max_allowed_r < self.min_obs_radius_px: return False, None, None

        final_max_px = min(max_allowed_r, self.max_obs_radius_px)
        obs_r_px = np.random.uniform(self.min_obs_radius_px, final_max_px)
        
        shift_dist = np.random.uniform(0, obs_r_px * 0.8)
        shift_angle = np.random.uniform(0, 2 * np.pi)
        center_x = px + shift_dist * np.cos(shift_angle)
        center_y = py + shift_dist * np.sin(shift_angle)

        if not (0 <= int(center_x) < w and 0 <= int(center_y) < h): return False, None, None
        if dist_map[int(center_y), int(center_x)] < obs_r_px: return False, None, None

        center_m = np.array([center_x, center_y]) * self.internal_res
        radius_m = obs_r_px * self.internal_res
        return True, center_m, radius_m

    def _generate_polygon_px(self, center_m, radius_m):
        # 纯数学计算，生成多边形的顶点像素坐标
        num_verts = np.random.randint(5, 9)
        angles = np.linspace(0, 2 * np.pi, num_verts, endpoint=False)
        angles += np.random.uniform(-0.2, 0.2, size=num_verts)
        radii = radius_m * np.random.uniform(0.7, 1.3, size=num_verts)
        
        x = center_m[0] + radii * np.cos(angles)
        y = center_m[1] + radii * np.sin(angles)
        poly_px = (np.column_stack((x, y)) / self.internal_res).astype(np.int32)
        return poly_px

class MovingObstacleWrapper(gym.Wrapper):
    """
    【动态验证专用】沿引导路径非匀速移动的障碍物
    障碍物会沿着全局路径朝向车辆移动，并带有随机的加减速与横向漂移，模拟真实的行人或对向车辆。
    """
    # 【修改 1】：默认数量降低为 5，防止拥堵
    def __init__(self, env, num_obstacles=5, dt=0.1, verbose=False):
        super().__init__(env)
        self.num_obstacles = num_obstacles
        self.dt = dt
        self.verbose = verbose
        
        self.internal_res = self.env.unwrapped.INTERNAL_RESOLUTION
        self.clean_map = None
        
        self.moving_obstacles = []  
        self.placed_obstacles = []  

    def reset(self, **kwargs):
        # 1. 恢复纯净地图并初始化
        if self.clean_map is None:
             self.clean_map = self.env.unwrapped.map_obstacle.copy()
        self.env.unwrapped.map_obstacle = self.clean_map.copy()
        self.moving_obstacles = []
        self.placed_obstacles = []
        
        # 2. 调用原始 Reset
        ret = self.env.reset(**kwargs)
        obs, info = ret if isinstance(ret, tuple) else (ret, {})
        
        # 3. 初始化动态障碍物
        if hasattr(self.env.unwrapped, 'total_poly_len') and self.env.unwrapped.poly_points_np is not None:
            total_len = self.env.unwrapped.total_poly_len
            
            for _ in range(self.num_obstacles):
                # 【修改 4】：初始生成时，也严格避开终点前 20 米
                spawn_s = np.random.uniform(10.0, max(11.0, total_len - 20.0))
                self.moving_obstacles.append({
                    'active': True,  # 新增活跃标记，用于终点保护销毁
                    's': spawn_s,
                    # 【修改 2】：降低基础速度至 0.5 ~ 1.5 m/s
                    'v': np.random.uniform(0.5, 1.5),
                    'base_v': np.random.uniform(0.5, 1.5),
                    'phase': np.random.uniform(0, 2*np.pi),
                    'lateral_offset': np.random.uniform(-1.5, 1.5), 
                    'radius_px': max(1, int(np.random.uniform(0.4, 1.0) / self.internal_res))
                })
                
        # 强制更新一次地图
        self._update_and_draw_obstacles(step_count=0)
        
        if hasattr(self.env.unwrapped, '_update_dist_map'):
             self.env.unwrapped._update_dist_map()
        if hasattr(self.env.unwrapped, '_get_obs'):
            obs = self.env.unwrapped._get_obs()
            
        return obs, info

    def step(self, action):
        step_count = getattr(self.env.unwrapped, 'current_step', 0)
        self._update_and_draw_obstacles(step_count)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def _update_and_draw_obstacles(self, step_count):
        self.env.unwrapped.map_obstacle = self.clean_map.copy()
        self.placed_obstacles = []
        
        # 获取车辆当前进度
        car_s = getattr(self.env.unwrapped, 'virtual_s', 0.0)
        total_len = getattr(self.env.unwrapped, 'total_poly_len', 100.0)
        
        for obs_data in self.moving_obstacles:
            # 如果该障碍物已经被销毁（处于终点保护区），则跳过不处理
            if not obs_data['active']:
                continue

            time_sec = step_count * self.dt
            # 【修改 3】：加减速波动振幅削弱到 0.5
            speed_fluctuation = 0.5 * np.sin(obs_data['phase'] + time_sec * 2.0) 
            random_jerk = np.random.uniform(-0.1, 0.1)
            
            obs_data['v'] = max(0.0, obs_data['base_v'] + speed_fluctuation + random_jerk)
            obs_data['s'] -= obs_data['v'] * self.dt
            
            # =======================================================
            # 基于相对距离的动态回收机制 (Dynamic Recycling)
            # =======================================================
            if obs_data['s'] < car_s - 3.0 or obs_data['s'] < 0:
                spawn_s = car_s + np.random.uniform(15.0, 35.0)
                
                # 【修改 4核心】：终点保护机制！
                # 如果打算重生的地方已经离终点不足 20 米，直接销毁该障碍物
                if spawn_s > total_len - 20.0:
                    obs_data['active'] = False
                    continue
                    
                obs_data['s'] = spawn_s
                obs_data['lateral_offset'] = np.random.uniform(-1.5, 1.5)
                obs_data['base_v'] = np.random.uniform(0.5, 1.5)
                obs_data['radius_px'] = max(1, int(np.random.uniform(0.4, 1.0) / self.internal_res))
                
            # === 坐标计算 ===
            pt_on_path = self.env.unwrapped._get_poly_point_at_s(obs_data['s'])
            s_next = min(obs_data['s'] + 1.0, total_len)
            pt_next = self.env.unwrapped._get_poly_point_at_s(s_next)
            
            direction = pt_next - pt_on_path
            norm = np.linalg.norm(direction)
            if norm > 0.001:
                direction /= norm
                normal_vec = np.array([-direction[1], direction[0]])
            else:
                normal_vec = np.array([0.0, 1.0])
                
            center_m = pt_on_path + normal_vec * obs_data['lateral_offset']
            
            # === 写入底层地图 ===
            px, py = int(center_m[0] / self.internal_res), int(center_m[1] / self.internal_res)
            h, w = self.env.unwrapped.map_obstacle.shape
            
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(self.env.unwrapped.map_obstacle, (px, py), obs_data['radius_px'], 1, -1)
                
                self.placed_obstacles.append({
                    'center': center_m, 
                    'radius': obs_data['radius_px'] * self.internal_res
                })
                
        if hasattr(self.env.unwrapped, '_update_dist_map'):
             self.env.unwrapped._update_dist_map()