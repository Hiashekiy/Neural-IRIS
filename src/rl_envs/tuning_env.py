import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys
import cv2

# --- 路径处理 ---
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from .common_env import CommonEnv
from src.planner.planner import Planner

class TuningEnv(CommonEnv):
    """
    【Stage 2: MPC调参环境 TuningEnv】
    核心职责：训练 Agent 根据路况动态调整 5维 MPC 核心权重参数。
    """
    def __init__(self, map_obstacle, **kwargs):
        super().__init__(map_obstacle, **kwargs)
        
        self.veh_wheelbase = 1.0  
        self.veh_width = 0.5      
        self.veh_circle_radius = np.hypot(self.veh_wheelbase / 2, self.veh_width / 2) 

        # ==========================================
        # 1. 动作空间 (5 维核心参数)
        # ==========================================
        # [0] v_ref: 目标车速
        # [1] w_obs: 障碍物斥力权重
        # [2] R_influence: 障碍物势场影响半径
        # [3] w_lat: 横向误差权重
        # [4] w_rate_steer: 转向角速度控制权重 (平顺度)
        
        # 保存物理边界用于 step 中的内部映射 (与动作空间对应)
        self.phys_low  = np.array([1.0,  0.1,  0.5,  0.1,  0.1], dtype=np.float32)
        self.phys_high = np.array([6.0, 20.0,  2.0,  3.0, 20.0], dtype=np.float32)
        
        # 暴露给强化学习算法的动作空间永远严格限制在 [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,), dtype=np.float32
        )
        
        self.planner = Planner(sample_time=self.dt, horizon_steps=10, veh_wheelbase=self.veh_wheelbase)
        self.planner.set_bounds(
            pos_ub=np.array([self.map_width_m, self.map_height_m]),
            v_min=-10.0, v_max=10.0,
            steer_min=-0.6, steer_max=0.6,    
            accel_min=-5.0, accel_max=5.0,     
            steer_rate_min=-0.6, steer_rate_max=0.6 
        )
        
        self.px_to_meter_scale = self.scale * self.INTERNAL_RESOLUTION
        self.cached_dist_map = None 
        
        self.mpc_guess = None
        self.last_ctrl = np.zeros(2)
        self.last_action = np.zeros(self.action_space.shape)
        self.mpc_guided_ref = None
        self.mpc_guided_phi = None
        
        self.ep_reward_info = {}
        self.ep_action_history = []

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # 初始化 5D 状态
        x_init, y_init, vx_init, vy_init = self.state[0], self.state[1], self.state[2], self.state[3]
        v_init = np.hypot(vx_init, vy_init)
        
        if v_init < 1e-3 and hasattr(self, 'poly_points_np') and len(self.poly_points_np) >= 2:
            dx = self.poly_points_np[1][0] - self.poly_points_np[0][0]
            dy = self.poly_points_np[1][1] - self.poly_points_np[0][1]
            theta_init = np.arctan2(dy, dx)
        else:
            theta_init = np.arctan2(vy_init, vx_init)
            
        psi_init = 0.0 
        self.state = np.array([x_init, y_init, theta_init, v_init, psi_init])
        
        # 重置 MPC Guess
        X_guess = np.tile(self.state.reshape(-1, 1), (1, self.planner.horizon_steps + 1))
        U_guess = np.zeros((self.planner.model.nu, self.planner.horizon_steps))
        self.mpc_guess = (X_guess, U_guess)
        
        self.last_ctrl = np.zeros(2)
        self.last_action = np.zeros(self.action_space.shape)
        self.mpc_guided_ref = None
        self.mpc_guided_phi = None
        self.ep_action_history = []
        
        # 重构记录字典
        self.ep_reward_info = {
            'rew_collision': 0.0, 'rew_success': 0.0, 'rew_truncated': 0.0,
            'rew_progress': 0.0, 'rew_vel': 0.0, 'rew_safety': 0.0,
            'rew_act_smooth': 0.0, 'rew_ctrl_smooth': 0.0
        }
        
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 2. 原生解码：将 [-1, 1] 映射到实际物理范围 [phys_low, phys_high]
        phys_action = self.phys_low + (action + 1.0) / 2.0 * (self.phys_high - self.phys_low)

        # 解析 5D 物理参数
        v_ref        = float(phys_action[0])
        w_obs        = float(phys_action[1])
        R_influence  = float(phys_action[2])
        w_lat        = float(phys_action[3])
        w_rate_steer = float(phys_action[4])
        
        # <<< [新增] 将实际应用的参数保存下来
        self.ep_action_history.append([v_ref, w_obs, R_influence, w_lat, w_rate_steer])

        # 固定的底层权重 (非强化学习训练的参数固定)
        w_lon        = 1.0
        w_heading    = 1.0
        w_vel        = 1.0
        w_accel      = 1.0
        w_steer      = 1.0
        speed_rew_c  = 0.0
        virt_lever   = 1.0 
        w_slack      = 1.0

        dt = self.planner.sample_time

        # 更新 Virtual S
        self.virtual_s = self._calculate_progress_on_poly(self.state[:2])
        self.virtual_s += v_ref * dt 
        if self.virtual_s > self.total_poly_len: 
            self.virtual_s = self.total_poly_len

        # 触发 MPC 规划 (已删除被硬编码截断的测试代码，恢复完整传参)
        next_state, next_ctrl, X_next, U_next, pos_guided, phi_guided, info = self.planner.step_once(
            self.state, self.last_ctrl, self.mpc_guess,
            self.map_obstacle, 
            path_points=self.poly_points_np.tolist(), 
            guide_cumulative_dists=self.poly_cumulative_dists, 
            current_s=self.virtual_s,                   
            target_velocity=v_ref,                      
            weight_lat_scale=w_lat,
            weight_lon_scale=w_lon,
            weight_heading_scale=w_heading,
            weight_vel_scale=w_vel,
            weight_obs_scale=w_obs,
            weight_accel_scale=w_accel,
            weight_steer_scale=w_steer,
            weight_rate_steer_scale=w_rate_steer,
            speed_reward_c_scale=speed_rew_c,
            weight_slack_scale=w_slack,
            R_influence=R_influence, 
            map_resolution=self.INTERNAL_RESOLUTION,
            debug_render=False 
        )

        self.mpc_guided_ref = np.array(pos_guided)
        self.mpc_guided_phi = np.array(phi_guided)
        if 'mpc_corridors' in info:
            self.mpc_corridors = info['mpc_corridors']

        # 状态更新
        current_pos = next_state[:2]
        current_heading = next_state[2]
        current_vel = next_state[3]    
        current_vel_norm = abs(current_vel) 
        
        current_ctrl = next_ctrl
        self.history_trajectory.append(current_pos)
        current_s_actual = self._calculate_progress_on_poly(current_pos)

        # 终止条件判定
        dist_to_goal = np.linalg.norm(current_pos - self.target_pos)
        collided = self._check_collision(current_pos, current_heading)
        
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

        # 计算奖励 (更新后的奖励函数)
        reward, step_rew_info = self._calculate_reward(
            current_pos=current_pos,
            current_heading=current_heading,
            current_vel_norm=current_vel_norm,
            current_ctrl=current_ctrl,
            current_s_actual=current_s_actual,
            action=action,
            collided=collided,
            is_success=is_success,
            truncated=truncated
        )
        
        for k, v in step_rew_info.items():
            self.ep_reward_info[k] += v

        # 更新内部迭代状态
        self.state = next_state
        self.mpc_guess = (X_next, U_next) 
        self.last_s = current_s_actual 
        self.last_ctrl = current_ctrl
        self.last_action = action.copy()
        self.current_step += 1
        
        final_obs = self._get_obs()
        
        debug_info = {}
        if (terminated or truncated) and not is_success:
            debug_info = {
                'debug_traj': self.history_trajectory,     
                'debug_guide': self.poly_points_np,        
                'debug_start': self.start_pos,             
                'debug_target': self.target_pos,           
                'debug_step': self.current_step
            }
            
        if terminated or truncated:
            info['ep_reward_info'] = self.ep_reward_info.copy()

            # <<< [新增] 在 episode 结束时计算本回合动作的平均值，并随 info 抛出
            if len(self.ep_action_history) > 0:
                mean_acts = np.mean(self.ep_action_history, axis=0)
                info['ep_action_mean'] = {
                    'v_ref': mean_acts[0],
                    'w_obs': mean_acts[1],
                    'R_influence': mean_acts[2],
                    'w_lat': mean_acts[3],
                    'w_rate_steer': mean_acts[4]
                }
        
        return final_obs, reward, terminated, truncated, {'is_success': is_success, **debug_info, **info}

    def _calculate_reward(self, current_pos, current_heading, current_vel_norm, current_ctrl, 
                          current_s_actual, action, collided, is_success, truncated):
        """重新设计的奖励系统：修复惩罚倒挂，鼓励高速穿插"""
        total_reward = 0.0
        rew_info = {
            'rew_collision': 0.0, 'rew_success': 0.0, 'rew_truncated': 0.0,
            'rew_progress': 0.0, 'rew_vel': 0.0, 'rew_safety': 0.0,
            'rew_act_smooth': 0.0, 'rew_ctrl_smooth': 0.0
        }

        # 1. 稀疏终止奖励 (彻底拉开档次)
        if collided: 
            rew_info['rew_collision'] = -200.0  # 撞车是最大恶极，扣200
        elif is_success: 
            rew_info['rew_success'] = 300.0     # 成功给大奖
        elif truncated: 
            rew_info['rew_truncated'] = -50.0   # 超时只扣30。绝对不能比撞车高！

        if not collided and not is_success:
            # 2. 进度与速度奖励 (加大奖励，逼迫提速)
            s_improvement = current_s_actual - self.last_s
            rew_info['rew_progress'] = s_improvement * 1.0  # 奖励翻倍
            rew_info['rew_vel'] = current_vel_norm * 0.1   

            # 3. 危险避让安全惩罚 (废除指数爆炸，改为线性)
            min_dist_m = self._get_min_dist_meters(current_pos, current_heading)
            SAFE_DIST_THRESHOLD = 0.8
            if min_dist_m < SAFE_DIST_THRESHOLD:
                diff = SAFE_DIST_THRESHOLD - min_dist_m
                # 线性惩罚：极限贴墙(0.1米)时单步只扣 -3.5。
                # 哪怕贴墙走 20 步也才 -70，依然比撞车(-200)划算，鼓励它勇敢钻窄缝。
                rew_info['rew_safety'] = -5.0 * diff 

            # 4. 平滑惩罚 (维持原样)
            if self.current_step > 0:
                action_diff = np.linalg.norm(action - self.last_action)
                rew_info['rew_act_smooth'] = -(action_diff * 0.05) 

            steer_rate = abs(current_ctrl[1])
            rew_info['rew_ctrl_smooth'] = -(steer_rate * 0.05) 

        for val in rew_info.values():
            total_reward += val
            
        return total_reward, rew_info

    # ==========================================
    # 辅助工具函数
    # ==========================================
    def _get_vehicle_circles(self, pos_meters, heading):
        rear_x, rear_y = pos_meters[0], pos_meters[1]
        front_x = rear_x + self.veh_wheelbase * np.cos(heading)
        front_y = rear_y + self.veh_wheelbase * np.sin(heading)
        return [(rear_x, rear_y), (front_x, front_y)]

    def _get_min_dist_meters(self, pos_meters, heading):
        if not hasattr(self, 'cached_dist_map') or self.cached_dist_map is None:
            binary_map = (1 - self.map_obstacle).astype(np.uint8)
            dist_px = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)
            self.cached_dist_map = dist_px * self.INTERNAL_RESOLUTION

        h, w = self.cached_dist_map.shape
        centers = self._get_vehicle_circles(pos_meters, heading)
        
        min_dist = float('inf')
        for cx, cy in centers:
            x_grid = int(cx / self.INTERNAL_RESOLUTION)
            y_grid = int(cy / self.INTERNAL_RESOLUTION)
            
            if x_grid < 0 or x_grid >= w or y_grid < 0 or y_grid >= h:
                dist_to_obs = 0.0 
            else:
                dist_to_obs = self.cached_dist_map[y_grid, x_grid]
                
            if dist_to_obs < min_dist:
                min_dist = dist_to_obs
                
        actual_clearance = min_dist - self.veh_circle_radius
        return max(0.0, actual_clearance)

    def _check_collision(self, pos_meters, heading):
        clearance = self._get_min_dist_meters(pos_meters, heading)
        return clearance <= 0.01