import numpy as np
import cv2
import os
import sys
import math

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.planner.bfs_path import find_random_path_bfs


def get_guide_point_at_s(target_s, path_points, cumulative_dists, total_len):
    """根据里程 s 获取路径上的坐标和切线角。"""
    target_s = np.clip(target_s, 0.0, total_len)

    idx = np.searchsorted(cumulative_dists, target_s) - 1
    idx = max(0, min(idx, len(path_points) - 2))

    s_start = cumulative_dists[idx]
    s_end = cumulative_dists[idx + 1]
    seg_len = s_end - s_start

    p_start = np.array(path_points[idx])
    p_end = np.array(path_points[idx + 1])

    if seg_len < 1e-6:
        pos = p_start
    else:
        ratio = (target_s - s_start) / seg_len
        pos = p_start + (p_end - p_start) * ratio

    delta = p_end - p_start
    angle = math.atan2(delta[1], delta[0])
    return pos, angle


def simulate_guided_motion(current_s, path_points, cumulative_dists, velocity, sample_time, num_steps):
    """基于路径里程 s 生成时域内参考位置与朝向。"""
    positions = []
    angles = []
    total_len = cumulative_dists[-1]

    for k in range(num_steps + 1):
        future_s = current_s + velocity * (k * sample_time)
        if future_s > total_len:
            future_s = total_len

        pos, ang = get_guide_point_at_s(future_s, path_points, cumulative_dists, total_len)
        positions.append(pos)
        angles.append(ang)

    return positions, angles

class MPCEnv:
    def __init__(self, map_obstacle, max_steps=1000, input_resolution=1.0, path_mode="train", planner_mode="constrained"):
        self.path_mode = path_mode
        self.planner_mode = planner_mode
        self.lidar_path_shortcut = None
        self.input_resolution = float(input_resolution)
        self.max_steps = max_steps
        self.dt = 0.1

        self.PROJ_WINDOW_SIZE = 10
        self.PROJ_MAX_JUMP = 5.0
        self.OBS_SAFE_MARGIN = 4.0
        self.GEO_NORM_SCALE = 25.0
        self.COL_CHECK_LEN = 20.0
        self.LIDAR_MAX_DIST = 25.0

        self.original_map = map_obstacle
        orig_h, orig_w = map_obstacle.shape
        self.INTERNAL_RESOLUTION = self.input_resolution
        
        self.internal_w = orig_w
        self.internal_h = orig_h
        self.map_obstacle = self.original_map.astype(np.float32)

        self.map_width_m = self.internal_w * self.INTERNAL_RESOLUTION
        self.map_height_m = self.internal_h * self.INTERNAL_RESOLUTION

        self.nav_graph = None
        self.cells = None
        self.original_poly_points_np = None
        self.global_path_cids = None

        self.lidar_num_rays = 9
        self.lidar_fov = np.pi

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
        self.cached_dist_map = None

        # mpc_env setup
        import os, json
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)['vehicle_parameters']
        
        self.veh_wheelbase = cfg['wheelbase_m']
        self.veh_width = cfg['car_width_m']
        self.veh_length = cfg['car_length_m']
        self.veh_overhang_front = cfg['overhang_front_m']
        self.veh_overhang_rear = self.veh_length - self.veh_wheelbase - self.veh_overhang_front
        
        self.veh_circle_radius = np.hypot(self.veh_wheelbase / 2, self.veh_width / 2)

        if planner_mode == "constrained":
            from src.planner.planner import Planner
        elif planner_mode == "unconstrained":
            from src.planner.planner_unconstrained import Planner
        else:
            raise ValueError(f"Unsupported planner_mode: {planner_mode}")
        self.planner = Planner(sample_time=self.dt, horizon_steps=20, veh_wheelbase=self.veh_wheelbase)
        self.planner.set_bounds(
            pos_ub=np.array([self.map_width_m, self.map_height_m]),
            v_min=-10.0, v_max=10.0,
            steer_min=-0.6, steer_max=0.6,
            accel_min=-5.0, accel_max=5.0,
            steer_rate_min=-0.6, steer_rate_max=0.6
        )

        self.px_to_meter_scale = self.INTERNAL_RESOLUTION
        self.mpc_guess = None
        self.last_ctrl = np.zeros(2)
        self.mpc_guided_ref = None
        self.mpc_guided_phi = None
        self.ep_action_history = []

    def reset(self, seed=None, options=None):
        self.history_trajectory = []
        self.current_step = 0
        self.last_heading = 0.0

        self._generate_random_path()

        if self.poly_points_np is not None and len(self.poly_points_np) > 0:
            self.start_pos = self.poly_points_np[0]
            self.target_pos = self.poly_points_np[-1]
            init_heading = self._calculate_path_heading(0.0)
            self.state = np.array([self.start_pos[0], self.start_pos[1], init_heading, 0.0, 0.0])
            self.last_heading = init_heading
            self.original_poly_points_np = self.poly_points_np.copy()
        else:
            self.state = np.zeros(5)

        self.virtual_s = 0.0
        self.last_s = 0.0
        self.history_trajectory.append(self.state[:2])

        X_guess = np.tile(self.state.reshape(-1, 1), (1, self.planner.horizon_steps + 1))
        U_guess = np.zeros((self.planner.model.nu, self.planner.horizon_steps))
        self.mpc_guess = (X_guess, U_guess)

        self.last_ctrl = np.zeros(2)
        self.mpc_guided_ref = None
        self.mpc_guided_phi = None
        self.ep_action_history = []

        return {}

    def step(self, action):
        v_ref = float(action[0])
        w_obs = float(action[1])
        R_influence = float(action[2])
        w_lat = float(action[3])
        w_rate_steer = float(action[4])

        self.ep_action_history.append([v_ref, w_obs, R_influence, w_lat, w_rate_steer])

        w_lon        = 1.0
        w_heading    = 1.0
        w_vel        = 1.0
        w_accel      = 1.0
        w_steer      = 1.0
        speed_rew_c  = 0.0
        virt_lever   = 1.0
        w_slack      = 1.0

        dt = self.planner.sample_time

        self.virtual_s = self._calculate_progress_on_poly(self.state[:2])
        self.virtual_s += v_ref * dt
        if self.virtual_s > self.total_poly_len:
            self.virtual_s = self.total_poly_len

        pos_guided, phi_guided = simulate_guided_motion(
            current_s=self.virtual_s,
            path_points=self.poly_points_np.tolist(),
            cumulative_dists=self.poly_cumulative_dists,
            velocity=v_ref,
            sample_time=self.planner.sample_time,
            num_steps=self.planner.horizon_steps,
        )
        guided_points = {"posi": pos_guided, "angle": phi_guided}

        next_state, next_ctrl, X_next, U_next, info = self.planner.step_once(
            self.state, self.last_ctrl, self.mpc_guess,
            self.map_obstacle,
            guided_points=guided_points,
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

        self.mpc_guided_ref = np.array(guided_points['posi'])
        self.mpc_guided_phi = np.array(guided_points['angle'])
        if "mpc_corridors" in info:
            self.mpc_corridors = info["mpc_corridors"]

        current_pos = next_state[:2]
        current_heading = next_state[2]
        current_vel = next_state[3]

        current_ctrl = next_ctrl
        self.history_trajectory.append(current_pos)
        
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

        info["is_success"] = is_success
        info["u_safe"] = current_ctrl
        info["u_nom"] = current_ctrl

        self.state = next_state
        self.last_ctrl = current_ctrl
        self.mpc_guess = (X_next, U_next)
        self.last_s = self.virtual_s
        self.current_step += 1

        return terminated, truncated, info

    def _calculate_progress_on_poly(self, pos):
        if self.poly_points_np is None:
            return 0.0
        points = self.poly_points_np
        seg_lens = self.poly_segment_lens
        cum_dists = self.poly_cumulative_dists
        prev_s = getattr(self, "last_s", None)

        num_segments = len(points) - 1
        if num_segments < 1: return 0.0

        window_size = self.PROJ_WINDOW_SIZE
        max_jump = self.PROJ_MAX_JUMP

        if prev_s is None:
            search_start, search_end = 0, num_segments
        else:
            idx_approx = np.clip(np.searchsorted(cum_dists, prev_s) - 1, 0, num_segments - 1)
            search_start = max(0, idx_approx - window_size)
            search_end = min(num_segments, idx_approx + window_size + 1)

        p_sub = points[search_start : search_end + 1]
        lens_sub = seg_lens[search_start : search_end]
        cum_dists_start = cum_dists[search_start : search_end]
        
        if len(lens_sub) == 0: return prev_s if prev_s is not None else 0.0

        p_a = p_sub[:-1]
        p_b = p_sub[1:]
        v_seg = p_b - p_a
        v_pt = pos - p_a
        
        lens_sub_sq = lens_sub ** 2
        lens_sub_sq = np.where(lens_sub_sq < 1e-8, 1e-8, lens_sub_sq)
        dot_prod = np.sum(v_pt * v_seg, axis=1)
        t = np.clip(dot_prod / lens_sub_sq, 0.0, 1.0)

        projections = p_a + v_seg * t[:, np.newaxis]
        dists_sq = np.sum((pos - projections)**2, axis=1)
        best_local_idx = np.argmin(dists_sq)
        best_s = cum_dists_start[best_local_idx] + t[best_local_idx] * lens_sub[best_local_idx]

        if prev_s is not None:
            diff = best_s - prev_s
            if diff > max_jump: best_s = prev_s + max_jump
            elif diff < -max_jump: best_s = prev_s - max_jump * 0.1
                
        return best_s

    def _generate_random_path(self, min_dist_m=20.0):
        min_dist_grid = min_dist_m / self.INTERNAL_RESOLUTION
        margin = max(1, int(2.0 / self.INTERNAL_RESOLUTION))
        
        result = find_random_path_bfs(self.map_obstacle, min_dist_grid, margin=margin)
        
        if result is not None:
            s_grid, t_grid, path_grid = result
            self.start_pos = np.array(s_grid) * self.INTERNAL_RESOLUTION
            self.target_pos = np.array(t_grid) * self.INTERNAL_RESOLUTION
            self.global_path_cids = None
            
            raw_path_meters = np.array(path_grid) * self.INTERNAL_RESOLUTION
            
            if len(raw_path_meters) > 3:
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
        if self.poly_points_np is None: 
            return np.zeros(2)
        s_val = np.clip(s_val, 0.0, max(self.total_poly_len, 1e-6))
        idx = np.clip(np.searchsorted(self.poly_cumulative_dists, s_val) - 1, 0, len(self.poly_points_np) - 2)
        s_start, s_end = self.poly_cumulative_dists[idx], self.poly_cumulative_dists[idx+1]
        ratio = (s_val - s_start) / (s_end - s_start) if s_end - s_start > 1e-6 else 0.0
        return self.poly_points_np[idx] + (self.poly_points_np[idx+1] - self.poly_points_np[idx]) * ratio

    def _calculate_path_heading(self, s_val, delta=0.5):
        if self.total_poly_len <= 0: 
            return 0.0
        p_next = self._get_poly_point_at_s(min(self.total_poly_len, s_val + delta))
        p_prev = self._get_poly_point_at_s(max(0, s_val - delta))
        vec = p_next - p_prev
        return np.arctan2(vec[1], vec[0]) if np.linalg.norm(vec) > 1e-6 else 0.0

    def _check_collision(self, pos_meters, heading):
        L = getattr(self, "veh_wheelbase", 1.4)
        W = getattr(self, "veh_width", 1.0)
        front_overhang = getattr(self, "veh_overhang_front", 0.5)
        rear_overhang = getattr(self, "veh_overhang_rear", 0.5)
        
        fl = (L + front_overhang, W / 2)
        fr = (L + front_overhang, -W / 2)
        rr = (-rear_overhang, -W / 2)
        rl = (-rear_overhang, W / 2)
        
        import numpy as np
        import cv2
        pts = np.array([fl, fr, rr, rl])
        
        c, s = np.cos(heading), np.sin(heading)
        rot = np.array([[c, -s], [s, c]])
        pts_rot = np.dot(pts, rot.T)
        
        pts_rot[:, 0] += pos_meters[0]
        pts_rot[:, 1] += pos_meters[1]
        
        pts_px = (pts_rot / self.INTERNAL_RESOLUTION).astype(np.int32)
        
        x_min = np.clip(np.min(pts_px[:, 0]), 0, self.internal_w - 1)
        x_max = np.clip(np.max(pts_px[:, 0]), 0, self.internal_w - 1)
        y_min = np.clip(np.min(pts_px[:, 1]), 0, self.internal_h - 1)
        y_max = np.clip(np.max(pts_px[:, 1]), 0, self.internal_h - 1)
        
        if np.min(pts_px[:, 0]) < 0 or np.max(pts_px[:, 0]) >= self.internal_w or \
           np.min(pts_px[:, 1]) < 0 or np.max(pts_px[:, 1]) >= self.internal_h:
            return True
            
        local_w = int(x_max - x_min + 1)
        local_h = int(y_max - y_min + 1)
        local_mask = np.zeros((local_h, local_w), dtype=np.uint8)
        
        local_pts = pts_px - np.array([x_min, y_min])
        cv2.fillConvexPoly(local_mask, local_pts, 1)
        
        map_crop = self.map_obstacle[int(y_min):int(y_max)+1, int(x_min):int(x_max)+1]
        if local_mask.shape != map_crop.shape:
            return True
            
        return bool(np.any((local_mask == 1) & (map_crop == 1)))
