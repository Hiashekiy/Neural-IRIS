# -*- coding: utf-8 -*-
"""
Stage 2 MPC 纯基线(Baseline) 测试
不加载任何强化学习模型，全程使用固定的默认缩放参数进行 MPC 控制。
包含 MPC 预测轨迹可视化与 CBF 介入状态监控。

用法:
    python scripts/test/test_default_mpc.py
    python scripts/test/test_default_mpc.py --map map1.png --episodes 5 --v-ref 4.0
    python scripts/test/test_default_mpc.py --no-render
"""
import os
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import argparse
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rl_envs.tuning_env import TuningEnv
from src.utils.obstacle_wrapper import StaticObstacleWrapper


def load_map(map_path: str) -> np.ndarray:
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"找不到地图 {map_path}")
    
    if map_path.endswith('.map'):
        with open(map_path, 'r') as f:
            lines = f.readlines()
        height, width, map_data_start = 0, 0, 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('height'): height = int(line.split()[1])
            elif line.startswith('width'): width = int(line.split()[1])
            elif line.startswith('map'):
                map_data_start = i + 1
                break
        grid = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            row_str = lines[map_data_start + i].strip()
            for j in range(width):
                if row_str[j] in ['.', 'G', 'S']: grid[i, j] = 0
                else: grid[i, j] = 1
        map_obstacle = grid
    else:
        maze = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        maze = cv2.resize(maze, None, fx=5.0, fy=5.0, interpolation=cv2.INTER_NEAREST)
        _, map_obstacle = cv2.threshold(maze, 50, 1, cv2.THRESH_BINARY)
        
    print(f"[Map] 加载 {os.path.basename(map_path)}  shape={map_obstacle.shape}")
    return map_obstacle


def make_test_env(map_obstacle: np.ndarray,
                  use_obstacles: bool = False,
                  density: float = 0.5):
    env = TuningEnv(map_obstacle=map_obstacle, input_resolution=0.25, max_steps=1000, path_mode='train')
    if use_obstacles:
        env = StaticObstacleWrapper(
            env,
            density=density,
            min_obs_radius=0.5,
            max_obs_radius=2.0,
            passage_width=3.5,
        )
        print(f"[ObstacleWrapper] 已启用 density={density}")
    return env


def render_step(render_objs, base_env, action_physical, traj_x, traj_y, velocities, status_text, infos=None):
    """ 每步调用，展示 地图+轨迹 | CNN输入 | 动作曲线 """
    res = base_env.INTERNAL_RESOLUTION
    
    if len(traj_x) > 1:
        offsets = np.c_[np.array(traj_x) / res, np.array(traj_y) / res]
        render_objs['traj_scatter'].set_offsets(offsets)
        render_objs['traj_scatter'].set_array(np.array(velocities))

    state = base_env.state
    cx, cy = state[0] / res, state[1] / res
    heading = state[2]
    
    L = 1.0 / res
    W = 0.5 / res
    rear_overhang = 0.5 / res
    front_overhang = 0.5 / res
    
    fl = (L + front_overhang, W / 2)
    fr = (L + front_overhang, -W / 2)
    rr = (-rear_overhang, -W / 2)
    rl = (-rear_overhang, W / 2)
    pts = np.array([fl, fr, rr, rl])
    
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    pts_rot = pts @ rot.T
    pts_rot[:, 0] += cx
    pts_rot[:, 1] += cy
    
    render_objs['car_patch'].set_xy(pts_rot)

    al = 4.0 / res
    render_objs['car_arrow'].xy = (cx + al * np.cos(heading), cy + al * np.sin(heading))
    render_objs['car_arrow'].set_position((cx, cy))

    vr = 35.0 / res
    render_objs['ax_m'].set_xlim(cx - vr, cx + vr)
    render_objs['ax_m'].set_ylim(cy + vr, cy - vr)
    render_objs['title_m'].set_text(status_text)

    if base_env.mpc_guess is not None:
        X_pred = base_env.mpc_guess[0] 
        render_objs['pred_line'].set_data(X_pred[0, :] / res, X_pred[1, :] / res)
        
        if hasattr(base_env, 'mpc_corridors') and base_env.mpc_corridors is not None:
            corridors = base_env.mpc_corridors
            poly_patches = render_objs['poly_patches']
            ell_patches = render_objs['ell_patches']
            for k in range(min(len(corridors), len(poly_patches))):
                A = corridors[k]['A']
                b = corridors[k]['b']
                from scipy.spatial import HalfspaceIntersection, ConvexHull
                import scipy.linalg
                try:
                    c_pt = corridors[k]['c_ell'] if corridors[k]['c_ell'] is not None else X_pred[:2, k]
                    
                    # Ensure we filter out unused A and b rows (avoiding singular matrix)
                    valid_mask = np.any(A != 0, axis=1)
                    A_valid = A[valid_mask]
                    b_valid = b[valid_mask]
                    
                    if len(A_valid) >= 3:
                        big_bounds = np.array([
                            [1, 0, c_pt[0] + 8.0],
                            [-1, 0, -(c_pt[0] - 8.0)],
                            [0, 1, c_pt[1] + 8.0],
                            [0, -1, -(c_pt[1] - 8.0)]
                        ])
    
                        A_with_bounds = np.vstack((A_valid, big_bounds[:, :2]))
                        b_with_bounds = np.concatenate((b_valid, big_bounds[:, 2]))
                        hs = HalfspaceIntersection(np.hstack((A_with_bounds, -b_with_bounds.reshape(-1, 1))), c_pt)
                        hull = ConvexHull(hs.intersections)
                        vertices = hs.intersections[hull.vertices]
                        
                        # Make the polygon closed by duplicating the first point to the end
                        vertices = np.vstack([vertices, vertices[0]])
                        
                        poly_patches[k].set_xy(vertices / res)
                        poly_patches[k].set_visible(True)
                    else:
                        poly_patches[k].set_visible(False)
                except Exception as e:
                    poly_patches[k].set_visible(False)
                    

                try:
                    c_ell = corridors[k]['c_ell']
                    P_inv = corridors[k]['P_inv']  
                    if c_ell is not None and P_inv is not None:
                        theta_vals = np.linspace(0, 2*np.pi, 30)
                        ell_pts = np.vstack([np.cos(theta_vals), np.sin(theta_vals)])
                        ell_world = c_ell[:, None] + P_inv @ ell_pts
                        ell_patches[k].set_data(ell_world[0, :] / res, ell_world[1, :] / res)
                        ell_patches[k].set_visible(True)
                    else:
                        ell_patches[k].set_visible(False)
                except Exception:
                    ell_patches[k].set_visible(False)


    if infos is not None and 'u_nom' in infos and 'u_safe' in infos:
        u_nom = infos['u_nom']
        u_safe = infos['u_safe']
        
        diff = np.linalg.norm(u_nom - u_safe)
        if diff > 1e-3:
            text_str = f"CBF ACTIVE!\nNominal: a={u_nom[0]:.2f}, w={u_nom[1]:.2f}\nSafe Ctrl: a={u_safe[0]:.2f}, w={u_safe[1]:.2f}"
            render_objs['cbf_text'].set_text(text_str)
            render_objs['cbf_text'].set_color('red')
        else:
            text_str = f"CBF Inactive\nCtrl: a={u_safe[0]:.2f}, w={u_safe[1]:.2f}"
            render_objs['cbf_text'].set_text(text_str)
            render_objs['cbf_text'].set_color('green')

    cnn_img = base_env._get_local_map_cnn(state[:2], heading)[0]
    render_objs['cnn_img'].set_data(cnn_img)

    if action_physical is not None:
        for bar, text, val in zip(render_objs['bars'], render_objs['bar_texts'], action_physical):
            bar.set_height(val)
            text.set_text(f'{val:.1f}')
            text.set_position((bar.get_x() + bar.get_width() / 2, val + 0.05))

    render_objs['fig'].canvas.draw_idle()
    render_objs['fig'].canvas.flush_events()
    plt.pause(0.001)


def run_episode(env, base_env, fixed_action: np.ndarray, render: bool, ep_idx: int, save_video: bool = False):
    obs, info = env.reset()

    episode_map  = base_env.map_obstacle.copy()
    episode_path = base_env.poly_points_np.copy() if base_env.poly_points_np is not None else None

    traj_x, traj_y, velocities = [], [], []
    done = False
    step = 0
    status = f"EP {ep_idx+1}  Running..."

    video_writer = None
    
    if render and save_video:
        video_dir = os.path.join(project_root, 'logs', 'videos')
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"episode_{ep_idx}.mp4")

    render_objs = {}
    if render:
        plt.ion()
        fig, (ax_m, ax_c, ax_a) = plt.subplots(
            1, 3, figsize=(18, 7),
            gridspec_kw={'width_ratios': [3, 1, 1]})
        plt.tight_layout()

        res = base_env.INTERNAL_RESOLUTION

        ax_m.imshow(1 - base_env.map_obstacle, cmap='gray', origin='upper')
        if base_env.poly_points_np is not None:
            p = base_env.poly_points_np
            ax_m.plot(p[:, 0] / res, p[:, 1] / res, 'g--', linewidth=1.5, alpha=0.6, label='Ref Path')
            sp = base_env.start_pos
            tp = base_env.target_pos
            ax_m.plot(sp[0] / res, sp[1] / res, 'g^', markersize=10, label='Start')
            ax_m.plot(tp[0] / res, tp[1] / res, 'r*', markersize=12, label='Goal')

        traj_scatter = ax_m.scatter([], [], c=[], cmap='jet', vmin=0, vmax=8, s=6, zorder=10)
        
        pred_line, = ax_m.plot([], [], color='m', linestyle='-', linewidth=2, alpha=0.8, label='MPC Pred')

        poly_patches = []
        ell_patches = []
        for _ in range(base_env.planner.horizon_steps):
            poly_patch = Polygon(np.zeros((3, 2)), closed=True, edgecolor='red', facecolor='salmon', zorder=5, alpha=0.3, linewidth=1.5)
            ax_m.add_patch(poly_patch)
            poly_patches.append(poly_patch)
            
            ell_line, = ax_m.plot([], [], color='blue', linestyle='--', linewidth=1.5, alpha=0.7, zorder=6)
            ell_patches.append(ell_line)

        car_patch = Polygon(np.zeros((4, 2)), closed=True, edgecolor='black', facecolor='cyan', zorder=15, alpha=0.8)
        ax_m.add_patch(car_patch)
        car_arrow = ax_m.annotate('', xy=(0,0), xytext=(0,0), arrowprops=dict(arrowstyle='->', color='red', lw=2))
        title_m = ax_m.set_title(status, fontsize=11)
        
        cbf_text = ax_m.text(0.02, 0.98, 'CBF Inactive', transform=ax_m.transAxes, 
                             color='green', fontsize=10, fontweight='bold', va='top', 
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        ax_m.legend(loc='upper right', fontsize=8)

        dummy_cnn = np.zeros((64, 64))
        cnn_img_plot = ax_c.imshow(dummy_cnn, cmap='gray', vmin=0, vmax=255, origin='upper')
        sz = 64
        ax_c.plot(sz / 2, sz * 0.8, 'r^', markersize=10)
        ax_c.set_title("CNN Input (64x64)", fontsize=10)
        ax_c.axis('off')

        labels  = ['v_ref', 'w_lat', 'w_lon', 'w_hdg', 'w_vel']
        colors  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

        bars = ax_a.bar(labels, fixed_action, color=colors, alpha=0.85)
        bar_texts = [ax_a.text(b.get_x() + b.get_width() / 2, val + 0.05, f'{val:.1f}', ha='center', va='bottom', fontsize=8) for b, val in zip(bars, fixed_action)]
        ax_a.set_ylim(0, max(11, fixed_action.max() + 2))
        ax_a.set_title("Fixed MPC Params", fontsize=10)
        ax_a.set_ylabel('Value')
        ax_a.tick_params(axis='x', labelsize=8, rotation=45)

        render_objs = {
            'fig': fig, 'ax_m': ax_m,
            'traj_scatter': traj_scatter, 'car_patch': car_patch, 
            'pred_line': pred_line,       
            'poly_patches': poly_patches, 
            'ell_patches': ell_patches,   
            'cbf_text': cbf_text,       
            'car_arrow': car_arrow, 'title_m': title_m,
            'cnn_img': cnn_img_plot,
            'bars': bars, 'bar_texts': bar_texts
        }

    while not done:
        obs, reward, terminated, truncated, infos = env.step(fixed_action)
        done = terminated or truncated
        step += 1
        if 'time_stats' in infos:
            t_stats = infos['time_stats']
            nn_t = t_stats.get('nn_inference_time', 0.0)
            poly_t = t_stats.get('polygon_generation_time', 0.0)
            solve_t = t_stats.get('solve_time', 0.0)
            mat_t = t_stats.get('matrix_build_time', 0.0)
            tot_t = t_stats.get('total_time', 0.0)
            print(f"[Execution Time Stats - Step {step}]")
            print(f"  -> NN Inference (cumulative): {nn_t*1000:.1f} ms")
            print(f"  -> Polygon Generation (cumulative): {poly_t*1000:.1f} ms")
            print(f"  -> Matrix Build Time (including NN & Poly): {mat_t*1000:.1f} ms")
            print(f"  -> MPC/HPIPM Solve Time: {solve_t*1000:.1f} ms")
            print(f"  => Total Step Planner Time: {tot_t*1000:.1f} ms\n")

        st = base_env.state
        traj_x.append(st[0])
        traj_y.append(st[1])
        velocities.append(float(abs(st[3])))

        if render:
            render_step(render_objs, base_env, fixed_action,
                        traj_x, traj_y, velocities, status, infos)
            
            if save_video:
                frame = capture_frame(render_objs['fig'])
                
                if video_writer is None:
                    h, w, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_path, fourcc, 15, (w, h))
                    print(f"[Video] 开始录制 {video_path} 尺寸: {w}x{h}")
                
                video_writer.write(frame)

    success = bool(infos.get('is_success', False))
    status  = f"EP {ep_idx+1}  {'SUCCESS ✅' if success else 'FAIL ❌'}  steps={step}"
    if render:
        render_step(render_objs, base_env, fixed_action,
                    traj_x, traj_y, velocities, status, infos)
        time.sleep(1.5)
        plt.close(fig)
        plt.ioff()

    avg_speed = float(np.mean(velocities)) if velocities else 0.0
    print(f"  {status}  avg_v={avg_speed:.2f} m/s")

    if video_writer is not None:
        video_writer.release()
        print(f"[Video] 已保存 {video_path}")

    return {
        'success':  success,
        'steps':    step,
        'avg_speed': avg_speed,
        'traj_x':   traj_x,
        'traj_y':   traj_y,
        'velocities': velocities,
        'map':      episode_map,
        'path':     episode_path,
        'start':    base_env.start_pos.copy(),
        'target':   base_env.target_pos.copy(),
    }


def plot_summary(results: list, map_obstacle: np.ndarray, save_path: str, fixed_action: np.ndarray):
    res = 0.5  
    n_ep = len(results)
    success_n = sum(r['success'] for r in results)

    fig, ax_map = plt.subplots(figsize=(10, 8))
    
    ax_map.imshow(1 - map_obstacle, cmap='gray', origin='upper')
    cmap = plt.cm.get_cmap('tab10', max(10, n_ep))
    
    for i, r in enumerate(results):
        color = cmap(i % 10)
        tx = np.array(r['traj_x']) / res
        ty = np.array(r['traj_y']) / res
        style = '-' if r['success'] else '--'
        alpha = 0.9 if r['success'] else 0.45
        ax_map.plot(tx, ty, linestyle=style, color=color, linewidth=1.8,
                    alpha=alpha, label=f"EP{i+1} {'OK' if r['success'] else 'X'}")
        ax_map.plot(r['start'][0]/res,  r['start'][1]/res,  '^', color=color, ms=7)
        ax_map.plot(r['target'][0]/res, r['target'][1]/res, '*', color=color, ms=9)
        
    ax_map.set_title(f"Baseline MPC Trajectories  ({success_n}/{n_ep} success)", fontsize=12)
    ax_map.legend(loc='upper right', fontsize=8, ncol=2)

    labels = ['v_ref', 'obs', 'track', 'lat', 'i_acc', 'i_str', 'r_acc', 'r_str']
    params_str = " | ".join([f"{l}:{v:.1f}" for l, v in zip(labels, fixed_action)])

    print("\n" + "="*50)
    print(" Baseline MPC Evaluation Summary")
    print("="*50)
    try:
        map_name = os.path.basename(save_path).split('_')[3]
    except IndexError:
        map_name = "Unknown"
    
    print(f"  Map        : {map_name}")
    print(f"  Success    : {success_n}/{n_ep} ({100*success_n/n_ep:.1f}%)")
    print(f"  Mean Steps : {np.mean([r['steps'] for r in results]):.1f}")
    print(f"  Fixed Act  : [{params_str}]")
    print("="*50 + "\n")

    plt.suptitle(
        f"Baseline Fixed Params  |  Map: {os.path.basename(save_path).split('_')[3]}  \n"
        f"SR={success_n}/{n_ep} ({100*success_n/n_ep:.0f}%)  |  Params: [{params_str}]",
        fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"[Summary] 已保存 {save_path}")
    plt.show()

def capture_frame(fig):
    buf = fig.canvas.buffer_rgba()
    frame = np.asarray(buf)[:, :, :3]
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    import random
    parser = argparse.ArgumentParser(description="Stage-2 MPC 默认参数基线测试")
    parser.add_argument('--map',      default='random', help='map')
    parser.add_argument('--episodes', type=int, default=5, help='运行 episode 数量')
    parser.add_argument('--no-render', action='store_true', help='关闭实时展示')
    parser.add_argument('--obstacles', action='store_true', help='启用静态随机障碍物')
    parser.add_argument('--density', type=float, default=1, help='障碍物采样密度 [0,1]，默认 0.5')
    parser.add_argument('--v-ref', type=float, default=10.0, help='指定基线测试的目标车速 v_ref')
    parser.add_argument('--save-video', action='store_true', help='是否保存视频')
    args = parser.parse_args()

    render = not args.no_render
    save_video = args.save_video

    map_dir = os.path.join(project_root, 'data', 'street-map')
    if args.map == 'random':
        import glob
        map_files = [os.path.basename(f) for f in glob.glob(os.path.join(map_dir, "*.map"))]
        if not map_files:
            raise FileNotFoundError(f"目录 {map_dir} 中没有找到 .map 文件")
    else:
        map_files = [args.map]

    default_action = np.array([
        args.v_ref,  
        1.0,         
        1.0,        
        1.0,        
        1.0,                 
    ], dtype=np.float32)

    print(f"\n开始测试纯 MPC 控制: {args.episodes} episodes  map={args.map}  render={render}")
    print(f"使用固定参数: {default_action}")
    
    results = []
    for ep in range(args.episodes):
        if args.map == 'random':
            chosen_map_file = random.choice(map_files)
            map_path = os.path.join(map_dir, chosen_map_file)
        else:
            map_path = os.path.join(project_root, 'data', 'maps', args.map)
            
        map_obstacle = load_map(map_path)
        env = make_test_env(map_obstacle, use_obstacles=args.obstacles, density=args.density)
        base_env = env.unwrapped
        
        r = run_episode(env, base_env, default_action, render, ep, save_video)
        r['map_name'] = chosen_map_file if args.map == 'random' else args.map
        results.append(r)

    success_n = sum(r['success'] for r in results)
    print(f"\n====== 测试结果 ======")
    print(f"  地图     : {args.map} (random from street-map)" if args.map == 'random' else f"  地图     : {args.map}")
    print(f"  success  : {success_n}/{args.episodes} ({100*success_n/args.episodes:.1f}%)")
    print(f"  avg_steps: {np.mean([r['steps'] for r in results]):.1f}")
    if any(r['velocities'] for r in results):
        print(f"  avg_speed: {np.mean([r['avg_speed'] for r in results]):.2f} m/s")

    save_path = os.path.join(
        project_root, 'logs',
        f"test_baseline_{args.map}_{args.episodes}ep.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plot_summary(results, results[0]['map'], save_path, default_action)

if __name__ == "__main__":
    main()