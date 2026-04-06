# -*- coding: utf-8 -*-
"""
Stage 2 MPC 参数调整模型测试 (动态障碍物版)
加载训练好的 DeepGatedActorCriticPolicy 模型，在多张地图上测试泛化性能。

用法:
    python scripts/test/test_param_adjust.py
    python scripts/test/test_param_adjust.py --map map1.png --episodes 10
    python scripts/test/test_param_adjust.py --no-render --episodes 20
    python scripts/test/test_param_adjust.py --obstacles --num-obs 5
    python scripts/test/test_param_adjust.py --save-video
"""
import os
import sys
import argparse
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')  # 交互展示，如果报错改为 'Agg'
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ==========================================
# 0. 路径设置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rl_envs.tuning_env import TuningEnv
from src.rl_networks.tune_gated_net import DeepGatedActorCriticPolicy
from src.utils.obstacle_wrapper import MovingObstacleWrapper # [修改] 引入动态移动障碍物

# ==========================================
# 1. 地图加载
# ==========================================
def load_map(map_filename: str) -> np.ndarray:
    map_path = os.path.join(project_root, 'data', 'maps', map_filename)
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"找不到地图: {map_path}")
    maze = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    maze = cv2.resize(maze, None, fx=5.0, fy=5.0, interpolation=cv2.INTER_NEAREST)
    _, map_obstacle = cv2.threshold(maze, 50, 1, cv2.THRESH_BINARY)
    print(f"[Map] 加载 {map_filename}  shape={map_obstacle.shape}")
    return map_obstacle


# ==========================================
# 2. 环境工厂（对齐训练时的 make_env）
# ==========================================
def make_test_env(map_obstacle: np.ndarray,
                  use_obstacles: bool = False,
                  num_obstacles: int = 5):
    env = TuningEnv(map_obstacle=map_obstacle, input_resolution=1.0, max_steps=1000, path_mode='train')
    if use_obstacles:
        # [修改] 替换为移动障碍物
        env = MovingObstacleWrapper(
            env,
            num_obstacles=num_obstacles,
            dt=0.1,
            verbose=True
        )
        print(f"[MovingObstacleWrapper] 已启用  num_obstacles={num_obstacles}")
    return env


# ==========================================
# 3. 实时可视化 (高性能数据更新模式)
# ==========================================
def render_step(render_objs, env_wrapper, action_physical, traj_x, traj_y, velocities, status_text):
    """  每步调用，展示: 地图+轨迹 | CNN输入 | 动作曲线  """
    base_env = env_wrapper.unwrapped
    res = base_env.INTERNAL_RESOLUTION
    
    # ----------------------------------------------------
    # 强制逐帧更新底图矩阵，使移动障碍物在视觉上显现实体
    # ----------------------------------------------------
    if 'map_img_plot' in render_objs:
        render_objs['map_img_plot'].set_data(1 - base_env.map_obstacle)
        
    # ---- 更新左图: 轨迹、车辆状态与视角 ----
    if len(traj_x) > 1:
        # 更新散点轨迹坐标和颜色映射
        offsets = np.c_[np.array(traj_x) / res, np.array(traj_y) / res]
        render_objs['traj_scatter'].set_offsets(offsets)
        render_objs['traj_scatter'].set_array(np.array(velocities))

    state = base_env.state
    cx, cy = state[0] / res, state[1] / res
    heading = state[2]
    
    # 更新车身边界框
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

    # 更新车头方向指示箭头
    al = 4.0 / res
    render_objs['car_arrow'].xy = (cx + al * np.cos(heading), cy + al * np.sin(heading))
    render_objs['car_arrow'].set_position((cx, cy))

    # 更新动态视角与标题
    vr = 35.0 / res
    render_objs['ax_m'].set_xlim(cx - vr, cx + vr)
    render_objs['ax_m'].set_ylim(cy + vr, cy - vr)
    render_objs['title_m'].set_text(status_text)
    
    # ----------------------------------------------------
    # 在此处统一绘制动态障碍物红圈
    # ----------------------------------------------------
    for line in render_objs['dynamic_lines']:
        line.remove()
    render_objs['dynamic_lines'].clear()
    
    if hasattr(env_wrapper, 'placed_obstacles') and env_wrapper.placed_obstacles:
        for obs_info in env_wrapper.placed_obstacles:
            ox, oy = obs_info['center'][0] / res, obs_info['center'][1] / res
            phys_radius_px = obs_info['radius'] / res
            circle = plt.Circle((ox, oy), phys_radius_px, color='red', fill=False, alpha=0.8, linewidth=2)
            render_objs['ax_m'].add_artist(circle)
            render_objs['dynamic_lines'].append(circle)

    # ---- 更新中图: CNN 输入 ----
    cnn_img = base_env._get_local_map_cnn(state[:2], heading)[0]
    render_objs['cnn_img'].set_data(cnn_img)

    # ---- 更新右图: 动作曲线 ----
    if action_physical is not None:
        for bar, text, val in zip(render_objs['bars'], render_objs['bar_texts'], action_physical):
            bar.set_height(val)
            text.set_text(f'{val:.1f}')
            text.set_position((bar.get_x() + bar.get_width() / 2, val + 0.05))

    # 触发 GUI 刷新，代替 plt.draw() 以提高性能
    render_objs['fig'].canvas.draw_idle()
    render_objs['fig'].canvas.flush_events()
    plt.pause(0.05) # [修改] 锁帧防止画面撕裂


# ==========================================
# 4. 单次 episode 运行
# ==========================================
def run_episode(model, env_wrapped, base_env, venv, render: bool, ep_idx: int, save_video: bool = False, video_path: str = None):
    obs = venv.reset()

    episode_map  = base_env.map_obstacle.copy()
    episode_path = base_env.poly_points_np.copy() if base_env.poly_points_np is not None else None

    traj_x, traj_y, velocities = [], [], []
    actions_physical = []   
    done = False
    step = 0
    status = f"EP {ep_idx+1}  Running..."

    # 直接从环境内部提取物理映射的边界
    phys_low  = base_env.phys_low
    phys_high = base_env.phys_high

    render_objs = {}
    if render:
        plt.ion()
        fig, (ax_m, ax_c, ax_a) = plt.subplots(
            1, 3, figsize=(18, 7),
            gridspec_kw={'width_ratios': [3, 1, 1]})
        plt.tight_layout()

        res = base_env.INTERNAL_RESOLUTION

        # 1. 初始化左图元素
        map_img_plot = ax_m.imshow(1 - base_env.map_obstacle, cmap='gray', origin='upper', vmin=0, vmax=1)
        if base_env.poly_points_np is not None:
            p = base_env.poly_points_np
            ax_m.plot(p[:, 0] / res, p[:, 1] / res, 'g--', linewidth=1.5, alpha=0.6, label='Ref Path')
            sp = base_env.start_pos
            tp = base_env.target_pos
            ax_m.plot(sp[0] / res, sp[1] / res, 'g^', markersize=10, label='Start')
            ax_m.plot(tp[0] / res, tp[1] / res, 'r*', markersize=12, label='Goal')

        traj_scatter = ax_m.scatter([], [], c=[], cmap='jet', vmin=0, vmax=8, s=6, zorder=10)
        car_patch = Polygon(np.zeros((4, 2)), closed=True, edgecolor='black', facecolor='cyan', zorder=15, alpha=0.8)
        ax_m.add_patch(car_patch)
        car_arrow = ax_m.annotate('', xy=(0,0), xytext=(0,0), arrowprops=dict(arrowstyle='->', color='red', lw=2))
        title_m = ax_m.set_title(status, fontsize=11)
        ax_m.legend(loc='upper right', fontsize=8)

        # 2. 初始化中图元素
        dummy_cnn = np.zeros((64, 64))
        cnn_img_plot = ax_c.imshow(dummy_cnn, cmap='gray', vmin=0, vmax=255, origin='upper')
        sz = 64
        ax_c.plot(sz / 2, sz * 0.8, 'r^', markersize=10)
        ax_c.set_title("CNN Input (64×64)", fontsize=10)
        ax_c.axis('off')

        # 3. 初始化右图元素 (严格匹配 5 维)
        labels  = ['v_ref', 'w_obs', 'R_inf', 'w_lat', 'w_r_str']
        colors  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
        bars = ax_a.bar(labels, np.zeros(5), color=colors, alpha=0.85)
        bar_texts = [ax_a.text(b.get_x() + b.get_width() / 2, 0.05, '0.0', ha='center', va='bottom', fontsize=8) for b in bars]
        ax_a.set_ylim(0, 155)
        ax_a.set_title("MPC Params (Physical)", fontsize=10)
        ax_a.set_ylabel('Value')
        ax_a.tick_params(axis='x', labelsize=8, rotation=45)

        # 打包渲染对象
        render_objs = {
            'fig': fig, 'ax_m': ax_m,
            'map_img_plot': map_img_plot, 'dynamic_lines': [],
            'traj_scatter': traj_scatter, 'car_patch': car_patch, 
            'car_arrow': car_arrow, 'title_m': title_m,
            'cnn_img': cnn_img_plot,
            'bars': bars, 'bar_texts': bar_texts,
            'ax_a': ax_a
        }

    while not done:
        # 网络预测输出的是严格在 [-1, 1] 之间的 action_norm
        action_norm, _ = model.predict(obs, deterministic=False) 
        
        # 将 normalized action 送给环境，环境会自动处理到实际物理大小
        obs, _, dones, infos = venv.step(action_norm)
        done = bool(dones[0])
        step += 1

        # 仅为了可视化提取实际物理 Action
        a0 = np.clip(action_norm[0], -1.0, 1.0)
        act_phys = phys_low + (a0 + 1.0) / 2.0 * (phys_high - phys_low)

        st = base_env.state
        traj_x.append(st[0])
        traj_y.append(st[1])
        velocities.append(float(abs(st[3])))
        actions_physical.append(act_phys.copy())

        if render:
            # 动态调整 y 轴的高度
            max_val = np.max(act_phys)
            render_objs['ax_a'].set_ylim(0, max(max_val * 1.2, 5))
            render_step(render_objs, env_wrapped, act_phys,
                        traj_x, traj_y, velocities, status)
            
            # 保存视频帧
            if save_video and video_path:
                fig = render_objs['fig']
                fig.canvas.draw()
                rgba = np.asarray(fig.canvas.buffer_rgba())
                frame_bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                
                if 'video_writer' not in render_objs:
                    h, w, _ = frame_bgr.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # 假定大约20fps的播放速度，可根据实际控制频率调整
                    render_objs['video_writer'] = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
                
                render_objs['video_writer'].write(frame_bgr)

    success = bool(infos[0].get('is_success', False))
    status  = f"EP {ep_idx+1}  {'SUCCESS ✅' if success else 'FAIL ❌'}  steps={step}"
    if render:
        render_step(render_objs, env_wrapped, act_phys,
                    traj_x, traj_y, velocities, status)
        
        # 保存最后一帧并释放视频写入器
        if save_video and video_path:
            fig = render_objs['fig']
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frame_bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            if 'video_writer' in render_objs:
                render_objs['video_writer'].write(frame_bgr)
                render_objs['video_writer'].release()
                print(f"  [Video] 视频已保存至: {video_path}")

        time.sleep(1.5)
        plt.close(fig)
        plt.ioff()

    avg_speed = float(np.mean(velocities)) if velocities else 0.0
    print(f"  {status}  avg_v={avg_speed:.2f} m/s")

    from src.utils.obstacle_wrapper import MovingObstacleWrapper
    w = env_wrapped
    while hasattr(w, 'env'):
        if isinstance(w, MovingObstacleWrapper):
            break
        w = w.env

    return {
        'success':  success,
        'steps':    step,
        'avg_speed': avg_speed,
        'traj_x':   traj_x,
        'traj_y':   traj_y,
        'velocities': velocities,
        'actions':  np.array(actions_physical),
        'map':      episode_map,
        'path':     episode_path,
        'start':    base_env.start_pos.copy(),
        'target':   base_env.target_pos.copy(),
    }


# ==========================================
# 5. 汇总图 (5维参数适配)
# ==========================================
def plot_summary(results: list, map_obstacle: np.ndarray, save_path: str):
    res = 0.5  
    n_ep = len(results)
    success_n = sum(r['success'] for r in results)

    # 布局 3x4 以容纳地图与 5 个直方图
    fig = plt.figure(figsize=(20, 11))
    gs  = fig.add_gridspec(3, 4)

    ax_map = fig.add_subplot(gs[:, :2])
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
    ax_map.set_title(f"All Trajectories  ({success_n}/{n_ep} success)", fontsize=12)
    ax_map.legend(loc='upper right', fontsize=8, ncol=2)

    # 5维参数标签
    labels = ['v_ref (m/s)', 'w_obs', 'R_influence', 'w_lat', 'w_rate_steer']
    
    # 获取右半部分的 5 个子图位置 (取消了右下角最后一个位置)
    axes_indices = [(0,2), (0,3), (1,2), (1,3), (2,2)]
    ax_acts = [fig.add_subplot(gs[row, col]) for row, col in axes_indices]

    all_actions = np.concatenate([r['actions'] for r in results if len(r['actions'])>0], axis=0)
    
    print("\n" + "="*50)
    print(" Stage-2 Param Adjust Evaluation Summary")
    print("="*50)
    try:
        map_name = os.path.basename(save_path).split('_')[3]
    except IndexError:
        map_name = "Unknown"
    
    print(f"  Map        : {map_name}")
    print(f"  Success    : {success_n}/{n_ep} ({100*success_n/n_ep:.1f}%)")
    print(f"  Mean Steps : {np.mean([r['steps'] for r in results]):.1f}")
    print("-" * 50)
    print("  MPC Parameters Mean Values & Histogram Data:")
    
    for i, (ax, lbl) in enumerate(zip(ax_acts, labels)):
        counts, bin_edges = np.histogram(all_actions[:, i], bins=30)
        mean_val = np.mean(all_actions[:, i])
        
        print(f"  >>> Parameter: {lbl}")
        print(f"      Mean       : {mean_val:.4f}")
        print(f"      Bin Edges  : {np.round(bin_edges, 4).tolist()}")
        print(f"      Counts     : {counts.tolist()}\n")

        ax.hist(all_actions[:, i], bins=30, color=f'C{i%10}', alpha=0.75, edgecolor='black')
        ax.axvline(mean_val, color='k', linestyle='--', linewidth=1.5,
                   label=f"mean={mean_val:.2f}")
        ax.set_title(lbl, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', labelsize=8)
        
    print("="*50 + "\n")

    plt.suptitle(
        f"Stage-2 Param Adjust Evaluation  |  Map: {os.path.basename(save_path).split('_')[3]}  "
        f"|  SR={success_n}/{n_ep} ({100*success_n/n_ep:.0f}%)  "
        f"|  Mean Steps={np.mean([r['steps'] for r in results]):.0f}",
        fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"[Summary] 已保存: {save_path}")
    plt.show()


# ==========================================
# 6. 主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Stage-2 MPC 参数调整测试 (动态障碍物版)")
    # [修改] 默认加载最新的高频训练动态避障模型
    parser.add_argument('--model',    default='param_adjust_gated_deep_dynamic_final',
                        help='模型文件名 (不含 .zip)，位于 models/param_adjust/')
    parser.add_argument('--stats',    default='param_adjust_gated_deep_dynamic_vecnorm.pkl',
                        help='VecNormalize 统计文件，位于 models/param_adjust/')
    parser.add_argument('--map',      default='maze.png',
                        help='地图文件名，位于 data/maps/')
    parser.add_argument('--episodes', type=int, default=5, help='运行 episode 数量')
    parser.add_argument('--no-render', action='store_true', help='关闭实时展示')
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--obstacles', action='store_true',
                        help='启用动态移动障碍物注入')
    # [修改] 配合 MovingObstacleWrapper 使用数量而不是概率密度
    parser.add_argument('--num-obs', type=int, default=3,
                        help='动态障碍物数量，默认 5')
    parser.add_argument('--save-video', action='store_true',
                        help='将渲染过程保存为 MP4 视频')
    args = parser.parse_args()

    render = not args.no_render

    map_obstacle = load_map(args.map)

    raw_env = make_test_env(map_obstacle,
                            use_obstacles=args.obstacles,
                            num_obstacles=args.num_obs)
    venv = DummyVecEnv([lambda: raw_env])

    stats_path = os.path.join(project_root, 'models', 'param_adjust', args.stats)
    if os.path.exists(stats_path):
        print(f"[VecNorm] 加载: {stats_path}")
        venv = VecNormalize.load(stats_path, venv)
        venv.training   = False
        venv.norm_reward = False
    else:
        print(f"[VecNorm] 未找到 {stats_path}，跳过归一化")

    actual_venv = venv.venv if isinstance(venv, VecNormalize) else venv
    env_wrapped = actual_venv.envs[0]          
    base_env    = env_wrapped.unwrapped        

    model_path = os.path.join(project_root, 'models', 'param_adjust', args.model)
    if not model_path.endswith('.zip'):
        model_path += '.zip'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[Model] 找不到模型文件: {model_path}")

    print(f"[Model] 加载: {model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = PPO.load(model_path, env=venv, device=device)
    print(f"[Model] 加载成功  device={device}")

    # 处理视频保存路径
    video_dir = os.path.join(project_root, 'logs', 'videos')
    if args.save_video and render:
        os.makedirs(video_dir, exist_ok=True)

    print(f"\n开始测试: {args.episodes} episodes  map={args.map}  render={render}")
    results = []
    for ep in range(args.episodes):
        video_path = None
        if args.save_video and render:
            video_path = os.path.join(video_dir, f"test_{os.path.splitext(args.map)[0]}_ep{ep+1}.mp4")
            
        r = run_episode(model, env_wrapped, base_env, venv, render, ep, args.save_video, video_path)
        results.append(r)

    success_n = sum(r['success'] for r in results)
    print(f"\n═══ 测试结果 ═══")
    print(f"  地图     : {args.map}")
    print(f"  success  : {success_n}/{args.episodes} ({100*success_n/args.episodes:.1f}%)")
    print(f"  avg_steps: {np.mean([r['steps'] for r in results]):.1f}")
    print(f"  avg_speed: {np.mean([r['avg_speed'] for r in results]):.2f} m/s")
    print(f"  动作均值  : v_ref={np.mean([r['actions'][:,0].mean() for r in results if len(r['actions'])]):.2f}")

    save_path = os.path.join(
        project_root, 'logs',
        f"test_summary_{os.path.splitext(args.map)[0]}_{args.episodes}ep.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_summary(results, map_obstacle, save_path)

if __name__ == "__main__":
    main()