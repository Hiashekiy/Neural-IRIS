"""
Stage 2: MPC 参数调整训练脚本 (Param Adjust) - 动态障碍物进阶版
策略网络: DeepGatedActorCriticPolicy (tune_gated_net.py)
环境:     TuningEnv (src/rl_envs/tuning_env.py)

使用方法 (加载基础模型继续训练):
    python scripts/train/train_param_adjust_dynamic.py --resume ./models/param_adjust/param_adjust_gated_deep_final
"""
import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from gymnasium.wrappers import RescaleAction

# ==========================================
# 0. 路径设置
# ==========================================
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.rl_envs.tuning_env import TuningEnv
from src.rl_networks.tune_gated_net import DeepGatedActorCriticPolicy
from src.utils.obstacle_wrapper import MovingObstacleWrapper # [修改] 引入动态移动障碍物包装器

# ==========================================
# 1. 训练超参数 (集中管理)
# ==========================================
# [修改] 更改实验名称，保存为全新的动态避障模型
EXP_NAME        = "param_adjust_gated_deep_dynamic" 
NUM_CPU         = 8          # 并行环境数量，根据 CPU 核心数调整
TOTAL_STEPS     = 3_000_000  # 总训练步数
INPUT_RESOLUTION = 1.0       # 传入 TuningEnv 的分辨率参数

# --- 学习率 ---
LR_INITIAL = 3e-4

# --- PPO 超参数 ---
N_STEPS    = 512    # 每次采样步数 (每个 env 收集 N_STEPS 步)
BATCH_SIZE = 2048   # minibatch 大小
N_EPOCHS   = 10     # 每次采样后更新的 epoch 数
GAMMA      = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
# 连续动作空间不需要熵正则化：Gaussian 的 std 本身提供探索。
# ent_coef != 0 时，熵梯度(0.005×5.68=0.028)会压制 policy_gradient(-0.015)，
# 导致 log_std 被钉死在初始值 0 (std≈1)，策略永远无法收敛。
ENT_COEF   = 0.0

# --- 路径 ---
LOG_ROOT       = os.path.join(root_path, 'logs')
MODEL_SAVE_DIR = os.path.join(root_path, 'models', 'param_adjust')
TB_LOG_DIR     = os.path.join(LOG_ROOT, 'tb_param_adjust')
FAILURE_IMG_DIR = os.path.join(LOG_ROOT, 'failures_param_adjust')

FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f"{EXP_NAME}_final")
FINAL_STATS_PATH = os.path.join(MODEL_SAVE_DIR, f"{EXP_NAME}_vecnorm.pkl")

CKPT_SAVE_FREQ = max(1, 50_000 // NUM_CPU)  # 每隔多少 env_steps 保存一次

# ==========================================
# 2. 回调函数
# ==========================================

def linear_schedule(initial_value: float):
    """线性衰减学习率调度器"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


class SuccessRateCallback(BaseCallback):
    """
    记录 episode 成功率到 TensorBoard。
    依赖 TuningEnv.step() 在 info 中写入 'is_success'。
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_successes: list = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # SB3 在 episode 结束时会在 info 中附带 'episode' 键
            if "episode" in info:
                self._episode_successes.append(float(info.get("is_success", 0.0)))

        if len(self._episode_successes) >= 100:
            success_rate = float(np.mean(self._episode_successes))
            self.logger.record("custom/success_rate_100ep", success_rate)
            self._episode_successes.clear()

        return True

class ActionMeanCallback(BaseCallback):
    """拦截 tuning_env 输出的物理参数均值，以便在控制台中格式化打印输出"""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.ep_actions_buffer = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "ep_action_mean" in info:
                self.ep_actions_buffer.append(info["ep_action_mean"])
        return True

    def _on_rollout_end(self) -> None:
        if len(self.ep_actions_buffer) > 0:
            keys = self.ep_actions_buffer[0].keys()
            for k in keys:
                mean_val = np.mean([ep.get(k, 0.0) for ep in self.ep_actions_buffer])
                # 以 action_mean/ 分组记录，SB3 自动排版成漂亮的表格
                self.logger.record(f"action_mean/{k}", mean_val)
            self.ep_actions_buffer.clear()

class RewardComponentsCallback(BaseCallback):
    """
    聚合记录每个 Episode 内拆解的各项奖励与惩罚项，输出到 Tensorboard 的 reward_components/ 分组下。
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.ep_rewards_buffer = []

    def _on_step(self) -> bool:
        # 每个 step 检查 info 字典是否抛出了回合统计项 (仅在 done 的时候会抛出)
        for info in self.locals.get("infos", []):
            if "ep_reward_info" in info:
                self.ep_rewards_buffer.append(info["ep_reward_info"])
        return True

    def _on_rollout_end(self) -> None:
        # 当 PPO 收集完 N_STEPS 数据准备进行网络更新时，计算本批次数据的均值并输出
        if len(self.ep_rewards_buffer) > 0:
            keys = self.ep_rewards_buffer[0].keys()
            for k in keys:
                mean_val = np.mean([ep.get(k, 0.0) for ep in self.ep_rewards_buffer])
                self.logger.record(f"reward_components/{k}", mean_val)
            self.ep_rewards_buffer.clear()

class SaveVecNormalizeCallback(BaseCallback):
    """每隔 save_freq 步保存一次 VecNormalize 统计数据，防止训练中断后丢失归一化参数。"""
    def __init__(self, save_freq: int, save_path: str, name_prefix: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_vecnorm.pkl")
                vec_env.save(path)
        return True


class FailureVisualizerCallback(BaseCallback):
    """
    当 episode 失败时，将失败轨迹渲染并保存为 PNG。
    依赖 TuningEnv.step() 在失败 episode 末尾往 info 中写入 'debug_*' 键。
    """
    def __init__(self, save_dir: str, max_saves: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.max_saves = max_saves
        self._save_count = 0

    def _on_step(self) -> bool:
        if self._save_count >= self.max_saves:
            return True
        for idx, info in enumerate(self.locals.get("infos", [])):
            if "debug_traj" in info:
                self._render_failure(info, idx)
        return True

    def _render_failure(self, info: dict, env_idx: int):
        try:
            scale = info['debug_scale']           # 米/像素

            def to_px(arr):
                return np.asarray(arr) / scale

            traj   = to_px(info['debug_traj'])    # (N, 2)
            start  = to_px(info['debug_start'])   # (2,)
            target = to_px(info['debug_target'])  # (2,)
            guide  = info.get('debug_guide')

            fig, ax = plt.subplots(figsize=(8, 8))
            # 地图背景：障碍=黑，空=白
            if 'debug_map' in info:
                ax.imshow(1 - info['debug_map'], cmap='gray', origin='upper')

            if guide is not None and len(guide) > 0:
                g = to_px(guide)
                ax.plot(g[:, 0], g[:, 1], 'g--', linewidth=1, alpha=0.6, label='ref path')

            if len(traj) > 0:
                ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=1.5, label='trajectory')
                ax.plot(traj[-1, 0], traj[-1, 1], 'rx', markersize=8)

            ax.plot(start[0],  start[1],  'go', markersize=8, label='start')
            ax.plot(target[0], target[1], 'b*', markersize=10, label='goal')

            step_info = info.get('debug_step', self.num_timesteps)
            ax.set_title(f"Failure @ step={step_info}  env={env_idx}")
            ax.legend(fontsize=8)

            fname = os.path.join(self.save_dir, f"fail_{self.num_timesteps}_{env_idx}.png")
            fig.savefig(fname, dpi=80, bbox_inches='tight')
            plt.close(fig)
            self._save_count += 1
        except Exception:
            pass

# ==========================================
# 3. 环境工厂
# ==========================================

def make_env(rank: int, map_obstacle: np.ndarray, input_res: float, seed: int = 0):
    """返回一个无参可调用对象，用于 SubprocVecEnv / DummyVecEnv。"""
    def _init():
        env = TuningEnv(map_obstacle, input_resolution=input_res, max_steps=1000)
        
        # ==========================================
        # [修改] 使用带有防蜂拥和终点保护机制的动态移动障碍物
        # ==========================================
        env = MovingObstacleWrapper(
            env,
            num_obstacles=5,
            dt=0.1,
            verbose=False
        )

        env.reset(seed=seed + rank)
        return env
    return _init

# ==========================================
# 4. 主训练函数
# ==========================================

def train(resume_path: str | None = None):
    print("=" * 60)
    print(f"[Stage 2] MPC 参数调整训练 (动态移动避障) ——  {EXP_NAME}")
    print(f"策略网络 : DeepGatedActorCriticPolicy")
    print(f"环境     : TuningEnv + MovingObstacleWrapper (动作空间: 5 维)")
    print(f"并行 env : {NUM_CPU}   总步数: {TOTAL_STEPS:,}")
    print("=" * 60)

    # ---- 目录创建 ----
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR,     exist_ok=True)
    os.makedirs(FAILURE_IMG_DIR, exist_ok=True)

    # ---- 加载地图 ----
    map_path = os.path.join(root_path, 'data', 'maps', 'maze.png')
    if not os.path.exists(map_path):
        # 向上一级再找
        map_path = os.path.join(root_path, 'data', 'maze.png')
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"找不到地图文件，请确认路径: {map_path}")

    maze_gray = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    maze_gray = cv2.resize(maze_gray, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    _, map_obstacle = cv2.threshold(maze_gray, 50, 1, cv2.THRESH_BINARY)
    print(f"地图加载: {map_path}  shape={map_obstacle.shape}")

    # ---- 创建并行环境 ----
    if NUM_CPU > 1:
        env = SubprocVecEnv([make_env(i, map_obstacle, INPUT_RESOLUTION) for i in range(NUM_CPU)])
    else:
        env = DummyVecEnv([make_env(0, map_obstacle, INPUT_RESOLUTION)])

    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=1000.0, gamma=GAMMA)

    # ---- 构建 / 恢复模型 ----
    if resume_path is not None and os.path.exists(resume_path + ".zip"):
        print(f">>> 从断点恢复: {resume_path}")
        model = PPO.load(
            resume_path,
            env=env,
            tensorboard_log=TB_LOG_DIR,
            # 学习率调度在 load 后需要重新设置
            learning_rate=linear_schedule(LR_INITIAL),
        )
        # 对应 VecNormalize 统计数据
        stats_path = resume_path.replace(".zip", "_vecnorm.pkl")
        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, env)
            model.set_env(env)
            print(f"VecNormalize 统计数据已恢复: {stats_path}")
    else:
        model = PPO(
            policy=DeepGatedActorCriticPolicy,
            env=env,
            verbose=1,
            learning_rate=linear_schedule(LR_INITIAL),
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            tensorboard_log=TB_LOG_DIR,
        )

    # ---- 回调 ----
    callbacks = [
        CheckpointCallback(
            save_freq=CKPT_SAVE_FREQ,
            save_path=MODEL_SAVE_DIR,
            name_prefix=EXP_NAME,
            save_vecnormalize=True,
        ),
        SuccessRateCallback(verbose=0),
        RewardComponentsCallback(verbose=0),  
        ActionMeanCallback(verbose=0), 
        FailureVisualizerCallback(save_dir=FAILURE_IMG_DIR, max_saves=500),
    ]

    # ---- 训练 ----
    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=callbacks,
            tb_log_name=EXP_NAME,
            reset_num_timesteps=(resume_path is None),
        )
    except KeyboardInterrupt:
        print("\n训练被手动中断，正在保存...")
    finally:
        model.save(FINAL_MODEL_PATH)
        env.save(FINAL_STATS_PATH)
        print(f"✅ 模型已保存: {FINAL_MODEL_PATH}.zip")
        print(f"✅ 归一化统计: {FINAL_STATS_PATH}")


# ==========================================
# 5. 入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: MPC 参数调整训练 (动态障碍物版)")
    parser.add_argument(
        "--resume",
        type=str,
        default="None",
        metavar="CHECKPOINT_PATH",
        help="从指定 checkpoint 恢复训练 (不含 .zip 后缀)",
    )
    args = parser.parse_args()
    
    # 将 "None" 字符串转换为实际的 None
    resume_target = None if args.resume == "None" else args.resume
    train(resume_path=resume_target)