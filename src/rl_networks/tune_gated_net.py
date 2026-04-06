import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ==========================================================
# 1. 感知层：轻量 CNN + Vec 联合特征提取器
# ==========================================================

class TuningCNNExtractor(BaseFeaturesExtractor):
    """
    [Stage 2 专用感知层 — 双流对称架构]

    图像分支 (1×64×64 局部障碍物地图):
        Conv2d(1→16,  k=3, s=2, p=1)  →  32×32×16  → ReLU
        Conv2d(16→32, k=3, s=2, p=1)  →  16×16×32  → ReLU
        Conv2d(32→32, k=3, s=2, p=1)  →   8×8×32   → ReLU
        Flatten  →  2048
        Linear(2048 → 64)
        LayerNorm(64)   ← 归一化激活尺度，防止大梯度冲击 GatedLayer
        ReLU

    向量分支 (34 维：速度/朝向/路径误差/LiDAR/几何特征):
        Linear(34 → 64)
        LayerNorm(64)   ← 与 CNN 分支尺度对齐
        ReLU

    输出: 64 (CNN) + 64 (Vec-MLP) = 128 维

    设计原则：
    - 两个分支维度对称 (64:64)，避免 CNN 梯度幅度主导 GatedLayer 的学习
    - LayerNorm 将每个分支的激活值归一化到相近量级
    - 减小 CNN 通道 (32→16→32→32) 降低参数量，减轻早期随机梯度干扰
    - 向量分支经过轻量 MLP 预处理，提取语义特征而非直通原始数值
    """
    CNN_OUTPUT_DIM: int = 64
    VEC_OUTPUT_DIM: int = 64

    def __init__(self, observation_space: spaces.Dict,
                 cnn_output_dim: int = 64, vec_output_dim: int = 64):
        super().__init__(observation_space, features_dim=cnn_output_dim + vec_output_dim)

        # --- CNN 分支（轻量化 + LayerNorm） ---
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # 32×32×16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 16×16×32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 8×8×32
            nn.ReLU(),
            nn.Flatten(),                                            # 2048
            nn.Linear(8 * 8 * 32, cnn_output_dim),
            nn.LayerNorm(cnn_output_dim),   # 关键：归一化激活，防止大梯度干扰
            nn.ReLU(),
        )

        # --- 向量分支（轻量 MLP + LayerNorm，与 CNN 分支尺度对齐） ---
        vec_in = observation_space.spaces["vec"].shape[0]  # 34
        self.vec_branch = nn.Sequential(
            nn.Linear(vec_in, vec_output_dim),
            nn.LayerNorm(vec_output_dim),   # 关键：与 CNN 输出尺度对齐
            nn.ReLU(),
        )

        # 正交初始化
        for m in list(self.cnn.modules()) + list(self.vec_branch.modules()):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, observations: dict) -> torch.Tensor:
        # img: (B, 1, 64, 64), uint8 → float32, 归一化到 [0, 1]
        img = observations["img"].float() / 255.0
        cnn_feat = self.cnn(img)                      # (B, 64)
        vec_feat  = self.vec_branch(observations["vec"])  # (B, 64)
        return torch.cat([cnn_feat, vec_feat], dim=1)    # (B, 128)


# ==========================================================
# 2. 门控层
# ==========================================================

class GatedLayer(nn.Module):
    """
    单层门控单元 (Gated Unit)
    公式: h = o * tanh( i * g )
    作用: 模拟 LSTM 的门控机制，但在 MLP 中实现，无需维护时序状态，反应更灵敏。
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc_i = nn.Linear(input_dim, hidden_dim) # Input Gate
        self.fc_g = nn.Linear(input_dim, hidden_dim) # Feature Gate
        self.fc_o = nn.Linear(input_dim, hidden_dim) # Output Gate
        
    def forward(self, x):
        i = torch.sigmoid(self.fc_i(x))
        g = torch.tanh(self.fc_g(x))
        o = torch.sigmoid(self.fc_o(x))
        return o * torch.tanh(i * g)

class DeepGatedNetwork(nn.Module):
    """
    双层门控特征提取器 (用于 Stage 2 & 3)
    """
    def __init__(self, feature_dim, last_layer_dim_pi=256, last_layer_dim_vf=256):
        super().__init__()
        
        # --- Actor Stream ---
        self.policy_net = nn.Sequential(
            GatedLayer(feature_dim, 256), # 第一层门控清洗特征
            GatedLayer(256, 256),         # 第二层门控深度提取
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, last_layer_dim_pi),
            nn.ReLU()
        )
        
        # --- Critic Stream ---
        self.value_net = nn.Sequential(
            GatedLayer(feature_dim, 256),
            GatedLayer(256, 256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, last_layer_dim_vf),
            nn.ReLU()
        )
        
        # SB3 要求定义的属性
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

    def forward(self, features):
        """同时返回 Actor 和 Critic 的 Latent Features"""
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)

# ==========================================================
# 4. 策略类：显式绑定感知层 + 门控决策层
# ==========================================================

class DeepGatedActorCriticPolicy(MultiInputActorCriticPolicy):
    """
    [Stage 2 专用]

    完整数据流:
        观测 Dict{img(1×64×64), vec(34)}
            ↓ TuningCNNExtractor
        162 维融合特征
            ↓ DeepGatedNetwork (Actor/Critic 双流)
        256 维 Latent
            ↓ SB3 action_net / value_net
        动作 / 价值
    """
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # 显式注入感知层，防止 SB3 误用默认的 Flatten 提取器
        kwargs.setdefault("features_extractor_class", TuningCNNExtractor)
        kwargs.setdefault("features_extractor_kwargs", {
            "cnn_output_dim": TuningCNNExtractor.CNN_OUTPUT_DIM,
            "vec_output_dim": TuningCNNExtractor.VEC_OUTPUT_DIM,
        })
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # SB3 的 __init__ 内部会自动调用 _build_mlp_extractor

    def _build_mlp_extractor(self) -> None:
        # self.features_dim 由 TuningCNNExtractor 设置 = CNN_OUTPUT_DIM + 34 = 162
        self.mlp_extractor = DeepGatedNetwork(
            self.features_dim,
            last_layer_dim_pi=256,
            last_layer_dim_vf=256,
        )