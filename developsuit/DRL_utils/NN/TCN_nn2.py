import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DualStreamTCNFeatureExtractor(nn.Module):
    """
    全观测 TCN 特征提取器：
    将单帧 obs 的所有维度都送入时序卷积，而不是只对力觉做 TCN。
    """
    def __init__(self, kin_dim=12, force_dim=6, history_len=5, tcn_feature_dim=128):
        super().__init__()
        self.kin_dim = kin_dim
        self.force_dim = force_dim
        self.obs_dim = kin_dim + force_dim
        self.history_len = history_len

        # ==========================================
        # 对完整观测序列做三分支多尺度时序卷积
        # ==========================================
        # 高频分支更关注最近几步的瞬时变化
        self.conv_high = nn.Conv1d(in_channels=self.obs_dim, out_channels=32, kernel_size=2)
        # 中频分支提取中等时间尺度的动态模式
        self.conv_mid = nn.Conv1d(in_channels=self.obs_dim, out_channels=32, kernel_size=3)
        # 低频分支覆盖整段历史，感知整体趋势
        self.conv_low = nn.Conv1d(in_channels=self.obs_dim, out_channels=32, kernel_size=history_len)

        # 计算卷积展平后的维度
        len_high = history_len - 2 + 1
        len_mid = history_len - 3 + 1
        len_low = history_len - history_len + 1
        self.tcn_flatten_dim = 32 * (len_high + len_mid + len_low)

        # 将多尺度 TCN 特征压缩为统一表示
        self.temporal_mlp = nn.Sequential(
            nn.Linear(self.tcn_flatten_dim, tcn_feature_dim),
            nn.LayerNorm(tcn_feature_dim),
            nn.ReLU()
        )

        self.output_dim = tcn_feature_dim

    def forward(self, obs_seq):
        # 输入 obs_seq 形状: [Batch, history_len, obs_dim]
        if obs_seq.dim() != 3:
            raise ValueError(
                f"DualStreamTCNFeatureExtractor 期望 3D 输入 [batch, history_len, obs_dim]，"
                f"实际收到 shape={tuple(obs_seq.shape)}"
            )

        expected_obs_dim = self.obs_dim
        if obs_seq.size(1) != self.history_len or obs_seq.size(2) != expected_obs_dim:
            raise ValueError(
                f"DualStreamTCNFeatureExtractor 输入维度不匹配："
                f"期望 [batch, {self.history_len}, {expected_obs_dim}]，"
                f"实际收到 {tuple(obs_seq.shape)}"
            )

        # ==========================================
        # 整段观测序列统一进入 TCN
        # ==========================================
        obs_input = obs_seq.transpose(1, 2)  # [Batch, obs_dim, history_len]

        out_high = F.relu(self.conv_high(obs_input))
        out_mid = F.relu(self.conv_mid(obs_input))
        out_low = F.relu(self.conv_low(obs_input))

        temporal_features_flat = torch.cat([
            out_high.flatten(1),
            out_mid.flatten(1),
            out_low.flatten(1)
        ], dim=1)

        return self.temporal_mlp(temporal_features_flat)


class ActorNetwork(nn.Module):
    def __init__(self, kin_dim=12, force_dim=6, action_dim=12, history_len=5, hidden_dim=256, action_scale=1.0, action_bias=0.0):
        super().__init__()

        # 1. 接入全观测 TCN 特征提取器
        self.extractor = DualStreamTCNFeatureExtractor(kin_dim=kin_dim, force_dim=force_dim, history_len=history_len)

        # 2. 时序特征再经 MLP 决策
        self.net = nn.Sequential(
            nn.Linear(self.extractor.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5, dtype=torch.float32))

        # Tensor 注册保持不变...
        if isinstance(action_scale, (int, float)):
            scale_tensor = torch.full((action_dim,), action_scale, dtype=torch.float32)
        else:
            scale_tensor = torch.tensor(action_scale, dtype=torch.float32)

        if isinstance(action_bias, (int, float)):
            bias_tensor = torch.full((action_dim,), action_bias, dtype=torch.float32)
        else:
            bias_tensor = torch.tensor(action_bias, dtype=torch.float32)

        self.register_buffer('action_scale', scale_tensor)
        self.register_buffer('action_bias', bias_tensor)

        self.to(device)

    def forward(self, obs_seq, deterministic=False):
        # 1. 提取出完整 obs 历史上的时序特征
        features = self.extractor(obs_seq)
        
        # 2. 决策推理 (过你的两个 MLP)
        x = self.net(features)
        mean = torch.tanh(self.mean_layer(x))
        std = torch.exp(self.log_std).expand_as(mean)

        if deterministic:
            z = mean
            noise_sq = torch.zeros_like(mean)
        else:
            noise = torch.randn_like(mean)
            z = mean + std * noise
            noise_sq = noise.pow(2)

        log_scale = self.log_std.expand_as(mean)
        action_log_probs = -0.5 * (noise_sq + 2 * log_scale + math.log(2 * math.pi))
        action_log_probs = action_log_probs.sum(dim=-1)

        norm_action = torch.clamp(z, min=-1.0, max=1.0)
        real_action = norm_action * self.action_scale + self.action_bias

        return real_action, action_log_probs, z

    def evaluate_actions(self, obs_seq, z):
        features = self.extractor(obs_seq)
        x = self.net(features)
        mean = torch.tanh(self.mean_layer(x))
        std = torch.exp(self.log_std).expand_as(mean)
        var = std.pow(2)
        log_scale = self.log_std.expand_as(mean)

        action_log_probs = -0.5 * (((z - mean) ** 2) / var + 2 * log_scale + math.log(2 * math.pi)).sum(dim=-1)
        dist_entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_scale).sum(dim=-1).mean()

        return action_log_probs, dist_entropy


class PPOValueNetwork(nn.Module):
    def __init__(self, kin_dim=12, force_dim=6, history_len=5, hidden_dim=256):
        super().__init__()
        
        # ==========================================
        # 🌟 为 Critic 实例化一个【独立】的全观测 TCN 特征提取器
        # ==========================================
        self.extractor = DualStreamTCNFeatureExtractor(
            kin_dim=kin_dim,
            force_dim=force_dim,
            history_len=history_len
        )
        
        # ==========================================
        # 价值网络的 MLP 大脑
        # 输入：全观测 TCN 提炼后的时序特征
        # 输出：1 维的标量 (State Value)，用于计算 GAE
        # ==========================================
        self.v_net = nn.Sequential(
            nn.Linear(self.extractor.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.to(device)

    def forward(self, obs_seq):
        # 1. 提取与 Actor 完全同时空视角的时序特征
        features = self.extractor(obs_seq)
        
        # 2. 评估当前状态的价值 V(s)
        return self.v_net(features)
