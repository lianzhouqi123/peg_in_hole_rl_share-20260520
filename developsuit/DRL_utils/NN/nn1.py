import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module):
    def __init__(self, state_dim=18, action_dim=12, hidden_dim=256, action_scale=1.0, action_bias=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
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

    def forward(self, state, deterministic=False):
        x = self.net(state)
        mean = torch.tanh(self.mean_layer(x))
        std = torch.exp(self.log_std).expand_as(mean)

        if deterministic:
            z = mean
            # 确定性模式下 noise 视为 0
            noise_sq = torch.zeros_like(mean)
        else:
            # 采样出纯净的噪声
            noise = torch.randn_like(mean)
            z = mean + std * noise
            # 直接复用噪声的平方，省去逆向推导的 3 步张量运算！
            noise_sq = noise.pow(2)

        # 【数学优化】：直接使用 noise_sq
        log_scale = self.log_std.expand_as(mean)
        action_log_probs = -0.5 * (noise_sq + 2 * log_scale + math.log(2 * math.pi))
        action_log_probs = action_log_probs.sum(dim=-1)

        norm_action = torch.clamp(z, min=-1.0, max=1.0)
        real_action = norm_action * self.action_scale + self.action_bias

        return real_action, action_log_probs, z

    def evaluate_actions(self, state, z):
        x = self.net(state)
        mean = torch.tanh(self.mean_layer(x))
        std = torch.exp(self.log_std).expand_as(mean)
        var = std.pow(2)
        log_scale = self.log_std.expand_as(mean)

        # 手写概率与熵的解析式
        action_log_probs = -0.5 * (((z - mean) ** 2) / var + 2 * log_scale + math.log(2 * math.pi)).sum(dim=-1)
        dist_entropy = (0.5 + 0.5 * math.log(2 * math.pi) + log_scale).sum(dim=-1).mean()

        return action_log_probs, dist_entropy


class PPOValueNetwork(nn.Module):
    def __init__(self, state_dim=18, hidden_dim=256):
        super().__init__()
        self.v_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.to(device)

    def forward(self, state):
        return self.v_net(state)