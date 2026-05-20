import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOBuffer:
    def __init__(self, num_steps, obs_dim, critic_obs_dim, action_dim):
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.step = 0

        # ========== 预分配 GPU 内存 ==========
        self.obs = torch.zeros((num_steps, obs_dim), dtype=torch.float32, device=device)
        self.critic_obs = torch.zeros((num_steps, critic_obs_dim), dtype=torch.float32, device=device)

        self.actions = torch.zeros((num_steps, action_dim), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((num_steps,), dtype=torch.float32, device=device)
        # 【关键协同】: Buffer 中存的是未经截断的纯高斯采样值 z
        self.zs = torch.zeros((num_steps, action_dim), dtype=torch.float32, device=device)

        self.values = torch.zeros((num_steps + 1,), dtype=torch.float32, device=device)
        self.masks = torch.ones((num_steps + 1,), dtype=torch.float32, device=device)
        self.time_limit_masks = torch.zeros((num_steps,), dtype=torch.bool, device=device)
        self.time_limit_values = torch.zeros((num_steps,), dtype=torch.float32, device=device)

        self.rewards = torch.zeros((num_steps,), dtype=torch.float32, device=device)
        self.returns = torch.zeros((num_steps,), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((num_steps,), dtype=torch.float32, device=device)

    def insert(self, obs, critic_obs, action, log_prob, z, value, reward, done,
               time_limit=None, time_limit_value=None):
        if self.step >= self.num_steps:
            raise ValueError("Buffer 已满，请先执行 PPO 更新后再 clear！")

        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(device).float().squeeze()
            return torch.tensor(x, dtype=torch.float32, device=device).squeeze()

        self.obs[self.step].copy_(to_tensor(obs))
        self.critic_obs[self.step].copy_(to_tensor(critic_obs))
        self.actions[self.step].copy_(to_tensor(action))
        self.log_probs[self.step].copy_(to_tensor(log_prob))
        self.zs[self.step].copy_(to_tensor(z))
        self.values[self.step].copy_(to_tensor(value))
        self.rewards[self.step].copy_(to_tensor(reward))
        self.masks[self.step + 1].copy_(torch.tensor(0.0 if done else 1.0, dtype=torch.float32, device=device))
        if time_limit is None:
            self.time_limit_masks[self.step].zero_()
            self.time_limit_values[self.step].zero_()
        else:
            self.time_limit_masks[self.step].copy_(torch.tensor(bool(time_limit), dtype=torch.bool, device=device))
            if time_limit_value is None:
                self.time_limit_values[self.step].zero_()
            else:
                self.time_limit_values[self.step].copy_(to_tensor(time_limit_value))

        self.step += 1

    def compute_returns_and_advantages(self, next_value, gamma=0.99, gae_lambda=0.95):
        self.values[self.step].copy_(torch.tensor(next_value, dtype=torch.float32, device=device))

        gae = 0
        for i in reversed(range(self.step)):
            bootstrap_value = torch.where(
                self.time_limit_masks[i],
                self.time_limit_values[i],
                self.values[i + 1] * self.masks[i + 1]
            )
            delta = self.rewards[i] + gamma * bootstrap_value - self.values[i]
            gae = delta + gamma * gae_lambda * self.masks[i + 1] * gae
            self.advantages[i] = gae
            self.returns[i] = gae + self.values[i]

        # 【核心操作】: Advantage 归一化 (此处逻辑极为标准，保持不变)
        adv = self.advantages[:self.step]
        self.advantages[:self.step] = (adv - adv.mean()) / (adv.std() + 1e-5)

    def mpl_generator(self, num_mini_batch, mini_batch_size=None):
        batch_size = self.step
        if mini_batch_size is None:
            mini_batch_size = batch_size // num_mini_batch

        sampler = torch.randperm(batch_size, device=device)

        for i in range(num_mini_batch):
            indices = sampler[i * mini_batch_size: (i + 1) * mini_batch_size]

            yield (
                self.critic_obs[indices],
                self.obs[indices],
                self.actions[indices],
                self.returns[indices],
                self.log_probs[indices],
                self.zs[indices], # 【关键协同】: 输出 zs 供 evaluate_actions 使用
                self.advantages[indices]
            )

    def clear(self):
        self.step = 0
