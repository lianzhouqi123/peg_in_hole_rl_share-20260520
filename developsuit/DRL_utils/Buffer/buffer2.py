import torch

# 全局变量：自动检测并选择运算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOBufferWarp:
    def __init__(self, num_steps, num_envs, obs_dim, critic_obs_dim, action_dim):
        """
        全张量化多环境并行 Buffer。
        数据结构全面升级为 [num_steps, num_envs, dim]，彻底消灭 CPU 介入。
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.step = 0

        # ========== 预分配 3D GPU 内存 ==========
        self.obs = torch.zeros((num_steps, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.critic_obs = torch.zeros((num_steps, num_envs, critic_obs_dim), dtype=torch.float32, device=device)

        self.actions = torch.zeros((num_steps, num_envs, action_dim), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.zs = torch.zeros((num_steps, num_envs, action_dim), dtype=torch.float32, device=device)

        # value 和 mask 需要多存一步，用于算 bootstrapped return
        self.values = torch.zeros((num_steps + 1, num_envs), dtype=torch.float32, device=device)
        self.masks = torch.ones((num_steps + 1, num_envs), dtype=torch.float32, device=device)
        self.time_limit_masks = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        self.time_limit_values = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

    def insert(self, obs, critic_obs, action, log_prob, z, value, reward, done,
               time_limit=None, time_limit_value=None):
        """
        插入单步数据。
        注意：所有传入的参数必须是形状为 [num_envs, ...] 的纯 GPU Tensor。
        内部没有任何 if-else 或 .item() 转换，保证 0 阻塞。
        """
        if self.step >= self.num_steps:
            raise ValueError("Buffer 已满，请先执行 PPO 更新后再 clear！")

        self.obs[self.step].copy_(obs)
        self.critic_obs[self.step].copy_(critic_obs)
        self.actions[self.step].copy_(action)
        self.log_probs[self.step].copy_(log_prob)
        self.zs[self.step].copy_(z)

        # 兼容 value 可能带有冗余的特征维度 [num_envs, 1] -> [num_envs]
        self.values[self.step].copy_(value.squeeze(-1) if value.dim() > 1 else value)
        self.rewards[self.step].copy_(reward)

        # done 通常是 bool 型的 tensor，转换为 float 后 1.0 - done 即可得到 mask
        # self.step + 1 处的 mask 会被用来计算 self.step 处的优势函数
        self.masks[self.step + 1].copy_(1.0 - done.float())
        if time_limit is None:
            self.time_limit_masks[self.step].zero_()
            self.time_limit_values[self.step].zero_()
        else:
            self.time_limit_masks[self.step].copy_(time_limit.bool())
            if time_limit_value is None:
                self.time_limit_values[self.step].zero_()
            else:
                self.time_limit_values[self.step].copy_(
                    time_limit_value.squeeze(-1) if time_limit_value.dim() > 1 else time_limit_value
                )

        self.step += 1

    def compute_returns_and_advantages(self, next_value, gamma=0.99, gae_lambda=0.95):
        """
        使用广播机制在 GPU 上并行计算所有环境的 GAE。
        """
        self.values[self.step].copy_(next_value.squeeze(-1) if next_value.dim() > 1 else next_value)

        # 临时张量，用于累加 GAE
        gae = torch.zeros(self.num_envs, dtype=torch.float32, device=device)

        for i in reversed(range(self.step)):
            # 这里所有的加减乘除都会在 num_envs 这个维度上自动广播并发计算
            bootstrap_value = torch.where(
                self.time_limit_masks[i],
                self.time_limit_values[i],
                self.values[i + 1] * self.masks[i + 1]
            )
            delta = self.rewards[i] + gamma * bootstrap_value - self.values[i]
            gae = delta + gamma * gae_lambda * self.masks[i + 1] * gae

            self.advantages[i].copy_(gae)
            self.returns[i].copy_(gae + self.values[i])

        # 获取有效步数内的优势函数并展平
        valid_adv = self.advantages[:self.step].view(-1)

        # 计算全局均值和方差，并原地归一化 (跨越所有时间步和所有环境)
        adv_mean = valid_adv.mean()
        adv_std = valid_adv.std() + 1e-8

        self.advantages[:self.step] = (self.advantages[:self.step] - adv_mean) / adv_std

    def mpl_generator(self, num_mini_batch, mini_batch_size=None):
        total_batch_size = self.step * self.num_envs

        if mini_batch_size is None:
            mini_batch_size = total_batch_size // num_mini_batch

        # 生成全局随机索引
        sampler = torch.randperm(total_batch_size, device=device)

        # 【极致优化】：在循环外一次性完成所有数据的高级索引洗牌！
        shuffled_obs = self.obs[:self.step].view(-1, self.obs_dim)[sampler]
        shuffled_critic_obs = self.critic_obs[:self.step].view(-1, self.critic_obs_dim)[sampler]
        shuffled_actions = self.actions[:self.step].view(-1, self.action_dim)[sampler]
        shuffled_returns = self.returns[:self.step].view(-1)[sampler]
        shuffled_log_probs = self.log_probs[:self.step].view(-1)[sampler]
        shuffled_zs = self.zs[:self.step].view(-1, self.action_dim)[sampler]
        shuffled_advantages = self.advantages[:self.step].view(-1)[sampler]

        for i in range(num_mini_batch):
            start = i * mini_batch_size
            end = start + mini_batch_size

            # 【极致优化】：这里使用 [start:end] 连续切片，返回的是零拷贝的视图 (View)
            yield (
                shuffled_critic_obs[start:end],
                shuffled_obs[start:end],
                shuffled_actions[start:end],
                shuffled_returns[start:end],
                shuffled_log_probs[start:end],
                shuffled_zs[start:end],
                shuffled_advantages[start:end]
            )

    def clear(self):
        self.step = 0
