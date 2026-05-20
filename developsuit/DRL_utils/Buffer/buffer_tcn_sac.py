import torch

# 全局变量：自动检测并选择运算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SACReplayBufferWarp:
    def __init__(self, capacity, obs_shape, action_dim):
        """
        全张量化多环境并行的 Off-Policy 经验回放池。
        支持 Warp 并行，也向下完美兼容单线程真机。
        """
        self.capacity = int(capacity)
        
        self.obs_shape = (obs_shape,) if isinstance(obs_shape, int) else tuple(obs_shape)
        self.action_dim = action_dim
        
        self.ptr = 0  
        self.current_size = 0  

        # ========== 预分配 1D 扁平化 GPU 内存 ==========
        self.obs = torch.zeros((self.capacity, *self.obs_shape), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((self.capacity, *self.obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(self.capacity, dtype=torch.float32, device=device)

    def push(self, obs, action, reward, next_obs, done):
        """
        批量/单步插入数据。
        """
        num_envs = obs.shape[0]
        
        # ==========================================
        # 🌟 【核心安全修复】: 维度对齐与标量保护
        # 防止单线程模式下的 0 维张量在环形切片时导致 IndexError
        # ==========================================
        if not isinstance(reward, torch.Tensor):
            reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
        if not isinstance(done, torch.Tensor):
            done = torch.as_tensor(done, dtype=torch.bool, device=device)
            
        # 强制将 0-dim 的标量展平为 1D 张量，形状为 (num_envs,)
        reward = reward.view(num_envs)
        done = done.view(num_envs).float() # 同时确保转为 float

        # 计算本次插入的结束位置
        end_idx = self.ptr + num_envs

        if end_idx <= self.capacity:
            # 🌟 情况 1：尾部空间充足，直接整体写入
            self.obs[self.ptr:end_idx].copy_(obs)
            self.actions[self.ptr:end_idx].copy_(action)
            self.rewards[self.ptr:end_idx].copy_(reward)
            self.next_obs[self.ptr:end_idx].copy_(next_obs)
            self.dones[self.ptr:end_idx].copy_(done) 
        else:
            # 🌟 情况 2：尾部空间不足，触发环形覆盖 (Wrap-around)
            overflow = end_idx - self.capacity
            fit = num_envs - overflow

            # 前半截：填满 Buffer 尾部 (由于上方有了 .view() 保护，这里的切片现在绝对安全)
            self.obs[self.ptr:self.capacity].copy_(obs[:fit])
            self.actions[self.ptr:self.capacity].copy_(action[:fit])
            self.rewards[self.ptr:self.capacity].copy_(reward[:fit])
            self.next_obs[self.ptr:self.capacity].copy_(next_obs[:fit])
            self.dones[self.ptr:self.capacity].copy_(done[:fit])

            # 后半截：覆盖 Buffer 头部
            self.obs[:overflow].copy_(obs[fit:])
            self.actions[:overflow].copy_(action[fit:])
            self.rewards[:overflow].copy_(reward[fit:])
            self.next_obs[:overflow].copy_(next_obs[fit:])
            self.dones[:overflow].copy_(done[fit:])

        # 更新环形指针和当前数据量
        self.ptr = (self.ptr + num_envs) % self.capacity
        self.current_size = min(self.current_size + num_envs, self.capacity)

    def sample(self, batch_size):
        """O(1) 极速随机采样"""
        idxs = torch.randint(0, self.current_size, size=(batch_size,), device=device)

        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            self.dones[idxs]
        )

    def size(self):
        return self.current_size