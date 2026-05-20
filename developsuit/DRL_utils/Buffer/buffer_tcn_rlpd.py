import torch
import os
import numpy as np

# 全局变量：自动检测并选择运算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RLPDReplayBufferWarp:
    def __init__(self, capacity, obs_shape, action_dim):
        """
        全张量化多环境并行的 Off-Policy 经验回放池。
        在 RLPD 中，此容器将被实例化两次：
        1. Online Buffer (存放真机在线探索数据，支持环形覆盖)
        2. Prior Buffer (存放离线成功先验数据，通常不再追加写入)
        """
        self.capacity = int(capacity)
        
        self.obs_shape = (obs_shape,) if isinstance(obs_shape, int) else tuple(obs_shape)
        self.action_dim = action_dim
        
        self.ptr = 0  
        self.current_size = 0  
        self.episode_lengths = []

        # ========== 预分配 1D 扁平化 GPU 内存 ==========
        self.obs = torch.zeros((self.capacity, *self.obs_shape), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((self.capacity, *self.obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(self.capacity, dtype=torch.float32, device=device)

    def push(self, obs, action, reward, next_obs, done):
        """批量/单步插入数据 (核心逻辑保持你完美的原版)"""
        num_envs = obs.shape[0]
        
        if not isinstance(reward, torch.Tensor):
            reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
        if not isinstance(done, torch.Tensor):
            done = torch.as_tensor(done, dtype=torch.bool, device=device)
            
        reward = reward.view(num_envs)
        done = done.view(num_envs).float() 

        end_idx = self.ptr + num_envs

        if end_idx <= self.capacity:
            self.obs[self.ptr:end_idx].copy_(obs)
            self.actions[self.ptr:end_idx].copy_(action)
            self.rewards[self.ptr:end_idx].copy_(reward)
            self.next_obs[self.ptr:end_idx].copy_(next_obs)
            self.dones[self.ptr:end_idx].copy_(done) 
        else:
            overflow = end_idx - self.capacity
            fit = num_envs - overflow

            self.obs[self.ptr:self.capacity].copy_(obs[:fit])
            self.actions[self.ptr:self.capacity].copy_(action[:fit])
            self.rewards[self.ptr:self.capacity].copy_(reward[:fit])
            self.next_obs[self.ptr:self.capacity].copy_(next_obs[:fit])
            self.dones[self.ptr:self.capacity].copy_(done[:fit])

            self.obs[:overflow].copy_(obs[fit:])
            self.actions[:overflow].copy_(action[fit:])
            self.rewards[:overflow].copy_(reward[fit:])
            self.next_obs[:overflow].copy_(next_obs[fit:])
            self.dones[:overflow].copy_(done[fit:])

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

    def record_episode_length(self, episode_len):
        self.episode_lengths.append(int(episode_len))

    def get_episode_length_median(self):
        if len(self.episode_lengths) == 0:
            return None
        return float(np.median(np.asarray(self.episode_lengths, dtype=np.float32)))

    # ==========================================
    # 🌟 新增：RLPD 专属离线数据序列化功能
    # ==========================================
    def save_data(self, save_path):
        """
        导出数据 (用于在收集脚本中保存 Prior Data)
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 将切片数据搬回 CPU 保存，防止长期占用显存
        data = {
            'obs': self.obs[:self.current_size].cpu(),
            'actions': self.actions[:self.current_size].cpu(),
            'rewards': self.rewards[:self.current_size].cpu(),
            'next_obs': self.next_obs[:self.current_size].cpu(),
            'dones': self.dones[:self.current_size].cpu(),
            'episode_lengths': torch.as_tensor(self.episode_lengths, dtype=torch.int32),
        }
        torch.save(data, save_path)
        print(f"✅ 成功将 {self.current_size} 条伪标签数据保存至: {save_path}")

    def load_data(self, load_path):
        """
        导入数据 (用于在 RLPD 训练开始前，瞬间填满 Prior Buffer)
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"❌ 找不到先验数据文件: {load_path}")

        data = torch.load(load_path, map_location=device, weights_only=True)
        load_size = data['obs'].shape[0]

        if load_size > self.capacity:
            print(f"⚠️ 警告: 导入的先验数据量 ({load_size}) 超过了 Buffer 容量 ({self.capacity})，将被截断！")
            load_size = self.capacity

        # O(1) 极速张量复制，瞬间填满 GPU 内存
        self.obs[:load_size].copy_(data['obs'][:load_size])
        self.actions[:load_size].copy_(data['actions'][:load_size])
        self.rewards[:load_size].copy_(data['rewards'][:load_size])
        self.next_obs[:load_size].copy_(data['next_obs'][:load_size])
        self.dones[:load_size].copy_(data['dones'][:load_size])

        self.current_size = load_size
        self.ptr = load_size % self.capacity

        if 'episode_lengths' in data:
            self.episode_lengths = [int(v) for v in data['episode_lengths'].tolist()]
        else:
            self.episode_lengths = []
            running_len = 0
            for done_flag in data['dones'][:load_size].tolist():
                running_len += 1
                if float(done_flag) >= 0.5:
                    self.episode_lengths.append(running_len)
                    running_len = 0

        print(f"✅ 成功从 {load_path} 导入 {load_size} 条先验伪标签数据！")
