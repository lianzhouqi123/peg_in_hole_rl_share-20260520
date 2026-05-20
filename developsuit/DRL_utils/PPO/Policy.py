import torch
import torch.nn as nn
from pathlib import Path


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr, min_lr=0.0):
    """
    内置的线性学习率衰减函数
    """
    # 计算当前应该使用的学习率
    lr = initial_lr - (initial_lr - min_lr) * (epoch / float(total_num_epochs))
    lr = max(lr, min_lr)  # 防止衰减过头

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PPO:
    def __init__(self, actor, critic, actor_lr, critic_lr, ppo_epoch, mini_batch_size, entropy_coef=1e-3, target_kl=0.015, device="cpu", seed=None, use_compile=True):
        self.device = device
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.use_compile = use_compile

        # 网络在初始化时应该已经在外部被放到了 device 上 (我们在 nn1.py 里写了 self.to(device))
        self.actor = actor
        self.critic = critic

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=1e-5, weight_decay=0)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=1e-5, weight_decay=0)

        self.ppo_epoch = ppo_epoch
        self.mini_batch_size = mini_batch_size
        self.clip_param = 0.2
        self.num_mini_batch = 1
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl
        self.max_grad_norm = 10

        self.v_loss_epoch = torch.zeros(1, device=self.device)
        self.p_loss_epoch = torch.zeros(1, device=self.device)

        if seed is not None:
            torch.manual_seed(seed)

    def lr_decay(self, episode, episodes, min_lr=0.0):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.actor_lr, min_lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr, min_lr)
    
    # ==========================================
    # 🌟 内部核心函数：被 CUDA Graph 完全接管，极致加速
    # ==========================================
    @torch.compile()
    def _compiled_get_actions_core(self, obs, deterministic):
        with torch.no_grad():
            return self.actor(obs, deterministic)

    @torch.compile()
    def _compiled_get_values_core(self, critic_obs):
        with torch.no_grad():
            return self.critic(critic_obs)

    # ==========================================
    # 🌟 外部公共接口：不要加装饰器！负责安全的数据搬运
    # ==========================================
    def get_actions(self, obs, deterministic=False):
        """
        与环境交互时调用。
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # 测试或调试阶段可关闭 compile，避开 Triton / CUDA 链接依赖。
        if self.use_compile:
            raw_actions, raw_log_probs, raw_z = self._compiled_get_actions_core(obs, deterministic)
        else:
            with torch.no_grad():
                raw_actions, raw_log_probs, raw_z = self.actor(obs, deterministic)

        # 2. 🌟 【金蝉脱壳】：在编译区之外显式分配动态张量并 copy_，避免直接暴露静态缓冲
        actions = torch.empty_like(raw_actions)
        log_probs = torch.empty_like(raw_log_probs)
        zs = torch.empty_like(raw_z)
        actions.copy_(raw_actions)
        log_probs.copy_(raw_log_probs)
        zs.copy_(raw_z)
        return actions, log_probs, zs

    def get_values(self, critic_obs):
        """
        获取状态价值。
        """
        if self.use_compile:
            raw_values = self._compiled_get_values_core(critic_obs)
        else:
            with torch.no_grad():
                raw_values = self.critic(critic_obs)
        
        # 2. 🌟 编译区外显式 copy_ 到动态张量
        values = torch.empty_like(raw_values)
        values.copy_(raw_values)
        return values

    def evaluate_actions(self, obs, z):
        """
        PPO 更新时调用。
        输入：obs, z (来自 Buffer，均已经是 device 上的 Tensor)
        输出：action_log_probs, dist_entropy (纯 Tensor)
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, z)
        return action_log_probs, dist_entropy

    def cal_value_loss(self, values, return_batch):
        # 确保 values 的 shape 从 [batch, 1] 压平为 [batch] 以对齐 return_batch
        value_loss = nn.functional.mse_loss(return_batch, values.squeeze(-1))
        return value_loss

    def ppo_update(self, sample):
        critic_obs_b, obs_b, actions_b, return_b, old_action_log_probs_b, z_b, adv_targ = sample

        action_log_probs, dist_entropy = self.evaluate_actions(obs_b, z_b)

        # 1. 算出 log_ratio 并转换为重要性权重
        log_ratio = action_log_probs - old_action_log_probs_b
        imp_weights = torch.exp(log_ratio)

        # ==========================================
        # 【新增】：计算近似 KL 散度 (无梯度上下文)
        # ==========================================
        with torch.no_grad():
            approx_kl = torch.mean((imp_weights - 1) - log_ratio)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        policy_loss = -torch.min(surr1, surr2).mean() - dist_entropy * self.entropy_coef

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        values = self.critic(critic_obs_b)
        value_loss = self.cal_value_loss(values, return_b)

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # 【极致优化】：返回时多带上一个 KL 散度张量，保持在 GPU 上
        return value_loss.detach(), policy_loss.detach(), approx_kl.detach()

    def train(self, buffer): # 设定工业标准的早停阈值 0.015
        # 初始化记录张量在 GPU 上
        self.v_loss_epoch.zero_()
        self.p_loss_epoch.zero_()
        count = 0

        # 【逻辑修复】：根据当前 buffer 数据量动态计算需要划分的 mini_batch 数量
        total_samples = buffer.step * buffer.num_envs
        actual_num_mini_batch = total_samples // self.mini_batch_size

        early_stop = False # 早停标志

        for epoch in range(self.ppo_epoch):
            # 如果内层循环触发了早停，外层 epoch 循环也立刻终止
            if early_stop:
                break

            data_generator = buffer.mpl_generator(actual_num_mini_batch, self.mini_batch_size)

            for sample in data_generator:
                value_loss, policy_loss, approx_kl = self.ppo_update(sample)

                # ==========================================
                # 【终极防塌补丁】：Target KL 拦截机制
                # ==========================================
                # 这里需要一次 host 同步来驱动早停控制流，避免重复 .item()
                kl_value = approx_kl.item()
                if kl_value > self.target_kl:
                    print(f"⚠️ [Epoch {epoch}] 触发 KL 早停 (KL: {kl_value:.4f} > {self.target_kl})，立刻中止更新以保护网络！")
                    early_stop = True
                    break # 跳出当前的 mini_batch 更新循环

                self.v_loss_epoch += value_loss
                self.p_loss_epoch += policy_loss
                count += 1

        # 【极致优化】：几百次网络反向传播完毕后，只在最后返回时做仅有的一次 CPU 同步！
        # 注意：如果第一波更新就触发了早停，count 可能为 0，你原本的兜底逻辑写得非常好。
        avg_v_loss = (self.v_loss_epoch / count).item() if count > 0 else 0.0
        avg_p_loss = (self.p_loss_epoch / count).item() if count > 0 else 0.0

        # 返回装在列表里的结果以兼容你原来的 runner
        return [avg_v_loss], [avg_p_loss]

    def prep_training(self):
        self.actor.train()
        self.critic.train()

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()

    def load_net(self, save_file, name=None):
        save_dir = Path(save_file)
        if name is None:
            actor_path = save_dir / "actor.pth"
            critic_path = save_dir / "critic.pth"
        else:
            actor_path = save_dir / f"actor{name}.pth"
            critic_path = save_dir / f"critic{name}.pth"

        # 1. 先把字典加载到内存里
        actor_state_dict = torch.load(actor_path, map_location=self.device, weights_only=True)
        critic_state_dict = torch.load(critic_path, map_location=self.device, weights_only=True)

        # ==========================================
        # 【核心修复】：抹除 torch.compile 带来的 _orig_mod. 前缀
        # 兼容编译保存与非编译保存的模型！
        # ==========================================
        actor_state_dict = {k.replace('_orig_mod.', ''): v for k, v in actor_state_dict.items()}
        critic_state_dict = {k.replace('_orig_mod.', ''): v for k, v in critic_state_dict.items()}

        # 2. 加载清洗后的字典进网络
        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)

    def save_net(self, save_file, name=None):
        save_dir = Path(save_file)
        save_dir.mkdir(parents=True, exist_ok=True)
        if name is None:
            torch.save(self.actor.state_dict(), save_dir / "actor.pth")
            torch.save(self.critic.state_dict(), save_dir / "critic.pth")
        else:
            torch.save(self.actor.state_dict(), save_dir / f"actor{name}.pth")
            torch.save(self.critic.state_dict(), save_dir / f"critic{name}.pth")
