import os
import time
import datetime
import csv
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BatchedRewardScaling:
    def __init__(self, num_envs, gamma=0.99):
        self.gamma = gamma
        self.R = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.running_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.running_var = torch.tensor(1.0, dtype=torch.float32, device=device)
        self.count = torch.tensor(1e-4, dtype=torch.float32, device=device)
        self.batch_count = torch.tensor(num_envs, dtype=torch.float32, device=device)

    def __call__(self, reward):
        self.R = self.gamma * self.R + reward
        batch_mean = self.R.mean()
        batch_var = self.R.var(unbiased=False)

        delta = batch_mean - self.running_mean
        tot_count = self.count + self.batch_count
        self.running_mean = self.running_mean + delta * self.batch_count / tot_count
        m_a = self.running_var * self.count
        m_b = batch_var * self.batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * self.batch_count / tot_count
        self.running_var = M2 / tot_count

        # 【防坍缩修复】：强行限制方差下限，绝不让它除以小于 1.0 的数！
        safe_var = torch.clamp(self.running_var, min=1.0)

        self.count = tot_count

        return reward / (torch.sqrt(safe_var) + 1e-8)

    def reset(self, done_mask):
        self.R[done_mask] = 0.0

    def get_para(self):
        return [self.count.item(), self.running_mean.item(), torch.sqrt(self.running_var).item()]

def run_ppo_training(env, ppo, buffer, config):
    max_episodes = config.get('max_episodes', 2000)
    num_steps = config.get('num_steps', env.step_max)
    success_rate_window = config.get('success_rate_window', 20)
    gamma = config.get('gamma', 0.99)
    gae_lambda = config.get('gae_lambda', 0.95)
    min_lr = config.get('min_lr', 1e-5)
    save_dir = config.get('save_dir', './result/ppo_train_warp/')
    save_freq = config.get('save_freq', 50)
    env_load_path = config.get('env_load_path')
    if_vise_open_rdm = config.get('if_vise_open_rdm')
    if_vise_base_rdm = config.get('if_vise_base_rdm')

    # 🌟 关键超参数：时序拆分
    history_len = config['history_len']
    single_obs_dim = config['single_obs_dim']  # 运动学 2 维 + 力觉 6 维

    num_envs = env.num_envs
    os.makedirs(save_dir, exist_ok=True)
    print(f"开始 Warp 并行 PPO 训练 (TCN双流架构)，环境数: {num_envs}，运行在设备: {device}")

    start_time_sec = time.time()
    reward_scal = BatchedRewardScaling(num_envs=num_envs, gamma=gamma)

    env.load_environment_state(env_load_path)
    env.reset(if_reset_data=True, if_vise_open_rdm=if_vise_open_rdm, if_vise_base_rdm=if_vise_base_rdm)

    # 1. 获取初始平铺观测 [num_envs, 40]
    flat_obs = env.get_observation()
    # 2. 🌟 重塑为 3D 序列 [num_envs, 5, 8]
    obs_seq = flat_obs.view(num_envs, history_len, single_obs_dim)

    reward_list = []
    best_avg_success_rate = 0.0
    success_rate_history = []
    reset_counts_history = []

    for episode in range(1, max_episodes + 1):
        ppo.prep_rollout()
        t0 = time.time()

        current_episode_reward = torch.zeros(num_envs, dtype=torch.float32, device=device)
        reset_counts = torch.zeros(6, dtype=torch.long, device=device)

        epoch_done_reward_sum = torch.tensor(0.0, device=device)
        epoch_done_pos_err_sum = torch.tensor(0.0, device=device)
        epoch_done_count = torch.tensor(0, dtype=torch.long, device=device)
 
        for step in range(num_steps):
            # ==========================================
            # 🌟 网络前向：送入 3D 序列 obs_seq
            # ==========================================
            real_action_t, log_prob_t, z_t = ppo.get_actions(obs_seq)
            value_t = ppo.get_values(obs_seq).squeeze(-1)

            # 与环境交互
            flat_next_obs, reward, done, info = env.step(real_action_t)
            
            # 🌟 将下一帧环境状态重塑为 3D 序列
            next_obs_seq = flat_next_obs.view(num_envs, history_len, single_obs_dim)

            time_limit_mask = info.get('time_limit', torch.zeros_like(done, dtype=torch.bool))
            
            # Critic 评估下一帧序列
            all_next_values = ppo.get_values(next_obs_seq).squeeze(-1)
            time_limit_value_t = torch.where(time_limit_mask, all_next_values, torch.zeros_like(value_t))
            
            scaled_reward = reward_scal(reward)

            # ==========================================
            # 🌟 存入 Buffer 的数据直接是 3D 序列形式！
            # 这样后续 ppo.train(buffer) 时就不需要再频繁 Reshape 了
            # ==========================================
            buffer.insert(
                obs=obs_seq, critic_obs=obs_seq, action=real_action_t,
                log_prob=log_prob_t, z=z_t, value=value_t, reward=scaled_reward, done=done,
                time_limit=time_limit_mask, time_limit_value=time_limit_value_t
            )

            current_episode_reward += reward
            timeout_mask = time_limit_mask

            # GPU 端统计：将 done 环境互斥归类，避免重复计数并补上 runaway_fail。
            success_done = done & info['is_success']
            unresolved_done = done & (~success_done)

            ik_done = unresolved_done & info['ik_fail']
            unresolved_done = unresolved_done & (~ik_done)

            drop_done = unresolved_done & info['drop_fail']
            unresolved_done = unresolved_done & (~drop_done)

            runaway_fail = info.get('runaway_fail', torch.zeros_like(done, dtype=torch.bool))
            runaway_done = unresolved_done & runaway_fail
            unresolved_done = unresolved_done & (~runaway_done)

            timeout_done = unresolved_done & timeout_mask
            unresolved_done = unresolved_done & (~timeout_done)

            other_done = unresolved_done

            reset_counts[0] += success_done.sum()
            reset_counts[1] += ik_done.sum()
            reset_counts[2] += drop_done.sum()
            reset_counts[3] += runaway_done.sum()
            reset_counts[4] += timeout_done.sum()
            reset_counts[5] += other_done.sum()

            epoch_done_reward_sum += torch.where(done, current_episode_reward, 0.0).sum()
            if 'pos_err' in info:
                valid_err = torch.nan_to_num(info['pos_err'], nan=0.0)
                epoch_done_pos_err_sum += torch.where(done, valid_err, 0.0).sum()
            epoch_done_count += done.sum()

            current_episode_reward = torch.where(done, 0.0, current_episode_reward)
            reward_scal.reset(done)

            # 隐式环境复位
            env.reset(if_reset_data=True, if_vise_open_rdm=if_vise_open_rdm, if_vise_base_rdm=if_vise_base_rdm, env_mask=done)
            
            # 重新获取平铺观测并重塑
            flat_obs = env.get_observation()
            obs_seq = flat_obs.view(num_envs, history_len, single_obs_dim)

        # ==========================================
        # 回合结束：使用最后的 obs_seq 算 GAE
        # ==========================================
        # 将回合末统计合并为一次 CPU 同步，减少多次 .item() / .cpu() 带来的 host 等待
        episode_stats_cpu = torch.cat([
            reset_counts.to(torch.float32),
            torch.stack([
                epoch_done_count.to(torch.float32),
                epoch_done_reward_sum,
                epoch_done_pos_err_sum
            ])
        ]).cpu()

        counts_cpu = episode_stats_cpu[:6].to(torch.long).numpy()
        count_val = int(episode_stats_cpu[6].item())
        if count_val > 0:
            avg_reward = (episode_stats_cpu[7] / count_val).item()
            avg_pos_err = (episode_stats_cpu[8] / count_val).item()
            reward_list.append(avg_reward)
        else:
            reward_list.append(reward_list[-1] if len(reward_list) > 0 else 0.0)
            avg_pos_err = float('nan')

        next_value_t = ppo.get_values(obs_seq).squeeze(-1)
        buffer.compute_returns_and_advantages(next_value=next_value_t, gamma=gamma, gae_lambda=gae_lambda)
        
        ppo.prep_training()
        ppo.train(buffer)
        buffer.clear()
        ppo.lr_decay(episode, max_episodes, min_lr=min_lr)

        reset_counts_history.append(counts_cpu.copy())
        recent_counts_cpu = np.sum(reset_counts_history[-success_rate_window:], axis=0)

        log_training_progress(episode, max_episodes, time.time() - t0, start_time_sec,
                              reward_list, recent_counts_cpu, avg_pos_err)

        total_resets = np.sum(recent_counts_cpu)
        if total_resets > 0:
            current_success_rate = recent_counts_cpu[0] / total_resets
            success_rate_history.append(current_success_rate)
            recent_rates = success_rate_history[-success_rate_window:]
            
            if len(recent_rates) == success_rate_window:
                avg_success_rate = np.mean(recent_rates)
                if (episode > max_episodes / 3) and (avg_success_rate >= best_avg_success_rate) and (avg_success_rate > 0.6):
                    best_avg_success_rate = avg_success_rate
                    ppo.save_net(save_dir, name="_best")
                    print(
                        f"🌟 [模型保存] 突破 1/3 预热期！发现新高平均成功率: "
                        f"{avg_success_rate*100:.1f}% (近{success_rate_window}局)，已保存至 best_actor.pth"
                    )

        if episode % save_freq == 0:
            ppo.save_net(save_dir, name=f"_{episode}")
            with open(os.path.join(save_dir, 'reward.csv'), 'w', newline='') as f:
                csv.writer(f).writerow(reward_list)

    return reward_list, "训练完成"

def log_training_progress(episode, max_episodes, t_cost, train_start_time,
                          reward_list, counts_cpu, avg_pos_err):
    
    # 提取当期(最新的)真实数据透视
    avg_reward = reward_list[-1]
    
    # 统计本局死因占比
    total_resets = np.sum(counts_cpu)
    if total_resets > 0:
        p_succ = counts_cpu[0] / total_resets * 100
        p_ik = counts_cpu[1] / total_resets * 100
        p_drop = counts_cpu[2] / total_resets * 100
        p_runaway = counts_cpu[3] / total_resets * 100
        p_time = counts_cpu[4] / total_resets * 100
        p_other = counts_cpu[5] / total_resets * 100
        reason_str = (
            f"[成{p_succ:.0f}% IK{p_ik:.0f}% 掉{p_drop:.0f}% "
            f"跑{p_runaway:.0f}% 时{p_time:.0f}% 其{p_other:.0f}%]"
        )
    else:
        reason_str = "[无重置]"

    # ==========================================
    # 时间与 ETA 计算
    # ==========================================
    current_time_dt = datetime.datetime.now()
    current_time_str = current_time_dt.strftime("%H:%M:%S")

    elapsed_sec = time.time() - train_start_time
    avg_time_per_epi = elapsed_sec / episode
    remaining_sec = avg_time_per_epi * (max_episodes - episode)

    eta_dt = current_time_dt + datetime.timedelta(seconds=remaining_sec)
    eta_str = eta_dt.strftime("%H:%M:%S")
    rem_time_str = str(datetime.timedelta(seconds=int(remaining_sec)))

    # 将 ETA 和剩余时间加回打印面板，并保持科学账单和死因
    pos_err_str = f"{avg_pos_err:.6f}" if not np.isnan(avg_pos_err) else "nan"

    print(f"[{current_time_str}] Epi {episode}/{max_episodes} | "
          f"Time: {t_cost:.2f}s | ETA: {eta_str} (剩余 {rem_time_str}) | "
          f"Rwd: {avg_reward:6.1f} | PosErr: {pos_err_str} | "
          f"死因:{reason_str}")
    
