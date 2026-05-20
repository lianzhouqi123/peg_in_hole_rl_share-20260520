import os, sys
import time
import csv
import torch
import numpy as np
import inspect
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULT_ROOT = PROJECT_ROOT / "result"
LOG_ROOT = PROJECT_ROOT / "log"

torch.set_float32_matmul_precision('high')

cuda_stub_paths = [
    "/usr/local/cuda/lib64/stubs",
    "/usr/lib/x86_64-linux-gnu/stubs",
    "/usr/lib/x86_64-linux-gnu",
]
current_lib_path = os.environ.get("LIBRARY_PATH", "")
os.environ["LIBRARY_PATH"] = ":".join(cuda_stub_paths) + ":" + current_lib_path


# 导入 TCN5 版环境与网络
from developsuit.envs.pibot_fc.env_fc_tcnhf2 import Env
from developsuit.DRL_utils.NN.TCN_nn2 import ActorNetwork, PPOValueNetwork
from developsuit.DRL_utils.PPO.Policy import PPO

# 停下torch编译
import torch._dynamo
torch._dynamo.disable()


run_mode = "train"
run_mode = "test"

# 路径设置
# load_file_test = RESULT_ROOT / "warp_train_tcn" / "1002"
load_file_test = RESULT_ROOT / "warp_train_fc" / "142"
# load_name = None
load_name = "_best"
# load_name = "_120"
env_load_path = RESULT_ROOT / "demo_grasp_stock_left_warp" / "state_pre_put_warp_FC_wrench_bias_wide_1e5.pt"

show_mode = "show"
# show_mode = "no_show"

# 测试选项
deterministic_test = False
deterministic_test = True
done_stop = False
# done_stop = True
done_sleep = True
# show_first_episode_wrench_plot = True
show_first_episode_wrench_plot = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. 统一超参数配置 =================
test_episodes = 20
config = {
    'num_envs': 2048,

    # 🌟 【架构升级】：替换原本的 obs_dim，显式定义 TCN 时序结构
    'history_len': 8,          # 5 步历史视野 (50ms)
    'kin_obs_dim': None,
    'force_obs_dim': None,
    'single_obs_dim': None,      # 单帧维度 (12维运动学直通 + 6维力觉TCN)
    'action_dim': None,
    'action_min': np.array([-0.03] * 6 + [50.0] * 6, dtype=np.float32),
    'action_max': np.array([0.03] * 6 + [200.0] * 6, dtype=np.float32),
    'hidden_dim': 256,

    'num_steps': 1000, 
    'step_max': 1500,

    'ppo_epoch': 5, 
    'mini_batch_size': 32768,  # 8 个 batch，共更新 24 次

    # 学习率保持平稳
    'actor_lr': 3e-5,
    'critic_lr': 1e-4,
    'gamma': 0.96,
    'gae_lambda': 0.95,
    'min_lr': 6e-6,
    'entropy_coef': 3e-3,
    'target_kl': 0.05,

    'max_episodes': 300,
    'env_load_path': env_load_path,
    
    'sensor_alpha': 0.11,
    'ctrl_alpha': 0.12,
    'admittance_dt': 0.01,
    'rl_dt': 0.01,

    'if_vise_open_rdm': False,
    "vise_open_dis": 0.0055,
    'if_vise_base_rdm': False,
    'vise_base_pos_range': np.array([[-0.01, 0.01], [-0.01, 0.01], [-5 /180 * np.pi, 5 /180 * np.pi]]),
    "noise_v_linear": 0.0,
    "noise_v_angular": 0.0, # 初始化时，添加的速度噪音

    'obs_noise_range': 0.003,  # 注入正负 2 毫米的最大视觉标定误差
    'camera_jitter_pos': 5e-5,  # 每步都施加
    'obs_ori_noise_range': 0.02,
    'camera_jitter_ori': 5e-4,

    'wrench_noise_std': np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2]),
    'wrench_drift_range': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    'max_delay_steps': 1,

    'controller_gain_mode': 'default',  # 'default' / 'randomize' / 'fixed'
    'controller_gain_rand_enabled': run_mode == "train",  # 兼容旧配置；未显式指定 mode 时才会用到
    'controller_kp_range': [5000.0, 6000.0],
    'controller_damping_ratio_range': [0.32, 0.4],
    'controller_gain_integral_scale': 0.0,
    'controller_kp_fixed': 5000.0,  # 当 mode='fixed' 时生效
    'controller_damping_ratio_fixed': 0.4,  # 当 mode='fixed' 时生效
}
def show_first_episode_wrench_plot_figure(step_ids, xwrench_list, xwrench_clean_list):
    xwrench_array = np.asarray(xwrench_list, dtype=np.float32)
    xwrench_clean_array = np.asarray(xwrench_clean_list, dtype=np.float32)
    labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

    fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=150)
    for dim, label in enumerate(labels):
        ax1.plot(step_ids, xwrench_array[:, dim], label=label, linewidth=1.5)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("xwrench")
    ax1.set_title("First Episode xwrench Trace")
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=3)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=150)
    for dim, label in enumerate(labels):
        ax2.plot(step_ids, xwrench_clean_array[:, dim], label=label, linewidth=1.5)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("xwrench_clean")
    ax2.set_title("First Episode xwrench_clean Trace")
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=3)
    fig2.tight_layout()

    plt.show()

def main():
    # ================= 2. 实例化对象 =================
    # 初始化环境 (注意：请确保 Env 实例化符合你现在的封装，这里默认用你传参的方式)
    env = Env(show_mode=show_mode, config=config)

    # 从环境中读取 TCN 观测结构
    config['obs_dim'] = env.obs_dim
    config['kin_obs_dim'] = env.kin_obs_dim
    config['force_obs_dim'] = env.force_obs_dim
    config['single_obs_dim'] = env.single_obs_dim
    config['history_len'] = env.history_len
    config['action_dim'] = env.action_dim
    config['action_range'] = env.action_range

    # ==========================================
    # 从环境中读取 action_range，并转换为 scale 和 bias
    # scale = (max - min) / 2
    # bias  = (max + min) / 2
    # ==========================================
    action_min, action_max = env.action_range
    action_scale = (action_max - action_min) / 2.0
    action_bias = (action_max + action_min) / 2.0

    # 打印出来确认一下是否正确
    print(f"动作放缩 Scale: {action_scale}")
    print(f"动作偏置 Bias: {action_bias}")

    # 实例化网络
    actor = ActorNetwork(
        kin_dim=config['kin_obs_dim'],
        force_dim=config['force_obs_dim'],
        action_dim=config['action_dim'],
        history_len=config['history_len'],
        hidden_dim=config['hidden_dim'],
        action_scale=action_scale,
        action_bias=action_bias
    )
    critic = PPOValueNetwork(
        kin_dim=config['kin_obs_dim'],
        force_dim=config['force_obs_dim'],
        history_len=config['history_len'],
        hidden_dim=config['hidden_dim']
    )

    # 实例化算法与数据池
    policy = PPO(
        actor=actor,
        critic=critic,
        actor_lr=config['actor_lr'],
        critic_lr=config['critic_lr'],
        ppo_epoch=config['ppo_epoch'],
        mini_batch_size=config['mini_batch_size'],
        device=device,
        use_compile=False
    )

    # ================= 3. 执行模式 =================
    print(f"========== 开始测试模式 (设备: {device}) ==========")
    try:
        policy.load_net(load_file_test, name=load_name)
        policy.actor.action_scale.copy_(
            torch.as_tensor(
                action_scale,
                dtype=policy.actor.action_scale.dtype,
                device=policy.actor.action_scale.device,
            )
        )
        policy.actor.action_bias.copy_(
            torch.as_tensor(
                action_bias,
                dtype=policy.actor.action_bias.dtype,
                device=policy.actor.action_bias.device,
            )
        )
        print(f"✅ 成功加载测试模型: {load_file_test}/actor{load_name if load_name else ''}.pth")
        print(f"已按当前脚本的 action_range 覆盖 Actor 动作缩放: scale[0]={policy.actor.action_scale[0].item():.4f}, bias[0]={policy.actor.action_bias[0].item():.4f}")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    if env_load_path:
        env.load_environment_state(env_load_path)
    
    first_episode_step_ids = []
    first_episode_xwrench = []
    first_episode_xwrench_clean = []

    for episode in range(1, test_episodes + 1):
        env.reset(if_reset_data=True, if_vise_open_rdm=config.get('if_vise_open_rdm'), if_vise_base_rdm=config.get('if_vise_base_rdm'))
        obs = env.get_observation()
        episode_reward = 0.0
        max_steps = env.step_max if hasattr(env, 'step_max') else config.get('num_steps', 500)

        print(f"env.xpose: {env.xpose}, env.qvel: {env.arm_qvel}")

        print(f"\n--- 开始测试回合 {episode}/{test_episodes} ---")

        for step in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).view(
                1, config['history_len'], config['single_obs_dim']
            )
            real_action_t, _, _ = policy.get_actions(obs_tensor, deterministic=deterministic_test)
            action_np = real_action_t.cpu().numpy().squeeze()
            policy_k = np.asarray(action_np[6:12], dtype=np.float32)

            # action_np = np.array([0, 0, 0, 0, 0, 0, 200, 200, 200, 200, 200, 200])

            # ==========================================
            # 【修改】：加入简单的 IK 计算异常保护
            # 防止由于极端网络输出导致的 IK 数学解崩溃中断测试
            # ==========================================
            try:
                next_obs, reward, done, info = env.step(action_np)

            except Exception as e:
                print(f"   [警告] 物理引擎步进异常 (可能是 IK 无解): {e}")
                done = True
                next_obs = obs
                info = {'is_success': False, 'ik_fail': True, 'pos_err': 99.0, 'force_norm': 0.0, 'ori_err': 99.0}
                reward = -100.0

            k_target = np.asarray(info.get('k_target', info.get('k', policy_k)), dtype=np.float32)
            print(
                f"   步数: {step:03d}/{max_steps} | "
                f"policy_k: {np.array2string(policy_k, precision=3)} | "
                f"k_target: {np.array2string(k_target, precision=3)}"
            )

            if episode == 1:
                current_xwrench = np.asarray(env.xwrench_eef_left, dtype=np.float32).copy()
                current_xwrench_clean = np.asarray(env.xwrench_clean, dtype=np.float32).copy()
                first_episode_step_ids.append(step)
                first_episode_xwrench.append(current_xwrench)
                first_episode_xwrench_clean.append(current_xwrench_clean)

            episode_reward += reward
            obs = next_obs

            if done:
                print("-" * 60)
                is_success = info.get('is_success', False)
                final_pos_err = info.get('pos_err', 0.0)
                final_ori_err = info.get('ori_err', 0.0)
                final_force_norm = info.get('force_norm', 0.0)
                if is_success:
                    print(f"✅ 回合 {episode} 成功！总步数: {step + 1}, 累加奖励: {episode_reward:.2f}")
                else:
                    print(f"❌ 回合 {episode} 失败。总步数: {step + 1}, 累加奖励: {episode_reward:.2f}")
                    print(
                        f"   最终状态 -> 位置误差: {final_pos_err:.4f}m, 姿态误差: {final_ori_err:.4f}rad, 接触力: {final_force_norm:.2f}N")

                if info.get('ik_fail', False):
                    print("   失败原因: 触发 IK 无解边界保护")

                if info.get('drop_fail', False):
                    print("   失败原因: 物体掉落 / 抓取力不足")

                if done_sleep:
                    time.sleep(1.0)

                if done_stop:
                    input("按 Enter 键继续下一次测试...")

                break

        if episode == 1 and show_first_episode_wrench_plot and first_episode_step_ids:
            show_first_episode_wrench_plot_figure(
                first_episode_step_ids,
                first_episode_xwrench,
                first_episode_xwrench_clean,
            )

if __name__ == '__main__':
    main()
