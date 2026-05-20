import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# 路径设置 (根据你的实际项目结构调整)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULT_ROOT = PROJECT_ROOT / "result"

torch.set_float32_matmul_precision('high')

cuda_stub_paths = [
    "/usr/local/cuda/lib64/stubs",
    "/usr/lib/x86_64-linux-gnu/stubs",
    "/usr/lib/x86_64-linux-gnu"
]
current_lib_path = os.environ.get("LIBRARY_PATH", "")
os.environ["LIBRARY_PATH"] = ":".join(cuda_stub_paths) + ":" + current_lib_path

# ==========================================
# 导入你的 Warp 环境与网络模块
# ==========================================
from developsuit.DRL_utils.PPO.Policy import PPO
from developsuit.DRL_utils.NN.TCN_nn2 import ActorNetwork, PPOValueNetwork

from developsuit.envs.pibot_fc.env_warp_fc_tcnhf2 import Env

# ==========================================
# 🎯 核心测试参数配置
# ==========================================
# 【极速测试秘籍】：将测试总数和并行环境数设为一样！
# 这样所有测试会同时在 GPU 上跑，几秒钟就能拿到 2048 局的严谨统计结果。
TOTAL_TEST_EPISODES = 500  # 测试总轮数
NUM_ENVS = 500             # 并行环境数 (如果显存不够，可以调小，脚本会自动循环补齐)

load_file_test = RESULT_ROOT / "warp_train_fc" / "142"  # 记得改环境编号
# load_name = None
load_name = "_best"
env_load_path = RESULT_ROOT / "demo_grasp_stock_left_warp" / "state_pre_put_warp_FC_wrench_bias_1e5.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_mode = "test"

config = {
    'num_envs': 2048,

    # 🌟 【架构升级】：替换原本的 obs_dim，显式定义 TCN 时序结构
    'history_len': 8,          # 5 步历史视野 (50ms)
    'kin_obs_dim': None,
    'force_obs_dim': None,
    'single_obs_dim': None,      # 单帧维度 (12维运动学直通 + 6维力觉TCN)
    'action_dim': None,
    'action_min': np.array([-0.01] * 6 + [50.0] * 6, dtype=np.float32),
    'action_max': np.array([0.01] * 6 + [200.0] * 6, dtype=np.float32),
    'hidden_dim': 256,

    'num_steps': 128, 
    'step_max': 2000,

    'ppo_epoch': 5, 
    'mini_batch_size': 32768,  # 8 个 batch，共更新 24 次

    # 学习率保持平稳
    'actor_lr': 3e-5,
    'critic_lr': 1e-4,
    'gamma': 0.96,
    'gae_lambda': 0.95,
    'min_lr': 6e-6,
    'entropy_coef': 3e-3,
    'target_kl': 0.06,

    'max_episodes': 180,
    'save_freq': 600,
    'env_load_path': env_load_path,
    
    'sensor_alpha': 0.11,
    'ctrl_alpha': 0.12,
    'admittance_dt': 0.01,
    'rl_dt': 0.03,

    'if_vise_open_rdm': False,
    "vise_open_dis": 0.0055,
    'if_vise_base_rdm': True,
    'vise_base_pos_range': np.array([[-0.01, 0.01], [-0.01, 0.01], [-5 /180 * np.pi, 5 /180 * np.pi]]),
    "noise_v_linear": 0.0,
    "noise_v_angular": 0.0, # 初始化时，添加的速度噪音

    'obs_noise_range': 0.003,  # 注入正负 2 毫米的最大视觉标定误差
    'camera_jitter_pos': 5e-5,  # 每步都施加
    'obs_ori_noise_range': 0.02,
    'camera_jitter_ori': 5e-4,

    'max_delay_steps': 1,

    # 'controller_gain_mode': 'default',  # 'default' / 'randomize' / 'fixed'
    # 'controller_gain_mode': 'randomize',
    'controller_gain_mode': 'default',
    'controller_gain_rand_enabled': run_mode == "train",  # 兼容旧配置；未显式指定 mode 时才会用到
    'controller_kp_range': [4000.0, 6000.0],
    'controller_damping_ratio_range': [0.32, 0.5],
    'controller_gain_integral_scale': 0.0,
    'controller_kp_fixed': 5000.0,  # 当 mode='fixed' 时生效
    'controller_damping_ratio_fixed': 0.4,  # 当 mode='fixed' 时生效
}

def main():
    print(f"========== 🚀 开始极速并行测试模式 (设备: {device}) ==========")
    print(f"目标测试总数: {TOTAL_TEST_EPISODES} 局 | 当前并行容量: {NUM_ENVS} 局/批")
    
    # 1. 实例化环境
    env = Env(num_envs=NUM_ENVS, show_mode="no_show", config=config)
    env.load_environment_state(config['env_load_path'])

    # 🌟 生成动态的 obs_shape 元组，供后续使用
    config['kin_obs_dim'] = env.kin_obs_dim
    config['force_obs_dim'] = env.force_obs_dim
    config['single_obs_dim'] = env.single_obs_dim
    config['obs_shape'] = (config['history_len'], config['single_obs_dim'])
    config['action_dim'] = env.action_dim
    config['action_range'] = env.action_range

    action_min, action_max = env.action_range
    action_scale = (action_max - action_min) / 2.0
    action_bias = (action_max + action_min) / 2.0

    print(f"动作放缩 Scale: {action_scale[0]} ...")
    print(f"动作偏置 Bias: {action_bias[0]} ...")

    # 1. 🌟 实例化 TCN 网络
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

    # 我们只需要前向推理，所以 lr 等参数随便填，不影响测试。
    # 显式关闭 compile，避免测试脚本被 Triton / torch.compile 卡住。
    policy = PPO(
        actor=actor,
        critic=critic,
        actor_lr=0,
        critic_lr=0,
        ppo_epoch=1,
        mini_batch_size=1,
        device=device,
        use_compile=False,
    )

    # 4. 加载权重
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
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # 设置为评估模式
    policy.prep_rollout()

    # ==========================================
    # 🎮 并行测试主循环
    # ==========================================
    total_successes = 0
    total_fail_reasons = {
        'ik_fail': 0,
        'drop_fail': 0,
        'runaway_fail': 0,
        'timeout_fail': 0,
    }
    step_count = 0

    start_time = time.time()
    total_batches = int(np.ceil(TOTAL_TEST_EPISODES / NUM_ENVS))

    completed_episodes = 0
    for batch_idx in range(total_batches):
        batch_target = min(NUM_ENVS, TOTAL_TEST_EPISODES - completed_episodes)
        eval_mask = torch.zeros(NUM_ENVS, dtype=torch.bool, device=device)
        eval_mask[:batch_target] = True
        retired_mask = torch.zeros(NUM_ENVS, dtype=torch.bool, device=device)
        batch_successes = 0

        env.reset(if_reset_data=True,
                  if_vise_open_rdm=config['if_vise_open_rdm'],
                  if_vise_base_rdm=config['if_vise_base_rdm'])
        flat_obs = env.get_observation()
        obs = flat_obs.view(NUM_ENVS, config['history_len'], config['single_obs_dim'])

        while not torch.all(retired_mask[eval_mask]):
            # 1. 极速推理 (直接输入 3D 时序张量，零 CPU 开销)
            actions, _, _ = policy.get_actions(obs, deterministic=True)

            # 2. 环境步进
            flat_next_obs, reward, done, info = env.step(actions, if_change_k=True)
            next_obs = flat_next_obs.view(NUM_ENVS, config['history_len'], config['single_obs_dim'])
            step_count += 1

            done_mask = done.bool()
            new_done_mask = done_mask & eval_mask & (~retired_mask)

            if new_done_mask.any():
                newly_finished = new_done_mask.sum().item()
                newly_success = info['is_success'][new_done_mask].sum().item()
                batch_successes += newly_success

                ik_fail_count = info['ik_fail'][new_done_mask].sum().item()
                drop_fail_count = info['drop_fail'][new_done_mask].sum().item()
                runaway_fail_count = info.get(
                    'runaway_fail',
                    torch.zeros_like(done_mask, dtype=torch.bool, device=device),
                )[new_done_mask].sum().item()
                timeout_fail_count = newly_finished - newly_success - ik_fail_count - drop_fail_count - runaway_fail_count

                total_fail_reasons['ik_fail'] += ik_fail_count
                total_fail_reasons['drop_fail'] += drop_fail_count
                total_fail_reasons['runaway_fail'] += runaway_fail_count
                total_fail_reasons['timeout_fail'] += timeout_fail_count

                retired_mask = retired_mask | new_done_mask

                # 局部重置这些已经计过分的环境，避免终止状态反复参与后续物理步进
                env.reset(if_reset_data=True,
                          if_vise_open_rdm=config['if_vise_open_rdm'],
                          if_vise_base_rdm=config['if_vise_base_rdm'],
                          env_mask=new_done_mask)
                flat_next_obs = env.get_observation()
                next_obs = flat_next_obs.view(NUM_ENVS, config['history_len'], config['single_obs_dim'])

                current_completed = completed_episodes + retired_mask[eval_mask].sum().item()
                current_total_success = total_successes + batch_successes
                current_sr = (current_total_success / current_completed) * 100
                print(
                    f"[Batch {batch_idx + 1}/{total_batches}] "
                    f"[{current_completed:04d}/{TOTAL_TEST_EPISODES}] 局已测完 | "
                    f"本步新完成: {newly_finished} | 本步成功: {newly_success} | "
                    f"当前成功率: {current_sr:.2f}%"
                )

            obs = next_obs

        completed_episodes += batch_target
        total_successes += batch_successes
        batch_sr = (batch_successes / batch_target) * 100
        print(f"批次 {batch_idx + 1}/{total_batches} 完成 | 批次成功率: {batch_sr:.2f}% ({batch_successes}/{batch_target})")

    # ==========================================
    # 📊 输出最终报告
    # ==========================================
    elapsed_time = time.time() - start_time
    final_success_rate = (total_successes / TOTAL_TEST_EPISODES) * 100

    print("\n" + "="*60)
    print("🏆 测试任务圆满结束！")
    print(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
    print(f"🔁 总物理步数: {step_count} 步")
    print(f"🎯 最终成功率: {final_success_rate:.2f}% ({total_successes}/{TOTAL_TEST_EPISODES})")
    print(
        "📉 失败分布: "
        f"IK {total_fail_reasons['ik_fail']} | "
        f"掉落 {total_fail_reasons['drop_fail']} | "
        f"逃逸 {total_fail_reasons['runaway_fail']} | "
        f"超时 {total_fail_reasons['timeout_fail']}"
    )
    print("="*60 + "\n")

if __name__ == '__main__':
    # 强制不分配过多 CPU 线程，Warp 全在 GPU 上跑
    torch.set_num_threads(1) 
    main()
