import os
import sys
import time
import csv
import torch
import inspect
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULT_ROOT = PROJECT_ROOT / "result"
LOG_ROOT = PROJECT_ROOT / "log"

torch.set_float32_matmul_precision('high')

cuda_stub_paths = [
    "/usr/local/cuda/lib64/stubs",
    "/usr/lib/x86_64-linux-gnu/stubs",
    "/usr/lib/x86_64-linux-gnu"
]
current_lib_path = os.environ.get("LIBRARY_PATH", "")
os.environ["LIBRARY_PATH"] = ":".join(cuda_stub_paths) + ":" + current_lib_path

# ==========================================
# 导入全新的全并行 GPU/Warp 模块
# ==========================================
from developsuit.envs.pibot_fc.env_warp_fc_tcnhf2_3 import Env
from developsuit.DRL_utils.NN.TCN_nn2 import ActorNetwork, PPOValueNetwork
from developsuit.DRL_utils.PPO.Policy import PPO
from developsuit.DRL_utils.Buffer.buffer_tcn import PPOBufferWarp
from developsuit.DRL_utils.runner.runner_tcn_warp import run_ppo_training
from testfile.plot_reward import plot_reward


run_mode = "train"
# run_mode = "test"

# 路径设置
save_file = RESULT_ROOT / "warp_train_fc" / "101"
load_file_train = None
load_name = None
env_load_path = RESULT_ROOT / "demo_grasp_stock_left_warp" / "state_pre_put_warp_FC_wrench_bias_wide_1e5.pt"

show_mode = "no_show"  # 强烈建议并行训练时关闭渲染以获取极限速度

log_file_all = LOG_ROOT / "warp_train_fc"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. 统一超参数配置 =================
config = {
    'num_envs': 1024 * 2,

    # 🌟 【架构升级】：替换原本的 obs_dim，显式定义 TCN 时序结构
    'history_len': 8,          # 5 步历史视野 (50ms)
    'kin_obs_dim': None,
    'force_obs_dim': None,
    'single_obs_dim': None,      # 单帧维度 (12维运动学直通 + 6维力觉TCN)
    'action_dim': None,
    'action_min': np.array([-0.03] * 6 + [50.0] * 6, dtype=np.float32),
    'action_max': np.array([0.03] * 6 + [200.0] * 6, dtype=np.float32),
    'hidden_dim': 256,

    'num_steps': 512,
    'step_max': 1024,

    'ppo_epoch': 5, 
    'mini_batch_size': 2 ** 17,  # 2048*512 下仍保持 8 个 batch，提升长时序稳定性

    # 学习率保持平稳
    'actor_lr': 3e-5,
    'critic_lr': 1e-4,
    'gamma': 0.99,
    'gae_lambda': 0.97,
    'min_lr': 6e-6,
    'entropy_coef': 1e-3,
    'target_kl': 0.04,
    'success_rate_window': 20,  # 保存 best 模型时统计近多少个训练回合的成功率

    'max_episodes': 1,
    'save_freq': 1000,
    'save_dir': save_file,
    'env_load_path': env_load_path,
    
    'sensor_alpha': 0.11,
    'ctrl_alpha': 0.12,
    'admittance_dt': 0.01,
    'rl_dt': 0.01,

    'if_vise_open_rdm': False,
    "vise_open_dis": 0.0055,
    'if_vise_base_rdm': True,
    'vise_base_pos_range': np.array([[-0.01, 0.01], [-0.01, 0.01], [-5 /180 * np.pi, 5 /180 * np.pi]]),
    "noise_v_linear": 0.0,
    "noise_v_angular": 0.0, # 初始化时，添加的速度噪音

    'obs_noise_range': 0.003,  # 0.01 小动作下先把观测噪声收窄，避免信噪比过低
    'camera_jitter_pos': 5e-5,  # 每步都施加
    'obs_ori_noise_range': 0.02,
    'camera_jitter_ori': 5e-4,

    'max_delay_steps': 1,

    'controller_gain_rand_enabled': False, #run_mode == "train",  # 训练时随机化底层关节控制器 kp / damping_ratio
    'controller_kp_range': [4000.0, 6000.0],  # 围绕默认 kp=6500 做保守扰动，避免初始状态过冲
    'controller_damping_ratio_range': [0.32, 0.5],  # 轻度扰动阻尼比，兼顾鲁棒性与稳定性
    'controller_gain_integral_scale': 0.0,  # reset 后增益变化时将旧积分项清零，避免第一拍冲击

    'joint_target_rand_enabled': run_mode == "train",  # 是否开启关节目标执行随机化，训练开/测试关
    'joint_target_delay_steps_range': [0, 2],  # 小动作训练先减轻执行延迟，否则容易把有效动作吞掉
    'joint_target_tau_range': [0.0, 0.04],  # 收窄目标跟踪时间常数，降低 10ms 小动作下的过慢响应
    'joint_target_scale_range': [0.99, 1.01],  # 缩小命令比例误差，保留 sim2real 扰动但不过度破坏信号
    'joint_target_bias_range': [-0.002, 0.002],  # 小动作阶段减小目标零偏，避免静态偏差占比过大
    'joint_target_vel_limit_range': [3.0, 6.0],  # 提高最小跟踪速度上限，减少执行侧“跟不上”
    'joint_target_deadband_range': [0.0, 0.001],  # 减小死区，避免 0.01 幅值下动作落入不响应区
    'joint_target_noise_std': 1e-4,  # 先降低执行噪声，等收敛后再逐步加回去
}

def main():
    save_file.mkdir(parents=True, exist_ok=True)

    # ================= 2. 实例化对象 =================
    print(f"正在初始化 Warp 环境，并行数量: {config.get('num_envs')}...")
    env = Env(num_envs=config.get('num_envs'), show_mode=show_mode, config=config)

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

    # 1. 🌟 实例化 TCN 双流网络 (传入 single_obs_dim)
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

    # 2. 实例化算法
    policy = PPO(
        actor=actor,
        critic=critic,
        actor_lr=config['actor_lr'],
        critic_lr=config['critic_lr'],
        ppo_epoch=config['ppo_epoch'],
        mini_batch_size=config['mini_batch_size'],
        entropy_coef=config['entropy_coef'],
        target_kl=config['target_kl'],
        device=device
    )

    if run_mode == "train":
        if load_file_train is not None:
            policy.load_net(load_file_train, name=load_name)
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
            print(f"✅ 成功加载初始模型: {load_file_train}/actor{load_name if load_name else ''}.pth")
            print(f"已按当前脚本的 action_range 覆盖 Actor 动作缩放: scale[0]={policy.actor.action_scale[0].item():.4f}, bias[0]={policy.actor.action_bias[0].item():.4f}")
        else:
            print("未指定初始模型，将从随机初始化开始训练。")
        
    print("正在使用 Triton 编译器融合神经网络算子，请稍候...")
    policy.actor = torch.compile(policy.actor)
    policy.critic = torch.compile(policy.critic)

    # 3. 🌟 实例化升级版的 3D 张量并行 Buffer (传入 obs_shape)
    buffer = PPOBufferWarp(
        num_steps=config['num_steps'],
        num_envs=config['num_envs'],
        obs_shape=config['obs_shape'],          # 使用元组形状 (5, 18)
        critic_obs_shape=config['obs_shape'],   # 使用元组形状 (5, 18)
        action_dim=config['action_dim']
    )

    # ================= 3. 执行模式 =================
    if run_mode == "train":
        print(f"========== 开始 Warp TCN 训练模式 (设备: {device}) ==========")

        # 调用基于全张量的 Runner
        reward_list, train_info = run_ppo_training(env=env, ppo=policy, buffer=buffer, config=config)

        ###########################################################
        # 存模型
        policy.save_net(save_file)
        print(f"训练完成，最终模型已保存至: {save_file}")
        with open(save_file / "reward.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(reward_list)

        ##########################################################
        # 画图
        fig_all = []
        fig_name_all = []

        try:
            fig = plot_reward(save_file, if_save=False, show_mode="no_show")
            fig_all.append(fig)
            fig_name_all.append("rewards")
        except Exception as e:
            print(f"画图失败，可能是没有安装 matplotlib 或后端不支持: {e}")

        ##########################################################
        # 生成日志
        file_info = {
            "demo_file_path": __file__,
            "train_env_file_path": inspect.getfile(Env),
            "network_file_path": inspect.getfile(ActorNetwork),
            "policy_file_path": inspect.getfile(PPO),
            "Buffer_file_path": inspect.getfile(PPOBufferWarp),
            "runner_file_path": inspect.getfile(run_ppo_training),
            "init_model_file_path": load_file_train,
            "save_file_path": save_file,
        }

        log_train(file_info, train_info, fig_all, fig_name_all)


def log_train(file_info, train_info, fig_all, fig_name_all):
    text = ""
    text += train_info
    text += "\n#########################################################################\n\n"
    text += "训练使用的文件（内容可能在后续训练中修改，文件名仅供参考）\n\n"

    for key in file_info:
        text += key + " : " + str(file_info[key]) + "\n"
    text += "\n#########################################################################\n\n"
    text += "训练使用的超参数\n\n"

    for key in config:
        text += key + " : " + str(config[key]) + "\n"
    text += "\n#########################################################################\n\n"

    env_src_path = Path(file_info["train_env_file_path"])
    if env_src_path.exists():
        text += f"环境源码：{env_src_path}\n\n"
        text += env_src_path.read_text(encoding="utf-8")
        text += "\n\n#########################################################################\n\n"

    log_time = datetime.now()
    formatted_log_time = log_time.strftime("%Y-%m-%d %H-%M-%S")
    demo_file_name = os.path.basename(file_info["demo_file_path"])[:-3]
    log_file_name = demo_file_name + "【" + formatted_log_time + "】"

    log_file_dic_path = log_file_all / log_file_name
    log_file_dic_path.mkdir(parents=True, exist_ok=True)

    log_file_path = log_file_dic_path / f"{log_file_name}.txt"

    with open(log_file_path, 'w', encoding='utf-8') as file:
        file.write(text)

    for ii in range(len(fig_all)):
        fig = fig_all[ii]
        fig_name = fig_name_all[ii]
        fig_path = log_file_dic_path / f"{fig_name}.png"
        fig.savefig(fig_path)


if __name__ == '__main__':
    main()
