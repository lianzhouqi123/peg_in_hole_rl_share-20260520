# Peg-in-Hole RL Share

这是一个用于分享的精简版强化学习训练项目，核心入口为
`train/Train_fc_tcnhf2_warp_3.py`。脚本使用 MuJoCo / MuJoCo Warp 构建并行仿真环境，并使用 PPO + TCN 双流网络训练插孔装配任务中的力控策略。

## 项目内容

- `train/Train_fc_tcnhf2_warp_3.py`：Warp 并行 PPO 训练入口。
- `developsuit/envs/pibot_fc/env_warp_fc_tcnhf2_3.py`：插孔任务环境与导纳控制逻辑。
- `developsuit/DRL_utils/NN/TCN_nn2.py`：TCN Actor / Critic 网络。
- `developsuit/DRL_utils/PPO/Policy.py`：PPO 策略与模型保存加载。
- `developsuit/DRL_utils/Buffer/buffer_tcn.py`：TCN 序列观测的 PPO buffer。
- `developsuit/DRL_utils/runner/runner_tcn_warp.py`：训练循环。
- `testfile/plot_reward.py`：训练奖励曲线绘制工具。

## 环境依赖

# 1. 克隆仓库并进入目录（与原命令一致）
git clone https://github.com/google-deepmind/mujoco_warp.git
cd mujoco_warp

# 2. 使用 conda 创建虚拟环境（假设使用 Python 3.11，可根据需求调整）
conda create -n mujoco_warp_rl python=3.11 -y

# 3. 激活虚拟环境
conda activate mujoco_warp_rl

# 4. 升级 pip（与原命令一致）
pip install --upgrade pip

# 5. 安装项目依赖（包括 dev 和 cuda 依赖）
# 注意：这里直接用 pip 替代 uv pip，conda 环境下 pip 可正常使用
pip install -e ".[dev,cuda]"

# 再安装pytorch和matplotlib
# 以往测试中，118比更高版本更快，因此选用。可再验证新高版本
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install matplotlib
```

如果 `mujoco-warp` 的安装方式随版本变化，请优先参考 MuJoCo Warp 官方文档。

## 运行训练

在项目根目录执行：

```bash
python train/Train_fc_tcnhf2_warp_3.py
```

脚本会自动把项目根目录加入 `PYTHONPATH`，通常不需要额外设置。若从其他目录运行，可以手动指定：

```bash
PYTHONPATH=. python train/Train_fc_tcnhf2_warp_3.py
```

## 关键配置

主要超参数集中在 `train/Train_fc_tcnhf2_warp_3.py` 的 `config` 字典中：

- `num_envs`：并行环境数量，默认 `2048`。
- `history_len`：TCN 使用的历史观测长度，默认 `8`。
- `action_min` / `action_max`：策略动作范围，前 6 维为速度指令，后 6 维为导纳刚度参数。
- `num_steps`：每轮 PPO 收集的仿真步数。
- `step_max`：单个环境 episode 的最大步数。
- `max_episodes`：训练轮数。
- `actor_lr` / `critic_lr`：Actor 和 Critic 学习率。
- `entropy_coef`：策略熵正则系数。
- `target_kl`：PPO 更新的 KL 约束阈值。
- `if_vise_base_rdm`、`joint_target_rand_enabled` 等参数用于域随机化。

默认脚本会从下面的环境状态文件恢复初始场景：

```text
result/demo_grasp_stock_left_warp/state_pre_put_warp_FC_wrench_bias_wide_1e5.pt
```

该文件属于运行产物，默认被 `.gitignore` 忽略。若分享仓库中不包含它，使用者需要自行生成该状态文件，或修改 `env_load_path` 指向已有的环境状态。

## 输出结果

训练结果默认保存到：

```text
result/warp_train_fc/101/
```

其中通常包括：

- `actor.pth`：最终 Actor 模型。
- `critic.pth`：最终 Critic 模型。
- `actor_best.pth` / `critic_best.pth`：满足成功率条件时保存的 best 模型。
- `reward.csv`：训练奖励记录。

训练日志默认保存到：

```text
log/warp_train_fc/
```

日志目录会按脚本名和时间戳命名，并保存本次训练的超参数、相关源码路径、环境源码快照和奖励曲线图片。

## GitHub 初始化

如果这是一个全新的分享仓库，可以在项目根目录执行：

```bash
git add -A
git commit -m "Initial share version"
git remote add origin https://github.com/<your-name>/peg_in_hole_rl_share-20260520.git
git push -u origin main
```

建议 GitHub 仓库名使用 `peg_in_hole_rl_share-20260520`，避免文件夹名中的方括号带来命令行转义问题。
