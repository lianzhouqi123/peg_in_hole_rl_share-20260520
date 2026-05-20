# Peg-in-Hole RL Share

这是一个用于分享的精简版强化学习训练项目，核心入口为
`train/Train_fc_tcnhf2_warp_3.py`。脚本使用 MuJoCo / MuJoCo Warp 构建并行仿真环境，并使用 PPO + TCN 双流网络训练插孔装配任务中的力控策略。

## 项目内容

- `train/Train_fc_tcnhf2_warp_3.py`：Warp 并行 PPO 训练入口。
- `train/Train_fc_tcnhf2.py`：单环境可视化 / 测试入口，用于加载模型并显示仿真过程。
- `developsuit/envs/pibot_fc/env_warp_fc_tcnhf2_3.py`：插孔任务环境与导纳控制逻辑。
- `developsuit/DRL_utils/NN/TCN_nn2.py`：TCN Actor / Critic 网络。
- `developsuit/DRL_utils/PPO/Policy.py`：PPO 策略与模型保存加载。
- `developsuit/DRL_utils/Buffer/buffer_tcn.py`：TCN 序列观测的 PPO buffer。
- `developsuit/DRL_utils/runner/runner_tcn_warp.py`：训练循环。
- `testfile/demo_grasp_stock_FC_warp_save_env.py`：Warp 并行生成训练初始状态池。
- `testfile/demo_grasp_stock_FC_save_env.py`：单环境可视化生成 / 调试初始状态。
- `testfile/test_success_rate_fc_tcnhf2_warp.py`：Warp 并行批量测试策略成功率。
- `testfile/plot_reward.py`：训练奖励曲线绘制工具。

## 环境依赖

下面是一组基于 conda 的 MuJoCo Warp 环境配置示例：

```bash
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

## 单环境初始状态生成与调试

主文件：`testfile/demo_grasp_stock_FC_save_env.py`

单环境可视化版本，用于调试抓取流程、观察轨迹，并保存一个 `.npz` 格式的抓取后、插入前初始状态。默认配置中：

- `show_mode = "show"`
- `if_plot = True`
- `if_save_env = True`
- `env_save_path = result/demo_grasp_stock_left/state_pre_put_5cm.npz`

运行命令：

```bash
python testfile/demo_grasp_stock_FC_save_env.py
```

该脚本会额外保存一个 `_target.npz` 文件，记录当前 demo 的 `target_x` 和 `target_q`。

## Warp 批量初始状态生成

主文件：`testfile/demo_grasp_stock_FC_warp_save_env.py`

Warp 并行版本，用于一次性生成大量训练初始状态池。默认配置中：

- `if_save_env = True`
- `total_envs = 100000`
- `batch_size = 10000`
- `env_save_path = result/demo_grasp_stock_left_warp/state_pre_put_warp_FC_wrench_bias_wide2_1e5.pt`

运行命令：

```bash
python testfile/demo_grasp_stock_FC_warp_save_env.py
```

该脚本会批量随机化工件 / 夹具位姿，执行抓取与预放置流程，筛掉 IK 失败、位姿误差过大或夹持力不足的环境，然后用 `torch.save` 保存可供 Warp 环境加载的状态池。

生成好状态文件后，请确认训练脚本中的 `env_load_path` 指向对应文件。例如 `train/Train_fc_tcnhf2_warp_3.py` 当前默认读取：

```text
result/demo_grasp_stock_left_warp/state_pre_put_warp_FC_wrench_bias_wide_1e5.pt
```

如果你生成的文件名不同，需要同步修改 `env_load_path`。

## Warp 并行 PPO 训练

主文件：`train/Train_fc_tcnhf2_warp_3.py`

Warp 并行 PPO 训练入口。脚本会创建 MuJoCo Warp 并行环境，构建 TCN Actor / Critic，使用 PPO 进行训练，并保存模型、奖励记录和训练日志。

在项目根目录执行：

```bash
python train/Train_fc_tcnhf2_warp_3.py
```

脚本会自动把项目根目录加入 `PYTHONPATH`，通常不需要额外设置。若从其他目录运行，可以手动指定：

```bash
PYTHONPATH=. python train/Train_fc_tcnhf2_warp_3.py
```

## 单环境可视化测试

主文件：`train/Train_fc_tcnhf2.py`

单环境可视化 / 测试入口，用于加载训练好的模型并显示仿真过程。默认配置中：

- `run_mode = "test"`
- `show_mode = "show"`
- `load_file_test = result/warp_train_fc/142`
- `load_name = "_best"`

运行命令：

```bash
python train/Train_fc_tcnhf2.py
```

如果模型目录或模型后缀不同，请在 `train/Train_fc_tcnhf2.py` 中修改 `load_file_test` 和 `load_name`。

## Warp 批量成功率评估

主文件：`testfile/test_success_rate_fc_tcnhf2_warp.py`

Warp 并行批量评估入口，用于在 GPU 上快速统计训练后策略的成功率。默认配置中：

- `TOTAL_TEST_EPISODES = 500`
- `NUM_ENVS = 500`
- `load_file_test = result/warp_train_fc/142`
- `load_name = "_best"`
- `env_load_path = result/demo_grasp_stock_left_warp/state_pre_put_warp_FC_wrench_bias_1e5.pt`
- `show_mode = "no_show"`

运行命令：

```bash
python testfile/test_success_rate_fc_tcnhf2_warp.py
```

脚本会加载 `actor_best.pth` / `critic_best.pth`，使用确定性动作进行批量测试，并在终端输出最终成功率以及 IK 失败、掉落、逃逸、超时等失败分布。测试前请按自己的模型目录和初始状态文件修改 `load_file_test`、`load_name` 和 `env_load_path`。

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
