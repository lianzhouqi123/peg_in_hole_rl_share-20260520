from developsuit.assets.robots.pibot.pibot_gripper_left_warp import PiBot_left_warp
from developsuit.utils.rm65_analytical_ik_torch import rm65_analytical_ik_torch
from developsuit.controllers.pibot_warp_controller import PiBot_Warp_Controller
from developsuit.utils.transform_utils_torch import *
import mujoco.viewer
import mujoco
import mujoco_warp as mjw
import warp as wp
import numpy as np
import os
import time
import torch

wp.init()
device = "cuda" if torch.cuda.is_available() else "cpu"


class Zk_Warp_Controller:
    def __init__(self, nworld, timestep):
        self.nworld = nworld
        self.device = device
        self.timestep = timestep

        # 将参数转化为 Tensor 以支持广播 [nworld]
        self.mm = torch.full((nworld,), 1.0, device=device)
        self.b = torch.full((nworld,), 100.0, device=device)
        self.k = torch.full((nworld,), 2000.0, device=device)
        self.b_contact_extra = 500.0

        # 滤波与自适应变量 (每个环境都有独立的状态)
        self.omega = torch.zeros(nworld, device=device)
        self.T_touch_old = torch.zeros(nworld, device=device)
        self.T_touch_filtered = torch.zeros(nworld, device=device)
        self.ddq_last = torch.zeros(nworld, device=device)

        # 接触标志位 [nworld]
        self.flag_touch_current = torch.zeros(nworld, dtype=torch.bool, device=device)

        # 常量设置
        self.force_filter_alpha = 0.4
        self.force_hyst_on = 0.0
        self.force_hyst_off = 0.0
        self.force_gain = 1.0
        self.ddq_smooth_alpha = 1.0

    def update(self, de, fd, left_touch_force, jac_1d, djac_1d, grp_distance, grp_vel):
        # 1. 力信号滤波
        T_touch_raw = torch.sum(torch.abs(left_touch_force), dim=-1) / 2.0 if left_touch_force.dim() > 1 else torch.abs(left_touch_force)
        self.T_touch_filtered = self.force_filter_alpha * T_touch_raw + (1 - self.force_filter_alpha) * self.T_touch_filtered

        # 2. 批量滞回判断 (Hysteresis)
        new_touch = torch.where(
            ~self.flag_touch_current,
            self.T_touch_filtered > self.force_hyst_on,  
            self.T_touch_filtered > self.force_hyst_off  
        )
        self.flag_touch_current = new_touch

        # 3. 计算运动学变量
        d = grp_distance
        dd = jac_1d * grp_vel

        # 4. --- 模式 A: 非接触 ---
        pos_error = d - de
        V_no_touch = -(self.b * dd + self.k * pos_error + fd) / self.mm

        # 5. --- 模式 B: 接触 ---
        b_total = self.b + self.b_contact_extra
        yita = 0.8 * b_total * self.timestep / (b_total * self.timestep + self.mm)
        omega_delta = yita / b_total * (fd - self.T_touch_old)

        # 🛡️ 护盾 1: 修复幽灵积分 (只在真正接触时才允许积分累加)
        self.omega = torch.where(self.flag_touch_current, self.omega + omega_delta, self.omega)
        
        force_error = fd - self.T_touch_filtered
        V_touch = -(b_total * (dd + self.omega) + self.force_gain * force_error) / self.mm

        # 6. 选择最终 V 并翻转方向
        V = torch.where(self.flag_touch_current, V_touch, V_no_touch)
        V = -V 

        # 🛡️ 护盾 2: 阻尼最小二乘法 (DLS) 彻底根除除以 0 的可能
        damping_sq = 1e-4  # 阻尼系数，避免 jac_1d 为 0 时发散
        numerator = V - djac_1d * grp_vel
        ddq_new = (numerator * jac_1d) / (jac_1d**2 + damping_sq)

        # 🛡️ 护盾 3: 绝对数值检疫网 (防止任何漏网的 NaN 或无穷大毒害物理引擎)
        ddq_new = torch.nan_to_num(ddq_new, nan=0.0)

        # 一阶平滑
        ddq = self.ddq_smooth_alpha * ddq_new + (1 - self.ddq_smooth_alpha) * self.ddq_last
        self.ddq_last = ddq.clone()

        # 更新历史力
        self.T_touch_old.copy_(self.T_touch_filtered)

        return ddq

class PiBotEnv:
    def __init__(self, num_envs=1, show_mode="no_show", timestep_control=0.01, config={}):
        self.num_envs = num_envs
        self.show_mode = show_mode
        # g = np.array([0., 0., 0.])
        g = np.array([0., 0., -9.8])
        self._timestep = 0.002
        self.timestep_control = timestep_control
        self.config = config
        self.vise_open_dis_value = torch.tensor(self.config.get("vise_open_dis"), dtype=torch.float32, device=device)

        # 创建机器人
        pibot_name = config.get('pibot_file_name', "pibot_fc.xml")
        self.arm = PiBot_left_warp(name=pibot_name)
        self.mj_model = self.arm.mj_model
        self.mj_data = self.arm.mj_data
        self.mj_model.opt.timestep = self._timestep
        self.mj_model.opt.gravity = g

        # 整合关节
        self.num_joints = len(self.arm.joint_ids)
        self.num_actuators = len(self.arm.actuator_ids)

        # # 2. 核心：推送到 GPU 并开启 batch
        # # mjw_model 是共享的（节省显存），mjw_data 会创建 num_envs 个独立副本
        self.mjw_model = mjw.put_model(self.mj_model)
        self.mjw_data = mjw.put_data(self.mj_model, self.mj_data, nworld=num_envs, njmax=150, nconmax=25)

        self.contact = self.mjw_data.contact

        if self.show_mode == "show":
            # 使用原生的 mj_model 和 mj_data (它们在 CPU 上，将作为渲染载体)
            self._viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            self._viewer.cam.distance = 2.75
            self._viewer.cam.elevation = -20
            self._viewer.cam.azimuth = 80
            self._viewer.cam.lookat = [0., 0.0, 0.5]

            # 初始化一个渲染计时器
            self._step_start = time.time()

        # 创建控制器
        self.controller = PiBot_Warp_Controller(self.num_envs, self.mjw_model, self.mjw_data, self.arm.joint_qpos_adrs,
                                                self.arm.joint_dof_adrs, self.arm.actuator_ids, self.timestep_control,
                                                kp=6500, damping_ratio=0.32, ki=5000, min_effort=-60.0, max_effort=60.0,
                                                device=device)
        self.zk_controller = Zk_Warp_Controller(nworld=self.num_envs, timestep=self.timestep_control)
        self.wrench_eef_left = torch.zeros([self.num_envs, 6], dtype=torch.float32, device=device)
        self.xwrench_eef_left = torch.zeros([self.num_envs, 6], dtype=torch.float32, device=device)

        # ==========================================
        # 🌟 修改点 1：移除旧的 filter_coef，新增传感器物理特性
        # ==========================================
        self.sensor_alpha = self.config.get('sensor_alpha', 0.45)  # 第一道硬件级滤波系数 (约 13Hz)
        
        # 从配置中读取噪声标准差和漂移范围 (支持默认值回退)
        wrench_noise_std = self.config.get('wrench_noise_std', np.array([0., 0., 0., 0.0, 0.0, 0.0]))
        self.has_wrench_noise = bool(np.any(np.asarray(wrench_noise_std) > 0))
        self.wrench_noise_std = torch.tensor(wrench_noise_std, dtype=torch.float32, device=device)
        
        wrench_drift_range = self.config.get('wrench_drift_range', np.array([.0, .0, .0, 0.0, 0.0, 0.0]))
        self.wrench_drift_range = torch.tensor(wrench_drift_range, dtype=torch.float32, device=device)
        
        self.if_wrench_drift_range = torch.linalg.norm(self.wrench_drift_range) > 1e-6
        self.wrench_drift = torch.zeros((self.num_envs, 6), device=device)

        # ==========================================
        # 🌟 核心修改 1：为 500Hz CUDA Graph 预分配高频噪声缓冲区
        # 形状: [num_micro_steps(5), num_envs, 6]
        # ==========================================
        self.num_micro_steps = int(self.timestep_control / self._timestep)  # 0.01 / 0.002 = 5
        self.noise_w_buffer = torch.zeros((self.num_micro_steps, self.num_envs, 6), dtype=torch.float32, device=device)
        self.noise_x_buffer = torch.zeros((self.num_micro_steps, self.num_envs, 6), dtype=torch.float32, device=device)

        # 信息
        self.arm_vel_max_sec = torch.ones([self.num_joints]) * 1

        # 1. 缓存所有底层 Warp 数组的 Torch 视图 (仅创建一次)
        self.qpos_torch = wp.to_torch(self.mjw_data.qpos)
        self.qvel_torch = wp.to_torch(self.mjw_data.qvel)
        self.site_xpos_torch = wp.to_torch(self.mjw_data.site_xpos)
        self.site_xmat_torch = wp.to_torch(self.mjw_data.site_xmat)
        self.sensordata_torch = wp.to_torch(self.mjw_data.sensordata)
        self.ctrl_torch = wp.to_torch(self.mjw_data.ctrl)
        self.qacc_warmstart_torch = wp.to_torch(self.mjw_data.qacc_warmstart)
        self.site_xpos_view = self.site_xpos_torch.view(self.num_envs, -1, 3)
        self.site_xmat_view = self.site_xmat_torch.view(self.num_envs, -1, 3, 3)

        # 2. 预分配雅可比计算所需的 GPU 内存
        nv = self.mj_model.nv
        self.jacp_wp = wp.zeros((self.num_envs, 3, nv), dtype=wp.float32, device=device)
        self.jacr_wp = wp.zeros((self.num_envs, 3, nv), dtype=wp.float32, device=device)
        self.jacp_torch = wp.to_torch(self.jacp_wp)
        self.jacr_torch = wp.to_torch(self.jacr_wp)

        # 预分配 Body ID 张量 (左臂末端和夹爪)
        self.eef_body_torch = torch.full((self.num_envs,), self.mj_model.site_bodyid[self.arm.eef_site_ids[0]],
                                         dtype=torch.int32, device=device)
        self.eef_body_wp = wp.from_torch(self.eef_body_torch, dtype=wp.int32)

        self.grp_body_torch = torch.full((self.num_envs,), self.arm.gripper_left_body_ids[0], dtype=torch.int32,
                                         device=device)
        self.grp_body_wp = wp.from_torch(self.grp_body_torch, dtype=wp.int32)

        self.num_micro_steps = int(self.timestep_control / self._timestep)

        # 4. 预定义数学常量 (避免循环中反复将常量推送到 GPU)
        self.z_axis_torch = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device).view(3, 1)
        self.theta_offset = 42.79 / 180.0 * torch.pi
        self.gripper_L = 0.055

        # 🚀 [新增 1]: 预存所有用于切片 (slice) 的标量索引为 Python 普通 int
        # 严禁在切片边界处使用 GPU Tensor，否则会引发严重的强制 CPU 同步！
        self._f_adr = int(self.arm.F_sensor_left_adrs[0])
        self._t_adr = int(self.arm.T_sensor_left_adrs[0])
        self._gyro_adrs = [int(adr) for adr in self.arm.base_gyro_sensor_adrs]
        self._eef_gyro_adrs = [int(adr) for adr in self.arm.eef_gyro_sensor_adrs]
        self._eef_vel_adrs = [int(adr) for adr in self.arm.eef_vel_sensor_adrs]
        self._eef_site_id = int(self.arm.eef_site_ids[0])
        self._grasp_site_id = int(self.arm.grasp_site_ids[0])
        self._eef_real_site_id = int(self.arm.eef_real_site_ids[0])
        self._base_site_id = int(self.arm.base_site_ids[0])
        self._insert_site_id = int(self.arm.insert_site_ids[0])
        self._insert_world_site_id = int(self.arm.insert_world_site_ids[0])
        self._wrench_site_id = int(self.arm.wrench_site_left_ids[0])
        self._left_gripper_site_id = int(self.arm.gripper_left_site_ids[0])
        self._left_gripper_dof_active = int(self.mj_model.jnt_dofadr[
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "Left_1_Joint_gripper_left")
        ])
        self._left_gripper_dof_support = int(self.mj_model.jnt_dofadr[
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "Left_Support_Joint_gripper_left")
        ])

        # 🚀 [新增 2]: 为 Warp 的物理引擎准备 CUDA Graph
        self._mjw_graph = None
        self._mjw_forward_graph = None
        self.ik_batch_idx = torch.arange(self.num_envs, device=device)

        self.vise_open_dis = torch.empty(self.num_envs, dtype=torch.float32, device=device)
        self.rand_base = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=device)
        self.vise_base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device)
        self.vise_base_pos_range = self.config.get("vise_base_pos_range", np.array([[-0.01, 0.01], [-0.01, 0.01], [0, 0]]))
        self.vise_base_pos_range = torch.tensor(self.vise_base_pos_range, dtype=torch.float32, device=device)
        self.stock_pose_init_default = torch.tensor(
            [0.0, -0.35, 1.15, 0.5, -0.5, -0.5, 0.5],
            dtype=torch.float32,
            device=device,
        )
        self.stock_pose_init_range = self.config.get(
            "stock_pose_init_range",
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        )
        self.stock_pose_init_range = torch.tensor(
            self.stock_pose_init_range,
            dtype=torch.float32,
            device=device,
        )

        # 高频派生状态缓存，减少每步重复 property 解析与四元数转换
        self.xpos_left_cache = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device)
        self.xvel_left_cache = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=device)
        self.xmat_left_cache = torch.zeros((self.num_envs, 3, 3), dtype=torch.float32, device=device)
        self.xpose_left_cache = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=device)
        self.eef_real_pose_cache = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=device)
        self.base_xpos_left_cache = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device)
        self.base_xmat_left_cache = torch.zeros((self.num_envs, 3, 3), dtype=torch.float32, device=device)
        self.grasp_pose_cache = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=device)
        self.insert_pose_cache = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=device)
        self.insert_world_pose_cache = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=device)

        # 将你的 IK 函数编译为极致优化的 CUDA Graph 和融合 Kernel
        self.ikine_analytical_left_torch = torch.compile(self.ikine_analytical_left_torch)

    def _refresh_pose_caches(self):
        eef_pos = self.site_xpos_view[:, self._eef_site_id, :]
        eef_mat = self.site_xmat_view[:, self._eef_site_id, :, :]
        eef_quat = mat2quat(eef_mat)

        eef_real_pos = self.site_xpos_view[:, self._eef_real_site_id, :]
        eef_real_mat = self.site_xmat_view[:, self._eef_real_site_id, :, :]
        eef_real_quat = mat2quat(eef_real_mat)

        grasp_pos = self.site_xpos_view[:, self._grasp_site_id, :]
        grasp_mat = self.site_xmat_view[:, self._grasp_site_id, :, :]
        grasp_quat = mat2quat(grasp_mat)

        base_pos = self.site_xpos_view[:, self._base_site_id, :]
        base_mat = self.site_xmat_view[:, self._base_site_id, :, :]

        insert_pos = self.site_xpos_view[:, self._insert_site_id, :]
        insert_mat = self.site_xmat_view[:, self._insert_site_id, :, :]
        insert_quat = mat2quat(insert_mat)

        insert_world_pos = self.site_xpos_view[:, self._insert_world_site_id, :]
        insert_world_mat = self.site_xmat_view[:, self._insert_world_site_id, :, :]
        insert_world_quat = mat2quat(insert_world_mat)

        self.xpos_left_cache.copy_(eef_pos)
        self.xmat_left_cache.copy_(eef_mat)
        self.xpose_left_cache[:, 0:3].copy_(eef_pos)
        self.xpose_left_cache[:, 3:7].copy_(eef_quat)
        self.eef_real_pose_cache[:, 0:3].copy_(eef_real_pos)
        self.eef_real_pose_cache[:, 3:7].copy_(eef_real_quat)

        self.base_xpos_left_cache.copy_(base_pos)
        self.base_xmat_left_cache.copy_(base_mat)

        self.grasp_pose_cache[:, 0:3].copy_(grasp_pos)
        self.grasp_pose_cache[:, 3:7].copy_(grasp_quat)

        self.insert_pose_cache[:, 0:3].copy_(insert_pos)
        self.insert_pose_cache[:, 3:7].copy_(insert_quat)
        self.insert_world_pose_cache[:, 0:3].copy_(insert_world_pos)
        self.insert_world_pose_cache[:, 3:7].copy_(insert_world_quat)

        # ==========================================
        # 🌟 新增：Sensor 速度的批量坐标系转换 (局部 -> 世界)
        # ==========================================
        # 1. 提取局部坐标系下的线速度与角速度 [num_envs, 3]
        local_linvel = torch.stack([
            self.sensordata_torch[:, adr: adr + 3] for adr in self._eef_vel_adrs
        ], dim=1).squeeze(1) 
        
        local_angvel = torch.stack([
            self.sensordata_torch[:, adr: adr + 3] for adr in self._eef_gyro_adrs
        ], dim=1).squeeze(1)

        # 2. 使用 torch.bmm (Batch Matrix-Matrix Product) 转换到世界坐标系
        # eef_mat 形状: [num_envs, 3, 3]
        # local_linvel.unsqueeze(-1) 形状: [num_envs, 3, 1]
        world_linvel = torch.bmm(eef_mat, local_linvel.unsqueeze(-1)).squeeze(-1) # [num_envs, 3]
        world_angvel = torch.bmm(eef_mat, local_angvel.unsqueeze(-1)).squeeze(-1) # [num_envs, 3]

        # 3. 写入缓存
        self.xvel_left_cache[:, 0:3].copy_(world_linvel)
        self.xvel_left_cache[:, 3:6].copy_(world_angvel)

    def fast_forward(self):
        """利用 Warp 原生 CUDA Graph 彻底消除 mjw.forward 的 CPU 发射开销"""
        if self._mjw_forward_graph is None:
            wp.capture_begin()
            mjw.forward(self.mjw_model, self.mjw_data)
            self._mjw_forward_graph = wp.capture_end()
            
        wp.capture_launch(self._mjw_forward_graph)
        self._refresh_pose_caches()

    def render(self):
        if self.show_mode == "show":
            # 1. 将 Env 0 的状态从 GPU (PyTorch/Warp) 拉回 CPU 的 NumPy 数组
            # 这里利用之前已经映射好的 self.qpos_all 张量
            # 使用 .detach().cpu().numpy() 安全转换为 numpy 数组
            env_0_qpos = self.qpos_torch[0].detach().cpu().numpy()
            env_0_qvel = self.qvel_torch[0].detach().cpu().numpy()

            # 2. 覆盖 CPU 实例的 data
            with self._viewer.lock():
                self.mj_data.qpos[:] = env_0_qpos
                self.mj_data.qvel[:] = env_0_qvel

                # 注意：如果你在仿真中还修改了 mocap (目标点) 或 body 的其他属性，
                # 也要把对应的值从 GPU 提出来赋值给 self.mj_data.mocap_pos 等。

                # 3. 触发一次 CPU 端的正向运动学计算，更新各个 link 的空间位置用于渲染
                mujoco.mj_forward(self.mj_model, self.mj_data)
                # 或者仅调用 mujoco.mj_kinematics(self.mj_model, self.mj_data) 速度更快，只算几何位姿

            # 4. 刷新渲染器画面
            self._viewer.sync()

            # 5. 帧率控制 (保持可视化的真实感，避免鬼畜)
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            # 重置计时器为下一次渲染做准备
            self._step_start = time.time()

    def reset(self, if_reset_data=False, if_vise_open_rdm=False, if_vise_base_rdm=False, env_mask=None):
        if env_mask is None:
            env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=device)

        # 扩充为 2D 掩码，全称无 CPU 介入，保持在 GPU 上极速广播
        mask_2d = env_mask.unsqueeze(-1)

        # ==========================================
        # 独立变量随机化生成
        # ==========================================
        if if_vise_open_rdm:
            # rand_vise = torch.empty(self.num_envs, dtype=torch.float32, device=device).uniform_(0.0, 0.0035)
            # self.vise_open_dis = torch.where(env_mask, rand_vise, self.vise_open_dis)
            raise ValueError("没有维护，请再仔细规划")
        else:
            self.vise_open_dis = torch.where(env_mask, self.vise_open_dis_value, self.vise_open_dis)

        if if_vise_base_rdm:
            rand_base = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=device)
            rand_base[:, 0].uniform_(self.vise_base_pos_range[0][0], self.vise_base_pos_range[0][1])
            rand_base[:, 1].uniform_(self.vise_base_pos_range[1][0], self.vise_base_pos_range[1][1])
            # 第三个自由度是 vise_qz_joint；必须单独赋值，不能残留未初始化显存
            rand_base[:, 2].uniform_(self.vise_base_pos_range[2][0], self.vise_base_pos_range[2][1])
            self.vise_base_pos = torch.where(mask_2d, rand_base, self.vise_base_pos)
        else:
            self.vise_base_pos = torch.where(mask_2d, 0.0, self.vise_base_pos)

        # 在 if_reset_data=True 的情况下，将这里新生成的随机量作为对读入状态的附加偏移
        vise_base_offset = self.vise_base_pos.clone()

        # ==========================================
        # 🌟 修改点 2：为重置的环境生成全新的零点漂移
        # ==========================================
        if self.if_wrench_drift_range:
            new_drift = torch.empty((self.num_envs, 6), device=device).uniform_(-1.0, 1.0) * self.wrench_drift_range
            self.wrench_drift = torch.where(mask_2d, new_drift, self.wrench_drift)

        # ==========================================
        # 物理状态重置与覆盖
        # ==========================================
        if if_reset_data:
            if hasattr(self, 'is_state_pool') and self.is_state_pool:
                sampled_idx = torch.randint(0, self.pool_size, (self.num_envs,), device=device)
                self.sampled_idx = sampled_idx
                
                # 分支 A：.pt 状态池 (极速抽样并用 where 覆盖)
                self.qpos_torch[:] = torch.where(mask_2d, self.state_pool['qpos'][sampled_idx], self.qpos_torch)
                self.qvel_torch[:] = torch.where(mask_2d, self.state_pool['qvel'][sampled_idx], self.qvel_torch)
                self.ctrl_torch[:] = torch.where(mask_2d, self.state_pool['ctrl'][sampled_idx], self.ctrl_torch)
                self.qacc_warmstart_torch[:] = torch.where(mask_2d, self.state_pool['qacc_warmstart'][sampled_idx], self.qacc_warmstart_torch)

                # self.vise_open_dis = torch.where(env_mask, self.state_pool['vise_open_dis'][sampled_idx], self.vise_open_dis)
                loaded_vise_base_pos = self.state_pool['vise_base_pos'][sampled_idx]
                vise_base_pos_tgt = loaded_vise_base_pos + vise_base_offset
                self.vise_base_pos = torch.where(mask_2d, vise_base_pos_tgt, self.vise_base_pos)
                self.qpos_torch[:, self.arm.joint_vise_base_qpos_adrs] = torch.where(
                    mask_2d,
                    vise_base_pos_tgt,
                    self.qpos_torch[:, self.arm.joint_vise_base_qpos_adrs],
                )

                vise_open_exp = self.vise_open_dis.unsqueeze(-1)
                self.qpos_torch[:, self.arm.joint_vise_front_qpos_adrs] = torch.where(mask_2d, vise_open_exp, self.qpos_torch[:, self.arm.joint_vise_front_qpos_adrs])
                self.qpos_torch[:, self.arm.joint_vise_back_qpos_adrs] = torch.where(mask_2d, vise_open_exp, self.qpos_torch[:, self.arm.joint_vise_back_qpos_adrs])

                self.controller.integral[:] = torch.where(mask_2d, self.state_pool['arm_integral'][sampled_idx], self.controller.integral)
                self.wrench_eef_left[:] = torch.where(mask_2d, self.state_pool['wrench_eef_left'][sampled_idx], self.wrench_eef_left)
                self.xwrench_eef_left[:] = torch.where(mask_2d, self.state_pool['xwrench_eef_left'][sampled_idx], self.xwrench_eef_left)
                if 'wrench_drift' in self.state_pool:
                    self.wrench_drift = torch.where(mask_2d, self.state_pool['wrench_drift'][sampled_idx], self.wrench_drift)

                self.zk_controller.omega = torch.where(env_mask, self.state_pool['zk_omega'][sampled_idx], self.zk_controller.omega)
                self.zk_controller.T_touch_old = torch.where(env_mask, self.state_pool['zk_T_touch_old'][sampled_idx], self.zk_controller.T_touch_old)
                self.zk_controller.flag_touch_current = torch.where(env_mask, self.state_pool['zk_flag_touch_current'][sampled_idx], self.zk_controller.flag_touch_current)
                self.zk_controller.T_touch_filtered = torch.where(env_mask, self.state_pool['zk_T_touch_filtered'][sampled_idx], self.zk_controller.T_touch_filtered)
                self.zk_controller.ddq_last = torch.where(env_mask, self.state_pool['zk_ddq_last'][sampled_idx], self.zk_controller.ddq_last)

            elif hasattr(self, 'reset_data') and self.reset_data is not None:
                # 分支 B：.npz 单状态 (利用广播机制批量赋值)
                nq, nv = self.mj_model.nq, self.mj_model.nv
                state_bytes = self.reset_data['state_bytes']

                qpos_tgt = torch.tensor(state_bytes[0:nq], dtype=torch.float32, device=device).expand(self.num_envs, -1)
                self.qpos_torch[:] = torch.where(mask_2d, qpos_tgt, self.qpos_torch)
                
                qvel_tgt = torch.tensor(state_bytes[nq:nq + nv], dtype=torch.float32, device=device).expand(self.num_envs, -1)
                self.qvel_torch[:] = torch.where(mask_2d, qvel_tgt, self.qvel_torch)

                vise_open_exp = self.vise_open_dis.unsqueeze(-1)
                loaded_vise_base_pos = qpos_tgt[:, self.arm.joint_vise_base_qpos_adrs]
                vise_base_pos_tgt = loaded_vise_base_pos + vise_base_offset
                self.vise_base_pos = torch.where(mask_2d, vise_base_pos_tgt, self.vise_base_pos)
                self.qpos_torch[:, self.arm.joint_vise_front_qpos_adrs] = torch.where(mask_2d, vise_open_exp, self.qpos_torch[:, self.arm.joint_vise_front_qpos_adrs])
                self.qpos_torch[:, self.arm.joint_vise_back_qpos_adrs] = torch.where(mask_2d, vise_open_exp, self.qpos_torch[:, self.arm.joint_vise_back_qpos_adrs])
                self.qpos_torch[:, self.arm.joint_vise_base_qpos_adrs] = torch.where(mask_2d, vise_base_pos_tgt, self.qpos_torch[:, self.arm.joint_vise_base_qpos_adrs])

                if 'physics_ctrl' in self.reset_data:
                    ctrl_tgt = torch.tensor(self.reset_data['physics_ctrl'], dtype=torch.float32, device=device).expand(self.num_envs, -1)
                    self.ctrl_torch[:] = torch.where(mask_2d, ctrl_tgt, self.ctrl_torch)
                if 'qacc_warmstart' in self.reset_data:
                    qacc_tgt = torch.tensor(self.reset_data['qacc_warmstart'], dtype=torch.float32, device=device).expand(self.num_envs, -1)
                    self.qacc_warmstart_torch[:] = torch.where(mask_2d, qacc_tgt, self.qacc_warmstart_torch)

                if 'zk_omega' in self.reset_data:
                    self.zk_controller.omega = torch.where(env_mask, float(self.reset_data['zk_omega'].item()), self.zk_controller.omega)
                    self.zk_controller.T_touch_old = torch.where(env_mask, float(self.reset_data['zk_T_touch_old'].item()), self.zk_controller.T_touch_old)
                    self.zk_controller.flag_touch_current = torch.where(env_mask, bool(self.reset_data['zk_flag_touch_current'].item()), self.zk_controller.flag_touch_current)
                    self.zk_controller.T_touch_filtered = torch.where(env_mask, float(self.reset_data['zk_T_touch_filtered'].item()), self.zk_controller.T_touch_filtered)
                    self.zk_controller.ddq_last = torch.where(env_mask, float(self.reset_data['zk_ddq_last'].item()), self.zk_controller.ddq_last)

                if 'arm_integral' in self.reset_data:
                    integ_tgt = torch.tensor(self.reset_data['arm_integral'], dtype=torch.float32, device=device).expand(self.num_envs, -1)
                    self.controller.integral[:] = torch.where(mask_2d, integ_tgt, self.controller.integral)
                if 'wrench_eef_left' in self.reset_data:
                    w_tgt = torch.tensor(self.reset_data['wrench_eef_left'], dtype=torch.float32, device=device).expand(self.num_envs, -1)
                    self.wrench_eef_left[:] = torch.where(mask_2d, w_tgt, self.wrench_eef_left)
                    xw_tgt = torch.tensor(self.reset_data['xwrench_eef_left'], dtype=torch.float32, device=device).expand(self.num_envs, -1)
                    self.xwrench_eef_left[:] = torch.where(mask_2d, xw_tgt, self.xwrench_eef_left)
                if 'wrench_drift' in self.reset_data:
                    drift_tgt = torch.tensor(self.reset_data['wrench_drift'], dtype=torch.float32, device=device).expand(self.num_envs, -1)
                    self.wrench_drift = torch.where(mask_2d, drift_tgt, self.wrench_drift)
            else:
                raise ValueError("未加载任何数据！请确保调用过 load_environment_state")

        else:
            env_mask_wp = wp.from_torch(env_mask.to(torch.bool))
            mjw.reset_data(self.mjw_model, self.mjw_data, reset=env_mask_wp)
            # 分支 C：非导入模式（纯净硬编码初始化）
            arm_init = torch.tensor([0.0, -70.0, 70.0, 0.0, 80.0, 0.0, 0.0], dtype=torch.float32, device=device) * (torch.pi / 180.0)

            self.qpos_torch[:, self.arm.joint_qpos_adrs] = torch.where(mask_2d, arm_init, self.qpos_torch[:, self.arm.joint_qpos_adrs])
            self.qpos_torch[:, self.arm.stock_qpos_adrs] = torch.where(mask_2d, self.stock_pose_init_default, self.qpos_torch[:, self.arm.stock_qpos_adrs])
            self.qpos_torch[:, self.arm.joint_vise_base_qpos_adrs] = torch.where(mask_2d, self.vise_base_pos, self.qpos_torch[:, self.arm.joint_vise_base_qpos_adrs])
            
            vise_open_exp = self.vise_open_dis.unsqueeze(-1)
            self.qpos_torch[:, self.arm.joint_vise_front_qpos_adrs] = torch.where(mask_2d, vise_open_exp, self.qpos_torch[:, self.arm.joint_vise_front_qpos_adrs])
            self.qpos_torch[:, self.arm.joint_vise_back_qpos_adrs] = torch.where(mask_2d, vise_open_exp, self.qpos_torch[:, self.arm.joint_vise_back_qpos_adrs])

            self.qvel_torch[:] = torch.where(mask_2d, 0.0, self.qvel_torch)
            
            self.ctrl_torch[:] = torch.where(mask_2d, 0.0, self.ctrl_torch)
            
            self.qacc_warmstart_torch[:] = torch.where(mask_2d, 0.0, self.qacc_warmstart_torch)

            self.zk_controller.omega = torch.where(env_mask, 0.0, self.zk_controller.omega)
            self.zk_controller.T_touch_old = torch.where(env_mask, 0.0, self.zk_controller.T_touch_old)
            self.zk_controller.flag_touch_current = torch.where(env_mask, False, self.zk_controller.flag_touch_current)
            self.zk_controller.T_touch_filtered = torch.where(env_mask, 0.0, self.zk_controller.T_touch_filtered)
            self.zk_controller.ddq_last = torch.where(env_mask, 0.0, self.zk_controller.ddq_last)

            self.controller.integral[:] = torch.where(mask_2d, 0.0, self.controller.integral)
            self.wrench_eef_left[:] = torch.where(mask_2d, 0.0, self.wrench_eef_left)
            self.xwrench_eef_left[:] = torch.where(mask_2d, 0.0, self.xwrench_eef_left)

        # 🚨【核心修改】：完全删除了 mjw.forward，交由子类统一处理！
        self.fast_forward()

    def ctrl(self, joints):
        # joint [num_envs, num_joints]
        self.controller.run(joints.reshape([self.num_envs, 7]))

    def touch_ctrl(self, joints, de, fd):
        target_grp_ddq = self.zk_controller.update(de, fd, self.left_touch_force, self.left_gripper_jac,
                                                   self.left_gripper_djac, self.grp_distance, self.arm_qvel[:, 6])
        self.controller.run(joints.reshape([self.num_envs, 7]), target_ddq_grp=target_grp_ddq, grp_joint_id=6)

    def ctrl_eef(self, pose_d, if_touch=False, de=0.0, fd_grp=0.0):
        """
        Batched EEF 控制器 (Warp 版本)
        :param pose_d: 期望位姿 [num_envs, 7] (x, y, z, w, x, y, z)
        :param if_touch: 是否启动接触力控闭环
        :param de: 期望夹爪闭合宽度 (可以是标量，也可以是 [num_envs] Tensor)
        :param fd_grp: 期望夹爪抓取力 (可以是标量，也可以是 [num_envs] Tensor)
        :return: if_available, 布尔张量 [num_envs], 表示各个环境的 IK 是否成功求解
        """
        # 1. 批量解析逆运动学
        nearest_q = self.ikine_analytical_left_torch(pose_d)

        # 2. 检查各环境 IK 是否成功 (通过对比是否与当前关节位置一致判断)
        current_q = self.arm_qpos[:, 0:6]
        ik_fail_mask = torch.all(nearest_q == current_q, dim=-1)
        if_available = ~ik_fail_mask

        # 3. 拼接夹爪支撑关节 (第 7 个关节) 使得控制器维度补齐 [num_envs, 7]
        q7 = self.arm_qpos[:, 6:7]
        target_joints = torch.cat([nearest_q, q7], dim=-1)

        # 4. 执行并行的底层力矩/接触力下发
        if if_touch:
            # 兼容输入：将标量转换为与环境数一致的 Batched Tensor
            if not isinstance(de, torch.Tensor):
                de_tensor = torch.full((self.num_envs,), de, dtype=torch.float32, device=device)
            else:
                de_tensor = de

            if not isinstance(fd_grp, torch.Tensor):
                fd_tensor = torch.full((self.num_envs,), fd_grp, dtype=torch.float32, device=device)
            else:
                fd_tensor = fd_grp

            self.touch_ctrl(target_joints, de_tensor, fd_tensor)
        else:
            self.ctrl(target_joints)

        return if_available

    def physics_step(self):
        # 1. 在循环外提前生成本回合所需的 5 个 500Hz 白噪声帧
        if self.has_wrench_noise:
            self.noise_w_buffer.normal_()
            self.noise_x_buffer.normal_()

        # ==========================================
        # 🌟 终极绝杀：只 Capture `mjw.step` 这个纯物理引擎内核
        # ==========================================
        if self._mjw_graph is None:
            wp.capture_begin()
            # 仅仅录制 1 步物理步进！没有任何 PyTorch 运算！
            mjw.step(self.mjw_model, self.mjw_data)
            self._mjw_graph = wp.capture_end()

        # ==========================================
        # 🚀 在 Python 循环中，极速交替回放图与 PyTorch 运算
        # ==========================================
        for ii in range(self.num_micro_steps):
            # A. 0 开销极速回放 1 步物理计算
            wp.capture_launch(self._mjw_graph)

            # B. 物理跑完了，回到 PyTorch 默认流，随意且安全地提取数据！
            raw_w = self.wrench_eef_left_raw
            raw_xw = self.xwrench_eef_left_raw
            
            # C. 肆无忌惮地使用 PyTorch 算子，绝不报错
            noisy_w = raw_w + self.wrench_drift + self.noise_w_buffer[ii] * self.wrench_noise_std
            noisy_xw = raw_xw + self.wrench_drift + self.noise_x_buffer[ii] * self.wrench_noise_std
            
            # D. In-place 滤波
            self.wrench_eef_left.mul_(1 - self.sensor_alpha).add_(noisy_w, alpha=self.sensor_alpha)
            self.xwrench_eef_left.mul_(1 - self.sensor_alpha).add_(noisy_xw, alpha=self.sensor_alpha)

        self._refresh_pose_caches()
        self.render()

    def load_environment_state(self, load_path):
        """
        兼容读取：同时支持老的 .npz 单状态文件 和 新的 .pt 高动态状态池文件
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"找不到状态文件: {load_path}")
            
        load_path_str = str(load_path)
        
        if load_path_str.endswith('.pt'):
            self.is_state_pool = True
            self.state_pool = torch.load(load_path, map_location=device)
            self.pool_size = self.state_pool['qpos'].shape[0]
            print(f"[*] 成功加载离线状态池 (.pt)，容量: {self.pool_size} 个环境快照。")
            
        elif load_path_str.endswith('.npz'):
            self.is_state_pool = False
            loaded = np.load(load_path, allow_pickle=True)
            self.reset_data = {k: loaded[k] for k in loaded.files}
            print(f"[*] 成功加载单状态文件 (.npz)。")
            
        else:
            raise ValueError("不支持的文件格式，请使用 .pt 或 .npz 后缀的文件。")

    def monitor_memory_peaks(self):
        """
        用于监测当前所有环境中的最大接触数(ncon)和标量约束数(nefc/njmax)。
        仅在调试配置 nconmax 和 njmax 时使用！
        """
        # 将 Warp 数据转为 Torch 张量并获取最大值 (这会触发隐式同步，很慢，但调试无所谓)
        current_max_ncon = int(wp.to_torch(self.mjw_data.nacon).item()/self.num_envs)
        current_max_nefc = wp.to_torch(self.mjw_data.nefc).max().item()

        return current_max_ncon, current_max_nefc

    @property
    def arm_qpos(self):
        return self.qpos_torch[:, self.arm.joint_qpos_adrs]

    @property
    def arm_qvel(self):
        return self.qvel_torch[:, self.arm.joint_dof_adrs]

    @property
    def arm_qpos_left(self):
        return self.qpos_torch[:, self.arm.joint_left_qpos_adrs]

    @property
    def arm_qvel_left(self):
        return self.qvel_torch[:, self.arm.joint_left_dof_adrs]

    @property
    def stock_pose(self):
        return self.qpos_torch[:, self.arm.stock_qpos_adrs]

    @property
    def grasp_pose(self):
        return self.grasp_pose_cache

    @property
    def stock_vel(self):
        return self.qvel_torch[:, self.arm.stock_qvel_adrs]

    @property
    def xpos_left(self):
        return self.xpos_left_cache
    
    @property
    def xvel_left(self):
        # 包含了 [num_envs, 6] 的完整速度
        return self.xvel_left_cache

    @property
    def eef_linvel(self):
        # 世界坐标系绝对线速度 [num_envs, 3]
        return self.xvel_left_cache[:, 0:3]
    
    @property
    def eef_omega(self):
        # 世界坐标系绝对角速度 [num_envs, 3]
        return self.xvel_left_cache[:, 3:6]

    @property
    def xmat_left(self):
        return self.xmat_left_cache

    @property
    def xpose_left(self):
        return self.xpose_left_cache

    @property
    def eef_real_pose(self):
        return self.eef_real_pose_cache

    @property
    def base_xpos_left(self):
        return self.base_xpos_left_cache

    @property
    def base_xmat_left(self):
        return self.base_xmat_left_cache

    @property
    def left_touch_force(self):
        """
        返回所有环境的左手触觉传感器数据
        返回形状: [nworld, num_left_sensors]
        """
        # 1. 将原生的 sensordata 映射为 Torch Tensor  # [nworld, total_sensor_dim]

        # 2. 使用预先获取的 sensor_adrs 进行批量切片
        # 假设 left_touch_sensors 是一个包含多个传感器 ID 的列表
        # 提取结果形状为 [nworld, num_left_sensors]
        return self.sensordata_torch[:, self.arm.left_touch_sensor_adrs]

    @property
    def vise_touch(self):
        return self.sensordata_torch[:, self.arm.vise_touch_sensor_adrs].sum(dim=1)
    
    @property
    def vise_surface_touch(self):
        return self.sensordata_torch[:, self.arm.vise_surface_touch_sensor_adrs].sum(dim=1)

    def ikine_analytical_left_torch(self, pose_d):
        """
        基于 Torch 的并行解析逆运动学
        pose_d: [num_envs, 7] (pos_xyz, quat_wxyz)
        """

        # 1. 准备数据
        xd = pose_d[:, 0:3]  # [num_envs, 3]
        qd = pose_d[:, 3:7]  # [num_envs, 4] (wxyz)

        # 2. 坐标变换：将目标转换到机械臂基座坐标系
        # xd_base = R^T @ (xd - p)
        relative_pos = xd - self.base_xpos_left
        xd_base = torch.bmm(relative_pos.unsqueeze(1), self.base_xmat_left).squeeze(1)  # 相当于 relative_pos @ base_R

        # 旋转变换：matd_base = R_base^T @ R_target
        matd = quat2mat(qd)  # [num_envs, 3, 3]
        matd_base_R = torch.matmul(self.base_xmat_left.transpose(-1, -2), matd)

        # 3. 批量求解 IK
        target_arm_q_all = rm65_analytical_ik_torch(
            target_pos=xd_base,
            target_R=matd_base_R,
            d6_tcp_m=0.161 + 0.2
        )

        # 4. 多解择优 (Nearest Solution)
        # target_arm_q_all 形状为 [num_envs, 8, 6]
        # current_q 形状为 [num_envs, 1, 6]
        current_q = self.arm_qpos[:, 0:6].unsqueeze(1)

        # 计算差值和距离
        diff = target_arm_q_all - current_q
        dist = torch.norm(diff, dim=-1)  # [num_envs, 8] (无效解对应的距离会是 NaN)

        # 【核心修改 1】：将 NaN 替换为正无穷
        # 这样 argmin 就会自动忽略它们，只在有效解中寻找最小值
        dist = torch.nan_to_num(dist, nan=float('inf'))

        # 找到每个环境中最接近的解的索引
        best_sol_idx = torch.argmin(dist, dim=-1)  # [num_envs]

        # 索引提取初步的最终解
        batch_idx = self.ik_batch_idx
        nearest_q = target_arm_q_all[batch_idx, best_sol_idx]  # [num_envs, 6]

        # 【核心修改 2】：处理“全军覆没”的极端情况
        # 如果 8 个距离全都是 inf，说明这个目标位姿在工作空间外或超出了关节极限
        # 在强化学习或控制中，最安全的做法是让它保持当前姿态不动
        all_invalid_mask = torch.isinf(dist).all(dim=-1).unsqueeze(-1)  # [num_envs, 1]
        nearest_q = torch.where(all_invalid_mask, self.arm_qpos[:, 0:6], nearest_q)

        return nearest_q

    @property
    def left_site_jac(self):
        """
        利用 mujoco-warp 底层 mjw.jac 内核计算左臂末端的 6D 空间雅可比矩阵。
        """
        # 从缓存视图中提取坐标，只做极低开销的 from_torch 包装
        site_xpos_torch = self.site_xpos_view[:, self._eef_site_id, :].contiguous()
        point_wp = wp.from_torch(site_xpos_torch, dtype=wp.vec3)

        # 触发 GPU 内核，结果会直接写入已预分配的 self.jacp_wp 内存中
        mjw.jac(self.mjw_model, self.mjw_data, self.jacp_wp, self.jacr_wp, point_wp, self.eef_body_wp)

        # 直接读取预先绑定好的 Torch 视图，0 转换开销
        dof_adrs = self.arm.joint_left_dof_adrs
        jacp_arm = self.jacp_torch[:, :, dof_adrs]
        jacr_arm = self.jacr_torch[:, :, dof_adrs]

        return torch.cat([jacp_arm, jacr_arm], dim=1)

    @property
    def grp_distance(self):
        """
        计算所有环境下左右夹爪指尖 site 之间的距离
        """
        # 直接使用缓存的 site 坐标池
        pos_0 = self.site_xpos_torch[:, self.arm.gripper_left_site_ids[0], :]
        pos_1 = self.site_xpos_torch[:, self.arm.gripper_left_site_ids[1], :]

        return torch.linalg.norm(pos_1 - pos_0, dim=-1)

    @property
    def left_gripper_jac(self):
        """
        基于平行连杆机构解析推导的开合方向 1D 雅可比 [nworld] 解析方法
        """
        # 直接读取缓存的 qpos_torch
        q = self.qpos_torch[:, self.arm.joint_left_gripper_qpos_adrs[0]]

        # 全部使用 __init__ 初始化的浮点常量，不产生中间 Tensor
        return -2.0 * self.gripper_L * torch.sin(self.theta_offset - q)

    @property
    def left_gripper_jac_mjw(self):
        """
        直接调用 mujoco-warp 底层的 mjw.jac 并行内核计算雅可比
        """
        # 设置测试点坐标
        site_xpos_torch = self.site_xpos_view[:, self._left_gripper_site_id, :].contiguous()
        point_wp = wp.from_torch(site_xpos_torch, dtype=wp.vec3)

        # 触发 GPU 内核，使用专属于夹爪的预分配 Body ID
        mjw.jac(self.mjw_model, self.mjw_data, self.jacp_wp, self.jacr_wp, point_wp, self.grp_body_wp)

        # 提取相关列 (复用 jacp_torch)
        jac_active_3d = self.jacp_torch[:, :, self._left_gripper_dof_active]
        jac_support_3d = self.jacp_torch[:, :, self._left_gripper_dof_support]
        jac_true_3d = jac_active_3d - jac_support_3d

        # 利用缓存的旋转矩阵和预置的 z 轴张量计算投影
        site_mat = self.site_xmat_view[:, self._left_gripper_site_id, :, :]
        close_dir = (site_mat @ self.z_axis_torch).squeeze(-1)

        return torch.sum(jac_true_3d * close_dir, dim=-1) * 2.0

    @property
    def left_gripper_djac(self):
        """
        基于平行连杆机构解析推导的开合方向 1D 雅可比
        """
        # 纯内存切片，无新对象分配
        q = self.qpos_torch[:, self.arm.joint_left_gripper_qpos_adrs[0]]
        dq = self.qvel_torch[:, self.arm.joint_left_gripper_dof_adrs[0]]

        return 2.0 * self.gripper_L * torch.cos(self.theta_offset - q) * dq

    @property
    def wrench_eef_left_raw(self):
        # 🚀 使用纯 int 进行切片，完全消除隐式同步
        F_eef = self.sensordata_torch[:, self._f_adr: self._f_adr + 3]
        T_eef = self.sensordata_torch[:, self._t_adr: self._t_adr + 3]
        return torch.cat([F_eef, T_eef], dim=-1)

    @property
    def xwrench_eef_left_raw(self):
        xmat = self.site_xmat_view[:, self._wrench_site_id, :, :]
        # 🚀 同上，使用 int 切片
        F_local = self.sensordata_torch[:, self._f_adr: self._f_adr + 3]
        T_local = self.sensordata_torch[:, self._t_adr: self._t_adr + 3]
        xF = torch.matmul(xmat, F_local.unsqueeze(-1)).squeeze(-1)
        xT = torch.matmul(xmat, T_local.unsqueeze(-1)).squeeze(-1)
        return torch.cat([xF, xT], dim=-1)

    @property
    def base_omega(self):
        # 🚀 使用 int 列表
        gyro_data = torch.stack([
            self.sensordata_torch[:, adr: adr + 3] for adr in self._gyro_adrs
        ], dim=1)
        base_mats = self.site_xmat_torch[:, self.arm.base_site_ids, :].view(self.num_envs, -1, 3, 3)
        omega_world = torch.matmul(base_mats, gyro_data.unsqueeze(-1))
        return omega_world.squeeze(-1)

    @property
    def insert_pose(self):
        return self.insert_pose_cache

    @property
    def insert_world_pose(self):
        return self.insert_world_pose_cache
