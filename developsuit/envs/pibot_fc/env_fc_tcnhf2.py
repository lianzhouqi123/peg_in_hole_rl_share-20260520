from collections import deque

import numpy as np

from developsuit.envs.pibot_base_env.pibotenv_FC_HF import PiBotEnv
from developsuit.utils.transform_utils import axisangle2quat, mat2eul, quat2axisangle, quat2mat, mat2quat


class CartesianAdmittanceController:
    """Numpy version of the Cartesian admittance controller used by the Warp env."""

    def __init__(
        self,
        md=np.diag([5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
        dd=np.diag([300.0, 300.0, 300.0, 300.0, 300.0, 300.0]),
        kd=np.diag([100.0, 100.0, 100.0, 100.0, 100.0, 100.0]),
        dt=0.01,
    ):
        self.Md = md.astype(np.float32, copy=True)
        self.Md_inv = np.linalg.inv(self.Md)
        self.Dd = dd.astype(np.float32, copy=True)
        self.Kd = kd.astype(np.float32, copy=True)
        self.dt = dt
        self.x_c = np.zeros(6, dtype=np.float32)
        self.x_c_dot = np.zeros(6, dtype=np.float32)
    
    def forward(self, x_d, F_ext, x_d_dot=None, x_d_ddot=None):
        x_d = np.asarray(x_d, dtype=np.float32).reshape(-1)
        F_ext = np.asarray(F_ext, dtype=np.float32).reshape(-1)
        x_d_dot = np.zeros(6, dtype=np.float32) if x_d_dot is None else np.asarray(x_d_dot, dtype=np.float32).reshape(-1)
        x_d_ddot = np.zeros(6, dtype=np.float32) if x_d_ddot is None else np.asarray(x_d_ddot, dtype=np.float32).reshape(-1)

        # 1. 计算原始位置误差 (虚拟弹簧的拉伸量)
        e = self.x_c - x_d
        
        # ==========================================
        # 🌟 改进 1：抗积分饱和 (Anti-Windup) / 弹簧限幅
        # ==========================================
        # 限制虚拟弹簧的最大拉伸长度，防止误差累积导致力矩爆炸
        max_trans_err = 0.015  # 最大允许平移误差 1.5 厘米
        max_rot_err = 0.15     # 最大允许旋转误差约 8.5 度

        e_trans = e[0:3]
        e_rot = e[3:6]

        trans_norm = np.linalg.norm(e_trans)
        if trans_norm > max_trans_err:
            # 等比例缩放误差向量
            e_trans = e_trans * (max_trans_err / trans_norm)
            # 🌟 极其关键：把外部跑偏的 x_d 强行拉回来！
            x_d[0:3] = self.x_c[0:3] - e_trans 

        rot_norm = np.linalg.norm(e_rot)
        if rot_norm > max_rot_err:
            e_rot = e_rot * (max_rot_err / rot_norm)
            x_d[3:6] = self.x_c[3:6] - e_rot
            
        # 重新拼接修正后的误差
        e = np.concatenate([e_trans, e_rot])
        # ==========================================

        de = self.x_c_dot - x_d_dot
        
        # 计算导纳加速度
        dde = self.Md_inv @ (F_ext - self.Dd @ de - self.Kd @ e)
        ddxc = x_d_ddot + dde

        self.x_c_dot += ddxc * self.dt
        
        # ==========================================
        # 🌟 改进 2：收紧极限速度 (治疗 XY 乒乓效应)
        # ==========================================
        # 原本的 0.5 m/s (50cm/s) 在精密装配中太狂暴了
        # 将平移速度限制在 5cm/s，旋转速度限制在 0.2 rad/s
        self.x_c_dot[0:3] = np.clip(self.x_c_dot[0:3], -0.05, 0.05)
        self.x_c_dot[3:6] = np.clip(self.x_c_dot[3:6], -0.2, 0.2)
        
        self.x_c += self.x_c_dot * self.dt
        
        # 🌟 改进 3：返回 x_c 的同时，返回被限幅拉回来的 x_d
        return self.x_c.copy(), x_d.copy()

    def reset(self):
        self.x_c.fill(0.0)
        self.x_c_dot.fill(0.0)


class Env(PiBotEnv):
    def __init__(self, show_mode="no_show", timestep_control=0.01, step_max=500, config={}):
        self.config = config

        self.admittance_dt = self.config.get("admittance_dt", timestep_control)
        self.rl_dt = self.config.get("rl_dt", 0.01)
        self.rl_micro_steps = int(self.rl_dt / self.admittance_dt)
        if self.rl_micro_steps <= 0:
            raise ValueError("rl_dt / admittance_dt 必须大于 0")

        super().__init__(show_mode=show_mode, timestep_control=self.admittance_dt, config=config)

        self.history_len = self.config.get("history_len", 5)
        self.noise_v_linear = self.config.get("noise_v_linear", 0.005)
        self.noise_v_angular = self.config.get("noise_v_angular", 0.01)

        self.eef_dn_controller = CartesianAdmittanceController(dt=self.admittance_dt)

        self.step_max = self.config.get("step_max", step_max)

        self.xd = np.zeros(6, dtype=np.float32)
        self.dxd = np.zeros(6, dtype=np.float32)
        self.dxd_prev = np.zeros(6, dtype=np.float32)
        self.wrench_bias = np.zeros(6, dtype=np.float32)
        self.ctrl_frame_rot_world_from_local = np.eye(3, dtype=np.float32)

        self.default_kd = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float32)
        self.default_dd = np.array([300.0, 300.0, 300.0, 250.0, 250.0, 250.0], dtype=np.float32)
        self.last_action_dxd = np.zeros(6, dtype=np.float32)
        self.last_action_k = self.default_kd.copy()

        self.prev_pos_err = 0.0
        self.prev_pos_err_vec = np.zeros(3, dtype=np.float32)
        self.prev_ori_err = 0.0
        self.i_step = 0

        self.if_touch = True
        self.de = 0.06
        self.fd_grp = 45.0
        self.touch_threshold = 1.0
        self.drop_fail_window = 5
        self.touch_fail_history = deque(maxlen=self.drop_fail_window)

        self.kin_obs_dim = 12
        self.force_obs_dim = 6
        self.single_obs_dim = self.kin_obs_dim + self.force_obs_dim
        self.obs_dim = self.single_obs_dim * self.history_len
        self.action_dim = 12
        self.obs_history = deque(maxlen=self.history_len)

        action_min = np.asarray(
            self.config.get('action_min', [-0.01] * 6 + [50.0] * 6),
            dtype=np.float32,
        )
        action_max = np.asarray(
            self.config.get('action_max', [0.01] * 6 + [200.0] * 6),
            dtype=np.float32,
        )
        self.action_range = (action_min, action_max)

        self.scale_pos = 10.0
        self.scale_ori = 10.0
        self.scale_vel = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.scale_wrench = np.array([1 / 30.0, 1 / 30.0, 1 / 30.0, 1 / 10.0, 1 / 10.0, 1 / 10.0], dtype=np.float32)

        self.penalty_reward = -300.0
        ref_step_budget = 0.01 * 0.05
        curr_step_budget = max(self.rl_dt * float(np.max(action_max[:3])), 1e-6)
        self.reward_progress_scale = float(np.clip(ref_step_budget / curr_step_budget, 1.0, 8.0))

        self.r_step = -0.05
        self.reward_xy_coef = 80.0 * self.reward_progress_scale
        self.reward_ori_far_coef = 20.0 * self.reward_progress_scale
        self.reward_ori_near_bonus = 40.0 * self.reward_progress_scale
        self.reward_z_approach_coef = 40.0 * self.reward_progress_scale
        self.reward_depth_insert_coef = 200.0 * self.reward_progress_scale
        self.prev_err_z = 0.0

        self.obs_noise_range = self.config.get("obs_noise_range", 0.0)
        self.camera_jitter_pos = self.config.get("camera_jitter_pos", 0.0)
        self.obs_ori_noise_range = self.config.get("obs_ori_noise_range", 0.0)
        self.camera_jitter_ori = self.config.get("camera_jitter_ori", 0.0)
        self.obs_pos_noise = np.zeros(3, dtype=np.float32)
        self.obs_ori_noise = np.zeros(3, dtype=np.float32)

        self.ctrl_alpha = self.config.get("ctrl_alpha", 0.12)
        self.ctrl_f_ext_filtered = np.zeros(6, dtype=np.float32)

        self.max_delay_steps = self.config.get("max_delay_steps", 0)
        self.action_history = np.zeros((self.max_delay_steps + 1, self.action_dim), dtype=np.float32)
        self.delay_steps = 0
        self.controller_num_joints = self.controller.integral.shape[0]
        default_controller_kp = np.asarray(self.controller.default_kp, dtype=np.float32)
        default_controller_damping = np.asarray(self.controller.default_damping_ratio, dtype=np.float32)
        gain_mode_default = "randomize" if self.config.get("controller_gain_rand_enabled", False) else "default"
        self.controller_gain_mode = str(self.config.get("controller_gain_mode", gain_mode_default)).lower()
        if self.controller_gain_mode not in {"default", "randomize", "fixed"}:
            raise ValueError("controller_gain_mode must be one of: 'default', 'randomize', 'fixed'")
        self.controller_gain_integral_scale = float(np.clip(self.config.get("controller_gain_integral_scale", 0.0), 0.0, 1.0))
        self.controller_kp_range = self._parse_controller_randomization_range(
            "controller_kp_range", default_controller_kp
        )
        self.controller_damping_ratio_range = self._parse_controller_randomization_range(
            "controller_damping_ratio_range", default_controller_damping
        )
        self.controller_kp_fixed = self._parse_controller_gain_vector(
            "controller_kp_fixed", default_controller_kp
        )
        self.controller_damping_ratio_fixed = self._parse_controller_gain_vector(
            "controller_damping_ratio_fixed", default_controller_damping
        )

    def _parse_controller_randomization_range(self, key, default):
        value = self.config.get(key, default)
        arr = np.asarray(value, dtype=np.float32)

        if arr.ndim == 0:
            arr = np.full(self.controller_num_joints, float(arr), dtype=np.float32)
            return np.stack([arr, arr], axis=-1)

        if arr.ndim == 1:
            if arr.shape[0] == 2:
                return np.tile(arr.reshape(1, 2), (self.controller_num_joints, 1)).astype(np.float32, copy=True)
            if arr.shape[0] == self.controller_num_joints:
                return np.stack([arr, arr], axis=-1).astype(np.float32, copy=True)

        if arr.ndim == 2 and arr.shape == (self.controller_num_joints, 2):
            lower = np.minimum(arr[:, 0], arr[:, 1])
            upper = np.maximum(arr[:, 0], arr[:, 1])
            return np.stack([lower, upper], axis=-1).astype(np.float32, copy=True)

        raise ValueError(
            f"{key} must be a scalar, shape [2], shape [{self.controller_num_joints}], "
            f"or shape [{self.controller_num_joints}, 2]"
        )

    def _parse_controller_gain_vector(self, key, default):
        value = self.config.get(key, default)
        arr = np.asarray(value, dtype=np.float32)

        if arr.ndim == 0:
            return np.full(self.controller_num_joints, float(arr), dtype=np.float32)

        arr = arr.reshape(-1).astype(np.float32, copy=False)
        if arr.shape[0] == self.controller_num_joints:
            return arr.copy()

        raise ValueError(f"{key} must be a scalar or shape [{self.controller_num_joints}]")

    def _sample_controller_randomization_range(self, range_array):
        lower = range_array[:, 0]
        upper = range_array[:, 1]
        return lower + np.random.rand(self.controller_num_joints).astype(np.float32) * (upper - lower)

    def _reset_controller_gain_mode(self):
        if self.controller_gain_mode == "default":
            target_kp = self.controller.default_kp
            target_damping_ratio = self.controller.default_damping_ratio
        elif self.controller_gain_mode == "fixed":
            target_kp = self.controller_kp_fixed
            target_damping_ratio = self.controller_damping_ratio_fixed
        else:
            target_kp = np.clip(
                self._sample_controller_randomization_range(self.controller_kp_range),
                a_min=1e-3,
                a_max=None,
            )
            target_damping_ratio = np.clip(
                self._sample_controller_randomization_range(self.controller_damping_ratio_range),
                a_min=1e-4,
                a_max=None,
            )

        self.controller.set_pd_gains(kp=target_kp, damping_ratio=target_damping_ratio)

        if self.controller_gain_mode != "default":
            self.controller.integral[:] = self.controller.integral * self.controller_gain_integral_scale

    def reset(self, if_reset_data=False, if_vise_open_rdm=False, if_vise_base_rdm=False):
        noise_v_linear = self.noise_v_linear
        noise_v_angular = self.noise_v_angular
        self.noise_v_linear = 0.0
        self.noise_v_angular = 0.0
        try:
            # 这步会调用父类重置物理状态 (如果 if_reset_data=True，应该产生 self.sampled_idx)
            super().reset(
                if_reset_data=if_reset_data,
                if_vise_open_rdm=if_vise_open_rdm,
                if_vise_base_rdm=if_vise_base_rdm,
            )
        finally:
            self.noise_v_linear = noise_v_linear
            self.noise_v_angular = noise_v_angular

        self._reset_controller_gain_mode()

        self.ctrl_frame_rot_world_from_local = self._get_current_ctrl_rot_world_from_local()
        current_pose = self.xpose_left[0]
        new_xd = self._pose_world_to_local(current_pose, self.ctrl_frame_rot_world_from_local)

        # ==========================================
        # 🌟 修复点：根据是否加载离线数据来初始化控制器
        # ==========================================
        loaded_wrench_bias = False
        if if_reset_data and hasattr(self, 'is_state_pool') and self.is_state_pool:
            if hasattr(self, 'sampled_idx') and 'wrench_bias' in self.state_pool:
                self.wrench_bias = self.state_pool['wrench_bias'][self.sampled_idx].numpy()
                loaded_wrench_bias = True
            elif 'wrench_bias' in self.state_pool:
                print("警告: 状态池包含 wrench_bias，但 sampled_idx 未暴露，跳过对齐加载。")
        elif if_reset_data and hasattr(self, 'reset_data') and self.reset_data is not None and 'wrench_bias' in self.reset_data:
            self.wrench_bias = self.reset_data['wrench_bias'][:].astype(np.float32, copy=True)
            loaded_wrench_bias = True

        # 常规重置：清空弹簧，锚定当前末端位姿
        self.xd = new_xd.copy()
        self.xd_1111 = new_xd.copy()
        self.dxd.fill(0.0)
        self.dxd_prev.fill(0.0)
        self.eef_dn_controller.reset()
        self.eef_dn_controller.x_c = self.xd.copy()

        self.eef_dn_controller.Kd = np.diag(self.default_kd)
        self.eef_dn_controller.Dd = np.diag(self.default_dd)
        self.last_action_dxd.fill(0.0)
        self.last_action_k = self.default_kd.copy()

        self.obs_pos_noise = np.random.uniform(-self.obs_noise_range, self.obs_noise_range, size=3).astype(np.float32)
        self.obs_ori_noise = np.random.uniform(-self.obs_ori_noise_range, self.obs_ori_noise_range, size=3).astype(np.float32)
        self.ctrl_f_ext_filtered = self._get_eef_real_wrench_local(self.ctrl_frame_rot_world_from_local)
        self.delay_steps = np.random.randint(0, self.max_delay_steps + 1)
        self.action_history.fill(0.0)

        if self.noise_v_angular > 0.0 or self.noise_v_linear > 0.0:
            noise_linear = np.random.uniform(-self.noise_v_linear, self.noise_v_linear, size=3)
            noise_angular = np.random.uniform(-self.noise_v_angular, self.noise_v_angular, size=3)
            noise_dxd = np.concatenate([noise_linear, noise_angular]).astype(np.float32)
            noise_dxd_world = self._rotate_spatial_to_world(self.ctrl_frame_rot_world_from_local, noise_dxd)

            jac = self.left_site_jac
            jac_t = jac.T
            lambda_sq = 1e-4
            A = jac_t @ jac + lambda_sq * np.eye(6, dtype=np.float32)
            B = jac_t @ noise_dxd_world
            noise_dq = np.linalg.solve(A, B)
            noise_dq = np.nan_to_num(noise_dq, nan=0.0).astype(np.float32, copy=False)

            self.dxd = noise_dxd.copy()
            self.last_action_dxd = noise_dxd.copy()
            self.eef_dn_controller.x_c_dot = noise_dxd.copy()

            current_qvel_left = np.asarray(self.physics.bind(self.arm.joint_lefts).qvel).reshape(-1)
            self.physics.bind(self.arm.joint_lefts).qvel[:] = current_qvel_left + noise_dq
            self.fast_forward()
        else:
            self.dxd.fill(0.0)
            self.last_action_dxd.fill(0.0)
            self.eef_dn_controller.x_c_dot.fill(0.0)

        # ==========================================
        # 🌟 修复点：如果有读取到的偏置，跳过这 10 步空跑
        # ==========================================
        if not loaded_wrench_bias:
            self.set_wrench_bias(10)

        g_pose = self.grasp_pose
        i_pose = self.insert_pose
        insert_world_pose = self.insert_world_pose
        rot_align = quat2mat(insert_world_pose[3:7])
        pos_err_vec_world = (g_pose[0:3] - i_pose[0:3]).astype(np.float32, copy=False)
        self.prev_pos_err_vec = self._rotate_vec_to_local(rot_align, pos_err_vec_world)
        self.prev_pos_err = float(np.linalg.norm(self.prev_pos_err_vec))

        rot_g = quat2mat(g_pose[3:7])
        rot_i = quat2mat(i_pose[3:7])
        rot_err = rot_i.T @ rot_g
        ori_err_vec_insert = quat2axisangle(mat2quat(rot_err))
        ori_err_vec_world = self._rotate_vec_to_world(rot_i, ori_err_vec_insert)
        ori_err_vec = self._rotate_vec_to_local(rot_align, ori_err_vec_world)
        self.prev_ori_err = float(np.linalg.norm(ori_err_vec))
        self.prev_err_z = float(np.abs(self.prev_pos_err_vec[2]))

        initial_frame = self._get_single_frame()
        self.obs_history.clear()
        for _ in range(self.history_len):
            self.obs_history.append(initial_frame.copy())
        self.touch_fail_history.clear()

        self.i_step = 0

    def set_wrench_bias(self, times=10):
        times = int(times)
        self.wrench_bias.fill(0.0)
        target_joints = self.arm_qpos.copy()
        for _ in range(times):
            self.ctrl_touch(target_joints, self.de, self.fd_grp)
            self.physics_step()
            self.wrench_bias += self.xwrench_eef_left.copy()
        
        self.wrench_bias /= times

    def step(self, action, if_change_k=True):
        action = np.asarray(action, dtype=np.float32)
        self._sync_controller_frame_to_current_local()
        ctrl_rot_world_from_local = self.ctrl_frame_rot_world_from_local.copy()

        # ==========================================
        # 🌟 终极防爆盾：环境级动作合法性保护
        # 拦截由于网络梯度爆炸导致的 NaN 或 Inf 动作
        # ==========================================
        if np.isnan(action).any() or np.isinf(action).any():
            print("\n🚨 [环境级保护触发] 接收到非法的 NaN 或 Inf 动作！强行截断当前回合。")
            
            # 返回当前观测状态，不让错误的动作污染物理引擎
            next_obs = self.get_observation() 
            # 给予严厉惩罚
            reward = float(self.penalty_reward)
            done = True
            
            # 伪造一个安全的 info 字典返回
            info = {
                "is_success": False,
                "ik_fail": False,
                "drop_fail": False,
                "runaway_fail": False,
                "invalid_action_fail": True,  # 新增一个死因标志
                "time_limit": False,
                "pos_err": 99.0,
                "dist_xy": 99.0,
                "ori_err": 99.0,
                "force_norm": 0.0,
                "penalty_scale": 1.0,
                "r_z_approach": 0.0,
                "r_components": (0.0, 0.0, 0.0, 0.0, 0.0),
                "k": self.last_action_k.copy(),
                "k_target": self.last_action_k.copy(),
                "k_applied_final": self.last_action_k.copy(),
                "k_applied_min": self.last_action_k.copy(),
                "k_applied_max": self.last_action_k.copy(),
            }
            return next_obs, reward, done, info
        # ==========================================

        if action.ndim > 1:
            action = action.reshape(-1)
        else:
            action = action.copy()

        if self.max_delay_steps > 0:
            self.action_history[1:] = self.action_history[:-1].copy()
        self.action_history[0] = action

        delayed_action = self.action_history[self.delay_steps]
        dxd_target = delayed_action[0:6]
        k_target = delayed_action[6:12]

        # dxd_target = np.zeros([6])
        
        dxd_step = (dxd_target - self.last_action_dxd) / self.rl_micro_steps
        k_step = (k_target - self.last_action_k) / self.rl_micro_steps

        accumulated_ik_fail = False
        accumulated_drop_fail = False
        applied_k_min = np.full(6, np.inf, dtype=np.float32)
        applied_k_max = np.full(6, -np.inf, dtype=np.float32)
        applied_k_final = self.last_action_k.copy()

        for i in range(self.rl_micro_steps):
            interp_dxd = self.last_action_dxd + dxd_step * (i + 1)
            interp_k = self.last_action_k + k_step * (i + 1)

            prev_xd = self.xd.copy()
            prev_dxd_prev = self.dxd_prev.copy()
            prev_xc = self.eef_dn_controller.x_c.copy()
            prev_xc_dot = self.eef_dn_controller.x_c_dot.copy()
            prev_kd = self.eef_dn_controller.Kd.copy()
            prev_dd = self.eef_dn_controller.Dd.copy()

            ddxd = (interp_dxd - self.dxd_prev) / self.admittance_dt
            self.dxd_prev = interp_dxd.copy()
            self.xd = self.xd + interp_dxd * self.admittance_dt

            current_noisy_wrench_local = self._get_eef_real_wrench_local(ctrl_rot_world_from_local)
            self.ctrl_f_ext_filtered = (
                self.ctrl_alpha * current_noisy_wrench_local
                + (1.0 - self.ctrl_alpha) * self.ctrl_f_ext_filtered
            )

            m_diag = np.diag(self.eef_dn_controller.Md)
            k_safe = np.maximum(interp_k, 1e-3)
            d_diag = 3.5 * np.sqrt(m_diag * k_safe)
            applied_k_min = np.minimum(applied_k_min, k_safe)
            applied_k_max = np.maximum(applied_k_max, k_safe)
            applied_k_final = k_safe.copy()

            if if_change_k:
                self.eef_dn_controller.Kd = np.diag(k_safe)
                self.eef_dn_controller.Dd = np.diag(d_diag)
            
            xc, self.xd = self.eef_dn_controller.forward(
                x_d=self.xd,
                F_ext=self.ctrl_f_ext_filtered,
                x_d_dot=interp_dxd,
                x_d_ddot=ddxd,
            )

            # xc = self.xd_1111.copy()
            # print(xc)
            pose_d = self._pose_local_to_world(xc, ctrl_rot_world_from_local)

            nearest_q = self.ikine_analytical_left(pose_d)
            ik_fail = nearest_q is None

            if ik_fail:
                self.xd = prev_xd
                self.dxd_prev = prev_dxd_prev
                self.eef_dn_controller.x_c = prev_xc
                self.eef_dn_controller.x_c_dot = prev_xc_dot
                self.eef_dn_controller.Kd = prev_kd
                self.eef_dn_controller.Dd = prev_dd
                nearest_q = self.arm_qpos[0:6].copy()

            q7 = self.arm_qpos[6:7].copy()
            target_joints = np.concatenate([nearest_q, q7]).astype(np.float32, copy=False)
            self.ctrl_touch(target_joints, self.de, self.fd_grp)
            self.physics_step()

            accumulated_ik_fail = accumulated_ik_fail or ik_fail
            self.touch_fail_history.append(np.mean(self.left_touch_force) < self.touch_threshold)
            accumulated_drop_fail = accumulated_drop_fail or (
                len(self.touch_fail_history) == self.drop_fail_window and all(self.touch_fail_history)
            )

        self._sync_controller_frame_to_current_local()
        dxd_target = self._reexpress_spatial(
            dxd_target, ctrl_rot_world_from_local, self.ctrl_frame_rot_world_from_local
        )
        self.last_action_dxd = dxd_target.copy()
        self.last_action_k = k_target.copy()
        self.dxd = dxd_target.copy()

        current_frame = self._get_single_frame()
        self.obs_history.append(current_frame.copy())

        next_obs = self.get_observation()
        reward, is_success, info = self.get_reward()

        current_pos_err = np.linalg.norm(self.grasp_pose[0:3] - self.insert_pose[0:3])
        is_nan = np.isnan(current_pos_err)
        runaway_fail = current_pos_err > 0.4
        fail = accumulated_ik_fail or accumulated_drop_fail or runaway_fail or is_nan

        if fail:
            reward = self.penalty_reward

        reward = float(np.nan_to_num(reward, nan=self.penalty_reward))
        is_success = bool(is_success and (not fail))

        info["ik_fail"] = accumulated_ik_fail
        info["drop_fail"] = accumulated_drop_fail
        info["runaway_fail"] = runaway_fail
        info["time_limit"] = False
        info["k"] = k_target.copy()
        info["k_target"] = k_target.copy()
        info["k_applied_final"] = applied_k_final.copy()
        info["k_applied_min"] = applied_k_min.copy()
        info["k_applied_max"] = applied_k_max.copy()

        self.i_step += 1
        time_limit = (not (is_success or fail)) and (self.i_step >= self.step_max - 1)
        done = is_success or fail or time_limit
        info["time_limit"] = time_limit

        return next_obs, reward, done, info

    @staticmethod
    def _rotate_vec_to_local(rot_world_from_local, vec_world):
        vec_world = np.asarray(vec_world, dtype=np.float32).reshape(3)
        return (rot_world_from_local.T @ vec_world.reshape(3, 1)).reshape(3)

    @staticmethod
    def _rotate_vec_to_world(rot_world_from_local, vec_local):
        vec_local = np.asarray(vec_local, dtype=np.float32).reshape(3)
        return (rot_world_from_local @ vec_local.reshape(3, 1)).reshape(3)

    def _rotate_spatial_to_local(self, rot_world_from_local, spatial_world):
        spatial_world = np.asarray(spatial_world, dtype=np.float32).reshape(6)
        return np.concatenate(
            [
                self._rotate_vec_to_local(rot_world_from_local, spatial_world[0:3]),
                self._rotate_vec_to_local(rot_world_from_local, spatial_world[3:6]),
            ]
        ).astype(np.float32, copy=False)

    def _rotate_spatial_to_world(self, rot_world_from_local, spatial_local):
        spatial_local = np.asarray(spatial_local, dtype=np.float32).reshape(6)
        return np.concatenate(
            [
                self._rotate_vec_to_world(rot_world_from_local, spatial_local[0:3]),
                self._rotate_vec_to_world(rot_world_from_local, spatial_local[3:6]),
            ]
        ).astype(np.float32, copy=False)

    def _pose_world_to_local(self, pose_world, rot_world_from_local):
        pose_world = np.asarray(pose_world, dtype=np.float32).reshape(7)
        pos_local = self._rotate_vec_to_local(rot_world_from_local, pose_world[0:3])
        rotvec_world = quat2axisangle(pose_world[3:7]).astype(np.float32, copy=False)
        rotvec_local = self._rotate_vec_to_local(rot_world_from_local, rotvec_world)
        return np.concatenate([pos_local, rotvec_local]).astype(np.float32, copy=False)

    def _pose_local_to_world(self, pose_local, rot_world_from_local):
        pose_local = np.asarray(pose_local, dtype=np.float32).reshape(6)
        pos_world = self._rotate_vec_to_world(rot_world_from_local, pose_local[0:3])
        rotvec_world = self._rotate_vec_to_world(rot_world_from_local, pose_local[3:6])
        quat_world = axisangle2quat(rotvec_world)
        return np.concatenate([pos_world, quat_world]).astype(np.float32, copy=False)

    def _reexpress_spatial(self, spatial_old_local, old_rot_world_from_local, new_rot_world_from_local):
        spatial_world = self._rotate_spatial_to_world(old_rot_world_from_local, spatial_old_local)
        return self._rotate_spatial_to_local(new_rot_world_from_local, spatial_world)

    def _reexpress_pose(self, pose_old_local, old_rot_world_from_local, new_rot_world_from_local):
        pose_world = self._pose_local_to_world(pose_old_local, old_rot_world_from_local)
        return self._pose_world_to_local(pose_world, new_rot_world_from_local)

    def _get_current_ctrl_rot_world_from_local(self):
        return quat2mat(self.eef_real_pose[3:7]).astype(np.float32, copy=False)

    def _sync_controller_frame_to_current_local(self):
        new_rot_world_from_local = self._get_current_ctrl_rot_world_from_local()
        old_rot_world_from_local = self.ctrl_frame_rot_world_from_local

        self.xd = self._reexpress_pose(self.xd, old_rot_world_from_local, new_rot_world_from_local)
        self.dxd = self._reexpress_spatial(self.dxd, old_rot_world_from_local, new_rot_world_from_local)
        self.dxd_prev = self._reexpress_spatial(self.dxd_prev, old_rot_world_from_local, new_rot_world_from_local)
        self.last_action_dxd = self._reexpress_spatial(
            self.last_action_dxd, old_rot_world_from_local, new_rot_world_from_local
        )
        self.ctrl_f_ext_filtered = self._reexpress_spatial(
            self.ctrl_f_ext_filtered, old_rot_world_from_local, new_rot_world_from_local
        )
        self.eef_dn_controller.x_c = self._reexpress_pose(
            self.eef_dn_controller.x_c, old_rot_world_from_local, new_rot_world_from_local
        )
        self.eef_dn_controller.x_c_dot = self._reexpress_spatial(
            self.eef_dn_controller.x_c_dot, old_rot_world_from_local, new_rot_world_from_local
        )

        if self.max_delay_steps > 0:
            for idx in range(self.action_history.shape[0]):
                self.action_history[idx, 0:6] = self._reexpress_spatial(
                    self.action_history[idx, 0:6], old_rot_world_from_local, new_rot_world_from_local
                )

        self.ctrl_frame_rot_world_from_local = new_rot_world_from_local.copy()

    def _get_eef_real_wrench_local(self, eef_real_rot_world_from_local):
        wrench_world = np.asarray(self.xwrench_clean, dtype=np.float32).reshape(6)
        return self._rotate_spatial_to_local(eef_real_rot_world_from_local, wrench_world)

    def _get_single_frame(self):
        g_pose = self.xpose_left[0]
        i_pose = self.insert_pose
        eef_real_rot_world_from_local = self._get_current_ctrl_rot_world_from_local()

        true_pos_err_world = g_pose[0:3] - i_pose[0:3]
        true_pos_err = self._rotate_vec_to_local(eef_real_rot_world_from_local, true_pos_err_world)
        obs_pos_err = true_pos_err + self.obs_pos_noise
        obs_pos_err += np.random.uniform(-self.camera_jitter_pos, self.camera_jitter_pos, size=3)

        rot_g = quat2mat(g_pose[3:7])
        rot_i = quat2mat(i_pose[3:7])
        rot_err = rot_i.T @ rot_g
        true_euler_err, _ = mat2eul(rot_err)
        true_euler_err = self._rotate_vec_to_local(eef_real_rot_world_from_local, true_euler_err)
        obs_euler_err = true_euler_err + self.obs_ori_noise
        obs_euler_err += np.random.uniform(-self.camera_jitter_ori, self.camera_jitter_ori, size=3)

        current_vel = np.asarray(self.eef_dn_controller.x_c_dot, dtype=np.float32).reshape(6)
        f_ext_local = self._get_eef_real_wrench_local(eef_real_rot_world_from_local)

        obs_pos_err_scaled = obs_pos_err * self.scale_pos
        obs_euler_err_scaled = obs_euler_err * self.scale_ori
        current_vel_scaled = current_vel * self.scale_vel
        f_ext_scaled = f_ext_local * self.scale_wrench

        single_obs = np.concatenate(
            [obs_pos_err_scaled, obs_euler_err_scaled, current_vel_scaled, f_ext_scaled]
        ).astype(np.float32, copy=False)

        single_obs = np.nan_to_num(single_obs, nan=0.0)
        return np.clip(single_obs, -5.0, 5.0)

    def get_observation(self):
        if len(self.obs_history) == 0:
            current_frame = self._get_single_frame()
            for _ in range(self.history_len):
                self.obs_history.append(current_frame.copy())
        return np.concatenate(list(self.obs_history), axis=0)

    def get_reward(self):
        g_pose = self.grasp_pose
        i_pose = self.insert_pose
        insert_world_pose = self.insert_world_pose

        # ==========================================
        # 1. 以 insert_pose 为目标，以 insert_world_pose 为对齐坐标系
        #    只对齐方向，不改变插入点原点定义。
        # ==========================================
        pos_err_vec_world = g_pose[0:3] - i_pose[0:3]
        rot_align = quat2mat(insert_world_pose[3:7])
        pos_err_vec = self._rotate_vec_to_local(rot_align, pos_err_vec_world)
        pos_err = float(np.linalg.norm(pos_err_vec))
        curr_dist_xy = float(np.linalg.norm(pos_err_vec[:2]))
        prev_dist_xy = float(np.linalg.norm(self.prev_pos_err_vec[:2]))

        rot_g = quat2mat(g_pose[3:7])
        rot_i = quat2mat(i_pose[3:7])
        rot_err = rot_i.T @ rot_g
        ori_err_vec_insert = quat2axisangle(mat2quat(rot_err))
        ori_err_vec_world = self._rotate_vec_to_world(rot_i, ori_err_vec_insert)
        ori_err_vec = self._rotate_vec_to_local(rot_align, ori_err_vec_world)
        ori_err = float(np.linalg.norm(ori_err_vec))

        # ==========================================
        # 2. 误差分量直接在 insert_world_site 坐标系下读取
        #    该系与虎钳固连，且 -Z 为插入方向。
        # ==========================================
        err_x = float(np.abs(pos_err_vec[0]))
        err_y = float(np.abs(pos_err_vec[1]))
        err_z = float(np.abs(pos_err_vec[2]))

        # ==========================================
        # 3. 接触力惩罚
        # ==========================================
        force_norm = float(np.abs(self.vise_touch))
        force_eef_norm = float(np.linalg.norm(self.xwrench_clean))

        # 只对 XY 平面距离给位置引力，让策略先学会平面找孔，再由深度奖励处理 Z 轴插入。
        r_xy_progress = self.reward_xy_coef * (prev_dist_xy - curr_dist_xy)

        dynamic_force_threshold = 15.0 + 30.0 * np.clip(1.0 - (curr_dist_xy / 0.05), 0.0, 1.0)

        is_near_hole = float(curr_dist_xy < 0.02)
        dynamic_ori_coef = self.reward_ori_far_coef + self.reward_ori_near_bonus * is_near_hole
        r_ori_progress = dynamic_ori_coef * (self.prev_ori_err - ori_err)

        raw_force_penalty = -0.02 * max(force_eef_norm - dynamic_force_threshold, 0.0)

        is_searching_blindly = float(curr_dist_xy > 0.015)
        penalty_scale = 1.0 - 0.9 * is_searching_blindly
        r_force_penalty = penalty_scale * raw_force_penalty

        # ==========================================
        # 4. 成功判定与专属深度奖励
        # ==========================================
        is_deep_enough = err_z < 0.002
        is_y_aligned = err_y < 0.003
        is_x_in_slot = err_x < 0.003
        is_ori_correct = ori_err < 0.1

        is_success = bool(is_deep_enough and is_y_aligned and is_x_in_slot and is_ori_correct)
        r_success = 300.0 if is_success else 0.0

        # 分阶段的 Z 轴奖励：
        is_in_chamfer = (err_x < 0.015) and (err_y < 0.015)
        r_z_approach = self.reward_z_approach_coef * (self.prev_err_z - err_z) * float(not is_in_chamfer)
        r_depth_insert = self.reward_depth_insert_coef * (self.prev_err_z - err_z) * float(is_in_chamfer)
        r_depth_progress = r_z_approach + r_depth_insert

        # ==========================================
        # 5. 反卡死惩罚
        # ==========================================
        current_vel_norm = float(np.linalg.norm(self.eef_dn_controller.x_c_dot[0:3]))
        is_stuck = (err_z > 0.04) and (current_vel_norm < 0.005)
        r_stuck_penalty = -0.0 if is_stuck else 0.0

        # ==========================================
        # 6. 状态更新与奖励合并
        # ==========================================
        self.prev_err_z = err_z
        self.prev_pos_err_vec = pos_err_vec.astype(np.float32, copy=True)
        self.prev_pos_err = pos_err
        self.prev_ori_err = ori_err

        reward = (
            r_xy_progress
            + r_ori_progress
            + r_force_penalty
            + r_success
            + self.r_step
            + r_depth_progress
            + r_stuck_penalty
        )
        reward = float(np.clip(reward, -100.0, 400.0))

        info = {
            "is_success": is_success,
            "pos_err": pos_err,
            "dist_xy": curr_dist_xy,
            "ori_err": ori_err,
            "force_norm": force_norm,
            "penalty_scale": penalty_scale,
            "reward_progress_scale": self.reward_progress_scale,
            "r_z_approach": r_z_approach,
            "r_components": (
                self.r_step,
                r_xy_progress,
                r_ori_progress,
                r_force_penalty,
                r_depth_progress,
            ),
        }
        return reward, is_success, info

    @property
    def xwrench_clean(self):
        return -(self.xwrench_eef_left - self.wrench_bias)
