import torch
import warp as wp


class PiBot_Warp_Controller:
    def __init__(self, nworld, mjw_model, mjw_data, jnt_qpos_adrs, jnt_dof_adrs, actuator_ids, timestep, kp, damping_ratio, ki, min_effort=-60.0,
                 max_effort=60.0, device="cuda"):
        self.nworld = nworld
        self.device = device
        self.mjw_model = mjw_model
        self.mjw_data = mjw_data
        self.jnt_qpos_adrs = jnt_qpos_adrs
        self.jnt_dof_adrs = jnt_dof_adrs
        self.actuator_ids = actuator_ids
        self.qpos_torch = wp.to_torch(self.mjw_data.qpos)
        self.qvel_torch = wp.to_torch(self.mjw_data.qvel)
        # 获取批量质量矩阵 M: [nworld, nv, nv]
        self.ctrl_torch = wp.to_torch(self.mjw_data.ctrl)
        # 获取批量偏置力 bias: [nworld, nv]
        self.M_full = wp.to_torch(self.mjw_data.qM)  # nM 是 MuJoCo 的质量矩阵
        self.bias_full = wp.to_torch(self.mjw_data.qfrc_bias)
        self._timestep = timestep
        self.num_joints = len(jnt_dof_adrs)

        # 增益转为 Tensor 并移动到 GPU，统一扩展为 [nworld, num_joints]
        self._kp = self._expand_gain_tensor(kp)
        self._damping_ratio = self._expand_gain_tensor(damping_ratio)
        self._kd = 2 * torch.sqrt(torch.clamp(self._kp, min=1e-8)) * self._damping_ratio
        self._ki = self._expand_gain_tensor(ki)
        self.default_kp = self._kp.clone()
        self.default_damping_ratio = self._damping_ratio.clone()
        self.default_ki = self._ki.clone()

        self._min_effort = min_effort
        self._max_effort = max_effort

        # 积分项初始化 (Shape: [nworld, num_joints])
        self.integral = torch.zeros((nworld, self.num_joints), device=device)

    def _expand_gain_tensor(self, value):
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)

        if tensor.ndim == 0:
            return tensor.expand(self.nworld, self.num_joints).clone()

        if tensor.ndim == 1:
            if tensor.shape[0] == self.num_joints:
                return tensor.unsqueeze(0).expand(self.nworld, -1).clone()
            if tensor.shape[0] == self.nworld:
                return tensor.unsqueeze(-1).expand(-1, self.num_joints).clone()

        if tensor.ndim == 2 and tensor.shape == (self.nworld, self.num_joints):
            return tensor.clone()

        raise ValueError(
            f"gain tensor must be scalar, shape [{self.num_joints}], shape [{self.nworld}], "
            f"or shape [{self.nworld}, {self.num_joints}]"
        )

    def set_pd_gains(self, kp=None, damping_ratio=None, env_mask=None):
        kp_tgt = self._kp if kp is None else self._expand_gain_tensor(kp)
        damping_ratio_tgt = self._damping_ratio if damping_ratio is None else self._expand_gain_tensor(damping_ratio)

        if env_mask is None:
            self._kp = kp_tgt.clone()
            self._damping_ratio = damping_ratio_tgt.clone()
        else:
            env_mask = torch.as_tensor(env_mask, dtype=torch.bool, device=self.device)
            mask_2d = env_mask.unsqueeze(-1)
            self._kp = torch.where(mask_2d, kp_tgt, self._kp)
            self._damping_ratio = torch.where(mask_2d, damping_ratio_tgt, self._damping_ratio)

        self._kd = 2 * torch.sqrt(torch.clamp(self._kp, min=1e-8)) * self._damping_ratio

    def run(self, target_qpos, target_ddq_grp=None, grp_joint_id=None):
        """
        target_qpos: [nworld, num_joints]
        target_ddq_grp: [nworld, 1] (可选)
        """
        # 1. 获取当前状态 (从 mjw_data 映射的 torch 视图中切片)
        # qpos 形状为 [nworld, nq]，根据 dof_ids 提取对应的关节
        qpos = self.qpos_torch[:, self.jnt_qpos_adrs]
        qvel = self.qvel_torch[:, self.jnt_dof_adrs]

        # 2. 计算 PID 误差
        q_error = target_qpos - qpos
        self.integral += q_error * self._timestep

        # 计算目标加速度 target_ddq: [nworld, num_joints]
        target_ddq = self._kp * q_error - self._kd * qvel + self._ki * self.integral

        # 处理特殊的夹爪控制 (如有)
        if target_ddq_grp is not None and grp_joint_id is not None:
            target_ddq[:, grp_joint_id] = target_ddq_grp.squeeze()
            self.integral[:, grp_joint_id] = 0

        # 3. 动力学补偿：Torque = M * ddq + bias
        # 提取相关关节的子矩阵
        # M 形状应为 [nworld, num_joints, num_joints]
        M = self.M_full[:, self.jnt_dof_adrs, :][:, :, self.jnt_dof_adrs]
        bias = self.bias_full[:, self.jnt_dof_adrs] # bias 形状为 [nworld, num_joints]

        # 4. 执行批量矩阵乘法 (BMM)
        # torch.bmm 要求输入为 [B, N, M] 和 [B, M, P]
        # target_ddq.unsqueeze(-1) -> [nworld, num_joints, 1]
        torque = torch.bmm(M, target_ddq.unsqueeze(-1)).squeeze(-1)
        torque += bias

        # 5. 限幅并写入控制
        torque = torch.clamp(torque, self._min_effort, self._max_effort)

        # 将结果写入 mjw_data.ctrl
        # 注意：这里假设执行器顺序与关节顺序一致，如果不一致需通过 id 映射
        self.ctrl_torch[:, self.actuator_ids] = torque
