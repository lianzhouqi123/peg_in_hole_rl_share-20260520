from developsuit.assets.robots.pibot.pibot_gripper_left import PiBot_left
from developsuit.controllers.pibot_controller import PiBot_Controller
from developsuit.utils.kine import *
from developsuit.utils.mujoco_utils import *
from developsuit.utils.rm65_analytical_ik import rm65_analytical_ik_quat
from dm_control import mjcf
import mujoco.viewer
import mujoco
import math as m
import time
import os


class Zk_Controller:
    def __init__(self, timestep):
        # 基础阻抗参数（沿用之前优化的稳定值）
        self.mm = 1
        self.b = 100
        self.k = 2000
        self.timestep = timestep

        # 自适应参数
        self.omega = 0
        self.T_touch_old = 0

        # 稳定性优化参数
        self.flag_touch_current = False
        self.force_filter_alpha = 0.4  # 增强滤波，降低噪声干扰
        self.T_touch_filtered = 0
        # 关键：增大滞回下限，接触后需要力跌到负数才退出（防止短暂力降就脱离）
        self.force_hyst_on = 0.  # 非接触→接触的阈值
        self.force_hyst_off = -0  # 接触→非接触的阈值（负数，制造强粘性）
        self.omega_limit = 50000  # 降低自适应参数上限，减少过冲
        self.ddq_limit = 50000

        # 新增：接触后阻尼增强参数
        self.b_contact_extra = 500  # 接触时额外增加的阻尼
        self.force_gain = 1  # 力误差反馈增益（<1，降低响应强度）
        self.ddq_smooth_alpha = 1.  # 关节加速度平滑系数

    def update(self, de, fd, left_touch_force, jac_1d, djac_1d, grp_distance, grp_vel):
        # 1. 力信号滤波（更强滤波，消除接触瞬间的力波动）
        T_touch_raw = np.sum(abs(left_touch_force)) / 2
        self.T_touch_filtered = self.force_filter_alpha * T_touch_raw + \
                                (1 - self.force_filter_alpha) * self.T_touch_filtered

        # 2. 强粘性滞回接触判断（核心：接触后需要力跌到负数才退出）
        if not self.flag_touch_current:
            # 非接触→接触：力超过 force_hyst_on
            flag_touch = self.T_touch_filtered > self.force_hyst_on
        else:
            # 接触→非接触：力低于 force_hyst_off（负数，必须力反向才退出）
            flag_touch = self.T_touch_filtered > self.force_hyst_off
        self.flag_touch_current = flag_touch

        # 3. 间距与间距变化率
        d = grp_distance
        dd = jac_1d * grp_vel

        # 5. 阻抗控制核心（接触后增强阻尼+降低力反馈增益）
        if not flag_touch:
            # 非接触：位置阻抗控制（不变）
            pos_error = d - de
            # pos_error_sat = np.clip(pos_error, -2 * self.force_hyst_on, 2 * self.force_hyst_on)
            V = -(self.b * dd + self.k * pos_error + fd) / self.mm
        else:
            # 接触：增强阻尼 + 低增益力反馈（抑制反弹）
            force_error = fd - self.T_touch_filtered
            # 总阻尼 = 基础阻尼 + 额外阻尼
            b_total = self.b + self.b_contact_extra
            yita = 0.8 * b_total * self.timestep / (b_total * self.timestep + self.mm)
            omega_delta = yita / b_total * (fd - self.T_touch_old)
            self.omega += omega_delta
            # self.omega = np.clip(self.omega, -self.omega_limit, self.omega_limit)
            V = -(b_total * (dd + self.omega) + self.force_gain * force_error) / self.mm

        # 6. 控制量限幅（更严格的限幅）
        V = -V  # 开合方向相反

        # 7. 关节角加速度计算 + 平滑处理（防止突变）
        ddq_new = (V - djac_1d * grp_vel) / jac_1d
        # 一阶平滑：避免加速度突变导致反弹
        ddq = self.ddq_smooth_alpha * ddq_new + (1 - self.ddq_smooth_alpha) * getattr(self, 'ddq_last', 0)
        # ddq = np.clip(ddq, -self.ddq_limit, self.ddq_limit)
        self.ddq_last = ddq  # 保存上一时刻加速度用于平滑

        # 8. 存数据
        self.T_touch_old = self.T_touch_filtered

        return ddq

class PiBotEnv:
    def __init__(self, show_mode="no_show", timestep_control=0.01, config={}):
        self.config=config
        self.show_mode = show_mode
        # g = np.array([0., 0., 0.])
        g = np.array([0., 0., -9.8])
        self._timestep = 0.002
        self.timestep_control = timestep_control
        self.noise_v_linear = self.config.get("noise_v_linear", 0.005)
        self.noise_v_angular =self.config.get("noise_v_angular", 0.01)
        self.vise_base_pos_range = self.config.get("vise_base_pos_range", np.array([[-0.01, 0.01], [-0.01, 0.01], [0, 0]]))

        # 创建机器人
        pibot_name = config.get('pibot_file_name', "pibot_fc.xml")
        self.arm = PiBot_left(name=pibot_name)
        self.mjcf_model = self.arm.mjcf_model
        self.mjcf_model.option.timestep = self._timestep
        self.mjcf_model.option.gravity = g

        # 整合关节
        self.joints = self.arm.joints
        self.stock_joint = self.arm.stock_joint
        self.actuators = self.arm.actuators
        self.num_joints = len(self.joints)
        self.num_actuators = len(self.actuators)
        self.vise_open_joints = self.arm.vise_open_joints
        self.vise_base_joints = self.arm.vise_base_joints

        (self.contact, self._viewer, self.physics, self.controller, self.bound_actuators,
         self._step_start, self.cnt_permit_id, self.reset_data) \
            = None, None, None, None, None, None, None, None
        self.physics_create()
        self.zk_controller = Zk_Controller(timestep=self.timestep_control)
        self.wrench_eef_left = np.zeros([6], dtype=np.float64)
        self.xwrench_eef_left = np.zeros([6], dtype=np.float64)
        self.sensor_alpha = self.config.get('sensor_alpha', 0.45)
        self.wrench_noise_std = np.array(
            self.config.get('wrench_noise_std', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        self.wrench_drift_range = np.array(
            self.config.get('wrench_drift_range', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        self.if_wrench_drift_range = np.linalg.norm(self.wrench_drift_range) > 1e-6
        self.wrench_drift = np.zeros(6, dtype=np.float32)
        self.vise_open_dis = np.zeros([2])
        self.vise_base_pos = np.zeros([3])
        self.vise_base_pos_range = self.config.get("vise_base_pos_range", np.array([[-0.01, 0.01], [-0.01, 0.01], [0, 0]]))
        self.stock_pose_init_default = np.array([0., -0.35, 1.15, 0.5, -0.5, -0.5, 0.5], dtype=np.float32)
        self.stock_pose_init_range = np.array(
            self.config.get("stock_pose_init_range", np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])),
            dtype=np.float32,
        )

        # 控制给入标志
        self.arm_control_flag = False

        self.arm_vel_max_sec = np.ones([self.num_joints]) * 1
        self.num_micro_steps = int(self.timestep_control / self._timestep)

    def physics_create(self):
        # 关闭老的物理模型
        if self._viewer is not None:
            if self._viewer.is_running():
                self._viewer.close()
        if self.physics is not None:
            del self.physics
        # 生成物理模型
        self.physics = mjcf.Physics.from_mjcf_model(self.mjcf_model)
        self.physics.data.time = 0.0
        self.contact = self.physics.data.contact
        if self.show_mode == "show":
            self._viewer = mujoco.viewer.launch_passive(self.physics.model.ptr, self.physics.data.ptr)
            # self._viewer.cam.distance = 2.75
            # self._viewer.cam.elevation = -35
            # self._viewer.cam.lookat = [0.25, 0.5, 0]
            self._viewer.cam.distance = 2.75
            self._viewer.cam.elevation = -20
            self._viewer.cam.azimuth = 80
            self._viewer.cam.lookat = [0., 0.0, 0.5]

        # 控制器
        self.create_controller()

    def create_controller(self):
        # kp = np.array([6500, 6500, 6500, 6500, 6500, 6500, 200,
        #                6500, 6500, 6500, 6500, 6500, 6500, 200], dtype=np.float32)
        # damping_ratio = np.array([0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 200,
        #                0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 200], dtype=np.float32)
        self.controller = PiBot_Controller(physics=self.physics, joints=self.joints, actuators=self.actuators,
                                           timestep=self.timestep_control, min_effort=-60.0, max_effort=60.0,
                                           kp=6500, damping_ratio=0.32, ki=5000)
        # self.bound_actuators = self.physics.bind(self.actuators)
        self._timestep = self.physics.model.opt.timestep

    def render(self):
        if self.show_mode == "show":
            # 可视化
            self._step_start = time.time()
            self._viewer.sync()
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def fast_forward(self):
        """与 Warp 版接口对齐，刷新一次前向运动学并同步可视化。"""
        self.physics.forward()
        self.render()

    @staticmethod
    def _quat_multiply_wxyz(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=np.float32)

    def _sample_stock_pose_init(self):
        stock_pose_init = self.stock_pose_init_default.copy()
        rand_stock = np.zeros(3, dtype=np.float32)
        rand_stock[0] = np.random.uniform(self.stock_pose_init_range[0, 0], self.stock_pose_init_range[0, 1])
        rand_stock[1] = np.random.uniform(self.stock_pose_init_range[1, 0], self.stock_pose_init_range[1, 1])
        rand_stock[2] = np.random.uniform(self.stock_pose_init_range[2, 0], self.stock_pose_init_range[2, 1])

        stock_pose_init[0] += rand_stock[0]
        stock_pose_init[1] += rand_stock[1]

        yaw = rand_stock[2]
        yaw_quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float32)
        stock_pose_init[3:7] = self._quat_multiply_wxyz(yaw_quat, stock_pose_init[3:7])
        stock_pose_init[3:7] /= np.linalg.norm(stock_pose_init[3:7]) + 1e-8

        return stock_pose_init

    # 初始化环境
    def reset(self, if_reset_data=False, if_vise_open_rdm=False, if_vise_base_rdm=False):
        """
        环境重置。
        :param noise_v_linear:  末端笛卡尔线速度噪声上限 (m/s)
        :param noise_v_angular: 末端笛卡尔角速度噪声上限 (rad/s)
        """
        # 随机化虎钳夹角 偏置已71mm，0.0045对应80mm
        if if_vise_open_rdm:
            # self.vise_open_dis = np.ones([2]) * np.random.uniform(0, 0.0035)
            raise ValueError("没有维护，请再仔细规划")
        else:
            self.vise_open_dis = np.ones([2]) * self.config.get("vise_open_dis")  # 81mm
            
        if if_vise_base_rdm:
            x = np.random.uniform(self.vise_base_pos_range[0, 0], self.vise_base_pos_range[0, 1])
            y = np.random.uniform(self.vise_base_pos_range[1, 0], self.vise_base_pos_range[1, 1])
            qz = np.random.uniform(self.vise_base_pos_range[2, 0], self.vise_base_pos_range[2, 1])
            self.vise_base_pos = np.array([x, y, qz])
        else:
            self.vise_base_pos = np.zeros([3])

        vise_base_offset = self.vise_base_pos.copy()

        if self.if_wrench_drift_range:
            self.wrench_drift = np.random.uniform(-1.0, 1.0, size=6).astype(np.float32) * self.wrench_drift_range
        else:
            self.wrench_drift = np.zeros(6, dtype=np.float32)

        if if_reset_data:
            if hasattr(self, 'is_state_pool') and self.is_state_pool:
                # ---------------------------------------------------
                # 分支 A：从 .pt 状态池中随机抽样并转化为 NumPy
                # ---------------------------------------------------
                idx = np.random.randint(0, self.pool_size)
                self.sampled_idx = idx

                # 1. 恢复物理核心状态 (将 Torch Tensor 转为 NumPy Array)
                self.physics.data.qpos[:] = self.state_pool['qpos'][idx].numpy()
                self.physics.data.qvel[:] = self.state_pool['qvel'][idx].numpy()
                self.physics.data.ctrl[:] = self.state_pool['ctrl'][idx].numpy()
                self.physics.data.qacc_warmstart[:] = self.state_pool['qacc_warmstart'][idx].numpy()

                # 使用池子中的随机化目标
                # self.vise_open_dis = self.state_pool['vise_open_dis'][idx].numpy()
                loaded_vise_base_pos = self.state_pool['vise_base_pos'][idx].numpy()
                self.vise_base_pos = loaded_vise_base_pos + vise_base_offset

                self.physics.bind(self.vise_open_joints).qpos[:] = self.vise_open_dis
                self.physics.bind(self.vise_base_joints).qpos[:] = self.vise_base_pos

                self.physics.forward()

                # 2. 恢复控制与滤波状态
                self.controller.integral[:] = self.state_pool['arm_integral'][idx].numpy()
                self.wrench_eef_left[:] = self.state_pool['wrench_eef_left'][idx].numpy()
                self.xwrench_eef_left[:] = self.state_pool['xwrench_eef_left'][idx].numpy()
                if 'wrench_drift' in self.state_pool:
                    self.wrench_drift[:] = self.state_pool['wrench_drift'][idx].numpy()

                # 标量使用 .item() 提取
                self.zk_controller.omega = self.state_pool['zk_omega'][idx].item()
                self.zk_controller.T_touch_old = self.state_pool['zk_T_touch_old'][idx].item()
                self.zk_controller.flag_touch_current = self.state_pool['zk_flag_touch_current'][idx].item()
                self.zk_controller.T_touch_filtered = self.state_pool['zk_T_touch_filtered'][idx].item()
                self.zk_controller.ddq_last = self.state_pool['zk_ddq_last'][idx].item()

            elif hasattr(self, 'reset_data') and self.reset_data is not None:
                # ---------------------------------------------------
                # 分支 B：兼容老的 .npz 单状态导入
                # ---------------------------------------------------
                self.physics.set_state(self.reset_data['state_bytes'])

                if 'physics_ctrl' in self.reset_data:
                    self.physics.data.ctrl[:] = self.reset_data['physics_ctrl'][:]
                if 'qacc_warmstart' in self.reset_data:
                    self.physics.data.qacc_warmstart[:] = self.reset_data['qacc_warmstart'][:]

                loaded_vise_base_pos = np.array(self.physics.bind(self.vise_base_joints).qpos).reshape([-1])
                self.vise_base_pos = loaded_vise_base_pos + vise_base_offset
                self.physics.bind(self.vise_open_joints).qpos[:] = self.vise_open_dis
                self.physics.bind(self.vise_base_joints).qpos[:] = self.vise_base_pos

                self.physics.forward()

                if 'zk_omega' in self.reset_data:
                    self.zk_controller.omega = self.reset_data['zk_omega'].item()
                    self.zk_controller.T_touch_old = self.reset_data['zk_T_touch_old'].item()
                    self.zk_controller.flag_touch_current = self.reset_data['zk_flag_touch_current'].item()
                    self.zk_controller.T_touch_filtered = self.reset_data['zk_T_touch_filtered'].item()
                    self.zk_controller.ddq_last = self.reset_data['zk_ddq_last'].item()

                if 'arm_integral' in self.reset_data:
                    self.controller.integral[:] = self.reset_data['arm_integral'][:]

                if 'wrench_eef_left' in self.reset_data:
                    self.wrench_eef_left[:] = self.reset_data['wrench_eef_left'][:]
                    self.xwrench_eef_left[:] = self.reset_data['xwrench_eef_left'][:]
                if 'wrench_drift' in self.reset_data:
                    self.wrench_drift[:] = self.reset_data['wrench_drift'][:]
            else:
                raise ValueError("未加载任何数据！请确保调用过 load_environment_state(load_path)")

            # =======================================================
            # 【终极补丁】：笛卡尔空间一致性动态加噪
            # =======================================================
            if self.noise_v_linear > 0.0 or self.noise_v_angular > 0.0:
                # 1. 笛卡尔噪声意图 [6]
                noise_linear = np.random.uniform(-self.noise_v_linear, self.noise_v_linear, 3)
                noise_angular = np.random.uniform(-self.noise_v_angular, self.noise_v_angular, 3)
                noise_dxd = np.concatenate([noise_linear, noise_angular])

                # 2. 欺骗小脑：修改 RL 滤波器与导纳控制器状态
                # （假设你的 dm_control 环境的属性结构与 warp 版相似）
                if hasattr(self, 'dxd'):
                    self.dxd = noise_dxd.copy()
                if hasattr(self, 'dxd_filtered'):
                    self.dxd_filtered = noise_dxd.copy()
                if hasattr(self, 'eef_dn_controller'):
                    self.eef_dn_controller.x_c_dot = noise_dxd.copy()

                # 3. 逆向映射：伪逆雅可比映射为关节速度噪声 dq = J_pinv * dx
                # 前提：你的 dm_control 版本中存在 self.left_site_jac 属性返回当前 6x6 雅可比矩阵
                if hasattr(self, 'left_site_jac_physics'):
                    jac = self.left_site_jac_physics
                    jac_t = jac.T
                    lambda_sq = 1e-4
                    A = jac_t @ jac + lambda_sq * np.eye(6)
                    B = jac_t @ noise_dxd
                    noise_dq = np.linalg.solve(A, B)
                    noise_dq = np.nan_to_num(noise_dq, nan=0.0)

                    # 4. 注射躯体：仅修改机械臂的 6 个关节速度，保护夹爪！
                    current_qvel_left = np.array(self.physics.bind(self.arm.joint_lefts).qvel).reshape([-1])
                    self.physics.bind(self.arm.joint_lefts).qvel[:] = current_qvel_left + noise_dq

                    # 再次前向传播以生效
                    self.physics.forward()
                else:
                    print("警告: 缺少 left_site_jac_physics 属性，无法在 dm_control 环境中执行笛卡尔加噪。")

        else:
            arm_init = np.array([0, -70, 70, 0., 80., 0., 0. ]) / 180 * np.pi

            # 重置物理系统，带入值
            with self.physics.reset_context():
                arm_init = arm_init.reshape([-1]).copy()
                self.physics.bind(self.joints).qpos[:] = arm_init
                self.physics.bind(self.stock_joint).qpos[:] = self.stock_pose_init_default
                self.physics.bind(self.vise_base_joints).qpos[:] = self.vise_base_pos
                self.physics.bind(self.vise_open_joints).qpos[:] = self.vise_open_dis
                self.physics.data.qvel[:] = 0.0
                self.physics.data.ctrl[:] = 0.0
                self.physics.data.qacc_warmstart[:] = 0.0

                self.arm_control_flag = False

                # 强制清零物理控制状态，避免残留动量
                if hasattr(self, 'dxd'):
                    self.dxd = np.zeros(6)
                if hasattr(self, 'dxd_filtered'):
                    self.dxd_filtered = np.zeros(6)
                if hasattr(self, 'eef_dn_controller'):
                    self.eef_dn_controller.x_c_dot = np.zeros(6)

                self.controller.integral[:] = 0.0
                self.wrench_eef_left[:] = 0.0
                self.xwrench_eef_left[:] = 0.0
                self.zk_controller.omega = 0.0
                self.zk_controller.T_touch_old = 0.0
                self.zk_controller.flag_touch_current = False
                self.zk_controller.T_touch_filtered = 0.0
                self.zk_controller.ddq_last = 0.0

            self.physics.step()

        self.fast_forward()

    def ctrl(self, joints):
        # max_arm_step = (self.arm_vel_max_sec * self._timestep).reshape([self.num_joints])
        # joint_step = np.clip(joint_step, -max_arm_step, max_arm_step)
        self.controller.run(joints.reshape([-1]))
        self.arm_control_flag = True

    def ctrl_touch(self, joints, de, fd):
        target_grp_ddq = self.zk_controller.update(
            de,
            fd,
            self.left_touch_force,
            self.left_gripper_jac,
            self.left_gripper_djac,
            self.grp_distance,
            self.arm_qvel[6],
        )

        joints = joints.reshape([-1])
        joints_ctrl = np.zeros([self.num_joints])
        joints_ctrl[:joints.shape[0]] = joints

        # 执行
        # max_arm_step = (self.arm_vel_max_sec * self._timestep).reshape([self.num_joints])
        # joint_step = np.clip(joint_step, -max_arm_step, max_arm_step)

        self.controller.run(joints_ctrl.reshape([-1]), target_ddq_grp=target_grp_ddq, grp_joint_id=6)
        self.arm_control_flag = True

    def ctrl_eef(self, pose_d, if_touch=False, de=0.06, fd_grp=45.0):
        """
        :param pose_d: [x, y, z, quat_w, quat_x, quat_y, quat_z]
        :param if_touch:
        :param de:期望夹爪闭合宽度
        :param fd_grp:期望夹爪抓取力
        :return:
        """
        joints = self.ikine_analytical_left(pose_d)
        if joints is None:
            return False

        if if_touch:
            self.ctrl_touch(joints, de, fd_grp)
        else:
            self.ctrl(joints)

        return True

    def physics_step(self):
        # 如果没被控过，则全给0
        if not self.arm_control_flag:
            joint_arm_step = np.zeros(self.num_joints)
            self.controller.run(joint_arm_step)
        # 被控标志清零
        self.arm_control_flag = False

        # 物理系统执行
        for _ in range(self.num_micro_steps):
            self.physics.step()
            # 与 Warp 版一致：在每个 500Hz 物理微步上都更新一次力觉噪声与一级低通。
            raw_wrench = self.wrench_eef_left_raw
            raw_xwrench = self.xwrench_eef_left_raw
            white_noise_w = np.random.randn(6).astype(np.float32) * self.wrench_noise_std
            white_noise_x = np.random.randn(6).astype(np.float32) * self.wrench_noise_std
            noisy_raw_wrench = raw_wrench + self.wrench_drift + white_noise_w
            noisy_raw_xwrench = raw_xwrench + self.wrench_drift + white_noise_x
            self.wrench_eef_left = self.sensor_alpha * noisy_raw_wrench + (1 - self.sensor_alpha) * self.wrench_eef_left
            self.xwrench_eef_left = self.sensor_alpha * noisy_raw_xwrench + (1 - self.sensor_alpha) * self.xwrench_eef_left

        self.render()

    def save_environment_state(self, save_path):
        """
        保存当前仿真状态（包含外部控制指令与底层求解器热启动缓存）
        """
        state_bytes = self.physics.get_state()

        extra_data = {
            'time': self.physics.time(),
            'state_bytes': state_bytes,

            # 1. 外部控制指令
            'physics_ctrl': self.physics.data.ctrl.copy(),

            # =======================================================
            # 【终极补丁】：保存 MuJoCo 求解器的热启动缓存！
            # =======================================================
            'qacc_warmstart': self.physics.data.qacc_warmstart.copy(),

            # 2. Zk_Controller 状态
            'zk_omega': float(np.asarray(self.zk_controller.omega).item()),
            'zk_T_touch_old': float(np.asarray(self.zk_controller.T_touch_old).item()),
            'zk_flag_touch_current': bool(np.asarray(self.zk_controller.flag_touch_current).item()),
            'zk_T_touch_filtered': float(np.asarray(self.zk_controller.T_touch_filtered).item()),
            'zk_ddq_last': float(np.asarray(getattr(self.zk_controller, 'ddq_last', 0.0)).item()),

            # 3. 控制器积分与滤波状态
            'arm_integral': self.controller.integral.copy(),
            'wrench_eef_left': self.wrench_eef_left.copy(),
            'xwrench_eef_left': self.xwrench_eef_left.copy()
        }

        np.savez(save_path, **extra_data)

    def load_environment_state(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"找不到状态文件: {load_path}")
            
        load_path_str = str(load_path)

        if load_path_str.endswith('.pt'):
            import torch  # 确保加载 torch 以读取 .pt 文件
            self.is_state_pool = True
            # 将张量加载到 CPU 内存中，方便后续转为 NumPy 数组
            self.state_pool = torch.load(load_path, map_location='cpu')
            self.pool_size = self.state_pool['qpos'].shape[0]
            print(f"[*] 成功加载离线状态池 (.pt)，容量: {self.pool_size} 个环境快照。")
            
        elif load_path_str.endswith('.npz'):
            self.is_state_pool = False
            loaded = np.load(load_path, allow_pickle=True)
            self.reset_data = {k: loaded[k] for k in loaded.files}
            print(f"[*] 成功加载单状态文件 (.npz)。")
            
        else:
            raise ValueError("不支持的文件格式，请使用 .pt 或 .npz 后缀的文件。")

    @property
    def arm_qpos(self):
        return np.array(self.physics.bind(self.arm.joints).qpos).reshape([-1])

    @property
    def arm_qvel(self):
        return np.array(self.physics.bind(self.arm.joints).qvel).reshape([-1])

    @property
    def arm_qpos_left(self):
        return np.array(self.physics.bind(self.arm.joint_lefts).qpos).reshape([-1])

    @property
    def arm_qvel_left(self):
        return np.array(self.physics.bind(self.arm.joint_lefts).qvel).reshape([-1])

    @property
    def stock_pose(self):
        return np.array(self.physics.bind(self.arm.stock_joint).qpos).reshape([-1])

    @property
    def grasp_pose(self):
        pos = np.array(self.physics.bind(self.arm.grasp_site).xpos).reshape([-1])
        mat = np.array(self.physics.bind(self.arm.grasp_site).xmat).reshape([3, 3])
        quat = mat2quat(mat)
        pose = np.concatenate([pos, quat])
        return pose

    @property
    def stock_vel(self):
        return np.array(self.physics.bind(self.arm.stock_joint).qvel).reshape([-1])

    @property
    def xpos(self):
        ee_poses = []
        for ii in range(len(self.arm.eef_site)):
            eef_site = self.arm.eef_site[ii]
            ee_pos = self.physics.bind(eef_site).xpos.reshape([-1])
            ee_poses.append(ee_pos)

        return np.vstack(ee_poses).reshape([-1, 3])

    @property
    def xpos_left(self):
        return self.xpos

    @property
    def xmat(self):
        ee_mats = []
        for ii in range(len(self.arm.eef_site)):
            eef_site = self.arm.eef_site[ii]
            ee_mat = self.physics.bind(eef_site).xmat.reshape([3, 3])
            ee_mats.append(ee_mat)

        return np.vstack(ee_mats).reshape([-1, 3, 3])

    @property
    def xmat_left(self):
        return self.xmat

    @property
    def xpose(self):
        ee_poses = []
        for ii in range(len(self.arm.eef_site)):
            eef_site = self.arm.eef_site[ii]
            ee_pos = self.physics.bind(eef_site).xpos.reshape([-1])
            ee_mat = self.physics.bind(eef_site).xmat.reshape([3, 3])
            ee_quat = mat2quat(ee_mat)
            ee_poses.append(np.hstack((ee_pos, ee_quat)))

        return np.vstack(ee_poses).reshape([-1, 7])

    @property
    def xpose_left(self):
        return self.xpose

    @property
    def xvel(self):
        ee_vels = []
        for ii in range(len(self.arm.eef_site)):
            eef_site = self.arm.eef_site[ii]
            eef_mat = np.array(self.physics.bind(eef_site).xmat).reshape([3, 3])
            local_linvel = np.array(self.physics.bind(self.arm.eef_vel_sensors[ii]).sensordata).reshape([3, 1])
            local_angvel = np.array(self.physics.bind(self.arm.eef_gyro_sensors[ii]).sensordata).reshape([3, 1])
            world_linvel = (eef_mat @ local_linvel).reshape([3])
            world_angvel = (eef_mat @ local_angvel).reshape([3])
            ee_vels.append(np.hstack((world_linvel, world_angvel)))

        return np.vstack(ee_vels).reshape([-1, 6])

    @property
    def xvel_left(self):
        return self.xvel[0]

    @property
    def base_xpos(self):
        base_poses = []
        for ii in range(len(self.arm.base_site)):
            base_site = self.arm.base_site[ii]
            base_pos = self.physics.bind(base_site).xpos.reshape([-1])
            base_poses.append(base_pos)

        return np.vstack(base_poses).reshape([-1, 3])

    @property
    def base_xmat(self):
        base_mats = []
        for ii in range(len(self.arm.base_site)):
            base_site = self.arm.base_site[ii]
            base_mat = self.physics.bind(base_site).xmat.reshape([3, 3])
            base_mats.append(base_mat)

        return np.stack(base_mats, axis=0).reshape([-1, 3, 3])

    @property
    def base_xpos_left(self):
        return self.base_xpos

    @property
    def base_xmat_left(self):
        return self.base_xmat

    @property
    def base_omega(self):
        omega_bases = []
        for ii in range(len(self.arm.base_gyro_sensors)):
            base_gyro_sensor = self.arm.base_gyro_sensors[ii]
            base_site = self.arm.base_site[ii]
            omega_base = np.array(self.physics.bind(base_gyro_sensor).sensordata).reshape([3, 1])
            base_mat = self.physics.bind(base_site).xmat.reshape([3, 3])
            omega_base = (base_mat @ omega_base).reshape([3])
            omega_bases.append(omega_base)

        omega_bases = np.vstack(omega_bases)
        return omega_bases

    @property
    def eef_linvel(self):
        return self.xvel[:, 0:3]

    @property
    def eef_omega(self):
        return self.xvel[:, 3:6]

    @property
    def left_touch_force(self):
        return np.array(self.physics.bind(self.arm.left_touch_sensors).sensordata).reshape([-1])

    @property
    def vise_touch(self):
        return np.array(self.physics.bind(self.arm.vise_touch_sensors).sensordata).sum()
    
    @property
    def vise_surface_touch(self):
        return np.array(self.physics.bind(self.arm.vise_surface_touch_sensors).sensordata).sum()

    # geom编号
    @property
    def geom_id(self):
        return np.array([self.physics.bind(self.arm.geoms).element_id]).reshape([-1])

    def get_ee_pose_left(self):
        # 由DH_m计算的ee_pos
        ee_pos, ee_quat = fkine_ee_m(self.arm.DH_m, self.arm_qpos[:6], self.arm.DH_m_end,
                                     R_base=self.base_xmat[0], base=self.base_xpos[0])
        ee_pose = np.hstack((ee_pos.reshape([3]), ee_quat.reshape([4])))

        return ee_pose

    def get_jac_djac_left(self):
        J, dJ = jac_m(self.arm.DH_m, self.arm_qpos[:6], self.arm.DH_m_end, dq=self.arm_qvel[:6],
                      R_base=self.base_xmat[0], base=self.base_xpos[0], omega_base=self.base_omega[0])

        return J, dJ

    def ikine_left(self, pose_d):
        pose_d = pose_d.reshape([-1])
        xd = pose_d[0:3]
        quatd = pose_d[3:7]
        qd, err = ikine_m(self.arm.DH_m, self.arm.DH_m_end, self.arm_qpos[:6], xd, quatd,
                          R_base=self.base_xmat[0], base=self.base_xpos[0], omega_base=self.base_omega[0])

        return qd.reshape([-1])

    def ikine_analytical_left(self, pose_d):
        pose_d = pose_d.reshape([-1])
        xd = pose_d[0:3].reshape([3, 1])
        matd = quat2mat(pose_d[3:7])
        xd_base = (self.base_xmat[0].T @ (xd - self.base_xpos[0].reshape([3, 1]))).reshape([-1])
        matd_base = mat2quat(self.base_xmat[0].T @ matd)
        target_arm_q_all = rm65_analytical_ik_quat(target_pos=xd_base, target_quat_wxyz=matd_base, d6_tcp_m=0.161 + 0.2)
        if target_arm_q_all is None:
            return None

        diff = target_arm_q_all - self.arm_qpos[0:6]
        distances = np.linalg.norm(diff, axis=1)
        best_idx = int(np.argmin(distances))
        best_q = target_arm_q_all[best_idx]

        return best_q.reshape([-1])

    @property
    def left_site_jac_physics(self):
        """
        获取指定site的6维空间雅可比矩阵（平移+旋转）
        返回:
            jacobian: np.ndarray - 6×nv的雅可比矩阵（nv为关节自由度）
        """
        site_name = self.arm.eef_site_name[0]
        # 1. 获取site ID并验证
        site_id = mujoco.mj_name2id(
            self.physics.model.ptr,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name
        )
        if site_id == -1:
            raise ValueError(f"未找到site: {site_name}，请检查模型中的site名称")

        # 3. 初始化雅可比存储数组
        nv = self.physics.model.ptr.nv  # 关节自由度数量
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))

        # 4. 调用原生API计算site雅可比
        mujoco.mj_jacSite(
            self.physics.model.ptr,
            self.physics.data.ptr,
            jacp,
            jacr,
            site_id
        )

        # 5. 组合为完整的6维雅可比
        jacobian = np.vstack((jacp, jacr))

        left_joint_ids = self.physics.bind(self.arm.joint_lefts).element_id
        left_joint_dof_ins = self.physics.model.jnt_dofadr[left_joint_ids]
        jacobian_left = jacobian[:, left_joint_dof_ins]

        return jacobian_left

    @property
    def left_site_jac(self):
        return self.left_site_jac_physics

    @property
    def grp_distance(self):
        pos = self.physics.bind(self.arm.gripper_left_site).xpos
        return np.linalg.norm(pos[1] - pos[0])

    @property
    def left_gripper_jac_physics(self):
        """
        获取指定site的闭环 1D 空间雅可比（开合方向投影）
        J*dq1 = dx = J1*dq1 + J2*dq2 -> dq1 = -dq2 -> J = J1 - J2
        J_1d = J * R * [0 0 1]
        q1: "Left_1_Joint_gripper_left", q2: "Left_Support_Joint_gripper_left"
        R: site固定系到全局

        由于q1和q2不严格相等（mujoco原因），因此和理论值有差别
        """
        # 1. 【关键修复】必须获取 Site 真正挂载的 Body ID (即 Left_Support_Link)
        site_element = self.physics.bind(self.arm.gripper_left_site[0])
        site_xpos = site_element.xpos
        body_id = self.physics.model.site_bodyid[site_element.element_id]

        nv = self.physics.model.ptr.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))

        # 传入正确的 body_id 计算完整拓扑雅可比
        mujoco.mj_jac(self.physics.model.ptr, self.physics.data.ptr, jacp, jacr, site_xpos, body_id)

        # 2. 【关键修复】获取主动关节和支撑关节的索引
        active_jnt_id = mujoco.mj_name2id(self.physics.model.ptr, mujoco.mjtObj.mjOBJ_JOINT,
                                          "Left_1_Joint_gripper_left")
        support_jnt_id = mujoco.mj_name2id(self.physics.model.ptr, mujoco.mjtObj.mjOBJ_JOINT,
                                           "Left_Support_Joint_gripper_left")

        active_dof = self.physics.model.jnt_dofadr[active_jnt_id]
        support_dof = self.physics.model.jnt_dofadr[support_jnt_id]

        # 3. 【关键修复】闭环运动学合成: J_true = J_active - J_support  J*dq1 = dx = J1*dq1 + J2*dq2 -> dq1 = -dq2 -> J = J1 - J2
        jac_active_3d = jacp[:, active_dof]
        jac_support_3d = jacp[:, support_dof]
        jac_true_3d = jac_active_3d - jac_support_3d

        # 4. 投影到开合方向
        close_dir = (site_element.xmat.reshape([3, 3]) @ np.array([.0, .0, 1.0]).reshape([3, 1])).reshape([-1])

        jac_1d = np.dot(jac_true_3d, close_dir)
        jac_1d *= 2.0  # 两个爪子

        return jac_1d

    @property
    def left_gripper_djac_physics(self):
        """
        获取指定site的闭环 1D 空间雅可比导数

        J*dq1 = dx = J1*dq1 + J2*dq2 -> dq1 = -dq2 -> J = J1 - J2
        J_1d = J * R * [0 0 1]

        dJ_1d = dJ * R * [0 0 1] + J * (omega * R) * [0 0 1]
        q1: "Left_1_Joint_gripper_left", q2: "Left_Support_Joint_gripper_left"
        R: site固定系到全局 omega：site固定系的角速度（在全局系下）

        由于q1和q2不严格相等（mujoco原因），因此和理论值有差别
        """
        site_element = self.physics.bind(self.arm.gripper_left_site[0])
        site_xpos = site_element.xpos
        body_id = self.physics.model.site_bodyid[site_element.element_id]

        nv = self.physics.model.ptr.nv
        jacp_dot = np.zeros((3, nv))
        jacr_dot = np.zeros((3, nv))

        # 1. 计算 \dot{J}
        mujoco.mj_jacDot(self.physics.model.ptr, self.physics.data.ptr, jacp_dot, jacr_dot, site_xpos, body_id)

        # 2. 为了计算补偿项，我们还需要当前的位置雅可比 J 和旋转雅可比
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jac(self.physics.model.ptr, self.physics.data.ptr, jacp, jacr, site_xpos, body_id)

        # 获取索引
        active_jnt_id = mujoco.mj_name2id(self.physics.model.ptr, mujoco.mjtObj.mjOBJ_JOINT,
                                          "Left_1_Joint_gripper_left")
        support_jnt_id = mujoco.mj_name2id(self.physics.model.ptr, mujoco.mjtObj.mjOBJ_JOINT,
                                           "Left_Support_Joint_gripper_left")

        active_dof = self.physics.model.jnt_dofadr[active_jnt_id]
        support_dof = self.physics.model.jnt_dofadr[support_jnt_id]

        # 3. 闭环合成: \dot{J}_true 和 J_true   J*dq1 = dx = J1*dq1 + J2*dq2 -> dq1 = -dq2 -> J = J1 - J2
        djac_true_3d = jacp_dot[:, active_dof] - jacp_dot[:, support_dof]
        jac_true_3d = jacp[:, active_dof] - jacp[:, support_dof]

        # 4. 计算当前的开合方向 close_dir
        close_dir = (site_element.xmat.reshape([3, 3]) @ np.array([.0, .0, 1.0]).reshape([3, 1])).reshape([-1])

        # 【核心物理修复】：计算投影轴由于机械臂挥动产生的旋转导数 dn/dt
        # 1) 利用旋转雅可比和当前关节速度，算出 Site 的绝对角速度 omega
        # omega_site = self.physics.data.cvel[body_id][:3]
        # 2) 向量旋转导数公式：dn/dt = omega x n
        close_dir_dot = np.cross(self.physics.data.cvel[body_id][:3], close_dir)

        # 5. 完整的微积分乘积法则: d(J·n)/dt = \dot{J}·n + J·\dot{n}
        term1 = np.dot(djac_true_3d, close_dir)
        term2 = np.dot(jac_true_3d, close_dir_dot)
        djac_1d = term1 + term2

        djac_1d *= 2.0  # 两个爪子

        return djac_1d

    @property
    def left_gripper_jac(self):
        """
        基于平行连杆机构解析推导的开合方向 1D 雅可比 [nworld] 解析方法
        """
        q = np.array(self.physics.bind(self.arm.joint_left_grippers).qpos)[0]

        theta_offset = 42.79 / 180.0 * np.pi
        gripper_L = 0.055

        return -2.0 * gripper_L * np.sin(theta_offset - q)

    @property
    def left_gripper_jac_mjw(self):
        """与 Warp 版命名对齐；FC_HF 下显式使用解析夹爪模型。"""
        return self.left_gripper_jac

    @property
    def left_gripper_djac(self):
        """
        基于平行连杆机构解析推导的开合方向 1D 雅可比
        """
        q = np.array(self.physics.bind(self.arm.joint_left_grippers).qpos)[0]
        dq = np.array(self.physics.bind(self.arm.joint_left_grippers).qvel)[0]

        theta_offset = 42.79 / 180.0 * np.pi
        gripper_L = 0.055

        return 2.0 * gripper_L * np.cos(theta_offset - q) * dq

    @property
    def wrench_eef_left_raw(self):
        F_eef = np.array(self.physics.bind(self.arm.F_sensor_left).sensordata, dtype=np.float32).reshape([-1])
        T_eef = np.array(self.physics.bind(self.arm.T_sensor_left).sensordata, dtype=np.float32).reshape([-1])
        return np.hstack((F_eef, T_eef))

    @property
    def xwrench_eef_left_raw(self):
        # 测的力是link6给link5的力 （link5受到的力）
        xmat = self.physics.bind(self.arm.wrench_site_left).xmat.reshape([3, 3])
        F_eef = np.array(self.physics.bind(self.arm.F_sensor_left).sensordata, dtype=np.float32).reshape([3, 1])
        T_eef = np.array(self.physics.bind(self.arm.T_sensor_left).sensordata, dtype=np.float32).reshape([3, 1])
        xF = (xmat @ F_eef).reshape([-1])
        xT = (xmat @ T_eef).reshape([-1])
        xwrench = np.hstack((xF, xT))
        return xwrench

    @property
    def insert_pose(self):
        pos = np.array(self.physics.bind(self.arm.insert_site).xpos).reshape([-1])
        mat = np.array(self.physics.bind(self.arm.insert_site).xmat).reshape([3, 3])
        quat = mat2quat(mat)
        pose = np.concatenate([pos, quat])
        return pose

    @property
    def insert_world_pose(self):
        pos = np.array(self.physics.bind(self.arm.insert_world_site).xpos).reshape([-1])
        mat = np.array(self.physics.bind(self.arm.insert_world_site).xmat).reshape([3, 3])
        quat = mat2quat(mat)
        pose = np.concatenate([pos, quat])
        return pose
    
    @property
    def eef_real_pose(self):
        pos = np.array(self.physics.bind(self.arm.eef_real_site).xpos).reshape([-1])
        mat = np.array(self.physics.bind(self.arm.eef_real_site).xmat).reshape([3, 3])
        quat = mat2quat(mat)
        pose = np.concatenate([pos, quat])
        return pose

    def monitor_memory_peaks(self):
        """
        与 Warp 版接口对齐：返回当前接触数和约束数峰值，供调试使用。
        """
        current_ncon = int(self.physics.data.ncon)
        current_nefc = int(self.physics.data.nefc)
        return current_ncon, current_nefc
