import os
import mujoco
import warp as wp
from developsuit.utils.transform_utils_torch import *

# 初始化 Warp（mujoco_warp 必须先初始化）
wp.init()
device="cuda" if torch.cuda.is_available() else "cpu"

class PiBot_left_warp:
    def __init__(self, num_envs=1, name="pibot_left.xml"):
        self.num_envs = num_envs

        # 加载 XML
        xml_path = os.path.join(os.path.dirname(__file__), name)
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # # 2. 核心：推送到 GPU 并开启 batch
        # # mjw_model 是共享的（节省显存），mjw_data 会创建 num_envs 个独立副本
        # self.mjw_model = mjw.put_model(self.mj_model)
        # self.mjw_data = mjw.put_data(self.mj_model, self.mj_data, nworld=num_envs)

        # 2. 查找所有元素的 ID（替代 dm_control 的 find 方法）
        # ===== 关节 ID 查找 =====
        joint_names = [
            "joint1_left", "joint2_left", "joint3_left", "joint4_left", "joint5_left", "joint6_left",
            "Left_1_Joint_gripper_left",
        ]
        self.joint_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_JOINT, joint_names)
        self.joint_qpos_adrs = self.mj_model.jnt_qposadr[self.joint_ids]
        self.joint_qpos_adrs = torch.tensor(self.joint_qpos_adrs, dtype=torch.long, device=device)
        self.joint_dof_adrs = self.mj_model.jnt_dofadr[self.joint_ids]
        self.joint_dof_adrs = torch.tensor(self.joint_dof_adrs, dtype=torch.long, device=device)

        joint_names_left = ["joint1_left", "joint2_left", "joint3_left", "joint4_left", "joint5_left", "joint6_left"]
        self.joint_left_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_JOINT, joint_names_left)
        self.joint_left_qpos_adrs = self.mj_model.jnt_qposadr[self.joint_left_ids]
        self.joint_left_qpos_adrs = torch.tensor(self.joint_left_qpos_adrs, dtype=torch.long, device=device)
        self.joint_left_dof_adrs = self.mj_model.jnt_dofadr[self.joint_left_ids]
        self.joint_left_dof_adrs = torch.tensor(self.joint_left_dof_adrs, dtype=torch.long, device=device)

        joint_name_left_gripper = ["Left_1_Joint_gripper_left"]
        self.joint_left_gripper_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_JOINT, joint_name_left_gripper)
        self.joint_left_gripper_qpos_adrs = self.mj_model.jnt_qposadr[self.joint_left_gripper_ids]
        self.joint_left_gripper_qpos_adrs = torch.tensor(self.joint_left_gripper_qpos_adrs, dtype=torch.long, device=device)
        self.joint_left_gripper_dof_adrs = self.mj_model.jnt_dofadr[self.joint_left_gripper_ids]
        self.joint_left_gripper_dof_adrs = torch.tensor(self.joint_left_gripper_dof_adrs, dtype=torch.long,
                                                         device=device)

        joint_name_vise_front = ["vise_front_joint"]
        self.joint_vise_front_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_JOINT, joint_name_vise_front)
        self.joint_vise_front_qpos_adrs = self.mj_model.jnt_qposadr[self.joint_vise_front_ids]
        self.joint_vise_front_qpos_adrs = torch.tensor(self.joint_vise_front_qpos_adrs, dtype=torch.long, device=device)
        joint_name_vise_back = ["vise_back_joint"]
        self.joint_vise_back_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_JOINT, joint_name_vise_back)
        self.joint_vise_back_qpos_adrs = self.mj_model.jnt_qposadr[self.joint_vise_back_ids]
        self.joint_vise_back_qpos_adrs = torch.tensor(self.joint_vise_back_qpos_adrs, dtype=torch.long, device=device)
        joint_vise_base = ["vise_x_joint", "vise_y_joint", "vise_qz_joint"]
        self.joint_vise_base_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_JOINT, joint_vise_base)
        self.joint_vise_base_qpos_adrs = self.mj_model.jnt_qposadr[self.joint_vise_base_ids]
        self.joint_vise_base_qpos_adrs = torch.tensor(self.joint_vise_base_qpos_adrs, dtype=torch.long, device=device)

        # 获取起始 ID
        start_id = self._find_element_ids(mujoco.mjtObj.mjOBJ_JOINT, ["stock"])[0]
        start_adr = self.mj_model.jnt_qposadr[start_id]
        # 针对 qpos (7维: 3位置 + 4四元数)
        self.stock_qpos_adrs = torch.arange(start_adr, start_adr + 7, dtype=torch.long, device=device)
        # 针对 qvel (6维: 3线速度 + 3角速度)
        # 注意：自由关节在 nv 中只占 6 维，对应的起始地址通常也是 start_id（如果之前没有跳变）
        # 但更安全的方法是查 mj_model.jnt_dofadr[start_id]
        start_nv_adr = self.mj_model.jnt_dofadr[start_id]
        self.stock_qvel_adrs = torch.arange(start_nv_adr, start_nv_adr + 6, dtype=torch.long, device=device)

        # ===== 位点（site）ID 查找 =====
        self.eef_site_names = ["end_effector_left"]
        self.eef_site_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_SITE, self.eef_site_names)
        self.eef_site_ids = torch.tensor(self.eef_site_ids, dtype=torch.long, device=device)

        self.eef_body_names = ["Link6_left"]
        self.eef_body_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_BODY, self.eef_body_names)
        self.eef_body_ids = torch.tensor(self.eef_body_ids, dtype=torch.long, device=device)

        base_site_names = ["base_site_left"]
        self.base_site_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_SITE, base_site_names)
        self.base_site_ids = torch.tensor(self.base_site_ids, dtype=torch.long, device=device)

        grasp_site_names = ["grasp_stock_site"]
        self.grasp_site_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_SITE, grasp_site_names)
        self.grasp_site_ids = torch.tensor(self.grasp_site_ids, dtype=torch.long, device=device)

        insert_site_names = ["insert_site"]
        self.insert_site_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_SITE, insert_site_names)
        self.insert_site_ids = torch.tensor(self.insert_site_ids, dtype=torch.long, device=device)

        eef_real_site_names = ["eef_real_left"]
        self.eef_real_site_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_SITE, eef_real_site_names)
        self.eef_real_site_ids = torch.tensor(self.eef_real_site_ids, dtype=torch.long, device=device)

        insert_world_site_names = ["insert_world_site"]
        self.insert_world_site_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_SITE, insert_world_site_names)
        self.insert_world_site_ids = torch.tensor(self.insert_world_site_ids, dtype=torch.long, device=device)

        self.gripper_left_site_names = ["gripper_left_1_site", "gripper_left_2_site"]
        self.gripper_left_site_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_SITE, self.gripper_left_site_names)
        self.gripper_left_site_ids = torch.tensor(self.gripper_left_site_ids, dtype=torch.long, device=device)

        self.gripper_left_body_names = ["Left_Support_Link_gripper_left", "Right_Support_Link_gripper_left"]
        self.gripper_left_body_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_BODY, self.gripper_left_body_names)
        self.gripper_left_body_ids = torch.tensor(self.gripper_left_body_ids, dtype=torch.long, device=device)

        self.wrench_site_left_names = ["eef_left_wrench"]
        self.wrench_site_left_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_SITE, self.wrench_site_left_names)
        self.wrench_site_left_ids = torch.tensor(self.wrench_site_left_ids, dtype=torch.long, device=device)

        # ===== 传感器（sensor）ID 查找 =====
        base_gyro_names = ["base_gyro_left"]
        self.base_gyro_sensor_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_SENSOR, base_gyro_names)
        self.base_gyro_sensor_adrs = [self._find_sensor_adr(sid) for sid in self.base_gyro_sensor_ids]
        self.base_gyro_sensor_adrs = torch.tensor(self.base_gyro_sensor_adrs, dtype=torch.long, device=device)

        left_touch_names = ["gripper_left_1", "gripper_left_2"]
        self.left_touch_sensor_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_SENSOR, left_touch_names)
        self.left_touch_sensor_adrs = [self._find_sensor_adr(sid) for sid in self.left_touch_sensor_ids]
        self.left_touch_sensor_adrs = torch.tensor(self.left_touch_sensor_adrs, dtype=torch.long, device=device)

        self.F_sensor_left_names = ["F_eef_left"]
        self.F_sensor_left_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_SENSOR, self.F_sensor_left_names)
        self.F_sensor_left_adrs = [self._find_sensor_adr(sid) for sid in self.F_sensor_left_ids]
        self.F_sensor_left_adrs = torch.tensor(self.F_sensor_left_adrs, dtype=torch.long, device=device)

        self.T_sensor_left_names = ["T_eef_left"]
        self.T_sensor_left_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_SENSOR, self.T_sensor_left_names)
        self.T_sensor_left_adrs = [self._find_sensor_adr(sid) for sid in self.T_sensor_left_ids]
        self.T_sensor_left_adrs = torch.tensor(self.T_sensor_left_adrs, dtype=torch.long, device=device)

        eef_gyro_names = ["eef_gyro_left"]
        self.eef_gyro_sensor_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_SENSOR, eef_gyro_names)
        self.eef_gyro_sensor_adrs = [self._find_sensor_adr(sid) for sid in self.eef_gyro_sensor_ids]
        self.eef_gyro_sensor_adrs = torch.tensor(self.eef_gyro_sensor_adrs, dtype=torch.long, device=device)

        eef_vel_names = ["eef_vel_left"]
        self.eef_vel_sensor_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_SENSOR, eef_vel_names)
        self.eef_vel_sensor_adrs = [self._find_sensor_adr(sid) for sid in self.eef_vel_sensor_ids]
        self.eef_vel_sensor_adrs = torch.tensor(self.eef_vel_sensor_adrs, dtype=torch.long, device=device)

        vise_touch_names = ["vise_front_touch", "vise_back_touch"]
        self.vise_touch_sensor_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_SENSOR, vise_touch_names)
        self.vise_touch_sensor_adrs = [self._find_sensor_adr(sid) for sid in self.vise_touch_sensor_ids]
        self.vise_touch_sensor_adrs = torch.tensor(self.vise_touch_sensor_adrs, dtype=torch.long, device=device)

        vise_surface_touch_names = ["vise_surface_front_touch", "vise_surface_back_touch"]
        self.vise_surface_touch_sensor_ids = self._find_element_ids(mujoco.mjtObj.mjOBJ_SENSOR, vise_surface_touch_names)
        self.vise_surface_touch_sensor_adrs = [self._find_sensor_adr(sid) for sid in self.vise_surface_touch_sensor_ids]
        self.vise_surface_touch_sensor_adrs = torch.tensor(self.vise_surface_touch_sensor_adrs, dtype=torch.long, device=device)

        # ===== 执行器（actuator）ID 查找 =====
        # 查找所有执行器（无需指定名称，直接获取全部ID）
        self.actuator_ids = list(range(self.mj_model.nu))  # nu 是执行器总数
        self.actuator_ids = torch.tensor(self.actuator_ids, dtype=torch.long, device=device)

        # 3. 保留原 DH 参数（无修改）
        self.DH_m = torch.tensor([[0., 0.2405, 0., 0.],
                              [torch.pi / 2, 0., 0., torch.pi / 2],
                              [torch.pi / 2, 0., 0.256, 0.],
                              [0., 0.210, 0., torch.pi / 2],
                              [0., 0., 0., -torch.pi / 2],
                              [0., 0.144, 0., torch.pi / 2]], device=device
                             )
        self.DH_m_end = torch.tensor([[0., 0.2, 0., 0.], ], device=device)

    def _find_element_ids(self, element_type, names):
        """
        通用元素查找方法：根据类型和名称列表返回ID列表
        :param element_type: mujoco.mjtObj 枚举（如 mjOBJ_JOINT）
        :param names: 元素名称列表
        :return: 对应ID列表（顺序与names一致）
        """
        ids = []
        for name in names:
            elem_id = mujoco.mj_name2id(self.mj_model, element_type, name)
            if elem_id == -1:
                raise ValueError(f"未找到元素：{name}（类型：{element_type}），请检查XML中的名称是否匹配")
            ids.append(elem_id)
        return ids

    def _find_sensor_adr(self, sensor_id):
        return self.mj_model.sensor_adr[sensor_id]
