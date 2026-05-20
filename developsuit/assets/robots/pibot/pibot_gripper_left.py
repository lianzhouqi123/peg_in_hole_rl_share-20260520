from dm_control import mjcf
from developsuit.utils.transform_utils import *
import os


class PiBot_left:
    def __init__(self, name="pibot_left.xml"):
        xml_path = os.path.join(
            os.path.dirname(__file__),
            name,
        )
        self.mjcf_root = mjcf.from_path(xml_path)
        self.mjcf_model = self.mjcf_root

        joint_name = ["joint1_left", "joint2_left", "joint3_left", "joint4_left", "joint5_left", "joint6_left",
                      "Left_1_Joint_gripper_left"]
        self.joints = [self.mjcf_root.find("joint", name) for name in joint_name]
        joint_name_left = ["joint1_left", "joint2_left", "joint3_left", "joint4_left", "joint5_left", "joint6_left"]
        self.joint_lefts = [self.mjcf_root.find("joint", name) for name in joint_name_left]
        joint_name_left_gripper = ["Left_1_Joint_gripper_left"]
        self.joint_left_grippers = [self.mjcf_root.find("joint", name) for name in joint_name_left_gripper]
        joint_vise_open_name = ["vise_front_joint", "vise_back_joint"]
        self.vise_open_joints = [self.mjcf_root.find("joint", name) for name in joint_vise_open_name]
        joint_vise_base_name = ["vise_x_joint", "vise_y_joint", "vise_qz_joint"]
        self.vise_base_joints = [self.mjcf_root.find("joint", name) for name in joint_vise_base_name]

        self.eef_site_name = ["end_effector_left"]
        self.eef_site = [self.mjcf_root.find("site", name) for name in self.eef_site_name]
        base_site_name = ["base_site_left"]
        self.base_site = [self.mjcf_root.find("site", name) for name in base_site_name]
        self.actuators = self.mjcf_root.find_all("actuator")
        self.stock_joint = self.mjcf_root.find("joint", "stock")
        base_gyro_name = ["base_gyro_left"]
        self.base_gyro_sensors = [self.mjcf_root.find("sensor", name) for name in base_gyro_name]
        eef_gyro_name = ["eef_gyro_left"]
        self.eef_gyro_sensors = [self.mjcf_root.find("sensor", name) for name in eef_gyro_name]
        eef_vel_name = ["eef_vel_left"]
        self.eef_vel_sensors = [self.mjcf_root.find("sensor", name) for name in eef_vel_name]
        body_name = ["Link6_left"]
        self.bodies = [self.mjcf_root.find("body", name) for name in body_name]
        grasp_site_name = ["grasp_stock_site"]
        self.grasp_site = [self.mjcf_root.find("site", name) for name in grasp_site_name]
        insert_site_name = ["insert_site"]
        self.insert_site = [self.mjcf_root.find("site", name) for name in insert_site_name]
        insert_world_site_name = ["insert_world_site"]
        self.insert_world_site = [self.mjcf_root.find("site", name) for name in insert_world_site_name]
        eef_real_site_name = ["eef_real_left"]
        self.eef_real_site = [self.mjcf_root.find("site", name) for name in eef_real_site_name]
        left_touch_names = ["gripper_left_1", "gripper_left_2"]
        self.left_touch_sensors = [self.mjcf_root.find("sensor", name) for name in left_touch_names]
        self.gripper_left_site_name = ["gripper_left_1_site", "gripper_left_2_site"]
        self.gripper_left_site = [self.mjcf_root.find("site",name) for name in self.gripper_left_site_name]
        self.gripper_left_body_name = ["Left_Support_Link_gripper_left", "Right_Support_Link_gripper_left"]
        self.wrench_site_left = [self.mjcf_root.find("site", "eef_left_wrench")]
        self.F_sensor_left = [self.mjcf_root.find("sensor", "F_eef_left")]
        self.T_sensor_left = [self.mjcf_root.find("sensor", "T_eef_left")]
        vise_touch_names = ["vise_front_touch", "vise_back_touch"]
        self.vise_touch_sensors = [self.mjcf_root.find("sensor", name) for name in vise_touch_names]
        vise_surface_touch_names = ["vise_surface_front_touch", "vise_surface_back_touch"]
        self.vise_surface_touch_sensors = [self.mjcf_root.find("sensor", name) for name in vise_surface_touch_names]
        

        # theta d a alpha
        self.DH_m = np.array([[0., 0.2405, 0., 0.],
                              [m.pi / 2, 0., 0., m.pi / 2],
                              [m.pi / 2, 0., 0.256, 0.],
                              [0., 0.210, 0., m.pi / 2],
                              [0., 0., 0., -m.pi / 2],
                              [0., 0.144, 0., m.pi / 2]],
                             )
        self.DH_m_end = np.array([[0., 0.2, 0., 0.], ])

    def get_eef_pose(self, physics):
        ee_poses = []
        for ii in range(len(self.eef_site)):
            eef_site = self.eef_site[ii]
            ee_pos = physics.bind(eef_site).xpos
            ee_quat = mat2quat(physics.bind(eef_site).xmat.reshape(3, 3))
            ee_pose = np.concatenate((ee_pos, ee_quat))
            ee_poses.append(ee_pose)

        return ee_poses
