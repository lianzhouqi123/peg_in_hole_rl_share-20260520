import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULT_ROOT = PROJECT_ROOT / "result"
LOG_ROOT = PROJECT_ROOT / "log"

from developsuit.envs.pibot_base_env.pibotenv_FC_HF import PiBotEnv
from developsuit.utils.kine import fkine_ee_m
from developsuit.utils.transform_utils import quat2mat
import matplotlib.pyplot as plt
import numpy as np
import time, os


show_mode="show"
# show_mode="no_show"
if_plot = True
# if_plot = False

# 保存pre put状态
if_save_env = True
if_env_env = False
env_save_path = RESULT_ROOT / "demo_grasp_stock_left" / "state_pre_put_5cm.npz"
if_load_env = False
# if_load_env = True

pos_rdm_range = np.array([[-0.02, 0.02], [-0.02, 0.02], [0.02, 0.04], [5 /180 * np.pi, 0]])
config = {
    'vise_base_pos_range': np.array([[-0.05, 0.05], [-0.05, 0.05], [-5 /180 * np.pi, 5 /180 * np.pi]]),
    "stock_pose_init_range": np.array([
        [0.0, 0.0],
        [-0.015, -0.005],
        [-0.005, 0.01],
    ]),
    'step_per_ik': 10,
    'step_move_max': 0.0003,
    'joint_step_max': np.deg2rad(0.05),
    "vise_open_dis": 0.0055,
}


def _normalize_quat_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quat_multiply_wxyz(q1, q2):
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q2 = np.asarray(q2, dtype=np.float64).reshape(4)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def _sample_pose_with_xy_yaw(base_pose, pose_range):
    sampled_pose = np.asarray(base_pose, dtype=np.float64).reshape(7).copy()
    pose_range = np.asarray(pose_range, dtype=np.float64).reshape(3, 2)

    local_offset = np.zeros(3, dtype=np.float64)
    local_offset[0] = np.random.uniform(pose_range[0, 0], pose_range[0, 1])
    local_offset[1] = np.random.uniform(pose_range[1, 0], pose_range[1, 1])
    local_offset[2] = np.random.uniform(pose_range[2, 0], pose_range[2, 1])

    grasp_rot_world = quat2mat(sampled_pose[3:7])
    sampled_pose[0:3] += grasp_rot_world @ local_offset

    return sampled_pose


def _quat_slerp_wxyz(quat0, quat1, fraction):
    quat0 = _normalize_quat_wxyz(quat0)
    quat1 = _normalize_quat_wxyz(quat1)
    fraction = float(np.clip(fraction, 0.0, 1.0))

    dot = np.dot(quat0, quat1)
    if dot < 0.0:
        quat1 = -quat1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        quat = quat0 + fraction * (quat1 - quat0)
        return _normalize_quat_wxyz(quat)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    if sin_theta_0 < 1e-8:
        return quat0.copy()

    theta = theta_0 * fraction
    sin_theta = np.sin(theta)
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    quat = s0 * quat0 + s1 * quat1
    return _normalize_quat_wxyz(quat)


def _quat_angle_wxyz(quat0, quat1):
    quat0 = _normalize_quat_wxyz(quat0)
    quat1 = _normalize_quat_wxyz(quat1)
    dot = np.clip(np.abs(np.dot(quat0, quat1)), -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _wrap_joints_to_pi(joints):
    joints = np.asarray(joints, dtype=np.float64).copy()
    return (joints + np.pi) % (2.0 * np.pi) - np.pi


def _joint_delta_shortest(q_target, q_current):
    q_target = np.asarray(q_target, dtype=np.float64).reshape(-1)
    q_current = np.asarray(q_current, dtype=np.float64).reshape(-1)
    return _wrap_joints_to_pi(q_target - q_current)


def _fk_target_pose_from_arm_q(my_env, arm_q):
    ee_pos, ee_quat = fkine_ee_m(
        my_env.arm.DH_m,
        np.asarray(arm_q, dtype=np.float64).reshape(6),
        my_env.arm.DH_m_end,
        R_base=my_env.base_xmat[0],
        base=my_env.base_xpos[0],
    )
    return np.hstack((ee_pos.reshape(3), ee_quat.reshape(4)))


def _position_arc_length_about_world_z(pos_start, pos_end, axis_origin_world):
    rel_start = np.asarray(pos_start, dtype=np.float64).reshape(3) - np.asarray(axis_origin_world, dtype=np.float64).reshape(3)
    rel_end = np.asarray(pos_end, dtype=np.float64).reshape(3) - np.asarray(axis_origin_world, dtype=np.float64).reshape(3)

    r_start = np.linalg.norm(rel_start[0:2])
    r_end = np.linalg.norm(rel_end[0:2])
    theta_start = np.arctan2(rel_start[1], rel_start[0])
    theta_end = np.arctan2(rel_end[1], rel_end[0])
    dtheta = _wrap_to_pi(theta_end - theta_start)
    r_mean = 0.5 * (r_start + r_end)
    dz = rel_end[2] - rel_start[2]
    dr = r_end - r_start

    return np.sqrt(dr ** 2 + (r_mean * dtheta) ** 2 + dz ** 2)


def _interpolate_pose_about_base_world_z_wxyz(pose_start, pose_end, fraction, axis_origin_world):
    pose_start = np.asarray(pose_start, dtype=np.float64).reshape(7)
    pose_end = np.asarray(pose_end, dtype=np.float64).reshape(7)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    axis_origin_world = np.asarray(axis_origin_world, dtype=np.float64).reshape(3)

    rel_start = pose_start[0:3] - axis_origin_world
    rel_end = pose_end[0:3] - axis_origin_world

    r_start = np.linalg.norm(rel_start[0:2])
    r_end = np.linalg.norm(rel_end[0:2])
    theta_start = np.arctan2(rel_start[1], rel_start[0])
    theta_end = np.arctan2(rel_end[1], rel_end[0])
    dtheta = _wrap_to_pi(theta_end - theta_start)

    r_interp = r_start + (r_end - r_start) * fraction
    theta_interp = theta_start + dtheta * fraction
    z_interp = rel_start[2] + (rel_end[2] - rel_start[2]) * fraction

    pose_interp = pose_start.copy()
    pose_interp[0] = axis_origin_world[0] + r_interp * np.cos(theta_interp)
    pose_interp[1] = axis_origin_world[1] + r_interp * np.sin(theta_interp)
    pose_interp[2] = axis_origin_world[2] + z_interp
    pose_interp[3:7] = _quat_slerp_wxyz(pose_start[3:7], pose_end[3:7], fraction)
    return pose_interp


def grasp_stock():
    # --- 配置参数 ---
    step_per_ik = config['step_per_ik']
    step_move_max = config['step_move_max']
    joint_step_max = config['joint_step_max']
    step_rot_max = np.deg2rad(2.0)
    max_ik_backtrack = 12
    min_progress_ratio = 1e-3

    my_env.reset()

    # 预分配 NumPy 存储空间 (比 List Append 更快更整洁)
    max_total_steps = 12000
    log_arm_q = np.zeros((max_total_steps, 7))
    log_arm_x = np.zeros((max_total_steps, 7))
    log_target_q = np.zeros((max_total_steps, 7))  # 7 维手臂 + 1 维夹爪
    log_target_x = np.zeros((max_total_steps, 7))
    log_touch_force = np.zeros((max_total_steps, 2))
    log_xwrench = np.zeros((max_total_steps, 6))

    # 状态变量
    target_x = my_env.xpose[0].copy().reshape([-1])
    target_arm_q = my_env.arm_qpos[0:6].copy()
    target_q = np.hstack((target_arm_q, np.array([0.])))

    i_step_all = 0

    # ==============================================================
    # 核心内部函数：利用 nonlocal 共享外部变量，取代冗余循环
    # ==============================================================
    def sub_step_move(target_x_end, max_steps, gripper_val, use_ctrl_touch=False, f_grp=0, if_ik=True):
        nonlocal i_step_all, target_x, target_q, target_arm_q

        target_x_end = target_x_end.copy()
        target_x_end[3:7] = _normalize_quat_wxyz(target_x_end[3:7])
        base_axis_origin_world = my_env.base_xpos[0].copy()

        t_start = time.perf_counter()
        steps_executed = 0
        stalled_ik_updates = 0

        for ii in range(max_steps):
            t_step_start = time.perf_counter()
            # 2. 插值规划与 IK计算
            if ii % step_per_ik == 0 and if_ik:
                pose_start = target_x.copy()
                pos_path_len = _position_arc_length_about_world_z(
                    pose_start[0:3], target_x_end[0:3], base_axis_origin_world
                )
                step_dist = step_per_ik * step_move_max
                rot_dist = _quat_angle_wxyz(pose_start[3:7], target_x_end[3:7])
                step_angle = step_per_ik * step_rot_max

                pos_fraction = 1.0 if pos_path_len < 1e-8 else min(1.0, step_dist / pos_path_len)
                rot_fraction = 1.0 if rot_dist < 1e-8 else min(1.0, step_angle / rot_dist)
                move_ratio = min(pos_fraction, rot_fraction)

                candidate_ratio = move_ratio
                candidate_pose = pose_start.copy()
                candidate_arm_q = None

                for _ in range(max_ik_backtrack):
                    candidate_pose = _interpolate_pose_about_base_world_z_wxyz(
                        pose_start, target_x_end, candidate_ratio, base_axis_origin_world
                    )
                    candidate_arm_q = my_env.ikine_analytical_left(candidate_pose)
                    if candidate_arm_q is not None:
                        break
                    candidate_ratio *= 0.5
                    if candidate_ratio < min_progress_ratio:
                        candidate_arm_q = None
                        break

                if candidate_arm_q is not None:
                    target_x = candidate_pose
                    target_x[3:7] = _normalize_quat_wxyz(target_x[3:7])
                    target_arm_q = candidate_arm_q
                    stalled_ik_updates = 0
                else:
                    stalled_ik_updates += 1
                    if stalled_ik_updates == 1 or stalled_ik_updates % 10 == 0:
                        print(
                            f"[TaskSpace Arc IK] unreachable intermediate pose, keep last reachable target. "
                            f"path={pos_path_len:.4f} m, rot={np.rad2deg(rot_dist):.2f} deg"
                        )
                    if stalled_ik_updates >= 20:
                        print("[TaskSpace Arc IK] stage stopped early because the world-Z arc task-space path is unreachable.")
                        break

            # 拼接目标关节角 (手臂 + 夹爪)
            target_q = np.hstack((target_arm_q, np.array([gripper_val])))

            # 3. 执行控制与物理步进
            if use_ctrl_touch:
                my_env.ctrl_touch(joints=target_q, de=0.06, fd=f_grp)
            else:
                my_env.ctrl(joints=target_q)

            my_env.physics_step()

            # 4. 记录数据
            if i_step_all < max_total_steps:
                log_arm_q[i_step_all] = my_env.arm_qpos[0:7].reshape([-1])
                log_arm_x[i_step_all] = my_env.xpose[0].reshape([-1])
                log_target_q[i_step_all] = target_q.reshape([-1])
                log_target_x[i_step_all] = target_x.reshape([-1])
                log_touch_force[i_step_all] = my_env.left_touch_force
                log_xwrench[i_step_all] = my_env.xwrench_eef_left

                i_step_all += 1

            # --- 单步帧率监测 ---
            steps_executed += 1
            t_step_end = time.perf_counter()
            step_fps = 1.0 / (t_step_end - t_step_start)

            if ii > 0 and ii % 100 == 0:
                print(f"[SubStep {ii}] Step Hz: {step_fps:.1f}")

        # --- 阶段汇总 ---
        duration = time.perf_counter() - t_start
        avg_step_fps = steps_executed / duration if duration > 0 else 0
        print("-" * 50)
        print(f"Stage Finished in {duration:.2f}s | Steps: {steps_executed} | Avg Hz: {avg_step_fps:.2f}")
        print("-" * 50)

    def sub_step_move_joint(target_x_end, max_steps, gripper_val, use_ctrl_touch=False, f_grp=0, if_ik=True):
        nonlocal i_step_all, target_x, target_q, target_arm_q

        target_x_end = target_x_end.copy()
        target_x_end[3:7] = _normalize_quat_wxyz(target_x_end[3:7])

        target_arm_q_end = None
        if if_ik:
            target_arm_q_end = my_env.ikine_analytical_left(target_x_end)
            if target_arm_q_end is None:
                print("[JointSpace IK] failed to find a valid end joint target, skip this stage.")
                return
            target_arm_q_end = _wrap_joints_to_pi(target_arm_q_end)

        t_start = time.perf_counter()
        steps_executed = 0

        for ii in range(max_steps):
            t_step_start = time.perf_counter()

            if ii % step_per_ik == 0 and if_ik:
                joint_delta = _joint_delta_shortest(target_arm_q_end, target_arm_q)
                joint_err_max = np.max(np.abs(joint_delta))
                step_joint = step_per_ik * joint_step_max

                if joint_err_max > step_joint:
                    move_ratio = step_joint / (joint_err_max + 1e-8)
                    target_arm_q = _wrap_joints_to_pi(target_arm_q + joint_delta * move_ratio)
                else:
                    target_arm_q = target_arm_q_end.copy()

                target_x = _fk_target_pose_from_arm_q(my_env, target_arm_q)
                target_x[3:7] = _normalize_quat_wxyz(target_x[3:7])

            target_q = np.hstack((target_arm_q, np.array([gripper_val])))

            if use_ctrl_touch:
                my_env.ctrl_touch(joints=target_q, de=0.06, fd=f_grp)
            else:
                my_env.ctrl(joints=target_q)

            my_env.physics_step()

            if i_step_all < max_total_steps:
                log_arm_q[i_step_all] = my_env.arm_qpos[0:7].reshape([-1])
                log_arm_x[i_step_all] = my_env.xpose[0].reshape([-1])
                log_target_q[i_step_all] = target_q.reshape([-1])
                log_target_x[i_step_all] = target_x.reshape([-1])
                log_touch_force[i_step_all] = my_env.left_touch_force
                log_xwrench[i_step_all] = my_env.xwrench_eef_left
                i_step_all += 1

            steps_executed += 1
            t_step_end = time.perf_counter()
            step_fps = 1.0 / (t_step_end - t_step_start)

            if ii > 0 and ii % 100 == 0:
                print(f"[JointSubStep {ii}] Step Hz: {step_fps:.1f}")

        duration = time.perf_counter() - t_start
        avg_step_fps = steps_executed / duration if duration > 0 else 0
        print("-" * 50)
        print(f"Joint Stage Finished in {duration:.2f}s | Steps: {steps_executed} | Avg Hz: {avg_step_fps:.2f}")
        print("-" * 50)

    # --- 执行任务阶段 (极其清爽的编排) ---

    if not if_load_env:
        # 1. 移动到毛料上方
        print("1. Starting to move above stock...")
        target_end = my_env.grasp_pose + np.array([0.08, 0., 0.0, 0., 0., 0., 0.])
        sub_step_move(target_end, 100, gripper_val=0.0)
        
        sampled_grasp_pose = _sample_pose_with_xy_yaw(
            my_env.grasp_pose.copy(),
            config["stock_pose_init_range"],
        )

        target_end = sampled_grasp_pose + np.array([0.08, 0., 0.0, 0., 0., 0., 0.])
        sub_step_move(target_end, 2000, gripper_val=0.0)

        # 2. 移动到毛料侧方 (接近)
        print("2. Reached above stock, starting to move to the side...")
        target_end = sampled_grasp_pose.copy()
        sub_step_move(target_end, 800, gripper_val=0.0)

        # 3. 夹取 (原地动作)
        print("3. Reached side of stock, starting to grasp...")
        sub_step_move(target_end, 500, gripper_val=0.065, use_ctrl_touch=True, f_grp=5, if_ik=False)
        sub_step_move(target_end, 300, gripper_val=0.065, use_ctrl_touch=True, f_grp=45, if_ik=False)

        # 4. 拿起
        print("4. Grasped stock, starting to lift...")
        target_end = my_env.xpose[0].reshape([-1]) + np.array([0., -0.1, 0.08, 0., 0., 0., 0.])
        sub_step_move(target_end, 500, gripper_val=0.065, use_ctrl_touch=True, f_grp=45)

        # 5. 移动到插入点上方
        print("5. Lifted stock, moving above insertion point...")
        # ==========================================
        # 3. 施加相对虎钳位姿的随机扰动 (NumPy 单线程版)
        # ==========================================
        # my_env.insert_pose 已经是被随机化过的虎钳位置 (单环境为 1D array)
        pos_noise = np.zeros(3)
        pos_noise[0] = np.random.uniform(pos_rdm_range[0, 0], pos_rdm_range[0, 1]) 
        pos_noise[1] = np.random.uniform(pos_rdm_range[1, 0], pos_rdm_range[1, 1]) 
        pos_noise[2] = np.random.uniform(pos_rdm_range[2, 0], pos_rdm_range[2, 1]) # Z 包含悬停基准高度与扰动

        new_pos = my_env.insert_pose[0:3] + pos_noise

        cone_half_angle = pos_rdm_range[3, 0]
        tilt_angle = np.random.uniform(0.0, cone_half_angle)
        azimuth = np.random.uniform(0.0, 2 * np.pi)

        qw = np.cos(tilt_angle / 2.0)
        qx = np.sin(tilt_angle / 2.0) * np.cos(azimuth)
        qy = np.sin(tilt_angle / 2.0) * np.sin(azimuth)
        qz = 0.0  # Numpy 中标量直接赋 0.0 即可
        
        q_noise = np.array([qw, qx, qy, qz])

        q_base = my_env.insert_pose[3:7]

        # 提取分量
        nw, nx, ny, nz = q_noise[0], q_noise[1], q_noise[2], q_noise[3]
        bw, bx, by, bz = q_base[0], q_base[1], q_base[2], q_base[3]

        # 四元数乘法
        new_w = nw * bw - nx * bx - ny * by - nz * bz
        new_x = nw * bx + nx * bw + ny * bz - nz * by
        new_y = nw * by - nx * bz + ny * bw + nz * bx
        new_z = nw * bz + nx * by - ny * bx + nz * bw

        q_new = np.array([new_w, new_x, new_y, new_z])
        # 强制单位化
        q_new = q_new / np.linalg.norm(q_new)

        # 🌟 合并为目标位姿
        target_end_noisy = np.concatenate([new_pos, q_new])

        pos_no_noise = my_env.insert_pose[0:3] + np.array([0., 0., 0.1])
        target_end = np.concatenate([pos_no_noise, q_new])
        target_end_up = target_end + np.array([0.0, 0.0, 0.08, 0, 0, 0, 0])
        sub_step_move_joint(target_end_up, 3000, gripper_val=0.065, use_ctrl_touch=True, f_grp=45)
        sub_step_move_joint(target_end, 1000, gripper_val=0.065, use_ctrl_touch=True, f_grp=45)
        sub_step_move(target_end_noisy, 1000, gripper_val=0.065, use_ctrl_touch=True, f_grp=45)

        if if_save_env:
            # 确保保存目录存在
            env_save_path.parent.mkdir(parents=True, exist_ok=True)

            # 1. 保存底层物理与控制器状态
            my_env.save_environment_state(env_save_path)

            # 2. 【新增】：单独保存当前 demo 的期望位姿 (target_x 和 target_q)
            target_save_path = env_save_path.with_name(env_save_path.stem + "_target.npz")
            np.savez(target_save_path, target_x=target_x, target_q=target_q)

            print(f"[*] 环境底层状态已保存至: {env_save_path}")
            print(f"[*] Demo专属目标(Target)已保存至: {target_save_path}\n" + "-" * 50)

    else:
        # 1. 恢复环境的物理状态和 PID/阻抗控制器积分项
        my_env.load_environment_state(env_save_path)

        # 2. 单独加载 target 状态作为 RL 初始的目标点
        target_save_path = env_save_path.with_name(env_save_path.stem + "_target.npz")
        target_data = np.load(target_save_path)

        # 拿回你的 target_x 和 target_q
        target_x = target_data['target_x']
        target_q = target_data['target_q']

    # # 6. 移动到插入点
    # print("6. Reached above insertion point, moving down to insert...")
    # target_end = my_env.insert_pose.copy()
    # sub_step_move(target_end, 600, gripper_val=0.065, use_ctrl_touch=True, f_grp=45)

    # # 7. 松手
    # print("7. Reached insertion point, starting to release...")
    # sub_step_move(target_end, 300, gripper_val=0.0, if_ik=False)

    # # 8. 移动到上方 (回撤)
    # print("8. Released stock, moving up for retreat...")
    # target_end = my_env.insert_pose + np.array([0., 0., 0.1, 0., 0., 0., 0.])
    # sub_step_move(target_end, 600, gripper_val=0.0)

    print("\nTask completed, plotting results...")

    # 切片取实际有效的数据进行绘图
    if if_plot:
        idx = slice(0, i_step_all)
        plot_results(log_arm_q[idx], log_arm_x[idx], log_target_q[idx], log_target_x[idx],
                     log_touch_force[idx], log_xwrench[idx])


def plot_results(arm_q, arm_x, target_q, target_x, touch_force, xwrench):
    # 因为输入已经是拼接好的 NumPy 数组，直接画图即可
    print("Max Joint Error (deg):", np.max(target_q[-1, 0:7] - arm_q[-1, 0:7]) / np.pi * 180)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 图 1: 位姿误差 (Pose Error)
    for i in range(3):
        axes[0, 0].plot(target_x[:, i] - arm_x[:, i], label=f'Dim {i}')
    axes[0, 0].set_title("Pose Error (m)")
    axes[0, 0].legend(frameon=False)

    # 图 2: 关节角误差 (Joint Error)
    for i in range(7):
        axes[0, 1].plot(target_q[:, i] - arm_q[:, i], label=f'J {i}')
    axes[0, 1].set_title("Joint Error (rad)")
    axes[0, 1].legend(frameon=False)

    # 图 3: 接触力 (Touch Force)
    for i in range(2):
        axes[1, 0].plot(touch_force[:, i], label=f'Touch {i}')
    axes[1, 0].set_title("Touch Force")
    axes[1, 0].legend(frameon=False)

    # 图 4: 世界系力矩/力 (World Wrench)
    for i in range(6):
        axes[1, 1].plot(xwrench[:, i], label=f'Wrench {i}')
    axes[1, 1].set_title("World Wrench")
    axes[1, 1].legend(frameon=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 初始化机械臂关节控制程序
    my_env = PiBotEnv(show_mode=show_mode, timestep_control=0.01, config=config)
    grasp_stock()
