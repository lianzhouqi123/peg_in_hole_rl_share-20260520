import os
import sys
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULT_ROOT = PROJECT_ROOT / "result"
LOG_ROOT = PROJECT_ROOT / "log"

from developsuit.envs.pibot_base_env.pibotenv_FC_HF_warp import PiBotEnv
from developsuit.utils.rm65_analytical_ik_torch import fk_mdh_batch, rm65_analytical_ik_torch, rm65_params_torch
from developsuit.utils.transform_utils_torch import mat2quat, quat2mat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if_plot = False
if_test_time = False

IK_D6_TCP_M = 0.161 + 0.2

# ==========================================
# 【配置项】
# ==========================================
if_save_env = True
env_save_path = RESULT_ROOT / "demo_grasp_stock_left_warp" / "state_pre_put_warp_FC_wrench_bias_wide2_1e5.pt"
total_envs = 100000
batch_size = 10000

# 随机域：[X偏移, Y偏移, Z悬停高度+Z扰动, [圆锥倾角分段点(rad), 圆锥倾角上限(rad)]]
pos_rdm_range = torch.tensor([[-0.025, 0.025], [-0.025, 0.025], [0.05, 0.06], [2 / 180 * np.pi, 6 / 180 * np.pi]], device=device)
save_pose_pos_thresh = 0.0025
save_pose_rot_thresh = np.deg2rad(2.5)
save_grip_force_thresh = 25.0

config = {
    "vise_base_pos_range": np.array([[-0.05, 0.05], [-0.05, 0.05], [-5 / 180 * np.pi, 5 / 180 * np.pi]]),
    "stock_pose_init_range": np.array([
        [0.0, 0.0],
        [-0.015, -0.005],
        [-0.005, 0.01],
    ]),
    "step_per_ik": 10,
    "step_move_max": 0.0003,
    "joint_step_max": np.deg2rad(0.05),
    "vise_open_dis": 0.0055,

    'wrench_noise_std': np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]),
    'wrench_drift_range': np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]),
}


def _normalize_quat_wxyz_torch(quat):
    quat = torch.as_tensor(quat, dtype=torch.float32, device=device)
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    quat_safe = quat / torch.clamp(norm, min=1e-8)
    identity = torch.zeros_like(quat_safe)
    identity[..., 0] = 1.0
    return torch.where(norm < 1e-8, identity, quat_safe)


def _quat_multiply_wxyz_torch(q1, q2):
    q1 = torch.as_tensor(q1, dtype=torch.float32, device=device)
    q2 = torch.as_tensor(q2, dtype=torch.float32, device=device)
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def _sample_pose_with_xy_yaw_torch(base_pose, pose_range):
    base_pose = torch.as_tensor(base_pose, dtype=torch.float32, device=device).clone()
    if base_pose.dim() == 1:
        base_pose = base_pose.unsqueeze(0)

    pose_range = torch.as_tensor(pose_range, dtype=torch.float32, device=device).reshape(3, 2)
    num_envs = base_pose.shape[0]

    local_offset = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    local_offset[:, 0].uniform_(float(pose_range[0, 0]), float(pose_range[0, 1]))
    local_offset[:, 1].uniform_(float(pose_range[1, 0]), float(pose_range[1, 1]))
    local_offset[:, 2].uniform_(float(pose_range[2, 0]), float(pose_range[2, 1]))

    sampled_pose = base_pose.clone()
    grasp_rot_world = quat2mat(base_pose[:, 3:7])
    world_offset = torch.bmm(grasp_rot_world, local_offset.unsqueeze(-1)).squeeze(-1)
    sampled_pose[:, 0:3] += world_offset

    return sampled_pose


def _quat_slerp_wxyz_torch(quat0, quat1, fraction):
    quat0 = _normalize_quat_wxyz_torch(quat0)
    quat1 = _normalize_quat_wxyz_torch(quat1)
    fraction = torch.as_tensor(fraction, dtype=quat0.dtype, device=quat0.device)
    while fraction.dim() < quat0.dim():
        fraction = fraction.unsqueeze(-1)
    fraction = torch.clamp(fraction, 0.0, 1.0)

    dot = torch.sum(quat0 * quat1, dim=-1, keepdim=True)
    quat1 = torch.where(dot < 0.0, -quat1, quat1)
    dot = torch.sum(quat0 * quat1, dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)

    linear_mask = dot > 0.9995
    quat_linear = _normalize_quat_wxyz_torch(quat0 + fraction * (quat1 - quat0))

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * fraction
    sin_theta = torch.sin(theta)
    s0 = torch.sin(theta_0 - theta) / torch.clamp(sin_theta_0, min=1e-8)
    s1 = sin_theta / torch.clamp(sin_theta_0, min=1e-8)
    quat_slerp = _normalize_quat_wxyz_torch(s0 * quat0 + s1 * quat1)

    return torch.where(linear_mask, quat_linear, quat_slerp)


def _quat_angle_wxyz_torch(quat0, quat1):
    quat0 = _normalize_quat_wxyz_torch(quat0)
    quat1 = _normalize_quat_wxyz_torch(quat1)
    dot = torch.sum(quat0 * quat1, dim=-1).abs()
    dot = torch.clamp(dot, -1.0, 1.0)
    return 2.0 * torch.acos(dot)


def _wrap_to_pi_torch(angle):
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def _wrap_joints_to_pi_torch(joints):
    return (joints + torch.pi) % (2.0 * torch.pi) - torch.pi


def _joint_delta_shortest_torch(q_target, q_current):
    return _wrap_joints_to_pi_torch(q_target - q_current)


def _position_arc_length_about_world_z_torch(pos_start, pos_end, axis_origin_world):
    rel_start = pos_start - axis_origin_world
    rel_end = pos_end - axis_origin_world

    r_start = torch.linalg.norm(rel_start[:, 0:2], dim=-1)
    r_end = torch.linalg.norm(rel_end[:, 0:2], dim=-1)
    theta_start = torch.atan2(rel_start[:, 1], rel_start[:, 0])
    theta_end = torch.atan2(rel_end[:, 1], rel_end[:, 0])
    dtheta = _wrap_to_pi_torch(theta_end - theta_start)
    r_mean = 0.5 * (r_start + r_end)
    dz = rel_end[:, 2] - rel_start[:, 2]
    dr = r_end - r_start

    return torch.sqrt(dr ** 2 + (r_mean * dtheta) ** 2 + dz ** 2)


def _interpolate_pose_about_base_world_z_wxyz_torch(pose_start, pose_end, fraction, axis_origin_world):
    pose_start = pose_start.reshape(-1, 7)
    pose_end = pose_end.reshape(-1, 7)
    axis_origin_world = axis_origin_world.reshape(-1, 3)
    fraction = torch.as_tensor(fraction, dtype=pose_start.dtype, device=pose_start.device).reshape(-1, 1)
    fraction = torch.clamp(fraction, 0.0, 1.0)

    rel_start = pose_start[:, 0:3] - axis_origin_world
    rel_end = pose_end[:, 0:3] - axis_origin_world

    r_start = torch.linalg.norm(rel_start[:, 0:2], dim=-1, keepdim=True)
    r_end = torch.linalg.norm(rel_end[:, 0:2], dim=-1, keepdim=True)
    theta_start = torch.atan2(rel_start[:, 1:2], rel_start[:, 0:1])
    theta_end = torch.atan2(rel_end[:, 1:2], rel_end[:, 0:1])
    dtheta = _wrap_to_pi_torch(theta_end - theta_start)

    r_interp = r_start + (r_end - r_start) * fraction
    theta_interp = theta_start + dtheta * fraction
    z_interp = rel_start[:, 2:3] + (rel_end[:, 2:3] - rel_start[:, 2:3]) * fraction

    pose_interp = pose_start.clone()
    pose_interp[:, 0:1] = axis_origin_world[:, 0:1] + r_interp * torch.cos(theta_interp)
    pose_interp[:, 1:2] = axis_origin_world[:, 1:2] + r_interp * torch.sin(theta_interp)
    pose_interp[:, 2:3] = axis_origin_world[:, 2:3] + z_interp
    pose_interp[:, 3:7] = _quat_slerp_wxyz_torch(pose_start[:, 3:7], pose_end[:, 3:7], fraction)
    return pose_interp


def _yaw_from_rotmat_torch(rotmat):
    return torch.atan2(rotmat[:, 1, 0], rotmat[:, 0, 0])


def _tensor_brief_str(tensor_row, precision=4):
    arr = torch.as_tensor(tensor_row).detach().cpu().numpy()
    return np.array2string(arr, precision=precision, suppress_small=False)


def _format_duration(seconds):
    seconds = max(float(seconds), 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - hours * 3600 - minutes * 60
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:05.2f}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:05.2f}s"
    return f"{secs:.2f}s"


def _ikine_analytical_left_with_mask(my_env, pose_d, env_ids=None):
    pose_d = pose_d.clone()
    pose_d[:, 3:7] = _normalize_quat_wxyz_torch(pose_d[:, 3:7])

    if env_ids is None:
        base_xpos_left = my_env.base_xpos_left
        base_xmat_left = my_env.base_xmat_left
        arm_qpos = my_env.arm_qpos[:, 0:6]
    else:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=pose_d.device)
        base_xpos_left = my_env.base_xpos_left[env_ids]
        base_xmat_left = my_env.base_xmat_left[env_ids]
        arm_qpos = my_env.arm_qpos[env_ids, 0:6]

    xd = pose_d[:, 0:3]
    qd = pose_d[:, 3:7]

    relative_pos = xd - base_xpos_left
    xd_base = torch.bmm(relative_pos.unsqueeze(1), base_xmat_left).squeeze(1)

    matd = quat2mat(qd)
    matd_base = torch.matmul(base_xmat_left.transpose(-1, -2), matd)

    target_arm_q_all = rm65_analytical_ik_torch(
        target_pos=xd_base,
        target_R=matd_base,
        d6_tcp_m=IK_D6_TCP_M,
    )

    current_q = arm_qpos.unsqueeze(1)
    diff = target_arm_q_all - current_q
    dist = torch.linalg.norm(diff, dim=-1)
    dist = torch.nan_to_num(dist, nan=float("inf"))

    best_sol_idx = torch.argmin(dist, dim=-1)
    batch_idx = torch.arange(pose_d.shape[0], device=pose_d.device)
    nearest_q = target_arm_q_all[batch_idx, best_sol_idx]

    valid_mask = ~torch.isinf(dist).all(dim=-1)
    nearest_q = torch.where(valid_mask.unsqueeze(-1), nearest_q, arm_qpos)
    return nearest_q, valid_mask


def _fk_target_pose_from_arm_q(my_env, arm_q, fk_params):
    T_base = fk_mdh_batch(arm_q, fk_params, n=6)
    ee_pos_base = T_base[:, 0:3, 3]
    ee_rot_base = T_base[:, 0:3, 0:3]

    ee_pos_world = my_env.base_xpos_left + torch.bmm(
        ee_pos_base.unsqueeze(1), my_env.base_xmat_left.transpose(-1, -2)
    ).squeeze(1)
    ee_rot_world = torch.matmul(my_env.base_xmat_left, ee_rot_base)
    ee_quat_world = _normalize_quat_wxyz_torch(mat2quat(ee_rot_world))
    return torch.cat([ee_pos_world, ee_quat_world], dim=-1)


def generate_state_pool(timestep_control=0.01):
    """
    分批次生成高动态离线状态池，防止显存溢出
    """
    total_demo_start = time.perf_counter()
    num_batches = math.ceil(total_envs / batch_size)
    print(f"🚀 准备生成 {total_envs} 个状态库，将分为 {num_batches} 个批次执行，每批 {batch_size} 个环境...")

    global_state_pool = {}
    envs_collected = 0

    my_env = PiBotEnv(num_envs=batch_size, show_mode="no_show", timestep_control=timestep_control, config=config)
    fk_params = rm65_params_torch(d6_tcp_m=IK_D6_TCP_M, device=device, dtype=torch.float32)

    for batch_idx in range(num_batches):
        batch_start_time = time.perf_counter()
        print(f"\n{'=' * 50}")
        print(f"🌟 正在处理第 {batch_idx + 1}/{num_batches} 批次")
        print(f"{'=' * 50}")

        my_env.reset(if_vise_base_rdm=True)

        step_per_ik = config["step_per_ik"]
        step_move_max = config["step_move_max"]
        joint_step_max = config["joint_step_max"]
        step_rot_max = np.deg2rad(2.0)
        max_ik_backtrack = 12
        min_progress_ratio = 1e-3

        target_x = my_env.xpose_left.clone()
        target_arm_q = my_env.arm_qpos[:, 0:6].clone()
        target_q = torch.cat([target_arm_q, my_env.arm_qpos[:, 6:7].clone()], dim=-1)
        invalid_env_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        failure_logged_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        failure_logs_emitted = 0
        failure_log_limit_per_batch = 20
        sampled_grasp_pose = None

        def _log_failed_env_details(
            failed_env_ids,
            stage_name,
            pose_start,
            target_x_end,
            pos_path_len,
            rot_dist,
            candidate_pose,
            candidate_ratio,
        ):
            nonlocal failure_logs_emitted, sampled_grasp_pose

            if failed_env_ids.numel() == 0 or failure_logs_emitted >= failure_log_limit_per_batch:
                return

            remaining_budget = failure_log_limit_per_batch - failure_logs_emitted
            failed_env_ids = failed_env_ids[:remaining_budget]

            debug_pose_pack = torch.cat(
                [
                    pose_start[failed_env_ids],
                    candidate_pose[failed_env_ids],
                    target_x_end[failed_env_ids],
                ],
                dim=0,
            )
            _, debug_valid_mask = _ikine_analytical_left_with_mask(my_env, debug_pose_pack, env_ids=failed_env_ids.repeat(3))
            num_failed = failed_env_ids.numel()
            start_valid_mask = debug_valid_mask[:num_failed]
            candidate_valid_mask = debug_valid_mask[num_failed: 2 * num_failed]
            end_valid_mask = debug_valid_mask[2 * num_failed:]
            base_yaw_deg = torch.rad2deg(_yaw_from_rotmat_torch(my_env.base_xmat_left[failed_env_ids]))

            for local_i, env_id in enumerate(failed_env_ids.tolist()):
                print(
                    f"[TaskSpace Arc IK][Debug] batch={batch_idx + 1} stage={stage_name} env={env_id} "
                    f"path={pos_path_len[env_id].item():.4f} m rot={np.rad2deg(rot_dist[env_id].item()):.2f} deg "
                    f"ratio={candidate_ratio[env_id].item():.6f} "
                    f"ik(start/candidate/end)={int(start_valid_mask[local_i].item())}/"
                    f"{int(candidate_valid_mask[local_i].item())}/{int(end_valid_mask[local_i].item())}"
                )
                print(f"  planner_start={_tensor_brief_str(pose_start[env_id])}")
                print(f"  candidate_pose={_tensor_brief_str(candidate_pose[env_id])}")
                print(f"  target_end={_tensor_brief_str(target_x_end[env_id])}")
                print(f"  actual_xpose_left={_tensor_brief_str(my_env.xpose_left[env_id])}")
                print(
                    f"  base_pos={_tensor_brief_str(my_env.base_xpos_left[env_id])} "
                    f"base_yaw_deg={base_yaw_deg[local_i].item():.2f} "
                    f"vise_base_pos={_tensor_brief_str(my_env.vise_base_pos[env_id])}"
                )
                print(f"  arm_qpos={_tensor_brief_str(my_env.arm_qpos[env_id, 0:6])}")
                if sampled_grasp_pose is not None:
                    print(f"  sampled_grasp_pose={_tensor_brief_str(sampled_grasp_pose[env_id])}")

            failure_logs_emitted += num_failed

        def sub_step_move(target_x_end, max_steps, gripper_val, use_touch_ctrl=False, f_grp=0, if_ik=True, stage_name=""):
            nonlocal target_x, target_q, target_arm_q, invalid_env_mask, failure_logged_mask

            target_x_end = target_x_end.clone()
            target_x_end[:, 3:7] = _normalize_quat_wxyz_torch(target_x_end[:, 3:7])
            base_axis_origin_world = my_env.base_xpos_left.clone()
            gripper_tensor = torch.full((batch_size, 1), float(gripper_val), dtype=target_x.dtype, device=device)

            t_start = time.perf_counter()
            steps_executed = 0
            stalled_ik_updates = torch.zeros(batch_size, dtype=torch.int64, device=device)

            for ii in range(max_steps):
                t_step_start = time.perf_counter()

                if ii % step_per_ik == 0 and if_ik:
                    pose_start = target_x.clone()
                    pos_path_len = _position_arc_length_about_world_z_torch(
                        pose_start[:, 0:3], target_x_end[:, 0:3], base_axis_origin_world
                    )
                    step_dist = step_per_ik * step_move_max
                    rot_dist = _quat_angle_wxyz_torch(pose_start[:, 3:7], target_x_end[:, 3:7])
                    step_angle = step_per_ik * step_rot_max

                    pos_fraction = torch.where(
                        pos_path_len < 1e-8,
                        torch.ones_like(pos_path_len),
                        torch.clamp(step_dist / torch.clamp(pos_path_len, min=1e-8), max=1.0),
                    )
                    rot_fraction = torch.where(
                        rot_dist < 1e-8,
                        torch.ones_like(rot_dist),
                        torch.clamp(step_angle / torch.clamp(rot_dist, min=1e-8), max=1.0),
                    )
                    candidate_ratio = torch.minimum(pos_fraction, rot_fraction)

                    candidate_pose_best = pose_start.clone()
                    candidate_arm_q_best = target_arm_q.clone()
                    success_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                    candidate_pose = pose_start.clone()

                    for _ in range(max_ik_backtrack):
                        candidate_pose = _interpolate_pose_about_base_world_z_wxyz_torch(
                            pose_start, target_x_end, candidate_ratio, base_axis_origin_world
                        )
                        candidate_arm_q, valid_mask = _ikine_analytical_left_with_mask(my_env, candidate_pose)

                        same_pose_mask = (
                            torch.linalg.norm(candidate_pose[:, 0:3] - pose_start[:, 0:3], dim=-1) < 1e-8
                        ) & (_quat_angle_wxyz_torch(candidate_pose[:, 3:7], pose_start[:, 3:7]) < 1e-6)
                        valid_mask = valid_mask | same_pose_mask

                        newly_success = valid_mask & (~success_mask)
                        if torch.any(newly_success):
                            candidate_pose_best[newly_success] = candidate_pose[newly_success]
                            candidate_arm_q_best[newly_success] = torch.where(
                                same_pose_mask[newly_success].unsqueeze(-1),
                                target_arm_q[newly_success],
                                candidate_arm_q[newly_success],
                            )
                            success_mask = success_mask | valid_mask

                        if torch.all(success_mask):
                            break

                        candidate_ratio = torch.where(success_mask, candidate_ratio, candidate_ratio * 0.5)
                        if torch.all(candidate_ratio[~success_mask] < min_progress_ratio):
                            break

                    if torch.any(success_mask):
                        target_x = torch.where(success_mask.unsqueeze(-1), candidate_pose_best, target_x)
                        target_x[:, 3:7] = _normalize_quat_wxyz_torch(target_x[:, 3:7])
                        target_arm_q = torch.where(success_mask.unsqueeze(-1), candidate_arm_q_best, target_arm_q)
                        stalled_ik_updates = torch.where(success_mask, 0, stalled_ik_updates + 1)
                    else:
                        stalled_ik_updates = stalled_ik_updates + 1

                    invalid_env_mask = invalid_env_mask | (stalled_ik_updates > 0)
                    newly_failed_mask = (stalled_ik_updates == 1) & (~failure_logged_mask)
                    if torch.any(newly_failed_mask):
                        _log_failed_env_details(
                            torch.nonzero(newly_failed_mask, as_tuple=False).squeeze(-1),
                            stage_name or "unnamed_stage",
                            pose_start,
                            target_x_end,
                            pos_path_len,
                            rot_dist,
                            candidate_pose,
                            candidate_ratio,
                        )
                        failure_logged_mask = failure_logged_mask | newly_failed_mask

                    if torch.any(stalled_ik_updates == 1) or torch.any((stalled_ik_updates % 10) == 0):
                        stalled_count = int((stalled_ik_updates > 0).sum().item())
                        if stalled_count > 0:
                            path_mean = pos_path_len[stalled_ik_updates > 0].mean().item()
                            rot_mean = rot_dist[stalled_ik_updates > 0].mean().item()
                            print(
                                "[TaskSpace Arc IK] unreachable intermediate pose, keep last reachable target. "
                                f"envs={stalled_count}, path={path_mean:.4f} m, rot={np.rad2deg(rot_mean):.2f} deg"
                            )

                    if torch.all(stalled_ik_updates >= 20):
                        print("[TaskSpace Arc IK] stage stopped early because the world-Z arc task-space path is unreachable.")
                        break

                target_q = torch.cat([target_arm_q, gripper_tensor], dim=-1)

                if use_touch_ctrl:
                    my_env.touch_ctrl(joints=target_q, de=0.06, fd=f_grp)
                else:
                    my_env.ctrl(joints=target_q)

                my_env.physics_step()

                steps_executed += 1
                # t_step_end = time.perf_counter()
                # step_fps = 1.0 / max(t_step_end - t_step_start, 1e-8)

                # if ii > 0 and ii % 100 == 0:
                #     print(f"[SubStep {ii}] Step Hz: {step_fps:.1f}")

            # duration = time.perf_counter() - t_start
            # avg_step_fps = steps_executed / duration if duration > 0 else 0.0
            # print("-" * 50)
            # print(f"Stage Finished in {duration:.2f}s | Steps: {steps_executed} | Avg Hz: {avg_step_fps:.2f}")
            # print("-" * 50)

        def sub_step_move_joint(target_x_end, max_steps, gripper_val, use_touch_ctrl=False, f_grp=0, if_ik=True):
            nonlocal target_x, target_q, target_arm_q, invalid_env_mask

            target_x_end = target_x_end.clone()
            target_x_end[:, 3:7] = _normalize_quat_wxyz_torch(target_x_end[:, 3:7])
            gripper_tensor = torch.full((batch_size, 1), float(gripper_val), dtype=target_x.dtype, device=device)

            target_arm_q_end = target_arm_q.clone()
            valid_end_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            if if_ik:
                target_arm_q_end, valid_end_mask = _ikine_analytical_left_with_mask(my_env, target_x_end)
                target_arm_q_end = _wrap_joints_to_pi_torch(target_arm_q_end)
                invalid_count = int((~valid_end_mask).sum().item())
                if invalid_count > 0:
                    invalid_env_mask = invalid_env_mask | (~valid_end_mask)
                    print(f"[JointSpace IK] failed to find a valid end joint target for {invalid_count} envs, keep their current targets.")

            t_start = time.perf_counter()
            steps_executed = 0

            for ii in range(max_steps):
                t_step_start = time.perf_counter()

                if ii % step_per_ik == 0 and if_ik:
                    joint_delta = _joint_delta_shortest_torch(target_arm_q_end, target_arm_q)
                    joint_err_max = torch.max(torch.abs(joint_delta), dim=-1).values
                    step_joint = step_per_ik * joint_step_max
                    move_ratio = torch.clamp(step_joint / torch.clamp(joint_err_max, min=1e-8), max=1.0)
                    move_ratio = torch.where(valid_end_mask, move_ratio, torch.zeros_like(move_ratio))

                    target_arm_q = _wrap_joints_to_pi_torch(target_arm_q + joint_delta * move_ratio.unsqueeze(-1))
                    target_x = _fk_target_pose_from_arm_q(my_env, target_arm_q, fk_params)

                target_q = torch.cat([target_arm_q, gripper_tensor], dim=-1)

                if use_touch_ctrl:
                    my_env.touch_ctrl(joints=target_q, de=0.06, fd=f_grp)
                else:
                    my_env.ctrl(joints=target_q)

                my_env.physics_step()

                steps_executed += 1
                # t_step_end = time.perf_counter()
                # step_fps = 1.0 / max(t_step_end - t_step_start, 1e-8)

                # if ii > 0 and ii % 100 == 0:
                #     print(f"[JointSubStep {ii}] Step Hz: {step_fps:.1f}")

            # duration = time.perf_counter() - t_start
            # avg_step_fps = steps_executed / duration if duration > 0 else 0.0
            # print("-" * 50)
            # print(f"Joint Stage Finished in {duration:.2f}s | Steps: {steps_executed} | Avg Hz: {avg_step_fps:.2f}")
            # print("-" * 50)

        # ==========================================
        # 2. 执行物理动作序列 (抓取毛料并提升)
        # ==========================================
        print("1. Starting to move above stock...")
        target_end = my_env.grasp_pose + torch.tensor([0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)
        sub_step_move(target_end, 100, gripper_val=0.0, stage_name="approach_stock_front")

        sampled_grasp_pose = _sample_pose_with_xy_yaw_torch(
            my_env.grasp_pose.clone(),
            config["stock_pose_init_range"],
        )

        target_end = sampled_grasp_pose + torch.tensor([0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)
        sub_step_move(target_end, 2000, gripper_val=0.0, stage_name="approach_sampled_stock")

        print("2. Reached above stock, starting to move to the side...")
        target_end = sampled_grasp_pose.clone()
        sub_step_move(target_end, 800, gripper_val=0.0, stage_name="move_to_stock_side")

        print("3. Reached side of stock, starting to grasp...")
        sub_step_move(target_end, 500, gripper_val=0.065, use_touch_ctrl=True, f_grp=5, if_ik=False)
        sub_step_move(target_end, 500, gripper_val=0.065, use_touch_ctrl=True, f_grp=45, if_ik=False)

        print("4. Grasped stock, starting to lift...")
        target_end = my_env.xpose_left.clone() + torch.tensor([0.0, 0, 0.08, 0.0, 0.0, 0.0, 0.0], device=device)
        sub_step_move(target_end, 500, gripper_val=0.065, use_touch_ctrl=True, f_grp=45, stage_name="lift_stock")

        print("5. Lifted stock, moving above insertion point...")
        pos_noise = torch.zeros((batch_size, 3), device=device)
        pos_noise[:, 0].uniform_(pos_rdm_range[0, 0], pos_rdm_range[0, 1])
        pos_noise[:, 1].uniform_(pos_rdm_range[1, 0], pos_rdm_range[1, 1])
        pos_noise[:, 2].uniform_(pos_rdm_range[2, 0], pos_rdm_range[2, 1])

        new_pos = my_env.insert_pose[:, 0:3] + pos_noise

        cone_half_angle_split = pos_rdm_range[3, 0]
        cone_half_angle_max = pos_rdm_range[3, 1]
        if cone_half_angle_max < cone_half_angle_split:
            raise ValueError("pos_rdm_range[3, 1] must be >= pos_rdm_range[3, 0] for segmented cone-angle sampling.")
        sample_first_segment = torch.rand(batch_size, device=device) < (1.0 / 3.0)
        tilt_angle = torch.empty(batch_size, device=device)
        first_count = int(sample_first_segment.sum().item())
        second_count = batch_size - first_count
        if first_count > 0:
            tilt_angle[sample_first_segment] = (
                torch.rand(first_count, device=device) * cone_half_angle_split
            )
        if second_count > 0:
            tilt_angle[~sample_first_segment] = (
                cone_half_angle_split
                + torch.rand(second_count, device=device)
                * (cone_half_angle_max - cone_half_angle_split)
            )
        azimuth = torch.empty(batch_size, device=device).uniform_(0.0, 2 * math.pi)

        qw = torch.cos(tilt_angle / 2.0)
        qx = torch.sin(tilt_angle / 2.0) * torch.cos(azimuth)
        qy = torch.sin(tilt_angle / 2.0) * torch.sin(azimuth)
        qz = torch.zeros_like(qw)
        q_noise = torch.stack([qw, qx, qy, qz], dim=-1)

        q_base = my_env.insert_pose[:, 3:7]

        nw, nx, ny, nz = q_noise[:, 0], q_noise[:, 1], q_noise[:, 2], q_noise[:, 3]
        bw, bx, by, bz = q_base[:, 0], q_base[:, 1], q_base[:, 2], q_base[:, 3]

        new_w = nw * bw - nx * bx - ny * by - nz * bz
        new_x = nw * bx + nx * bw + ny * bz - nz * by
        new_y = nw * by - nx * bz + ny * bw + nz * bx
        new_z = nw * bz + nx * by - ny * bx + nz * bw

        q_new = torch.stack([new_w, new_x, new_y, new_z], dim=-1)
        q_new = _normalize_quat_wxyz_torch(q_new)

        target_end_noisy = torch.cat([new_pos, q_new], dim=-1)

        pos_no_noise = my_env.insert_pose[:, 0:3] + torch.tensor([0.0, 0.0, 0.1], device=device)
        target_end = torch.cat([pos_no_noise, q_new], dim=-1)
        target_end_up = target_end + torch.tensor([0.0, 0.0, 0.08, 0, 0, 0, 0], device=device)
        sub_step_move_joint(target_end_up, 3000, gripper_val=0.065, use_touch_ctrl=True, f_grp=45)
        sub_step_move_joint(target_end, 1000, gripper_val=0.065, use_touch_ctrl=True, f_grp=45)
        sub_step_move_joint(target_end_noisy, 1000, gripper_val=0.065, use_touch_ctrl=True, f_grp=45)

        print("6. 计算并提取最后时刻的力矩偏置 (wrench_bias)...")
        wrench_bias = torch.zeros((batch_size, 6), dtype=torch.float32, device=device)
        for _ in range(50):
            # 每步重新下发当前保持控制，减少末端在偏置标定阶段继续漂移。
            my_env.touch_ctrl(joints=target_q, de=0.06, fd=45)
            my_env.physics_step()
            # 与训练时 xwrench_clean 的定义对齐，这里标定滤波后的世界系力觉偏置。
            wrench_bias += my_env.xwrench_eef_left
        wrench_bias /= 50.0

        pose_pos_err = torch.linalg.norm(my_env.xpose_left[:, 0:3] - target_end_noisy[:, 0:3], dim=-1)
        pose_rot_err = _quat_angle_wxyz_torch(my_env.xpose_left[:, 3:7], target_end_noisy[:, 3:7])
        pose_close_mask = (pose_pos_err < save_pose_pos_thresh) & (pose_rot_err < save_pose_rot_thresh)
        grip_force = my_env.zk_controller.T_touch_filtered
        grip_force_ok_mask = grip_force > save_grip_force_thresh

        # ==========================================
        # 4. 提取底层状态并保存到 CPU 内存字典中
        # ==========================================
        if not hasattr(my_env, "ctrl_torch"):
            my_env.ctrl_torch = wp.to_torch(my_env.mjw_data.ctrl)
        if not hasattr(my_env, "qacc_warmstart_torch"):
            my_env.qacc_warmstart_torch = wp.to_torch(my_env.mjw_data.qacc_warmstart)

        batch_state_pool = {
            "qpos": my_env.qpos_torch.cpu().clone(),
            "qvel": my_env.qvel_torch.cpu().clone(),
            "ctrl": my_env.ctrl_torch.cpu().clone(),
            "qacc_warmstart": my_env.qacc_warmstart_torch.cpu().clone(),
            "vise_open_dis": my_env.vise_open_dis.cpu().clone(),
            "vise_base_pos": my_env.vise_base_pos.cpu().clone(),
            "arm_integral": my_env.controller.integral.cpu().clone(),
            "wrench_eef_left": my_env.wrench_eef_left.cpu().clone(),
            "xwrench_eef_left": my_env.xwrench_eef_left.cpu().clone(),
            "wrench_drift": my_env.wrench_drift.cpu().clone(),
            "zk_omega": my_env.zk_controller.omega.cpu().clone(),
            "zk_T_touch_old": my_env.zk_controller.T_touch_old.cpu().clone(),
            "zk_flag_touch_current": my_env.zk_controller.flag_touch_current.cpu().clone(),
            "zk_T_touch_filtered": my_env.zk_controller.T_touch_filtered.cpu().clone(),
            "zk_ddq_last": my_env.zk_controller.ddq_last.cpu().clone(),
            "target_x": target_x.cpu().clone(),
            "target_q": target_q.cpu().clone(),
            "wrench_bias": wrench_bias.cpu().clone()
        }

        valid_env_mask_torch = (~invalid_env_mask) & pose_close_mask & grip_force_ok_mask
        valid_env_mask = valid_env_mask_torch.cpu()
        saved_env_count = int(valid_env_mask.sum().item())
        dropped_env_count = batch_size - saved_env_count

        dropped_ik_count = int(invalid_env_mask.sum().item())
        dropped_pose_count = int((~pose_close_mask).sum().item())
        dropped_force_count = int((~grip_force_ok_mask).sum().item())

        if dropped_env_count > 0:
            print(
                f"[State Pool] drop {dropped_env_count} envs from batch {batch_idx + 1} "
                f"(IK fail: {dropped_ik_count}, pose mismatch: {dropped_pose_count}, grip force low: {dropped_force_count})."
            )
            if torch.any(~valid_env_mask_torch):
                failed_mask = ~valid_env_mask_torch
                print(
                    f"[State Pool] failed env stats | "
                    f"pos_err_mean={pose_pos_err[failed_mask].mean().item():.4f} m, "
                    f"rot_err_mean={np.rad2deg(pose_rot_err[failed_mask].mean().item()):.2f} deg, "
                    f"grip_force_mean={grip_force[failed_mask].mean().item():.2f}"
                )

        for key, tensor_data in batch_state_pool.items():
            if key not in global_state_pool:
                global_state_pool[key] = []
            global_state_pool[key].append(tensor_data[valid_env_mask].clone())

        envs_collected += saved_env_count

        batch_duration = time.perf_counter() - batch_start_time
        elapsed_total = time.perf_counter() - total_demo_start
        avg_batch_duration = elapsed_total / float(batch_idx + 1)
        remaining_batches = num_batches - batch_idx - 1
        eta_seconds = avg_batch_duration * remaining_batches
        print(
            f"[Timing] batch {batch_idx + 1}/{num_batches} finished in {_format_duration(batch_duration)} | "
            f"elapsed {_format_duration(elapsed_total)} | "
            f"ETA {_format_duration(eta_seconds)} | "
            f"saved {saved_env_count} envs, dropped {dropped_env_count} envs"
        )

    if if_save_env:
        print(f"\n🔄 正在合并 {num_batches} 个批次的数据...")
        final_state_pool = {}
        for key, tensor_list in global_state_pool.items():
            final_state_pool[key] = torch.cat(tensor_list, dim=0)

        env_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_state_pool, env_save_path)
        print(f"✅ 大功告成！总计 {envs_collected} 个环境的高动态全状态池已保存至:\n   {env_save_path}")

    total_demo_duration = time.perf_counter() - total_demo_start
    print(
        f"[Timing] generate_state_pool finished in {_format_duration(total_demo_duration)} | "
        f"collected {envs_collected} envs across {num_batches} batches"
    )


if __name__ == "__main__":
    generate_state_pool()
