import torch
import time


# =========================
# Utilities (支持任意维度)
# =========================
def wrap_to_pi_batch(x: torch.Tensor) -> torch.Tensor:
    """将任意形状的张量角度归一化到 (-pi, pi]"""
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


# =========================
# RM65 MDH Parameters
# =========================
def rm65_params_torch(d6_tcp_m: float = 0.1612, device: str = 'cpu', dtype: torch.dtype = torch.float32) -> dict:
    mm = 1e-3
    a = torch.tensor([0.0, 0.0, 256.0, 0.0, 0.0, 0.0], dtype=dtype, device=device) * mm
    alpha = torch.deg2rad(torch.tensor([0.0, 90.0, 0.0, 90.0, -90.0, 90.0], dtype=dtype, device=device))
    d = torch.tensor([240.5, 0.0, 0.0, 210.0, 0.0, d6_tcp_m / mm], dtype=dtype, device=device) * mm
    offset = torch.deg2rad(torch.tensor([0.0, 90.0, 90.0, 0.0, 0.0, 0.0], dtype=dtype, device=device))

    jlim_deg = torch.tensor([
        [-178.0, 178.0], [-130.0, 130.0], [-135.0, 135.0],
        [-178.0, 178.0], [-128.0, 128.0], [-360.0, 360.0],
    ], dtype=dtype, device=device)
    jlim = torch.deg2rad(jlim_deg)

    return dict(a=a, alpha=alpha, d=d, offset=offset, jlim=jlim)


# =========================
# Batched FK (支持 [..., 6] 输入)
# =========================
def fk_mdh_batch(q: torch.Tensor, P: dict, n: int = 6) -> torch.Tensor:
    """
    输入: q 形状为 [..., 6]
    输出: 齐次变换矩阵 [..., 4, 4]
    """
    batch_shape = q.shape[:-1]
    device, dtype = q.device, q.dtype

    # 初始化基坐标系 [..., 4, 4]
    T = torch.eye(4, device=device, dtype=dtype).expand(*batch_shape, 4, 4).clone()

    for i in range(n):
        theta = q[..., i] + P['offset'][i]

        ca, sa = torch.cos(P['alpha'][i]), torch.sin(P['alpha'][i])
        ct, st = torch.cos(theta), torch.sin(theta)
        a, d = P['a'][i], P['d'][i]

        zero = torch.zeros_like(theta)
        one = torch.ones_like(theta)
        a_t = torch.full_like(theta, a)
        d_t = torch.full_like(theta, d)

        # 【修改点】：给 -sa 和 ca 乘上 one，撑开 Batch 维度
        row1 = torch.stack([ct, -st, zero, a_t], dim=-1)
        row2 = torch.stack([st * ca, ct * ca, -sa * one, -sa * d_t], dim=-1)
        row3 = torch.stack([st * sa, ct * sa, ca * one, ca * d_t], dim=-1)
        row4 = torch.stack([zero, zero, zero, one], dim=-1)

        T_i = torch.stack([row1, row2, row3, row4], dim=-2)
        T = T @ T_i  # 批量矩阵乘法

    return T


def _get_R03_batch(q1: torch.Tensor, q2: torch.Tensor, q3: torch.Tensor, P: dict) -> torch.Tensor:
    """内部函数：快速批量求解 R03，输入形如 [..., 8]，输出 [..., 8, 3, 3]"""
    batch_shape = q1.shape
    device, dtype = q1.device, q1.dtype
    T = torch.eye(4, device=device, dtype=dtype).expand(*batch_shape, 4, 4).clone()
    qs = [q1, q2, q3]

    for i in range(3):
        theta = qs[i] + P['offset'][i]
        ca, sa = torch.cos(P['alpha'][i]), torch.sin(P['alpha'][i])
        ct, st = torch.cos(theta), torch.sin(theta)
        a, d = P['a'][i], P['d'][i]

        zero = torch.zeros_like(theta)
        one = torch.ones_like(theta)
        a_t = torch.full_like(theta, a)
        d_t = torch.full_like(theta, d)

        # 【修改点】：同上，给 -sa 和 ca 乘上 one
        row1 = torch.stack([ct, -st, zero, a_t], dim=-1)
        row2 = torch.stack([st * ca, ct * ca, -sa * one, -sa * d_t], dim=-1)
        row3 = torch.stack([st * sa, ct * sa, ca * one, ca * d_t], dim=-1)
        row4 = torch.stack([zero, zero, zero, one], dim=-1)
        T_i = torch.stack([row1, row2, row3, row4], dim=-2)
        T = T @ T_i

    return T[..., :3, :3]


# =========================
# 核心 Batched IK 求解器
# =========================
def rm65_analytical_ik_torch(
        target_pos: torch.Tensor,
        target_R: torch.Tensor,
        *,
        d6_tcp_m: float = 0.1612,
        prune_by_limits: bool = True,
        tol: float = 1e-9,
) -> torch.Tensor:
    """
    输入:
      target_pos: [..., 3]
      target_R:   [..., 3, 3]
    输出:
      sols: [..., 8, 6] 包含 8 组解析解。无效的解将被填充为 torch.nan。
    """
    device, dtype = target_pos.device, target_pos.dtype
    batch_shape = target_pos.shape[:-1]

    # 压平为 1D 批次以简化处理逻辑: [B, 3] 和 [B, 3, 3]
    p_targ = target_pos.reshape(-1, 3)
    R_targ = target_R.reshape(-1, 3, 3)
    B = p_targ.shape[0]

    P = rm65_params_torch(d6_tcp_m=d6_tcp_m, device=device, dtype=dtype)
    a, d, jlim = P["a"], P["d"], P["jlim"]
    d1, a2, d4, d6 = d[0], a[2], d[3], d[5]

    # 定义 8 个分支的结构矩阵 [B, 8]
    # 0,1,2,3 -> k=+r ; 4,5,6,7 -> k=-r
    k_signs = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1], device=device, dtype=dtype).unsqueeze(0).expand(B, 8)
    # 0,1,4,5 -> elbow up ; 2,3,6,7 -> elbow down
    q3_signs = torch.tensor([1, 1, -1, -1, 1, 1, -1, -1], device=device, dtype=dtype).unsqueeze(0).expand(B, 8)
    # 偶数正常手腕，奇数翻转手腕
    wrist_flips = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=device, dtype=dtype).unsqueeze(0).expand(B, 8)

    # 记录每个分支是否有效
    valid_mask = torch.ones((B, 8), dtype=torch.bool, device=device)

    # --- Step 1) 腕点 ---
    z6 = R_targ[..., 2]
    p_wc = p_targ - d6 * z6
    xw, yw, zw = p_wc[:, 0], p_wc[:, 1], p_wc[:, 2]

    # --- Step 2) 求解 q1, q2, q3 ---
    r = torch.hypot(xw, yw)
    ang = torch.atan2(yw, xw)

    q1_plus = wrap_to_pi_batch(ang + torch.pi)
    q1_minus = wrap_to_pi_batch(ang)
    is_sing_q1 = (r < tol).unsqueeze(1)  # [B, 1]

    k = r.unsqueeze(1) * k_signs  # [B, 8]
    q1_base = torch.where(k_signs > 0, q1_plus.unsqueeze(1), q1_minus.unsqueeze(1))
    q1 = torch.where(is_sing_q1, 0.0, q1_base)

    zp = (zw - d1).unsqueeze(1)  # [B, 1]
    L2 = k ** 2 + zp ** 2  # [B, 8]

    cos_q3 = (L2 - a2 ** 2 - d4 ** 2) / (2.0 * a2 * d4)
    # 标记不可达配置
    valid_mask &= (torch.abs(cos_q3) <= 1.0 + 1e-8)

    cos_q3 = torch.clamp(cos_q3, -1.0, 1.0)
    q3_base_val = torch.acos(cos_q3)
    q3 = q3_base_val * q3_signs  # [B, 8]

    gamma = torch.atan2(k, zp.expand_as(k))
    beta = torch.atan2(d4 * torch.sin(q3), a2 + d4 * torch.cos(q3))
    q2 = gamma - beta  # [B, 8]

    # --- Step 3) 求解腕部 q4, q5, q6 ---
    # 获取 [B, 8, 3, 3] 的前三轴旋转矩阵
    R03 = _get_R03_batch(q1, q2, q3, P)
    R_targ_exp = R_targ.unsqueeze(1).expand(B, 8, 3, 3)

    # R36 = R03^T @ R_target
    R36 = R03.transpose(-1, -2) @ R_targ_exp  # [B, 8, 3, 3]

    r11, r13 = R36[..., 0, 0], R36[..., 0, 2]
    r21, r22, r23 = R36[..., 1, 0], R36[..., 1, 1], R36[..., 1, 2]
    r31, r33 = R36[..., 2, 0], R36[..., 2, 2]

    s5 = torch.hypot(r21, r22)
    c5 = -r23

    # 正常解
    q5_norm = torch.atan2(s5, c5)
    q4_norm = torch.atan2(r33, r13)
    q6_norm = torch.atan2(-r22, r21)

    # 翻转手腕解
    q5_flip = -q5_norm
    q4_flip = wrap_to_pi_batch(q4_norm + torch.pi)
    q6_flip = wrap_to_pi_batch(q6_norm + torch.pi)

    q5 = torch.where(wrist_flips == 1, q5_flip, q5_norm)
    q4 = torch.where(wrist_flips == 1, q4_flip, q4_norm)
    q6 = torch.where(wrist_flips == 1, q6_flip, q6_norm)

    # 腕部奇异性处理 (q5 接近 0 或 pi)
    is_wrist_sing = (s5 < tol)
    q5_sing = torch.where(c5 > 0, 0.0, torch.pi)
    q4_sing = torch.zeros_like(q4)
    q6_sing = torch.atan2(r31, r11)

    q5 = torch.where(is_wrist_sing, q5_sing, q5)
    q4 = torch.where(is_wrist_sing, q4_sing, q4)
    q6 = torch.where(is_wrist_sing, q6_sing, q6)

    # --- Step 4) 打包并利用限制条件裁剪 ---
    sols = torch.stack([q1, q2, q3, q4, q5, q6], dim=-1)  # [B, 8, 6]
    sols = wrap_to_pi_batch(sols)

    if prune_by_limits:
        lo = jlim[:, 0].view(1, 1, 6)
        hi = jlim[:, 1].view(1, 1, 6)
        in_limits = torch.all((sols >= lo) & (sols <= hi), dim=-1)  # [B, 8]
        valid_mask &= in_limits

    # 将无效的分支填充为 NaN
    sols = torch.where(valid_mask.unsqueeze(-1), sols, float('nan'))

    # 恢复原有的 Batch 维度
    return sols.reshape(*batch_shape, 8, 6)


def run_torch_batch_test(N=2048, device="cuda"):
    print(f"=== 开始 Torch Batch 测试 (N={N}, Device={device}) ===")

    # 1. 准备参数 (直接调用新版的 rm65_params_torch 以对齐设备和数据类型)
    params = rm65_params_torch(d6_tcp_m=0.1612, device=device)
    jlim = params["jlim"]

    # 2. 随机生成 N 组关节角作为真值 (Ground Truth)
    q_true = torch.rand((N, 6), device=device) * (jlim[:, 1] - jlim[:, 0]) + jlim[:, 0]

    # 3. 并行前向动力学 (FK) 获取目标位姿
    # 输入 q_true 形状 [N, 6]，输出 T_target 形状 [N, 4, 4]
    t_fk_start = time.perf_counter()
    T_target = fk_mdh_batch(q_true, params, n=6)
    p_target = T_target[..., :3, 3]  # [N, 3]
    R_target = T_target[..., :3, :3]  # [N, 3, 3]

    if device == "cuda":
        torch.cuda.synchronize()
    print(f"Batch FK 耗时: {(time.perf_counter() - t_fk_start) * 1000:.3f} ms")

    # 4. 全并行逆运动学 (IK) 求解
    t_ik_start = time.perf_counter()

    # 调用重构后的 Torch 版 IK
    # q_sols 形状为 [N, 8, 6]，无效解会被填充为 NaN
    q_sols = rm65_analytical_ik_torch(p_target, R_target, d6_tcp_m=0.1612, prune_by_limits=True)

    if device == "cuda":
        torch.cuda.synchronize()
    t_ik_end = time.perf_counter()

    total_ik_ms = (t_ik_end - t_ik_start) * 1000.0
    print(f"Batch IK 总耗时 (共解 {N * 8} 个分支): {total_ik_ms:.3f} ms")
    print(f"平均每个 Batch 耗时: {total_ik_ms / N:.6f} ms")

    # 5. 验证结果 (将解代回 FK 检查误差)
    # 因为 q_sols 中包含 NaN，我们可以先将其替换为 0 以免 FK 报错（或者直接传进去，让 FK 产出 NaN 位姿）
    # 这里直接传进去，FK 矩阵中含 NaN 的部分在计算距离时依然是 NaN
    T_sol = fk_mdh_batch(q_sols, params, n=6)  # 输入 [N, 8, 6] -> 输出 [N, 8, 4, 4]

    p_sol = T_sol[..., :3, 3]  # [N, 8, 3]
    R_sol = T_sol[..., :3, :3]  # [N, 8, 3, 3]

    # 扩展目标以匹配 8 个分支，利用广播机制进行减法
    p_target_exp = p_target.unsqueeze(1)  # [N, 1, 3]
    R_target_exp = R_target.unsqueeze(1)  # [N, 1, 3, 3]

    # 计算位置误差 (Euclidean) 和 姿态误差 (Frobenius)
    pos_err = torch.norm(p_sol - p_target_exp, dim=-1)  # [N, 8]
    rot_err = torch.norm(R_sol - R_target_exp, p='fro', dim=(-2, -1))  # [N, 8]

    # 统计成功率 (阈值：1mm 和 0.01 姿态误差)
    # 注意：含 NaN 的地方，(NaN < 1e-3) 在 PyTorch 中会安全地返回 False
    branch_success = (pos_err < 1e-3) & (rot_err < 1e-2)  # [N, 8]

    # 只要 8 个分支中有 1 个解达标，这一个 Batch 就算求解成功
    batch_success = branch_success.any(dim=1)  # [N]
    n_success = torch.sum(batch_success).item()

    # 6. 报告
    print("-" * 30)
    print(f"Batch 成功率: {n_success}/{N} ({n_success / N * 100:.2f}%)")
    if n_success > 0:
        print(f"有效分支的平均位置误差: {torch.mean(pos_err[branch_success]) * 1000:.6f} mm")
        print(f"有效分支的最大位置误差: {torch.max(pos_err[branch_success]) * 1000:.6f} mm")
        print(f"有效分支的平均姿态误差: {torch.mean(rot_err[branch_success]):.6e}")

    # 打印第一个成功的例子的详细对比
    if n_success > 0:
        # 找到第一个成功的 Batch 索引
        idx = torch.where(batch_success)[0][0].item()
        # 找到这个 Batch 中第一个成功的分支索引
        branch_idx = torch.where(branch_success[idx])[0][0].item()

        print("-" * 30)
        print(f"示例对比 (Batch Index {idx}, 匹配到的分支 Branch {branch_idx}):")
        print(f"  True q (Ground Truth) : {q_true[idx].cpu().numpy()}")
        print(f"  Sol  q (IK 的有效分支): {q_sols[idx, branch_idx].cpu().numpy()}")
        print(f"  Pos Err: {pos_err[idx, branch_idx].item() * 1000:.6f} mm")
    else:
        print("未找到成功的解。")


if __name__ == "__main__":
    # 第一次运行会包含 CUDA 初始化开销，建议 N 设大一点看平均性能
    run_torch_batch_test(N=2048, device="cuda" if torch.cuda.is_available() else "cpu")
