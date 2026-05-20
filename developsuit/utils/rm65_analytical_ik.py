import numpy as np
import time

# =========================
# Utilities
# =========================
def wrap_to_pi(x: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi

def close_to_zero(x: float, tol: float = 1e-9) -> bool:
    return abs(x) < tol

def rot_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """
    Quaternion (w, x, y, z) -> Rotation matrix.
    """
    w, x, y, z = q
    # normalize
    n = np.sqrt(w*w + x*x + y*y + z*z)

    # # ==== DEBUG: 检查四元数范数 ====
    # print('=============[DEBUG]=============')
    # if n < 1e-8:
    #     print("[RM65-LOWLEVEL] rot_from_quat_wxyz: zero or tiny-norm quaternion detected!")
    #     print(f"  q = {q}, norm = {n}")
    #     # 这里你有两种选择：raise 让上层看到错误，或者返回单位阵继续跑
    #     # 我建议调试阶段先 raise，这样能精准定位第一次出现问题的地方：
    #     raise ValueError(f"Zero-norm quaternion passed to rot_from_quat_wxyz: q={q}")
    #     # 如果你想先不打断流程，也可以改成：
    #     # return np.eye(3, dtype=float)
    # # =================================

    w, x, y, z = w/n, x/n, y/n, z/n
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=float)
    return R

# =========================
# MDH Forward Kinematics
# =========================
def mdh_A(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """
    Modified DH (Craig) transform:

        T_{i-1}^{i} = RotX(alpha_{i-1}) * TransX(a_{i-1}) * RotZ(theta_i) * TransZ(d_i)

    where:
      a:     a_{i-1}
      alpha: alpha_{i-1}
      d:     d_i
      theta: theta_i  (NOTE: theta_i = q_i + offset_i)
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    # Expanded 4x4 (for numerical stability and speed)
    return np.array([
        [ct,       -st,        0.0,        a],
        [st*ca,  ct*ca,     -sa,   -sa*d],
        [st*sa,  ct*sa,      ca,    ca*d],
        [0.0,     0.0,      0.0,    1.0],
    ], dtype=float)

def fk_mdh(q: np.ndarray, a: np.ndarray, alpha: np.ndarray, d: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """FK for 6R under MDH."""
    T = np.eye(4)
    for i in range(6):
        theta = q[i] + offset[i]
        T = T @ mdh_A(a[i], alpha[i], d[i], theta)
    return T

def fk_mdh_n(q: np.ndarray, a: np.ndarray, alpha: np.ndarray, d: np.ndarray, offset: np.ndarray, n: int) -> np.ndarray:
    """FK up to joint n (n<=6)."""
    T = np.eye(4)
    for i in range(n):
        theta = q[i] + offset[i]
        T = T @ mdh_A(a[i], alpha[i], d[i], theta)
    return T

# =========================
# RM65 MDH Parameters (from your table)
# Units: meters, radians
# =========================
def rm65_params(d6_tcp_m: float = 0.1612) -> dict:
    """
    d6_tcp_m:
      - use 0.0 if your DH frame is flange frame (tool offset handled separately)
      - use 0.1612 if you want DH end-effector at TCP (as you provided)
    """
    mm = 1e-3
    a = np.array([0, 0, 256, 0, 0, 0], dtype=float) * mm
    alpha = np.deg2rad(np.array([0, 90, 0, 90, -90, 90], dtype=float))
    d = np.array([240.5, 0, 0, 210, 0, d6_tcp_m / mm], dtype=float) * mm
    offset = np.deg2rad(np.array([0, 90, 90, 0, 0, 0], dtype=float))

    # Joint limits from your message (degrees) -> radians
    jlim_deg = np.array([
        [-178, 178],
        [-130, 130],
        [-135, 135],
        [-178, 178],
        [-128, 128],
        [-360, 360],
    ], dtype=float)
    jlim = np.deg2rad(jlim_deg)

    return dict(a=a, alpha=alpha, d=d, offset=offset, jlim=jlim)

# =========================
# Analytical IK for RM65 (MDH)
# =========================
def rm65_analytical_ik(
    target_pos: np.ndarray,
    target_R: np.ndarray,
    *,
    d6_tcp_m: float = 0.1612,
    prune_by_limits: bool = True,
    tol: float = 1e-9,
    q_seed: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    Solve IK for RM65 under Modified DH parameters.

    Inputs
    ------
    target_pos: (3,) in meters
    target_R:   (3,3) rotation matrix of desired end-effector frame in base
    d6_tcp_m:   d6 used in MDH table (meters). See rm65_params() docstring.

    Output
    ------
    solutions: (K,6) joint angles in radians (each wrapped to (-pi, pi])
    """
    P = rm65_params(d6_tcp_m=d6_tcp_m)
    a, alpha, d, offset, jlim = P["a"], P["alpha"], P["d"], P["offset"], P["jlim"]
    if q_seed is not None:
        q_seed = np.asarray(q_seed, dtype=float).reshape(-1)
        if q_seed.size < 6:
            raise ValueError("q_seed must have at least 6 elements.")
        q_seed = q_seed[:6]

    # ------------------------------------------------------------
    # Step 1) Wrist center (sphere wrist decoupling)
    # ------------------------------------------------------------
    # In MDH here, last translation is TransZ(d6) along z6.
    # So the wrist center (intersection of wrist axes) is:
    #
    #   p_wc = p_06 - d6 * z6
    #        = p_06 - d6 * R_06 @ [0,0,1]
    #
    z6 = target_R[:, 2]
    p_wc = target_pos - d[5] * z6
    xw, yw, zw = p_wc

    # ------------------------------------------------------------
    # Step 2) Solve q1, q2, q3 from wrist-center position
    # ------------------------------------------------------------
    # Using MDH for RM65 (with offsets already included in theta2/theta3),
    # the wrist center p_wc (= p04) simplifies to (derived from MDH chain):
    #
    #   xw = -(a2*sin(q2) + d4*sin(q2+q3)) * cos(q1)
    #   yw = -(a2*sin(q2) + d4*sin(q2+q3)) * sin(q1)
    #   zw =  d1 + a2*cos(q2) + d4*cos(q2+q3)
    #
    # Let:
    #   k  = a2*sin(q2) + d4*sin(q2+q3)
    #   z' = zw - d1 = a2*cos(q2) + d4*cos(q2+q3)
    #
    # Then in the (k, z') plane it's a classic 2-link IK:
    #   [k, z'] corresponds to link lengths a2 and d4 with elbow angle q3.
    #
    d1, a2, d4 = d[0], a[2], d[3]
    r = np.hypot(xw, yw)

    # Two q1 branches come from k being +r or -r:
    #   if xw = -k cos(q1), yw = -k sin(q1)
    #   choose:
    #     (q1 = atan2(yw, xw) + pi, k = +r)  OR
    #     (q1 = atan2(yw, xw),       k = -r)
    q1_candidates = []
    if r < tol:
        # Shoulder singular: wrist center on base z-axis -> q1 indeterminate.
        # Here we pick q1=0 as a default; you may want to keep last q1 or search multiple seeds.
        q1_candidates.append((0.0, 0.0))
    else:
        ang = np.arctan2(yw, xw)
        q1_candidates.append((wrap_to_pi(ang + np.pi), +r))
        q1_candidates.append((wrap_to_pi(ang),         -r))

    sols = []

    for q1, k in q1_candidates:
        zp = zw - d1
        L2 = k*k + zp*zp

        # Law of cosines:
        #   L^2 = a2^2 + d4^2 + 2*a2*d4*cos(q3)
        # => cos(q3) = (L^2 - a2^2 - d4^2) / (2*a2*d4)
        cos_q3 = (L2 - a2*a2 - d4*d4) / (2.0 * a2 * d4)

        if abs(cos_q3) > 1.0 + 1e-8:
            continue
        cos_q3 = float(np.clip(cos_q3, -1.0, 1.0))

        q3_list = [np.arccos(cos_q3), -np.arccos(cos_q3)]  # elbow-up / elbow-down

        for q3 in q3_list:
            # Define:
            #   gamma = atan2(k, z')  (direction to wrist center in the plane)
            #   beta  = atan2(d4*sin(q3), a2 + d4*cos(q3))
            # Then:
            #   q2 = gamma - beta
            gamma = np.arctan2(k, zp)
            beta = np.arctan2(d4 * np.sin(q3), a2 + d4 * np.cos(q3))
            q2 = gamma - beta

            # ------------------------------------------------------------
            # Step 3) Solve wrist q4, q5, q6 from orientation
            # ------------------------------------------------------------
            # Compute R03 from FK(q1,q2,q3), then:
            #   R36 = R03^T * R06
            q_first3 = np.array([q1, q2, q3, 0, 0, 0], dtype=float)
            T03 = fk_mdh_n(q_first3, a, alpha, d, offset, n=3)
            R03 = T03[:3, :3]
            R36 = R03.T @ target_R

            # For this RM65 wrist under MDH with:
            #   alpha3 = +90°, alpha4 = -90°, alpha5 = +90°
            # the wrist rotation becomes:
            #
            #   R36 = Rx(90) Rz(q4) Rx(-90) Rz(q5) Rx(90) Rz(q6)
            #
            # From symbolic expansion:
            #   r13 = sin(q5)*cos(q4)
            #   r33 = sin(q4)*sin(q5)
            #   r21 = sin(q5)*cos(q6)
            #   r22 = -sin(q5)*sin(q6)
            #   r23 = -cos(q5)
            #
            # So (when sin(q5) != 0):
            #   q5 = atan2( sqrt(r21^2+r22^2), -r23 )
            #   q4 = atan2( r33, r13 )
            #   q6 = atan2( -r22, r21 )
            r11, r12, r13 = R36[0, 0], R36[0, 1], R36[0, 2]
            r21, r22, r23 = R36[1, 0], R36[1, 1], R36[1, 2]
            r31, r32, r33 = R36[2, 0], R36[2, 1], R36[2, 2]

            s5 = np.hypot(r21, r22)
            c5 = -r23

            if s5 < tol:
                # Wrist singularity (q5 ~ 0 or pi): q4 and q6 are coupled.
                # Keep continuity with the seed wrist pose when available.
                q5 = 0.0 if c5 > 0 else np.pi
                phi = np.arctan2(r31, r11)
                singular_solutions = []

                if q_seed is not None:
                    q4_seed = wrap_to_pi(q_seed[3])
                    q6_seed = wrap_to_pi(q_seed[5])
                    singular_solutions.append(
                        np.array([q1, q2, q3, q4_seed, q5, wrap_to_pi(phi - q4_seed)], dtype=float)
                    )
                    singular_solutions.append(
                        np.array([q1, q2, q3, wrap_to_pi(phi - q6_seed), q5, q6_seed], dtype=float)
                    )
                else:
                    singular_solutions.append(np.array([q1, q2, q3, 0.0, q5, wrap_to_pi(phi)], dtype=float))

                for sol in singular_solutions:
                    if not any(np.allclose(sol, existing, atol=1e-8) for existing in sols):
                        sols.append(sol)
            else:
                q5 = np.arctan2(s5, c5)
                q4 = np.arctan2(r33, r13)
                q6 = np.arctan2(-r22, r21)

                sols.append(np.array([q1, q2, q3, q4, q5, q6], dtype=float))

                # Second wrist solution:
                #   (q4, q5, q6) -> (q4+pi, -q5, q6+pi)
                sols.append(np.array([q1, q2, q3,
                                      wrap_to_pi(q4 + np.pi),
                                      -q5,
                                      wrap_to_pi(q6 + np.pi)], dtype=float))

    if len(sols) == 0:
        return None

    sols = np.array([[wrap_to_pi(v) for v in s] for s in sols], dtype=float)

    # # ==== DEBUG: 打印解的情况 ====
    # print('=============[DEBUG]=============')
    # if sols is None or len(sols) == 0:
    #     print("[RM65-LOWLEVEL] rm65_analytical_ik_quat: NO solutions returned.")
    # else:
    #     print(f"[RM65-LOWLEVEL] rm65_analytical_ik_quat: {len(sols)} solutions returned.")
    #     # 你也可以只打印第一组解，避免刷屏：
    #     print("  first solution (rad) =", sols[0])
    # # =================================

    # ------------------------------------------------------------
    # Step 4) Prune by joint limits
    # ------------------------------------------------------------
    if prune_by_limits:
        lo, hi = jlim[:, 0], jlim[:, 1]
        # Note: J6 range is +/-360°, wrap_to_pi will map it to (-pi, pi],
        # If you truly want multi-turn solutions, you need a separate unwrapping strategy.
        mask = np.all((sols >= lo) & (sols <= hi), axis=1)
        sols = sols[mask]

    return sols if len(sols) else None

# =========================
# Convenience wrapper: quaternion input
# =========================
def rm65_analytical_ik_quat(
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    *,
    d6_tcp_m: float = 0.1612,
    prune_by_limits: bool = True,
    q_seed: np.ndarray | None = None,
) -> np.ndarray | None:
    target_pos = np.asarray(target_pos, dtype=float)
    target_quat_wxyz = np.asarray(target_quat_wxyz, dtype=float)
    # # ==== DEBUG: 打印输入 ====
    # print('=============[DEBUG]=============')
    # print("[RM65-LOWLEVEL] rm65_analytical_ik_quat called")
    # print("  target_pos_world   =", target_pos)
    # print("  target_quat_world  =", target_quat_wxyz,
    #       " |quat| =", float(np.linalg.norm(target_quat_wxyz)))
    # # =========================

    R = rot_from_quat_wxyz(target_quat_wxyz)

    # ==== DEBUG: 打印旋转矩阵 ====
    # print("  target_R_world (first row) =", R[0])
    # =============================

    return rm65_analytical_ik(target_pos, R, d6_tcp_m=d6_tcp_m, prune_by_limits=prune_by_limits, q_seed=q_seed)


# =========================
# (Optional) random batch test
# =========================
if __name__ == "__main__":
    P = rm65_params(d6_tcp_m=0.1612)
    a, alpha, d, offset, jlim = P["a"], P["alpha"], P["d"], P["offset"], P["jlim"]

    # -----------------------------
    # Random batch test settings
    # -----------------------------
    N = 200
    rng = np.random.default_rng(0)

    pos_errs = []
    rot_errs = []
    solve_times_ms = []
    n_solutions_list = []
    n_success = 0
    n_fail = 0

    # Helper: sample joint angles within limits (uniform in joint space)
    def sample_q_within_limits(rng, jlim):
        lo, hi = jlim[:, 0], jlim[:, 1]
        return rng.uniform(lo, hi)

    # Helper: rotation matrix distance (Frobenius norm)
    def rot_fro_norm(R1, R2):
        return np.linalg.norm(R1 - R2, ord="fro")

    # -----------------------------
    # Run tests
    # -----------------------------
    t_batch0 = time.perf_counter()

    for k in range(N):
        # 1) sample a random reachable configuration
        q_true = sample_q_within_limits(rng, jlim)
        # q_true = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # DEBUG: 固定一个测试点，方便调试

        # 2) forward kinematics to get target pose
        T = fk_mdh(q_true, a, alpha, d, offset)
        p_des = T[:3, 3]
        R_des = T[:3, :3]

        # 3) solve IK + time it
        t0 = time.perf_counter()
        sols = rm65_analytical_ik(p_des, R_des, d6_tcp_m=0.1612, prune_by_limits=True)
        t1 = time.perf_counter()
        solve_times_ms.append((t1 - t0) * 1000.0)

        if sols is None or len(sols) == 0:
            n_fail += 1
            n_solutions_list.append(0)
            continue

        n_success += 1
        n_solutions_list.append(len(sols))

        # 4) verify: choose the best solution (min combined pose error)
        best_pos_err = np.inf
        best_rot_err = np.inf

        for q_sol in sols:
            T_sol = fk_mdh(q_sol, a, alpha, d, offset)
            pos_err = np.linalg.norm(T_sol[:3, 3] - p_des)
            rot_err = rot_fro_norm(T_sol[:3, :3], R_des)

            # simple combined criterion (position dominates; rotation scaled)
            score = pos_err + 0.1 * rot_err
            if score < best_pos_err + 0.1 * best_rot_err:
                best_pos_err = pos_err
                best_rot_err = rot_err

        pos_errs.append(best_pos_err)
        rot_errs.append(best_rot_err)

    t_batch1 = time.perf_counter()
    total_ms = (t_batch1 - t_batch0) * 1000.0

    # -----------------------------
    # Report
    # -----------------------------
    def safe_stats(x):
        if len(x) == 0:
            return dict(mean=np.nan, std=np.nan, max=np.nan, p95=np.nan)
        x = np.array(x, dtype=float)
        return dict(
            mean=float(np.mean(x)),
            std=float(np.std(x)),
            max=float(np.max(x)),
            p95=float(np.percentile(x, 95)),
        )

    pos_s = safe_stats(pos_errs)
    rot_s = safe_stats(rot_errs)
    time_s = safe_stats(solve_times_ms)

    print(f"Batch size: {N}")
    print(f"Success: {n_success}/{N}  Fail: {n_fail}/{N}")
    print(f"Total wall time: {total_ms:.3f} ms  (avg per case: {total_ms/N:.3f} ms)")

    print("\n#solutions per success (min/mean/max):")
    if n_success > 0:
        ns = np.array([n for n in n_solutions_list if n > 0], dtype=int)
        print(f"  min={int(ns.min())}, mean={float(ns.mean()):.2f}, max={int(ns.max())}")
    else:
        print("  (no successful cases)")

    print("\nPosition error (meters):")
    print(f"  mean={pos_s['mean']:.3e}, std={pos_s['std']:.3e}, p95={pos_s['p95']:.3e}, max={pos_s['max']:.3e}")

    print("\nRotation error (Frobenius norm):")
    print(f"  mean={rot_s['mean']:.3e}, std={rot_s['std']:.3e}, p95={rot_s['p95']:.3e}, max={rot_s['max']:.3e}")

    print("\nSolve time per IK call (ms):")
    print(f"  mean={time_s['mean']:.3f}, std={time_s['std']:.3f}, p95={time_s['p95']:.3f}, max={time_s['max']:.3f}")

    # 测试用例：所有关节角度为0
    q = np.zeros(6)

    print("=" * 60)
    print("测试1: 验证零位姿态")
    print("=" * 60)

    T = fk_mdh(q, a, alpha, d, offset)
    print(f"零位末端位置: {T[:3, 3] * 1000} mm")
    print(f"零位末端姿态:\n{T[:3, :3]}")

    # 手工计算零位（考虑offset）
    # θ1=0, θ2=90°, θ3=90°, θ4=0, θ5=0, θ6=0
    print("\n手工计算验证:")

    # T01
    T01 = mdh_A(a[0], alpha[0], d[0], offset[0])
    print(f"T01 position: {T01[:3, 3] * 1000} mm")

    # T02
    T02 = T01 @ mdh_A(a[1], alpha[1], d[1], offset[1])
    print(f"T02 position: {T02[:3, 3] * 1000} mm")

    # T03
    T03 = T02 @ mdh_A(a[2], alpha[2], d[2], offset[2])
    print(f"T03 position: {T03[:3, 3] * 1000} mm")

    # T04 (腕部中心)
    T04 = T03 @ mdh_A(a[3], alpha[3], d[3], offset[3])
    print(f"T04 position (腕部中心): {T04[:3, 3] * 1000} mm")

    print("\n" + "=" * 60)
    print("测试2: 验证IK求解")
    print("=" * 60)

    # 随机测试
    np.random.seed(42)
    jlim = P["jlim"]

    n_tests = 10
    success = 0

    for i in range(n_tests):
        # 随机关节角度
        q_true = np.random.uniform(jlim[:, 0], jlim[:, 1])
        
        # FK
        T_target = fk_mdh(q_true, a, alpha, d, offset)
        p_target = T_target[:3, 3]
        R_target = T_target[:3, :3]
        
        # IK
        sols = rm65_analytical_ik(p_target, R_target, d6_tcp_m=0.1612)
        
        if sols is None:
            print(f"测试 {i+1}: 无解")
            continue
        
        # 验证解
        found_match = False
        for q_sol in sols:
            T_sol = fk_mdh(q_sol, a, alpha, d, offset)
            pos_err = np.linalg.norm(T_sol[:3, 3] - p_target)
            rot_err = np.linalg.norm(T_sol[:3, :3] - R_target, ord='fro')
            
            if pos_err < 1e-6 and rot_err < 1e-6:
                found_match = True
                break
        
        if found_match:
            success += 1
            print(f"测试 {i+1}: ✓ (找到 {len(sols)} 个解)")
        else:
            print(f"测试 {i+1}: ✗ 误差过大")
            print(f"  位置误差: {pos_err*1000:.6f} mm")
            print(f"  姿态误差: {rot_err:.6f}")

    print(f"\n成功率: {success}/{n_tests}")

    print("\n" + "=" * 60)
    print("测试3: 检查腕部中心计算")
    print("=" * 60)

    # 特定姿态测试
    q_test = np.array([0, 0, 0, 0, 0, 0])
    T = fk_mdh(q_test, a, alpha, d, offset)

    # 计算腕部中心（应该是T04的位置）
    T04 = fk_mdh(np.concatenate([q_test[:3], [0, 0, 0]]), a, alpha, d, offset)
    p_wc_true = T04[:3, 3]

    # 用代码中的方法计算
    z6 = T[:3, 2]
    p_wc_calc = T[:3, 3] - d[5] * z6

    print(f"真实腕部中心: {p_wc_true * 1000} mm")
    print(f"计算腕部中心: {p_wc_calc * 1000} mm")
    print(f"误差: {np.linalg.norm(p_wc_true - p_wc_calc) * 1000:.6f} mm")
