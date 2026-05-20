from developsuit.utils.transform_utils import *
import math as m
import matplotlib.pyplot as plt


# m表示使用的是改进DH参数
def fkine_m(DH, q, DH_mode=None, R_base=np.eye(3), base=np.zeros([3, 1])):
    # 标准DH参数
    q = q.reshape([-1, 1])
    DH = np.copy(DH)
    Ndof = q.size
    if DH_mode is None:
        DH_mode = ["hinge"] * Ndof

    temp = np.eye(4)
    temp[0:3, 0:3] = R_base  # 基座旋转矩阵
    temp[0:3, 3] = base.reshape([3])  # 基座向量
    T = np.zeros([4, 4, Ndof])  # 初始化T

    # DH第一列为theta的offset, 在这上面加关节角
    for ii in range(Ndof):
        if DH_mode[ii] == "hinge":
            DH[ii, 0] += q[ii, 0]
        if DH_mode[ii] == "slide":
            DH[ii, 1] += q[ii, 0]

    for ii in range(Ndof):
        ct = m.cos(DH[ii, 0])
        st = m.sin(DH[ii, 0])
        ca = m.cos(DH[ii, 3])
        sa = m.sin(DH[ii, 3])

        temp = temp @ np.array([[ct, -st, 0., DH[ii, 2]],
                                [ca * st, ca * ct, - sa, - DH[ii, 1] * sa],
                                [sa * st, sa * ct, ca, DH[ii, 1] * ca],
                                [0., 0., 0., 1.]])

        T[:, :, ii] = temp

    return T


def jac_m(DH, q, DH_end, dq=None, DH_mode=None, R_base=np.eye(3), base=np.zeros([3, 1]), omega_base=np.zeros([3, 1])):
    # https://zhuanlan.zhihu.com/p/205342861?utm_id=0
    q = q.reshape([-1])
    DH = np.copy(DH)
    Ndof = DH.shape[0]
    if DH_mode is None:
        DH_mode = ["hinge"] * Ndof

    if dq is None:
        dq = np.zeros([Ndof, 1])
    dq = dq.reshape([Ndof, 1])

    T = fkine_m(DH, q, DH_mode=DH_mode, R_base=R_base, base=base)  # 所有的T
    R = T[0:3, 0:3, :]  # 各系旋转矩阵
    x = T[0:3, 3, :].reshape([3, -1])  # 各系的原点位置
    x_end, quat_end = fkine_ee_m(DH, q, DH_end, DH_mode=DH_mode, R_base=R_base, base=base)
    x_end = x_end.reshape([3, 1])  # 末端位置
    x = np.hstack((x, x_end))

    omega_save = np.zeros([3, Ndof])

    R_old = R_base.copy()
    omega_old = omega_base.reshape([3, 1])
    for ii in range(Ndof):
        if DH_mode[ii] == "hinge":
            omega = omega_old + R_old @ np.array([0, 0, dq[ii, 0]]).reshape([3, 1]) # 各坐标系的角速度
        else:
            omega = omega_old.copy()

        # 保存参数
        omega_save[:, ii] = omega.reshape([3])  # 对应于 q(ii+6d)

        # 更新参数
        R_old = R[:, :, ii]
        omega_old = omega.copy()

    # 对应于网页式(8)
    dx = np.zeros([3, 1])  # end 相对于 6(=end) 的速度
    dX = np.zeros([3, Ndof + 1])  # 0为end相对base，1为end相对1,Ndof为end相对于最后一个关节, 即与关节角对应
    for ii in range(Ndof - 1, 0, -1):  # Ndof-6d : -6d : 6d
        if DH_mode[ii] == "hinge":
            dx += np.cross(omega_save[:, ii].reshape([1, 3]), (x[:, ii + 1] - x[:, ii]).reshape([1, 3])).reshape([3, 1])
        else:
            dx += R[:, :, ii] @ np.array([0, 0, dq[ii, 0]]).reshape([3, 1])
        dX[:, ii + 1] = dx.reshape([3])

    # 基座
    dx += np.cross(omega_base.reshape([3]), (x[:, 0] - base).reshape([3])).reshape([3, 1])
    dX[:, 0] = dx.reshape([3])

    # qi 在 i 系下描述, 0为基座
    Z = np.zeros([3, Ndof + 1])  # Z轴矢量
    U = np.zeros([3, Ndof + 1])  # Z矢量叉乘（x_end - xi）
    dZ = np.zeros([3, Ndof + 1])
    dU = np.zeros([3, Ndof + 1])

    # 对应于网页式(3)
    Z[:, 0] = (R_base @ np.array([0., 0., 1.]).reshape([3, 1])).reshape([3])
    # 对应于网页式(3)
    U[:, 0] = np.cross(Z[:, 0].reshape([3]), (x_end.reshape([3]) - base.reshape([3]))).reshape([3])
    # 对应于网页式(10)
    dZ[:, 0] = np.cross(omega_base.reshape([3]), Z[:, 0].reshape([3])).reshape([3])
    # 对应于网页式(7) 改为了e * p，对应(6)中的第一行
    dU[:, 0] = (((np.cross(Z[:, 0].reshape([3]), dX[:, 0].reshape([3]))
                  + np.cross(dZ[:, 0].reshape([3]), x_end.reshape([3]) - base.reshape([3])))).reshape([3]))

    for ii in range(Ndof):
        # 对应于网页式(3)
        Z[:, ii + 1] = (R[:, :, ii] @ np.array([0., 0., 1.]).reshape([3, 1])).reshape([3])
        # 对应于网页式(3)
        U[:, ii + 1] = np.cross(Z[:, ii + 1].reshape([3]), (x_end.reshape([3]) - x[:, ii].reshape([3]))).reshape([3])
        # 对应于网页式(10)
        dZ[:, ii + 1] = np.cross(omega_save[:, ii].reshape([3]), Z[:, ii + 1].reshape([3])).reshape([3])
        # 对应于网页式(7)
        dU[:, ii + 1] = (((np.cross(Z[:, ii + 1].reshape([3]), dX[:, ii + 1].reshape([3]))
                           + np.cross(dZ[:, ii + 1].reshape([3]), x_end.reshape([3]) - x[:, ii].reshape([3]))))
                         .reshape([3]))

    J = np.zeros([6, Ndof])
    dJ = np.zeros([6, Ndof])
    for ii in range(Ndof):
        if DH_mode[ii] == "hinge":
            J[0:3, ii] = U[:, ii + 1]  # 对应于网页式(4)(5)
            J[3:6, ii] = Z[:, ii + 1]
            dJ[0:3, ii] = dU[:, ii + 1]  # 对应于网页式(6)
            dJ[3:6, ii] = dZ[:, ii + 1]
        elif DH_mode[ii] == "slide":
            J[0:3, ii] = Z[:, ii + 1]
            dJ[0:3, ii] = dZ[:, ii + 1]

    return J, dJ


def fkine_ee_m(DH, q, DH_end, DH_mode=None, R_base=np.eye(3), base=np.zeros([3, 1])):
    T = fkine_m(DH, q, DH_mode=DH_mode, R_base=R_base, base=base)
    # 由于改进DH，所以要加上末端到末连杆系的一段
    DH_end = DH_end.reshape([-1])
    theta, d, a, alpha = DH_end[0], DH_end[1], DH_end[2], DH_end[3]
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    T_end = T[:, :, -1] @ np.array([[ct, -st, 0., a],
                                [ca * st, ca * ct, - sa, - d * sa],
                                [sa * st, sa * ct, ca, d * ca],
                                [0., 0., 0., 1.]])
    x_end = T_end[0:3, 3].reshape([3, 1])
    R_end = T_end[0:3, 0:3]
    quat_end = mat2quat(R_end).reshape([4, 1])

    return x_end, quat_end


# def ikine_m(DH, DH_end, qc, xd, quatd, DH_mode=None, R_base=np.eye(3), base=np.zeros([3, 1]), omega_base=np.zeros([3, 1]),
#           threshold=1e-3, max_iter=3e3):
#     # quat cos位在第一位
#     DH = np.copy(DH)
#     qc = qc.reshape([-1, 1])
#     xd = xd.reshape([-1, 1])
#     quatd = quatd.reshape([4, 1])
#     x, quat = fkine_ee_m(DH, qc, DH_end, DH_mode=DH_mode, R_base=R_base, base=base)
#
#     euld, _ = quat2eul(quatd)
#     euld = euld.reshape([3, 1])
#
#     epsino = m.pi / 2 - 5e-1
#     # 判断欧拉角模式，避开奇异点
#     if abs(euld[1, 0]) < epsino:
#         mode = "ZYX"
#     else:
#         mode = "XZY"
#
#     eul, _ = quat2eul(quat, mode)
#     eul = eul.reshape([3, 1])
#     euld, _ = quat2eul(quatd, mode)
#     euld = euld.reshape([3, 1])
#
#     x = np.vstack((x, eul))
#     xd = np.vstack((xd, euld))
#     J, _ = jac_m(DH, qc, DH_end, DH_mode=DH_mode, R_base=R_base, base=base, omega_base=omega_base)
#
#     err_p = np.linalg.norm(x[0:3] - xd[0:3])
#     err_eul = np.linalg.norm(x[3:6] - xd[3:6])
#     err = err_p + err_eul
#
#     ii = 0
#     while 1:
#         ii += 1
#         delta_x = xd - x
#         pinvJ = np.linalg.pinv(J)
#         delta_q = pinvJ @ delta_x
#         if err < 1.0:
#             qc = qc + (0.05 * err + 0.0001) * delta_q
#         else:
#             qc = qc + 0.1 * delta_q
#         qc = clip_q(qc)
#         x, quat = fkine_ee_m(DH, qc, DH_end, DH_mode=DH_mode, R_base=R_base, base=base)
#         eul, _ = quat2eul(quat, mode)
#         eul = eul.reshape([3, 1])
#         x = np.vstack((x, eul))
#         J, _ = jac_m(DH, qc, DH_end, DH_mode=DH_mode, R_base=R_base, base=base, omega_base=omega_base)
#
#         err_p = np.linalg.norm(x[0:3] - xd[0:3])
#         err_eul = np.linalg.norm(x[3:6] - xd[3:6])
#         err = err_p + err_eul
#
#         if err < threshold:
#             break
#
#         if ii > max_iter:
#             print('**ikine** breaks after %d iterations with error %.4f.\n'.format(ii, err))
#             break
#
#     return qc, err

def ikine_m(DH, DH_end, qc, xd, quatd, DH_mode=None, R_base=np.eye(3), base=np.zeros([3, 1]),
            omega_base=np.zeros([3, 1]),
            threshold=1e-3, max_iter=1000):
    # 初始化逻辑完全保留
    DH = np.copy(DH)
    qc = qc.reshape([-1, 1])
    xd = xd.reshape([-1, 1])
    quatd = quatd.reshape([4, 1])
    x, quat = fkine_ee_m(DH, qc, DH_end, DH_mode=DH_mode, R_base=R_base, base=base)

    euld, _ = quat2eul(quatd)
    euld = euld.reshape([3, 1])

    epsino = m.pi / 2 - 5e-1
    if abs(euld[1, 0]) < epsino:
        mode = "ZYX"
    else:
        mode = "XZY"

    eul, _ = quat2eul(quat, mode)
    eul = eul.reshape([3, 1])
    euld, _ = quat2eul(quatd, mode)
    euld = euld.reshape([3, 1])

    x = np.vstack((x, eul))
    xd = np.vstack((xd, euld))
    J, _ = jac_m(DH, qc, DH_end, DH_mode=DH_mode, R_base=R_base, base=base, omega_base=omega_base)

    err_p = np.linalg.norm(x[0:3] - xd[0:3])
    err_eul = np.linalg.norm(x[3:6] - xd[3:6])
    err = err_p + err_eul

    ii = 0
    # ===== 提速核心修改 =====
    # 1. 动态阻尼：初期极小（提速），后期小幅增加（稳精度）
    damping_init = 1e-5  # 初始阻尼（比之前更小）
    damping_final = 1e-4  # 后期阻尼（仍小，仅防震荡）
    # 2. 分阶段步长：误差>阈值10倍时用大 step，<10倍时衰减
    fast_err_threshold1 = 0.5  # 快速收敛阶段阈值
    step_fast1 = 0.1  # 快速阶段步长（比原0.1大）
    fast_err_threshold2 = 0.2  # 快速收敛阶段阈值
    step_fast2 = 0.005  # 快速阶段步长（比原0.1大）
    step_decay_fast = 0.9995  # 快速阶段衰减慢（保持大步长）
    step_decay_slow = 0.999  # 精细阶段衰减快（稳精度）
    min_step = 1e-8  # 最小步长

    err_list = []
    while 1:
        ii += 1
        delta_x = xd - x

        # 动态调整阻尼：迭代次数越多，阻尼越接近final（线性过渡）
        damping = damping_init + (damping_final - damping_init) * (ii / max_iter)
        # 计算带动态阻尼的伪逆
        J_T = J.T
        pinvJ = J_T @ np.linalg.inv(J @ J_T + damping * np.eye(6))

        delta_q = pinvJ @ delta_x

        # 分阶段步长策略：
        if err > fast_err_threshold1:
            # 阶段1：误差大，用大步长+慢衰减，快速降误差
            step = step_fast1 * (step_decay_fast ** ii)
        elif err > fast_err_threshold2:
            step = step_fast2 * (step_decay_fast ** ii)
        else:
            step = (0.01 * err + 1e-8) * (step_decay_slow ** ii)
        # 限制最小步长（放宽，提速）
        step = max(step, min_step)
        qc = qc + step * delta_q

        qc = clip_q(qc)
        # 以下逻辑完全保留
        x, quat = fkine_ee_m(DH, qc, DH_end, DH_mode=DH_mode, R_base=R_base, base=base)
        eul, _ = quat2eul(quat, mode)
        eul = eul.reshape([3, 1])
        x = np.vstack((x, eul))
        J, _ = jac_m(DH, qc, DH_end, DH_mode=DH_mode, R_base=R_base, base=base, omega_base=omega_base)

        err_p = np.linalg.norm(x[0:3] - xd[0:3])
        err_eul = np.linalg.norm(x[3:6] - xd[3:6])
        err = err_p + 0.2 * err_eul

        err_list.append(err)

        if err < threshold:
            break

        if ii > max_iter:
            print(f'**ikine** breaks after {ii} iterations with error {err:.4f}.\n')
            break

    plt.figure()
    plt.plot(err_list)
    plt.show()

    return qc, err


def ikine_3d_m(DH, qc, DH_end, xd, DH_mode=None, R_base=np.eye(3), base=np.zeros([3, 1]), omega_base=np.zeros([3, 1]),
             threshold=1e-6, max_iter=5e2):
    DH = np.copy(DH)
    qc = qc.reshape([-1, 1])
    xd = xd.reshape([-1, 1])
    x, quat = fkine_ee_m(DH, qc, DH_end, DH_mode=DH_mode, R_base=R_base, base=base)
    x = x.reshape([-1, 1])
    J, _ = jac_m(DH, qc, DH_end, DH_mode=DH_mode, R_base=R_base, base=base, omega_base=omega_base)
    J = J[0:3, :]  # 只要位置

    ii = 0
    while 1:
        ii += 1
        delta_x = xd - x
        pinvJ = np.linalg.pinv(J)
        delta_q = pinvJ @ delta_x
        qc = qc + 0.1 * delta_q
        qc = clip_q(qc)
        x, quat = fkine_ee_m(DH, qc, DH_end, DH_mode=DH_mode, R_base=R_base, base=base)
        x = x.reshape([-1, 1])
        J, _ = jac_m(DH, qc, DH_end, DH_mode=DH_mode, R_base=R_base, base=base, omega_base=omega_base)
        J = J[0:3, :]  # 只要位置
        err = np.linalg.norm(x[0:3, 0] - xd[0:3, 0])

        if err < threshold:
            break

        if ii > max_iter:
            print('**ikine** breaks after %d iterations with error %.4f.\n'.format(ii, err))
            break

    return qc, err


def fkine(DH, q, DH_mode=None, R_base=np.eye(3), base=np.zeros([3, 1])):
    # 标准DH参数
    q = q.reshape([-1, 1])
    DH = np.copy(DH)
    Ndof = q.size
    if DH_mode is None:
        DH_mode = ["hinge"] * Ndof

    temp = np.eye(4)
    temp[0:3, 0:3] = R_base  # 基座旋转矩阵
    temp[0:3, 3] = base.reshape([3])  # 基座向量
    T = np.zeros([4, 4, Ndof])  # 初始化T

    # DH第一列为theta的offset, 在这上面加关节角
    for ii in range(Ndof):
        if DH_mode[ii] == "hinge":
            DH[ii, 0] += q[ii, 0]
        if DH_mode[ii] == "slide":
            DH[ii, 1] += q[ii, 0]

    for ii in range(Ndof):
        ct = m.cos(DH[ii, 0])
        st = m.sin(DH[ii, 0])
        ca = m.cos(DH[ii, 3])
        sa = m.sin(DH[ii, 3])

        temp = temp @ np.array([[ct, -st * ca, st * sa, DH[ii, 2] * ct],
                                [st, ct * ca, -ct * sa, DH[ii, 2] * st],
                                [0, sa, ca, DH[ii, 1]],
                                [0, 0, 0, 1]])

        T[:, :, ii] = temp

    return T


def jac(DH, q, dq=None, DH_mode=None, R_base=np.eye(3), base=np.zeros([3, 1]), omega_base=np.zeros([3, 1])):
    # https://zhuanlan.zhihu.com/p/205342861?utm_id=0
    q = q.reshape([-1])
    DH = np.copy(DH)
    Ndof = DH.shape[0]
    if DH_mode is None:
        DH_mode = ["hinge"] * Ndof

    if dq is None:
        dq = np.zeros([Ndof, 1])
    dq = dq.reshape([Ndof, 1])

    T = fkine(DH, q, DH_mode=DH_mode, R_base=R_base, base=base)  # 所有的T
    R = T[0:3, 0:3, :]  # 各系旋转矩阵
    x = T[0:3, 3, :].reshape([3, -1])  # 各系的原点位置
    x_end = x[:, -1].reshape([-1, 1])  # 末端位置

    omega_save = np.zeros([3, Ndof])

    R_old = R_base
    omega_old = omega_base
    for ii in range(Ndof):
        if DH_mode[ii] == "hinge":
            omega = omega_old + R_old @ np.array([0, 0, dq[ii, 0]]).reshape([-1, 1])  # 各坐标系的角速度
        else:
            omega = omega_old

        # 保存参数
        omega_save[:, ii] = omega.reshape([3])  # 对应于 q(ii+6d)

        # 更新参数
        R_old = R[:, :, ii]
        omega_old = omega

    # 对应于网页式(8)
    dx = np.zeros([3, 1])  # end 相对于 6(=end) 的速度
    dX = np.zeros([3, Ndof])  # 0为end相对base，Ndof-1为end相对于5, 即与关节角对应
    for ii in range(Ndof - 1, 0, -1):  # Ndof-6d : -6d : 6d
        if DH_mode[ii] == "hinge":
            dx += np.cross(omega_save[:, ii].reshape([1, 3]), (x[:, ii] - x[:, ii - 1]).reshape([1, 3])).reshape([3, 1])
        else:
            dx += R[:, :, ii - 1] @ np.array([0, 0, dq[ii, 0]]).reshape([3, 1])
        dX[:, ii] = dx.reshape([3])

    dX[:, 0] = dx.reshape([3]) + np.cross(omega_save[:, 0].reshape([3]),
                                          (x[:, 0].reshape([3]) - base.reshape([3]))).reshape([3])

    # qi 在 i-1系下描述
    Z = np.zeros([3, Ndof])  # Z轴矢量
    U = np.zeros([3, Ndof])  # Z矢量叉乘（x_end - xi）
    dZ = np.zeros([3, Ndof])
    dU = np.zeros([3, Ndof])

    # 对应于网页式(3)
    Z[:, 0] = (R_base @ np.array([0, 0, 1]).reshape([3, 1])).reshape([3])
    # 对应于网页式(3)
    U[:, 0] = np.cross(Z[:, 0].reshape([3]), (x_end.reshape([3]) - base.reshape([3]))).reshape([3])
    # 对应于网页式(10)
    dZ[:, 0] = np.cross(omega_base.reshape([3]), Z[:, 0].reshape([3])).reshape([3])
    # 对应于网页式(7)
    dU[:, 0] = (((np.cross(Z[:, 0].reshape([3]), dX[:, 0].reshape([3]))
                  + np.cross(dZ[:, 0].reshape([3]), x_end.reshape([3]) - base.reshape([3])))).reshape([3]))

    for ii in range(Ndof - 1):
        # 对应于网页式(3)
        Z[:, ii + 1] = (R[:, :, ii] @ np.array([0, 0, 1]).reshape([3, 1])).reshape([3])
        # 对应于网页式(3)
        U[:, ii + 1] = np.cross(Z[:, ii + 1].reshape([3]), (x_end.reshape([3]) - x[:, ii].reshape([3]))).reshape([3])
        # 对应于网页式(10)
        dZ[:, ii + 1] = np.cross(omega_save[:, ii].reshape([3]), Z[:, ii + 1].reshape([3])).reshape([3])
        # 对应于网页式(7)
        dU[:, ii + 1] = (((np.cross(Z[:, ii + 1].reshape([3]), dX[:, ii + 1].reshape([3]))
                           + np.cross(dZ[:, ii + 1].reshape([3]), x_end.reshape([3]) - x[:, ii].reshape([3]))))
                         .reshape([3]))

    J = np.zeros([6, Ndof])
    dJ = np.zeros([6, Ndof])
    for ii in range(Ndof):
        if DH_mode[ii] == "hinge":
            J[0:3, ii] = U[:, ii]  # 对应于网页式(4)(5)
            J[3:6, ii] = Z[:, ii]
            dJ[0:3, ii] = dU[:, ii]  # 对应于网页式(6)
            dJ[3:6, ii] = dZ[:, ii]
        elif DH_mode[ii] == "slide":
            J[0:3, ii] = Z[:, ii]
            dJ[0:3, ii] = dZ[:, ii]

    return J, dJ


def fkine_ee(DH, q, DH_mode=None, R_base=np.eye(3), base=np.zeros([3, 1])):
    T = fkine(DH, q, DH_mode=DH_mode, R_base=R_base, base=base)
    x_end = T[0:3, 3, -1].reshape([3, 1])
    R_end = T[0:3, 0:3, -1]
    quat_end = mat2quat(R_end).reshape([4, 1])

    return x_end, quat_end


def ikine(DH, qc, xd, quatd, DH_mode=None, R_base=np.eye(3), base=np.zeros([3, 1]), omega_base=np.zeros([3, 1]),
          threshold=1e-5, max_iter=5e2):
    # quat cos位在第一位
    DH = np.copy(DH)
    qc = qc.reshape([-1, 1])
    xd = xd.reshape([-1, 1])
    quatd = quatd.reshape([4, 1])
    x, quat = fkine_ee(DH, qc, DH_mode=DH_mode, R_base=R_base, base=base)

    euld = quat2eul(quatd).reshape([3, 1])

    epsino = m.pi / 2 - 5e-1
    # 判断欧拉角模式，避开奇异点
    if abs(euld[1, 0]) < epsino:
        mode = "ZYX"
    else:
        mode = "XZY"

    eul = quat2eul(quat, mode).reshape([3, 1])
    euld = quat2eul(quatd, mode).reshape([3, 1])

    x = np.vstack((x, eul))
    xd = np.vstack((xd, euld))
    J, _ = jac(DH, qc, DH_mode=DH_mode, R_base=R_base, base=base, omega_base=omega_base)

    err_p = np.linalg.norm(x[0:3] - xd[0:3])
    err_eul = np.linalg.norm(x[3:6] - xd[3:6])
    err = err_p + err_eul

    ii = 0
    while 1:
        ii += 1
        delta_x = xd - x
        pinvJ = np.linalg.pinv(J)
        delta_q = pinvJ @ delta_x
        # if err < 6d:
        #     qc = qc + 0.06 * err * delta_q
        # else:
        qc = qc + 0.1 * delta_q
        qc = clip_q(qc)
        x, quat = fkine_ee(DH, qc, DH_mode=DH_mode, R_base=R_base, base=base)
        eul = quat2eul(quat, mode).reshape([3, 1])
        x = np.vstack((x, eul))
        J, _ = jac(DH, qc, DH_mode=DH_mode, R_base=R_base, base=base, omega_base=omega_base)

        err_p = np.linalg.norm(x[0:3] - xd[0:3])
        err_eul = np.linalg.norm(x[3:6] - xd[3:6])
        err = err_p + err_eul

        if err < threshold:
            break

        if ii > max_iter:
            print('**ikine** breaks after %d iterations with error %.4f.\n'.format(ii, err))
            break

    return qc, err