"""
Utility functions of matrix and vector transformations (PyTorch version).
Pure PyTorch implementation with ZERO CPU-GPU sync overhead.

NOTE: convention for quaternions is (w, x, y, z)
"""

import torch

# 定义常量（适配PyTorch）
PI = torch.pi
EPS = torch.finfo(torch.float32).eps * 4.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def axisangle2quat(rotvec):
    """
    【新增】Batched 旋转向量转四元数 (wxyz)
    输入: [..., 3]
    输出: [..., 4]
    """
    rotvec = torch.as_tensor(rotvec, dtype=torch.float32, device=device)
    angle = torch.norm(rotvec, dim=-1, keepdim=True)

    # 防止除以 0 导致 NaN
    axis = rotvec / (angle + 1e-6)
    half_angle = angle * 0.5

    q = torch.cat([torch.cos(half_angle), axis * torch.sin(half_angle)], dim=-1)

    # 【极小角度保护】使用 torch.where 在 GPU 上处理，绝不使用 if angle < 1e-6 导致 CPU 同步
    identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device).expand_as(q)
    return torch.where(angle < 1e-6, identity_q, q)


def quat2axisangle(quat):
    """
    将四元数 [..., 4] (wxyz) 转换为角轴向量 [..., 3]
    全 GPU 张量运算，防除零报错保护
    """
    # 强制归一化以防万一
    norm = torch.norm(quat, dim=-1, keepdim=True)
    quat = quat / torch.clamp(norm, min=1e-8)

    w = quat[..., 0:1]
    vec = quat[..., 1:4]

    # 限制 w 在 [-1, 1] 之间防止 acos 产生 NaN
    angle = 2.0 * torch.acos(torch.clamp(w, -1.0, 1.0))
    sin_half = torch.sqrt(torch.clamp(1.0 - w * w, min=1e-8))

    # 当角度极小时，防止除以 0，默认旋转轴为 X 轴
    small_angle_mask = (sin_half < 1e-6).expand_as(vec)
    default_axis = torch.zeros_like(vec)
    default_axis[..., 0] = 1.0

    axis = torch.where(small_angle_mask, default_axis, vec / sin_half)

    return axis * angle


def mat2quat(rmat):
    """
    【核心优化】使用代数闭式解替换 eigh 特征值分解。
    通过 torch.where 掩码处理四种象限的分支，全并行、无 CPU 阻塞。
    输入: [..., 3, 3]
    输出: [..., 4] (顺序为 w, x, y, z)
    """
    rmat = torch.as_tensor(rmat, dtype=torch.float32, device=device)

    m00, m01, m02 = rmat[..., 0, 0], rmat[..., 0, 1], rmat[..., 0, 2]
    m10, m11, m12 = rmat[..., 1, 0], rmat[..., 1, 1], rmat[..., 1, 2]
    m20, m21, m22 = rmat[..., 2, 0], rmat[..., 2, 1], rmat[..., 2, 2]

    # 计算矩阵的迹
    tr = m00 + m11 + m22

    # 分支 1: Trace > 0
    # 注意：clamp 0.0 是防止浮点误差导致负数开方产生 NaN
    S1 = torch.sqrt(torch.clamp(tr + 1.0, min=0.0)) * 2.0
    qw1 = 0.25 * S1
    qx1 = (m21 - m12) / (S1 + 1e-8)
    qy1 = (m02 - m20) / (S1 + 1e-8)
    qz1 = (m10 - m01) / (S1 + 1e-8)
    q1 = torch.stack([qw1, qx1, qy1, qz1], dim=-1)

    # 分支 2: m00 最大
    S2 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)) * 2.0
    qw2 = (m21 - m12) / (S2 + 1e-8)
    qx2 = 0.25 * S2
    qy2 = (m01 + m10) / (S2 + 1e-8)
    qz2 = (m02 + m20) / (S2 + 1e-8)
    q2 = torch.stack([qw2, qx2, qy2, qz2], dim=-1)

    # 分支 3: m11 最大
    S3 = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=0.0)) * 2.0
    qw3 = (m02 - m20) / (S3 + 1e-8)
    qx3 = (m01 + m10) / (S3 + 1e-8)
    qy3 = 0.25 * S3
    qz3 = (m12 + m21) / (S3 + 1e-8)
    q3 = torch.stack([qw3, qx3, qy3, qz3], dim=-1)

    # 分支 4: m22 最大
    S4 = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=0.0)) * 2.0
    qw4 = (m10 - m01) / (S4 + 1e-8)
    qx4 = (m02 + m20) / (S4 + 1e-8)
    qy4 = (m12 + m21) / (S4 + 1e-8)
    qz4 = 0.25 * S4
    q4 = torch.stack([qw4, qx4, qy4, qz4], dim=-1)

    # 生成条件掩码并合并 (无 CPU 分支跳跃)
    cond1 = tr > 0.0
    cond2 = (m00 > m11) & (m00 > m22)
    cond3 = m11 > m22

    q = torch.where(cond1.unsqueeze(-1), q1,
            torch.where(cond2.unsqueeze(-1), q2,
                torch.where(cond3.unsqueeze(-1), q3, q4)))

    # 规范化四元数，确保 w >= 0 (双倍覆盖唯一化)
    q = torch.where(q[..., 0:1] < 0, -q, q)
    return q


def quat2mat(quaternions):
    """
    【核心优化】提前分配内存，直接索引写入，消灭高频 stack 导致的内存碎片和调度延迟。
    输入: [..., 4] (顺序为 w, x, y, z)
    输出: [..., 3, 3]
    """
    q = torch.as_tensor(quaternions, dtype=torch.float32, device=device)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # 计算缩放系数，避免除以零
    n = torch.sum(q ** 2, dim=-1)
    s = torch.where(n > 1e-8, 2.0 / n, torch.zeros_like(n))

    # 计算中间项
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    xx, xy, xz = x * x * s, x * y * s, x * z * s
    yy, yz, zz = y * y * s, y * z * s, z * z * s

    # 预分配全零矩阵，直接在指定设备分配
    mat = torch.zeros((*q.shape[:-1], 3, 3), dtype=torch.float32, device=device)

    # 原位写入，消除 Python 层面拼接张量的 Overhead
    mat[..., 0, 0] = 1.0 - yy - zz
    mat[..., 0, 1] = xy - wz
    mat[..., 0, 2] = xz + wy

    mat[..., 1, 0] = xy + wz
    mat[..., 1, 1] = 1.0 - xx - zz
    mat[..., 1, 2] = yz - wx

    mat[..., 2, 0] = xz - wy
    mat[..., 2, 1] = yz + wx
    mat[..., 2, 2] = 1.0 - xx - yy

    return mat


def quat2eul(quat, mode="ZYX"):
    q = torch.as_tensor(quat, dtype=torch.float32, device=device)
    q3, q0, q1, q2 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    if mode == "ZYX":
        sing_val = q0 * q2 - q1 * q3
        threshold = 0.5 - 1e-8

        rx = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))
        ry = torch.asin(torch.clamp(2 * sing_val, -1.0, 1.0))
        rz = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))

        pos_mask = sing_val > threshold
        rx = torch.where(pos_mask, torch.zeros_like(rx), rx)
        ry = torch.where(pos_mask, torch.full_like(ry, torch.pi / 2), ry)
        rz = torch.where(pos_mask, -2 * torch.atan2(q1, q0), rz)

        neg_mask = sing_val < -threshold
        rx = torch.where(neg_mask, torch.zeros_like(rx), rx)
        ry = torch.where(neg_mask, torch.full_like(ry, -torch.pi / 2), ry)
        rz = torch.where(neg_mask, 2 * torch.atan2(q1, q0), rz)

    elif mode == "XZY":
        rx = torch.atan2(2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q3 ** 2))
        ry = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
        rz = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1.0, 1.0))
    else:
        raise ValueError("Unsupported mode")

    return torch.stack([rx, ry, rz], dim=-1), mode


def mat2eul(mat, mode="ZYX"):
    """
    将旋转矩阵转换为欧拉角。
    【已合并 XYZ 模式，完全兼容 Warp】
    """
    mat = torch.as_tensor(mat, dtype=torch.float32, device=device)

    if mode == "ZYX":
        sy = -mat[..., 0, 2]
        ry = torch.asin(torch.clamp(sy, -1.0, 1.0))

        cos_y = torch.cos(ry)
        singular_mask = torch.abs(cos_y) < 1e-8

        rz = torch.atan2(mat[..., 0, 1], mat[..., 0, 0])
        rx = torch.atan2(mat[..., 1, 2], mat[..., 2, 2])

        rz_sing = torch.acos(torch.clamp(mat[..., 2, 0], -1.0, 1.0))

        rz = torch.where(singular_mask, rz_sing, rz)
        rx = torch.where(singular_mask, torch.zeros_like(rx), rx)

        return torch.stack([rx, ry, rz], dim=-1), mode

    elif mode == "XYZ":
        # XYZ 等价于 Pitch, Roll, Yaw 对应的另一种万向节锁定处理
        sy = torch.sqrt(mat[..., 0, 0]**2 + mat[..., 1, 0]**2)
        singular = sy < 1e-6

        x = torch.where(singular,
                        torch.atan2(-mat[..., 1, 2], mat[..., 1, 1]),
                        torch.atan2(mat[..., 2, 1], mat[..., 2, 2]))
        y = torch.where(singular,
                        torch.atan2(-mat[..., 2, 0], sy),
                        torch.atan2(-mat[..., 2, 0], sy))
        z = torch.where(singular,
                        torch.zeros_like(sy),
                        torch.atan2(mat[..., 1, 0], mat[..., 0, 0]))
        return torch.stack([x, y, z], dim=-1), mode
    else:
        raise ValueError("Unsupported mode")


def _rot_base(angle, axis):
    angle = torch.as_tensor(angle, dtype=torch.float32, device=device)
    c, s = torch.cos(angle), torch.sin(angle)

    mat = torch.zeros((*angle.shape, 3, 3), dtype=torch.float32, device=device)

    if axis == 'x':
        mat[..., 0, 0] = 1.0
        mat[..., 1, 1] = c
        mat[..., 1, 2] = s
        mat[..., 2, 1] = -s
        mat[..., 2, 2] = c
    elif axis == 'y':
        mat[..., 0, 0] = c
        mat[..., 0, 2] = -s
        mat[..., 1, 1] = 1.0
        mat[..., 2, 0] = s
        mat[..., 2, 2] = c
    elif axis == 'z':
        mat[..., 0, 0] = c
        mat[..., 0, 1] = s
        mat[..., 1, 0] = -s
        mat[..., 1, 1] = c
        mat[..., 2, 2] = 1.0

    return mat


def rotx(angle): return _rot_base(angle, 'x')

def roty(angle): return _rot_base(angle, 'y')

def rotz(angle): return _rot_base(angle, 'z')

def clip_q(q):
    q = torch.as_tensor(q, dtype=torch.float32, device=device)
    return (q + torch.pi) % (2 * torch.pi) - torch.pi