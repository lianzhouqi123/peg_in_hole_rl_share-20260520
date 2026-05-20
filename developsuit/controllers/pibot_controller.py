import numpy as np
import math
from developsuit.utils.mujoco_utils import (
    get_fullM,
)


class PiBot_Controller:
    def __init__(self, physics, joints, actuators, timestep, min_effort, max_effort, kp, damping_ratio, ki, mm=None):
        self._physics = physics
        self._joints = joints
        self._actuators = actuators
        self._min_effort = min_effort
        self._max_effort = max_effort
        self._kp = np.asarray(kp, dtype=np.float32)
        self._damping_ratio = np.asarray(damping_ratio, dtype=np.float32)
        self._kd = 2 * np.sqrt(self._kp) * self._damping_ratio
        self._ki = np.asarray(ki, dtype=np.float32)
        self.default_kp = self._kp.copy()
        self.default_damping_ratio = self._damping_ratio.copy()
        self.default_ki = self._ki.copy()
        self._jnt_dof_ids = self._physics.bind(self._joints).dofadr
        self._timestep = timestep
        if mm is None:
            self.mm = 1
        else:
            self.mm = mm
        self.integral = np.zeros([self._jnt_dof_ids.shape[0]])
        pass

    def _expand_gain_vector(self, value):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            return np.full(self._jnt_dof_ids.shape[0], float(arr), dtype=np.float32)
        arr = arr.reshape(-1).astype(np.float32, copy=False)
        if arr.shape[0] != self._jnt_dof_ids.shape[0]:
            raise ValueError(
                f"gain vector must have length {self._jnt_dof_ids.shape[0]}, got {arr.shape[0]}"
            )
        return arr.copy()

    def set_pd_gains(self, kp=None, damping_ratio=None):
        if kp is not None:
            self._kp = self._expand_gain_vector(kp)
        if damping_ratio is not None:
            self._damping_ratio = self._expand_gain_vector(damping_ratio)
        self._kd = 2 * np.sqrt(np.clip(self._kp, a_min=1e-8, a_max=None)) * self._damping_ratio

    def run(self, target, target_ddq_grp=None, grp_joint_id=None):
        M_full = get_fullM(
            self._physics.model.ptr,
            self._physics.data.ptr,
        )
        M = M_full[self._jnt_dof_ids[:], :][:, self._jnt_dof_ids[:]]
        dq = self._physics.bind(self._joints).qvel[:]
        q_error = target.reshape([-1]) - self._physics.bind(self._joints).qpos[:]
        self.integral += q_error * self._timestep
        target_ddq = np.array((self._kp * q_error - self._kd * dq + self._ki * self.integral) / self.mm)
        if target_ddq_grp is not None:
            target_ddq[grp_joint_id] = target_ddq_grp
            self.integral[grp_joint_id] = 0
        torque = np.dot(M, target_ddq)
        torque += self._physics.bind(self._joints).qfrc_bias[:]
        # torque = np.clip(torque, self._min_effort, self._max_effort)
        self._physics.bind(self._actuators).ctrl[:] = torque[:]
        pass
