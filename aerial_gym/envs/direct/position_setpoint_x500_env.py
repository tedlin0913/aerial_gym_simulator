"""
PositionSetpointX500Env — Isaac Lab DirectRLEnv for x500 sim-to-real position setpoint.

Ports position_setpoint_task_sim2real_px4 (x500 robot) to Isaac Lab.

Observation (15-dim) — same as end-to-end task:
    [0:3]   position error (world frame, noise std=0.001)
    [3:9]   rotation 6D (first two cols of R; noise via Euler round-trip std=π/1032)
    [9:12]  linear velocity (world frame, noise std=0.002)
    [12:15] body angular velocity (noise std=0.001)

Differences from PositionSetpointEndToEndEnv:
  - x500 robot (1.656 kg, 20 N max thrust per motor, prop link names)
  - crash_dist = 6.5 m (more lenient — x500 is used for outdoor flight)

Motor body order matches X500Cfg allocation_matrix columns:
    [front_left_prop, front_right_prop, back_left_prop, back_right_prop]
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from aerial_gym.envs.direct.aerial_gym_base_env import AerialGymBaseEnv, AerialGymBaseEnvCfg
from aerial_gym.envs.assets.x500_asset import X500_CFG
from aerial_gym.control.control_allocation import ControlAllocator
from aerial_gym.config.robot_config.x500_config import X500Cfg
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("position_setpoint_x500_env")


@configclass
class PositionSetpointX500EnvCfg(AerialGymBaseEnvCfg):
    """Configuration for x500 sim-to-real position setpoint task."""

    robot = X500_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    episode_length_s: float = 5.0
    decimation: int = 2

    action_space: int = 4
    # obs: pos_error(3) + rot6d(6) + world_linvel(3) + body_angvel(3) = 15
    observation_space: int = 15
    state_space: int = 0

    target_position_x: float = 0.0
    target_position_y: float = 0.0
    target_position_z: float = 1.5

    # Sensor noise (same as original PX4 task)
    pos_error_noise_std: float = 0.001
    orientation_noise_std: float = 0.003054   # π/1032
    linvel_noise_std: float = 0.002
    angvel_noise_std: float = 0.001

    # Lenient crash distance — x500 has more space outdoors
    crash_dist: float = 6.5


class PositionSetpointX500Env(AerialGymBaseEnv):
    """
    x500 sim-to-real position setpoint environment (Isaac Lab).

    Uses the x500 quadrotor (1.656 kg, 20 N max thrust per motor) with
    rotation 6D observation encoding.

    Motor body names override:
        front_left_prop  → allocation matrix column 0
        front_right_prop → allocation matrix column 1
        back_left_prop   → allocation matrix column 2
        back_right_prop  → allocation matrix column 3
    (Matches X500Cfg.control_allocator_config.application_mask = [4, 1, 3, 2])
    """

    cfg: PositionSetpointX500EnvCfg

    # Override: x500 uses prop link names, not motor_0..3
    _motor_body_names: list[str] = [
        "front_left_prop",
        "front_right_prop",
        "back_left_prop",
        "back_right_prop",
    ]

    def __init__(self, cfg: PositionSetpointX500EnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._setup_drag_coefficients(X500Cfg)

        self._target_position = self._terrain.env_origins.clone()
        self._target_position[:, 0] += cfg.target_position_x
        self._target_position[:, 1] += cfg.target_position_y
        self._target_position[:, 2] += cfg.target_position_z

        self._obs_buf = torch.zeros(self.num_envs, cfg.observation_space, device=self.device)
        self._prev_pos_error = torch.zeros(self.num_envs, 3, device=self.device)

        self._reward_params: dict = {}

        logger.info(
            f"PositionSetpointX500Env: {self.num_envs} envs, "
            f"obs={cfg.observation_space}, target={self._target_position[0].tolist()}"
        )

    # =========================================================================
    # AerialGymBaseEnv interface
    # =========================================================================

    def _setup_control_allocator(self):
        dt = self.cfg.sim.dt * self.cfg.decimation
        return ControlAllocator(
            num_envs=self.num_envs,
            dt=dt,
            config=X500Cfg.control_allocator_config,
            device=self.device,
        )

    def _build_observation_tensor(self) -> None:
        from aerial_gym.utils.math import get_euler_xyz_tensor, quat_from_euler_xyz_tensor, ssa
        from aerial_gym.envs.direct.rotation_utils import (
            euler_angles_to_matrix,
            matrix_to_rotation_6d,
        )

        cfg = self.cfg
        pos_error = self._target_position - self.obs_dict["robot_position"]

        euler = ssa(get_euler_xyz_tensor(self.obs_dict["robot_orientation"]))
        euler_noisy = euler + torch.randn_like(euler) * cfg.orientation_noise_std
        R = euler_angles_to_matrix(euler_noisy, "ZYX")
        rot6d = matrix_to_rotation_6d(R)

        self._obs_buf[:, 0:3]  = pos_error + torch.randn_like(pos_error) * cfg.pos_error_noise_std
        self._obs_buf[:, 3:9]  = rot6d
        self._obs_buf[:, 9:12] = (self.obs_dict["robot_linvel"]
                                   + torch.randn_like(self.obs_dict["robot_linvel"])
                                   * cfg.linvel_noise_std)
        self._obs_buf[:, 12:15] = (self.obs_dict["robot_body_angvel"]
                                    + torch.randn_like(self.obs_dict["robot_body_angvel"])
                                    * cfg.angvel_noise_std)

    def _get_rewards(self) -> torch.Tensor:
        pos_error_world = self._target_position - self.obs_dict["robot_position"]

        rewards, crashes = compute_reward(
            pos_error_world,
            self.obs_dict["robot_orientation"],
            self.obs_dict["robot_linvel"],
            self.obs_dict["robot_body_angvel"],
            self.obs_dict["crashes"],
            self.obs_dict["robot_actions"],
            self.obs_dict["robot_prev_actions"],
            self._prev_pos_error,
            self.cfg.crash_dist,
        )

        self._prev_pos_error[:] = pos_error_world
        self.obs_dict["crashes"][:] = crashes
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self.obs_dict["crashes"].bool()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids_for_target = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_for_target = env_ids

        self._target_position[env_ids_for_target] = (
            self._terrain.env_origins[env_ids_for_target].clone()
        )
        self._target_position[env_ids_for_target, 0] += self.cfg.target_position_x
        self._target_position[env_ids_for_target, 1] += self.cfg.target_position_y
        self._target_position[env_ids_for_target, 2] += self.cfg.target_position_z

        self._prev_pos_error[env_ids_for_target] = 0.0


# ---------------------------------------------------------------------------
# Reward function — same shape as position_setpoint_end_to_end_env.py
# (copied rather than shared to keep JIT compilation separate per task)
# ---------------------------------------------------------------------------

import torch
from torch import Tensor
from typing import Tuple


@torch.jit.script
def _exp_func(x: Tensor, gain: float, exp: float) -> Tensor:
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def _exp_penalty_func(x: Tensor, gain: float, exp: float) -> Tensor:
    return gain * (torch.exp(-exp * x * x) - 1)


@torch.jit.script
def _quat_axis(q: Tensor, axis: int = 0) -> Tensor:
    """Extract local axis vector from xyzw quaternion. axis: 0=x, 1=y, 2=z"""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    if axis == 0:
        return torch.stack([
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ], dim=-1)
    elif axis == 1:
        return torch.stack([
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        ], dim=-1)
    else:
        return torch.stack([
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ], dim=-1)


@torch.jit.script
def compute_reward(
    pos_error: Tensor,
    quats: Tensor,
    linvels_world: Tensor,
    angvels_body: Tensor,
    crashes: Tensor,
    current_action: Tensor,
    prev_action: Tensor,
    prev_pos_error: Tensor,
    crash_dist: float,
) -> Tuple[Tensor, Tensor]:

    target_dist = torch.norm(pos_error[:, :3], dim=1)
    prev_target_dist = torch.norm(prev_pos_error, dim=1)

    pos_error_weighted = pos_error.clone()
    pos_error_weighted[:, 2] = pos_error[:, 2] * 11.0
    pos_reward = (
        torch.sum(_exp_func(pos_error_weighted[:, :3], 10.0, 10.0), dim=1)
        + torch.sum(_exp_func(pos_error_weighted[:, :3], 2.0, 2.0), dim=1)
    )

    ups = _quat_axis(quats, 2)
    tiltage = 1.0 - ups[:, 2]
    upright_reward = _exp_func(tiltage, 2.5, 5.0)

    forw = _quat_axis(quats, 0)
    alignment = 1.0 - forw[:, 0]
    alignment_reward = _exp_func(alignment, 6.0, 5.0)

    angvel_reward = torch.sum(_exp_func(angvels_body, 0.3, 10.0), dim=1)
    vel_reward = torch.sum(_exp_func(linvels_world, 1.0, 5.0), dim=1)

    action_cost = torch.sum(_exp_penalty_func(current_action, 0.01, 10.0), dim=1)

    closer_by_dist = prev_target_dist - target_dist
    towards_goal_reward = torch.where(
        closer_by_dist >= 0.0,
        10.0 * closer_by_dist,
        15.0 * closer_by_dist,
    )

    action_difference = current_action - prev_action
    action_difference_penalty = torch.sum(_exp_penalty_func(action_difference, 1.3, 6.0), dim=1)

    reward = towards_goal_reward + (
        pos_reward * (alignment_reward + vel_reward + angvel_reward + action_difference_penalty)
        + (angvel_reward + vel_reward + upright_reward + pos_reward + action_cost)
    ) / 100.0

    crashes = torch.where(target_dist > crash_dist, torch.ones_like(crashes), crashes)

    return reward, crashes
