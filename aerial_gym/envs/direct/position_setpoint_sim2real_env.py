"""
PositionSetpointSim2RealEnv — Isaac Lab DirectRLEnv for sim-to-real position setpoint.

Extends PositionSetpointEnv with:
  - 17-dim observation: pos_error(3) + orientation(4) + body_linvel(3) + body_angvel(3) + prev_actions(4)
  - Gaussian sensor noise on all measurements (realistic IMU / state estimator noise)
  - Richer reward with velocity penalty, closer_reward (potential-based), yaw alignment
  - crash threshold: dist > 10 m (more lenient than base task's 8 m)

Preserves the original compute_reward() JIT function unchanged.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from aerial_gym.envs.direct.aerial_gym_base_env import AerialGymBaseEnv, AerialGymBaseEnvCfg
from aerial_gym.envs.assets.base_quad_asset import BASE_QUAD_CFG
from aerial_gym.control.control_allocation import ControlAllocator
from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("position_setpoint_sim2real_env")


@configclass
class PositionSetpointSim2RealEnvCfg(AerialGymBaseEnvCfg):
    """Configuration for sim-to-real position setpoint task."""

    robot = BASE_QUAD_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    episode_length_s: float = 5.0
    decimation: int = 2

    action_space: int = 4
    # obs: pos_error(3) + quat(4) + body_linvel(3) + body_angvel(3) + prev_actions(4) = 17
    observation_space: int = 17
    state_space: int = 0

    target_position_x: float = 0.0
    target_position_y: float = 0.0
    target_position_z: float = 1.5

    # Sensor noise std devs (matches original task noise levels)
    pos_error_noise_std: float = 0.03
    orientation_noise_std: float = 0.02   # applied to euler angles before re-encoding as quat
    linvel_noise_std: float = 0.02
    angvel_noise_std: float = 0.02


class PositionSetpointSim2RealEnv(AerialGymBaseEnv):
    """
    Sim-to-real position setpoint environment (Isaac Lab).

    Observation vector (17-dim):
        [0:3]  position error (world frame, with noise)
        [3:7]  robot orientation quaternion (xyzw, noise via Euler round-trip)
        [7:10] body-frame linear velocity (with noise)
        [10:13] body-frame angular velocity (with noise)
        [13:17] previous motor actions

    Reward: see compute_reward() below — potential-based, yaw penalty, speed penalty
    """

    cfg: PositionSetpointSim2RealEnvCfg

    def __init__(self, cfg: PositionSetpointSim2RealEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._setup_drag_coefficients(BaseQuadCfg)

        # Target position in world frame (env_origin + offset)
        self._target_position = self._terrain.env_origins.clone()
        self._target_position[:, 0] += cfg.target_position_x
        self._target_position[:, 1] += cfg.target_position_y
        self._target_position[:, 2] += cfg.target_position_z

        self._obs_buf = torch.zeros(self.num_envs, cfg.observation_space, device=self.device)

        # prev_dist used by closer_reward in compute_reward
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)

        # Reward params (no parameter_dict for this task — coefficients are hardcoded in compute_reward)
        # Pass empty dict; compute_reward uses hardcoded constants
        self._reward_params: dict = {}

        logger.info(
            f"PositionSetpointSim2RealEnv: {self.num_envs} envs, "
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
            config=BaseQuadCfg.control_allocator_config,
            device=self.device,
        )

    def _build_observation_tensor(self) -> None:
        """
        Build 17-dim observation with Gaussian sensor noise.

        [0:3]   pos_error world frame + noise
        [3:7]   orientation (noise via Euler round-trip: encode→add noise→decode)
        [7:10]  body_linvel + noise
        [10:13] body_angvel + noise
        [13:17] prev_actions (no noise)
        """
        from aerial_gym.utils.math import get_euler_xyz_tensor, quat_from_euler_xyz_tensor, ssa

        cfg = self.cfg

        pos_error = self._target_position - self.obs_dict["robot_position"]

        # Orientation noise: add Gaussian noise to Euler angles then re-encode
        # Matches original: euler_angles_noisy = euler_angles + N(0, 0.02)
        euler = ssa(get_euler_xyz_tensor(self.obs_dict["robot_orientation"]))  # xyzw → euler
        euler_noisy = euler + torch.randn_like(euler) * cfg.orientation_noise_std
        quat_noisy = quat_from_euler_xyz_tensor(euler_noisy)                  # euler → xyzw

        self._obs_buf[:, 0:3]  = pos_error + torch.randn_like(pos_error) * cfg.pos_error_noise_std
        self._obs_buf[:, 3:7]  = quat_noisy
        self._obs_buf[:, 7:10] = (self.obs_dict["robot_body_linvel"]
                                   + torch.randn_like(self.obs_dict["robot_body_linvel"])
                                   * cfg.linvel_noise_std)
        self._obs_buf[:, 10:13] = (self.obs_dict["robot_body_angvel"]
                                    + torch.randn_like(self.obs_dict["robot_body_angvel"])
                                    * cfg.angvel_noise_std)
        self._obs_buf[:, 13:17] = self.obs_dict["robot_prev_actions"]

    def _get_rewards(self) -> torch.Tensor:
        from aerial_gym.utils.math import quat_apply_inverse, get_euler_xyz_tensor, ssa

        pos_error_world = self._target_position - self.obs_dict["robot_position"]
        pos_error_vehicle = quat_apply_inverse(
            self.obs_dict["robot_vehicle_orientation"], pos_error_world
        )
        yaw_error = ssa(get_euler_xyz_tensor(self.obs_dict["robot_orientation"]))[:, 2]

        rewards, crashes = compute_reward(
            pos_error_vehicle,
            self._prev_dist,
            yaw_error,
            self.obs_dict["robot_body_linvel"],
            self.obs_dict["robot_body_angvel"],
            self.obs_dict["crashes"],
            1.0,
            self.obs_dict["robot_actions"],
            self.obs_dict["robot_prev_actions"],
        )

        # Update prev_dist for next step's closer_reward
        self._prev_dist[:] = torch.norm(pos_error_vehicle, dim=1)
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

        self._prev_dist[env_ids_for_target] = 0.0


# ---------------------------------------------------------------------------
# Reward function — copied verbatim from position_setpoint_task_sim2real.py
# (kept as a module-level JIT function so it stays unchanged for sim2real parity)
# ---------------------------------------------------------------------------

import torch
from torch import Tensor
from typing import Dict, Tuple


@torch.jit.script
def _exp_func(x: Tensor, gain: float, exp: float) -> Tensor:
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def _abs_exp_func(x: Tensor, gain: float, exp: float) -> Tensor:
    return gain * torch.exp(-exp * torch.abs(x))


@torch.jit.script
def _abs_exp_penalty_func(x: Tensor, gain: float, exp: float) -> Tensor:
    return gain * (torch.exp(-exp * torch.abs(x)) - 1)


@torch.jit.script
def compute_reward(
    pos_error: Tensor,
    prev_dist: Tensor,
    yaw_error: Tensor,
    robot_linvels: Tensor,
    robot_angvels: Tensor,
    crashes: Tensor,
    curriculum_level_multiplier: float,
    current_action: Tensor,
    prev_actions: Tensor,
) -> Tuple[Tensor, Tensor]:
    dist = torch.norm(pos_error, dim=1)

    pos_reward = (
        _exp_func(dist, 2.0, 1.0)
        + _exp_func(dist, 3.0, 10.0)
        + _abs_exp_func(dist, 3.0, 50.0)
    )

    robot_speed = torch.norm(robot_linvels, dim=1)
    speed_reward = _exp_func(robot_speed, 1.0, 3.0)

    dist_reward = (20.0 - dist) / 40.0

    action_penalty = torch.sum(_abs_exp_penalty_func(current_action, 0.2, 4.0), dim=1)
    action_difference_penalty = torch.sum(
        _abs_exp_penalty_func(current_action - prev_actions, 0.3, 6.0), dim=1
    )

    closer_reward = 400.0 * (prev_dist - dist)
    yaw_error_reward = _abs_exp_func(yaw_error, 2.0, 3.0)

    total_reward = (
        pos_reward
        + dist_reward
        + pos_reward * (speed_reward + action_penalty + closer_reward / 10.0)
        + action_penalty
        + action_difference_penalty
        + closer_reward
        + yaw_error_reward
    )

    total_reward = curriculum_level_multiplier * total_reward

    crashes = torch.where(dist > 10.0, torch.ones_like(crashes), crashes)
    total_reward = torch.where(crashes > 0.0, -50.0 * torch.ones_like(total_reward), total_reward)

    return total_reward, crashes
