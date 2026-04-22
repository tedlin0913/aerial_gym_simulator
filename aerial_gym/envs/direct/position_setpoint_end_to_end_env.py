"""
PositionSetpointEndToEndEnv — Isaac Lab DirectRLEnv for sim-to-real end-to-end position setpoint.

Ports position_setpoint_task_sim2real_end_to_end to Isaac Lab.

Observation (15-dim):
    [0:3]   position error (world frame, with noise std=0.001)
    [3:9]   rotation 6D (first two cols of rotation matrix; noise via Euler round-trip std=π/1032)
    [9:12]  linear velocity (world frame, with noise std=0.002)
    [12:15] body angular velocity (with noise std=0.001)

Differences from PositionSetpointSim2RealEnv:
  - Uses rotation 6D instead of quaternion (continuous rotation representation)
  - World-frame linear velocity (not body-frame)
  - No prev_actions in observation
  - Upright/alignment terms in reward
  - crash_dist = 1.5 m
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from aerial_gym.envs.direct.aerial_gym_base_env import AerialGymBaseEnv, AerialGymBaseEnvCfg
from aerial_gym.envs.assets.base_quad_asset import BASE_QUAD_CFG
from aerial_gym.control.control_allocation import ControlAllocator
from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("position_setpoint_end_to_end_env")


@configclass
class PositionSetpointEndToEndEnvCfg(AerialGymBaseEnvCfg):
    """Configuration for sim-to-real end-to-end position setpoint task."""

    robot = BASE_QUAD_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    episode_length_s: float = 5.0
    decimation: int = 2

    action_space: int = 4
    # obs: pos_error(3) + rot6d(6) + world_linvel(3) + body_angvel(3) = 15
    observation_space: int = 15
    state_space: int = 0

    target_position_x: float = 0.0
    target_position_y: float = 0.0
    target_position_z: float = 1.5

    # Sensor noise std devs (matches original end-to-end task)
    pos_error_noise_std: float = 0.001
    orientation_noise_std: float = 0.003054   # π/1032 ≈ 0.003054 rad
    linvel_noise_std: float = 0.002
    angvel_noise_std: float = 0.001

    # Crash distance (strict — encourages tight hovering)
    crash_dist: float = 1.5


class PositionSetpointEndToEndEnv(AerialGymBaseEnv):
    """
    Sim-to-real end-to-end position setpoint environment (Isaac Lab).

    Uses rotation 6D representation for smooth, singularity-free orientation encoding.

    Observation vector (15-dim):
        [0:3]   position error (world frame, with noise)
        [3:9]   rotation 6D — first two columns of R, flattened [R[:,0], R[:,1]]
        [9:12]  world-frame linear velocity (with noise)
        [12:15] body-frame angular velocity (with noise)

    Reward: potential-based closer_reward + position + upright + alignment + velocity penalties
    """

    cfg: PositionSetpointEndToEndEnvCfg

    def __init__(self, cfg: PositionSetpointEndToEndEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._setup_drag_coefficients(BaseQuadCfg)

        # Target position in world frame (env_origin + offset)
        self._target_position = self._terrain.env_origins.clone()
        self._target_position[:, 0] += cfg.target_position_x
        self._target_position[:, 1] += cfg.target_position_y
        self._target_position[:, 2] += cfg.target_position_z

        self._obs_buf = torch.zeros(self.num_envs, cfg.observation_space, device=self.device)

        # prev_pos_error for closer_reward
        self._prev_pos_error = torch.zeros(self.num_envs, 3, device=self.device)

        self._reward_params: dict = {}

        logger.info(
            f"PositionSetpointEndToEndEnv: {self.num_envs} envs, "
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
        Build 15-dim observation with Gaussian sensor noise.

        [0:3]   pos_error world frame + noise
        [3:9]   rotation 6D (first two cols of R; noise via Euler round-trip)
        [9:12]  world_linvel + noise
        [12:15] body_angvel + noise
        """
        from aerial_gym.utils.math import get_euler_xyz_tensor, quat_from_euler_xyz_tensor, ssa
        from aerial_gym.envs.direct.rotation_utils import (
            quaternion_to_matrix,
            matrix_to_rotation_6d,
            euler_angles_to_matrix,
        )

        cfg = self.cfg

        pos_error = self._target_position - self.obs_dict["robot_position"]

        # Orientation noise: add Gaussian noise to Euler angles then re-encode to rotation matrix
        euler = ssa(get_euler_xyz_tensor(self.obs_dict["robot_orientation"]))  # xyzw → (roll, pitch, yaw)
        euler_noisy = euler + torch.randn_like(euler) * cfg.orientation_noise_std
        # Euler round-trip via rotation matrix → 6D
        R = euler_angles_to_matrix(euler_noisy, "ZYX")  # (N, 3, 3)
        rot6d = matrix_to_rotation_6d(R)                # (N, 6)

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
# Reward — imported from standalone module so it can be tested without SimApp.
# ---------------------------------------------------------------------------

from aerial_gym.envs.direct.end_to_end_reward import compute_reward  # noqa: F401

