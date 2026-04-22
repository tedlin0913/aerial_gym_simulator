"""
PositionSetpointLMF2Env — Isaac Lab DirectRLEnv for lmf2 sim-to-real position setpoint.

Ports position_setpoint_task_acceleration_sim2real (lmf2 robot) to Isaac Lab.

Observation (17-dim) — same structure as sim2real task:
    [0:3]   position error (world frame, noise std=0.03)
    [3:7]   orientation quaternion xyzw (noise via Euler round-trip std=0.02)
    [7:10]  body-frame linear velocity (noise std=0.02)
    [10:13] body-frame angular velocity (noise std=0.02)
    [13:17] previous actions

Differences from PositionSetpointSim2RealEnv:
  - lmf2 robot (1.24 kg, 10 N max thrust per motor, base_link force application)
  - Acceleration-tuned reward coefficients (harder penalty for going farther from goal)
  - crash_dist = 10 m

Note: The original acceleration_sim2real task uses vehicle-frame action
transformation in the reward. This port applies the reward in world/vehicle frame
for pos_error but uses motor-thrust actions directly (normalized ControlAllocator
[-1,1] range) rather than acceleration commands.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from aerial_gym.envs.direct.aerial_gym_base_env import AerialGymBaseEnv, AerialGymBaseEnvCfg
from aerial_gym.envs.assets.lmf2_asset import LMF2_CFG
from aerial_gym.control.control_allocation import ControlAllocator
from aerial_gym.config.robot_config.lmf2_config import LMF2Cfg
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("position_setpoint_lmf2_env")


@configclass
class PositionSetpointLMF2EnvCfg(AerialGymBaseEnvCfg):
    """Configuration for lmf2 acceleration sim-to-real position setpoint task."""

    robot = LMF2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    episode_length_s: float = 5.0
    decimation: int = 2

    action_space: int = 4
    # obs: pos_error(3) + quat(4) + body_linvel(3) + body_angvel(3) + prev_actions(4) = 17
    observation_space: int = 17
    state_space: int = 0

    target_position_x: float = 0.0
    target_position_y: float = 0.0
    target_position_z: float = 1.5

    pos_error_noise_std: float = 0.03
    orientation_noise_std: float = 0.02
    linvel_noise_std: float = 0.02
    angvel_noise_std: float = 0.02

    crash_dist: float = 10.0


class PositionSetpointLMF2Env(AerialGymBaseEnv):
    """
    lmf2 acceleration sim-to-real position setpoint environment (Isaac Lab).

    Uses "base_link" force application: the ControlAllocator computes the full
    body wrench (Fz + roll/pitch/yaw moments) and it is applied at the root body.
    No per-motor body force needed.
    """

    cfg: PositionSetpointLMF2EnvCfg

    def __init__(self, cfg: PositionSetpointLMF2EnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._setup_drag_coefficients(LMF2Cfg)

        self._target_position = self._terrain.env_origins.clone()
        self._target_position[:, 0] += cfg.target_position_x
        self._target_position[:, 1] += cfg.target_position_y
        self._target_position[:, 2] += cfg.target_position_z

        self._obs_buf = torch.zeros(self.num_envs, cfg.observation_space, device=self.device)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)

        self._reward_params: dict = {}

        logger.info(
            f"PositionSetpointLMF2Env: {self.num_envs} envs, "
            f"obs={cfg.observation_space}, target={self._target_position[0].tolist()}"
        )

    def _setup_control_allocator(self):
        dt = self.cfg.sim.dt * self.cfg.decimation
        return ControlAllocator(
            num_envs=self.num_envs,
            dt=dt,
            config=LMF2Cfg.control_allocator_config,
            device=self.device,
        )

    def _build_observation_tensor(self) -> None:
        from aerial_gym.utils.math import get_euler_xyz_tensor, quat_from_euler_xyz_tensor, ssa

        cfg = self.cfg
        pos_error = self._target_position - self.obs_dict["robot_position"]

        euler = ssa(get_euler_xyz_tensor(self.obs_dict["robot_orientation"]))
        euler_noisy = euler + torch.randn_like(euler) * cfg.orientation_noise_std
        quat_noisy = quat_from_euler_xyz_tensor(euler_noisy)

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
            self.cfg.crash_dist,
        )

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
# Reward — imported from standalone module so it can be tested without SimApp.
# ---------------------------------------------------------------------------

from aerial_gym.envs.direct.lmf2_reward import compute_reward  # noqa: F401
