"""
PositionSetpointEnv — Isaac Lab DirectRLEnv for position setpoint tracking.

Replaces:
    PositionSetpointTask (uses EnvManager/IGE_env_manager) — Isaac Gym

With:
    PositionSetpointEnv(AerialGymBaseEnv)             — Isaac Lab

The reward computation and observation structure are preserved unchanged.
This env wraps the existing compute_reward() @torch.jit.script function.

Usage:
    /home/cow_server01/pg-dev/isaacsim/python.sh train_position_setpoint.py

Or via sample-factory (see scripts/train_position_setpoint_isaaclab.py).
"""

from __future__ import annotations

import math
import torch

from isaaclab.utils import configclass

from aerial_gym.envs.direct.aerial_gym_base_env import AerialGymBaseEnv, AerialGymBaseEnvCfg
from aerial_gym.envs.assets.base_quad_asset import BASE_QUAD_CFG
from aerial_gym.control.control_allocation import ControlAllocator
from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("position_setpoint_env")


@configclass
class PositionSetpointEnvCfg(AerialGymBaseEnvCfg):
    """Configuration for position setpoint task."""

    # Episode: 500 steps × 0.01s = 5s (matches position_setpoint_task_config.py)
    episode_length_s: float = 5.0
    decimation: int = 2

    # Action: 4 motor thrusts in [-1, 1]
    action_space: int = 4
    # Observation: [pos_error(3), orientation(4), body_linvel(3), body_angvel(3)] = 13
    observation_space: int = 13
    state_space: int = 0

    # Task-specific
    target_position_x: float = 0.0
    target_position_y: float = 0.0
    target_position_z: float = 0.0


class PositionSetpointEnv(AerialGymBaseEnv):
    """
    Position setpoint tracking environment (Isaac Lab).

    Observation vector (13-dim, same as original):
        [0:3]  position error in vehicle frame (target - robot)
        [3:7]  robot orientation quaternion (xyzw)
        [7:10] robot body-frame linear velocity
        [10:13] robot body-frame angular velocity

    Reward: compute_reward() from position_setpoint_task.py (unchanged)
    """

    cfg: PositionSetpointEnvCfg

    def __init__(self, cfg: PositionSetpointEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Target position (fixed for now, same as original task_config default)
        self._target_position = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_position[:, 0] = cfg.target_position_x
        self._target_position[:, 1] = cfg.target_position_y
        self._target_position[:, 2] = cfg.target_position_z

        # Observation buffer
        self._obs_buf = torch.zeros(self.num_envs, cfg.observation_space, device=self.device)

        # Reward parameters (matches position_setpoint_task_config.py)
        self._reward_params = {
            key: torch.tensor(val, device=self.device)
            for key, val in {
                "pos_error_gain1": [2.0, 2.0, 2.0],
                "pos_error_exp1": [1 / 3.5, 1 / 3.5, 1 / 3.5],
                "pos_error_gain2": [2.0, 2.0, 2.0],
                "pos_error_exp2": [2.0, 2.0, 2.0],
                "dist_reward_coefficient": 7.5,
                "max_dist": 15.0,
                "action_diff_penalty_gain": [1.0, 1.0, 1.0],
                "absolute_action_reward_gain": [2.0, 2.0, 2.0],
                "crash_penalty": -100.0,
            }.items()
        }

        logger.info(
            f"PositionSetpointEnv: {self.num_envs} envs, "
            f"target={self._target_position[0].tolist()}"
        )

    # =========================================================================
    # AerialGymBaseEnv interface
    # =========================================================================

    def _setup_robot_cfg(self):
        """Use the base_quadrotor URDF asset."""
        return BASE_QUAD_CFG

    def _setup_control_allocator(self):
        """Build ControlAllocator from BaseQuadCfg (same config as before migration)."""
        dt = self.cfg.sim.dt * self.cfg.decimation  # RL control dt
        return ControlAllocator(
            num_envs=self.num_envs,
            dt=dt,
            config=BaseQuadCfg.control_allocator_config,
            device=self.device,
        )

    def _build_observation_tensor(self) -> None:
        """
        Build 13-dim observation vector (same structure as process_obs_for_task()).

        [0:3]  pos_error in vehicle frame: target - robot_pos, rotated to vehicle frame
        [3:7]  robot orientation (xyzw)
        [7:10] robot body-frame linear velocity
        [10:13] robot body-frame angular velocity
        """
        from aerial_gym.utils.math import quat_apply_inverse

        pos_error_world = self._target_position - self.obs_dict["robot_position"]
        pos_error_vehicle = quat_apply_inverse(
            self.obs_dict["robot_vehicle_orientation"], pos_error_world
        )

        self._obs_buf[:, 0:3]  = pos_error_vehicle
        self._obs_buf[:, 3:7]  = self.obs_dict["robot_orientation"]    # xyzw
        self._obs_buf[:, 7:10] = self.obs_dict["robot_body_linvel"]
        self._obs_buf[:, 10:13] = self.obs_dict["robot_body_angvel"]

    def _get_rewards(self) -> torch.Tensor:
        """Compute reward using the existing @torch.jit.script compute_reward function."""
        # Import here to avoid circular import at module level
        from aerial_gym.task.position_setpoint_task.position_setpoint_task import compute_reward

        pos_error_world = self._target_position - self.obs_dict["robot_position"]
        pos_error_vehicle = _apply_quat_inverse(
            self.obs_dict["robot_vehicle_orientation"], pos_error_world
        )

        rewards, crashes = compute_reward(
            pos_error_vehicle,
            self.obs_dict["robot_linvel"],
            self.obs_dict["robot_orientation"],        # xyzw — quat_axis() expects xyzw
            self.obs_dict["robot_body_angvel"],
            self.obs_dict["crashes"],
            1.0,  # curriculum_level_multiplier (fixed for Phase 1)
            self.obs_dict["robot_actions"],
            self.obs_dict["robot_prev_actions"],
            self._reward_params,
        )

        # compute_reward updates crashes in-place (dist > 8m → crash)
        self.obs_dict["crashes"][:] = crashes

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        terminated: crashed (out of bounds or collision)
        time_out: episode length exceeded (handled by DirectRLEnv episode_length_buf)
        """
        terminated = self.obs_dict["crashes"].bool()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset with randomized target position (matches original task behavior)."""
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids_for_target = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_for_target = env_ids

        # Fixed target at origin (matches original position_setpoint_task_config)
        self._target_position[env_ids_for_target] = 0.0


# ---------------------------------------------------------------------------
# Helper: quat_apply_inverse without the gymnastics of importing from task
# ---------------------------------------------------------------------------

def _apply_quat_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply inverse of quaternion q (xyzw) to vector v. Returns rotated vector."""
    from aerial_gym.utils.math import quat_apply_inverse
    return quat_apply_inverse(q, v)
