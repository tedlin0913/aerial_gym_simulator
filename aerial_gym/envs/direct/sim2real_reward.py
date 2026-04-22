"""
sim2real_reward.py — Pure-torch JIT reward for PositionSetpointSim2RealEnv.

Extracted here so it can be imported (and unit-tested) without loading the full
Isaac Lab simulation stack.
"""

import torch
from torch import Tensor
from typing import Tuple


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
