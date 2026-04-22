"""
lmf2_reward.py — Pure-torch JIT reward for PositionSetpointLMF2Env.

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
    crash_dist: float,
) -> Tuple[Tensor, Tensor]:

    dist = torch.norm(pos_error, dim=1)

    pos_reward = (
        _exp_func(dist, 2.0, 1.0)
        + _exp_func(dist, 3.0, 10.0)
        + _abs_exp_func(dist, 3.0, 50.0)
    )

    close_pos_reward = _exp_func(dist, 2.0, 1.0)

    robot_speed = torch.norm(robot_linvels, dim=1)
    speed_reward = _exp_func(robot_speed, 2.0, 2.5)

    action_penalty = torch.sum(_abs_exp_penalty_func(current_action, 0.3, 4.0), dim=1)
    action_difference_penalty = torch.sum(
        _abs_exp_penalty_func(current_action - prev_actions, 0.4, 6.0), dim=1
    )

    # Asymmetric closer_reward: penalize moving away more than rewarding approach
    closer_reward = torch.where(
        dist < prev_dist,
        400.0 * (prev_dist - dist),
        1200.0 * (prev_dist - dist),
    )

    yaw_error_reward = _abs_exp_func(yaw_error, 3.0, 5.0)

    total_reward = (
        (pos_reward + pos_reward * (closer_reward / 9.0 + action_penalty / 3.0 + speed_reward / 1.5))
        + action_penalty
        + action_difference_penalty
        + closer_reward
        + yaw_error_reward
        + close_pos_reward
        + speed_reward * 0.2
    )

    total_reward = curriculum_level_multiplier * total_reward

    crashes = torch.where(dist > crash_dist, torch.ones_like(crashes), crashes)
    total_reward = torch.where(crashes > 0.0, -50.0 * torch.ones_like(total_reward), total_reward)

    return total_reward, crashes
