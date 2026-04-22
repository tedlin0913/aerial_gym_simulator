"""
end_to_end_reward.py — Pure-torch JIT reward for PositionSetpointEndToEndEnv.

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
def _exp_penalty_func(x: Tensor, gain: float, exp: float) -> Tensor:
    return gain * (torch.exp(-exp * x * x) - 1)


@torch.jit.script
def _quat_axis(q: Tensor, axis: int = 0) -> Tensor:
    """Extract local axis vector from xyzw quaternion. axis: 0=x, 1=y, 2=z"""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    if axis == 0:  # local x
        return torch.stack([
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ], dim=-1)
    elif axis == 1:  # local y
        return torch.stack([
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        ], dim=-1)
    else:  # local z (up)
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

    # Z axis weighted heavily (vertical hover is harder)
    pos_error_weighted = pos_error.clone()
    pos_error_weighted[:, 2] = pos_error[:, 2] * 11.0
    pos_reward = (
        torch.sum(_exp_func(pos_error_weighted[:, :3], 10.0, 10.0), dim=1)
        + torch.sum(_exp_func(pos_error_weighted[:, :3], 2.0, 2.0), dim=1)
    )

    # Upright: local-z axis should point up (world z)
    ups = _quat_axis(quats, 2)
    tiltage = 1.0 - ups[:, 2]
    upright_reward = _exp_func(tiltage, 2.5, 5.0)

    # Forward alignment: local-x axis should point along world x (yaw=0 preference)
    forw = _quat_axis(quats, 0)
    alignment = 1.0 - forw[:, 0]
    alignment_reward = _exp_func(alignment, 6.0, 5.0)

    angvel_reward = torch.sum(_exp_func(angvels_body, 0.3, 10.0), dim=1)
    vel_reward = torch.sum(_exp_func(linvels_world, 1.0, 5.0), dim=1)

    action_cost = torch.sum(_exp_penalty_func(current_action, 0.01, 10.0), dim=1)

    # Closer-to-goal reward (potential-based)
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
