"""
rotation_utils.py — Rotation representation helpers for Isaac Lab envs.

Replaces pytorch3d.transforms for the functions used in PX4 / end-to-end tasks:
    quaternion_to_matrix   xyzw  → (N, 3, 3)
    matrix_to_rotation_6d  (N, 3, 3) → (N, 6)  [first two cols]
    matrix_to_euler_angles (N, 3, 3) → (N, 3)  ZYX euler
    euler_angles_to_matrix (N, 3)   → (N, 3, 3) ZYX euler

All functions use isaaclab.utils.math internally.
Convention: quaternion = xyzw (aerial_gym standard, w at index 3).
"""

from __future__ import annotations

import torch
from torch import Tensor


def quaternion_to_matrix(q_xyzw: Tensor) -> Tensor:
    """
    Convert xyzw quaternion(s) to rotation matrix.

    Args:
        q_xyzw: (..., 4) quaternion in xyzw convention (w is last)

    Returns:
        (..., 3, 3) rotation matrix
    """
    from isaaclab.utils.math import matrix_from_quat

    # Isaac Lab's matrix_from_quat expects wxyz
    q_wxyz = torch.roll(q_xyzw, 1, dims=-1)
    return matrix_from_quat(q_wxyz)


def matrix_to_rotation_6d(R: Tensor) -> Tensor:
    """
    Convert rotation matrix to 6D continuous rotation representation.
    Takes the first two columns of R (flattened): [R[:,0], R[:,1]]

    Args:
        R: (..., 3, 3) rotation matrix

    Returns:
        (..., 6) — first 3 elements are col 0, next 3 are col 1
    """
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def matrix_to_euler_angles(R: Tensor, convention: str) -> Tensor:
    """
    Convert rotation matrix to Euler angles.

    Args:
        R: (..., 3, 3) rotation matrix
        convention: e.g. "ZYX" (matches pytorch3d convention name)

    Returns:
        (..., 3) Euler angles [roll, pitch, yaw] in radians (smallest signed angle)
    """
    from isaaclab.utils.math import quat_from_matrix
    from aerial_gym.utils.math import get_euler_xyz_tensor, ssa

    if convention != "ZYX":
        raise ValueError(f"Only ZYX convention supported, got {convention}")

    # R → quat wxyz → xyzw, then use aerial_gym's correct euler extraction
    q_wxyz = quat_from_matrix(R)
    q_xyzw = torch.roll(q_wxyz, -1, dims=-1)  # wxyz → xyzw
    euler = get_euler_xyz_tensor(q_xyzw)       # [roll, pitch, yaw] in [0, 2π)
    return ssa(euler)                          # map to [-π, π]


def euler_angles_to_matrix(angles: Tensor, convention: str) -> Tensor:
    """
    Convert Euler angles to rotation matrix.

    Args:
        angles: (..., 3) Euler angles [roll, pitch, yaw] in radians
        convention: "ZYX" only

    Returns:
        (..., 3, 3) rotation matrix
    """
    from isaaclab.utils.math import matrix_from_euler

    if convention != "ZYX":
        raise ValueError(f"Only ZYX convention supported, got {convention}")

    # matrix_from_euler("ZYX") expects [yaw, pitch, roll] (Z angle at index 0).
    # Our callers pass [roll, pitch, yaw] (aerial_gym / get_euler_xyz_tensor order).
    # Reorder: [..., 0]=roll, [..., 1]=pitch, [..., 2]=yaw  →  [yaw, pitch, roll]
    angles_zyx = angles[..., [2, 1, 0]]
    return matrix_from_euler(angles_zyx, convention="ZYX")
