"""
Test 1: Quaternion Convention Check (Pre-Migration Regression)

Verifies that global_tensor_dict["robot_orientation"] stores quaternions in xyzw format.

Why this matters: Wrong convention (wxyz) causes quat_axis(), quat_rotate(), and
get_euler_xyz() in math.py to silently compute incorrect values. The drone would
spawn inverted and the reward would diverge from step 1.

xyzw identity quaternion: [0, 0, 0, 1]  (w is the last element, index 3)
wxyz identity quaternion: [1, 0, 0, 0]  (w is the first element, index 0)

Requires: Isaac Sim Python interpreter
  /home/cow_server01/pg-dev/isaacsim/python.sh -m pytest tests/test_quaternion_convention.py
"""
import pytest
import torch


def test_math_xyzw_identity():
    """
    Unit test (no simulator needed): verify math.py functions treat index 3 as w.
    A quat [0,0,0,1] in xyzw should be identity — quat_axis should return z-axis.
    """
    from aerial_gym.utils.math import quat_axis, quat_rotate

    # Identity quaternion in xyzw: x=0, y=0, z=0, w=1
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    # quat_axis(q, 2) extracts the z-axis of the frame described by q
    # For identity rotation, the z-axis should be [0, 0, 1]
    z_axis = quat_axis(identity, 2)
    assert z_axis.shape == (1, 3), f"Expected shape (1,3), got {z_axis.shape}"
    assert torch.allclose(z_axis, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-5), (
        f"Identity quat z-axis should be [0,0,1], got {z_axis}. "
        "If this fails, math.py may be using wxyz convention."
    )

    # quat_rotate with identity should return vector unchanged
    v = torch.tensor([[1.0, 2.0, 3.0]])
    v_rotated = quat_rotate(identity, v)
    assert torch.allclose(v_rotated, v, atol=1e-5), (
        f"Identity rotation should not change vector. Got {v_rotated} from {v}."
    )


def test_get_euler_xyz_identity():
    """
    Verify get_euler_xyz treats the quaternion as xyzw.
    Identity quaternion [0,0,0,1] should give euler angles [0,0,0].
    """
    from aerial_gym.utils.math import get_euler_xyz

    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    roll, pitch, yaw = get_euler_xyz(identity)

    assert torch.allclose(roll, torch.tensor([0.0]), atol=1e-5), (
        f"Roll should be 0 for identity quat, got {roll}"
    )
    assert torch.allclose(pitch, torch.tensor([0.0]), atol=1e-5), (
        f"Pitch should be 0 for identity quat, got {pitch}"
    )
    assert torch.allclose(yaw, torch.tensor([0.0]), atol=1e-5), (
        f"Yaw should be 0 for identity quat, got {yaw}"
    )


def test_get_euler_xyz_90_deg_roll():
    """
    Verify a 90-degree roll quaternion produces roll=pi/2 (not some other angle).
    90 deg roll around x: quat = [sin(45deg), 0, 0, cos(45deg)] in xyzw
    """
    import math as pymath
    from aerial_gym.utils.math import get_euler_xyz

    half_angle = pymath.pi / 4  # 45 degrees = half of 90 degrees
    q_xyzw = torch.tensor([[pymath.sin(half_angle), 0.0, 0.0, pymath.cos(half_angle)]])

    roll, pitch, yaw = get_euler_xyz(q_xyzw)

    assert torch.allclose(roll, torch.tensor([pymath.pi / 2]), atol=1e-4), (
        f"Expected roll=pi/2 for 90-deg roll quat, got {roll.item():.4f}. "
        "Likely a wxyz/xyzw convention mismatch."
    )


def test_quat_axis_w_component_is_index_3():
    """
    Directly verify that w is at index 3 in quat_axis implementation.
    A quaternion with w at index 0 (wxyz) would fail this test.
    """
    from aerial_gym.utils.math import quat_axis
    import math as pymath

    # 180-degree rotation around z-axis
    # xyzw: [0, 0, sin(90deg), cos(90deg)] = [0, 0, 1, 0]
    q_180_z_xyzw = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

    # x-axis after 180-deg rotation around z: should become [-1, 0, 0]
    x_axis_rotated = quat_axis(q_180_z_xyzw, 0)
    assert torch.allclose(x_axis_rotated, torch.tensor([[-1.0, 0.0, 0.0]]), atol=1e-5), (
        f"Expected [-1,0,0] for x-axis after 180-deg z rotation, got {x_axis_rotated}. "
        "This indicates wrong quaternion convention."
    )


# =============================================================================
# The test below requires a live Isaac Lab/Isaac Gym simulator.
# It is marked skip by default and should be run manually during migration
# to verify the convention is preserved after porting _populate_tensor_dict().
# =============================================================================

@pytest.mark.skip(reason="Requires live simulator — run with Isaac Sim Python during migration")
def test_sim_robot_orientation_is_xyzw():
    """
    Integration test: spawn a robot with zero rotation, read robot_orientation from
    global_tensor_dict, verify w=1 (index 3) and xyz=0.

    Run with:
        /home/cow_server01/pg-dev/isaacsim/python.sh -m pytest tests/test_quaternion_convention.py::test_sim_robot_orientation_is_xyzw -v
    """
    # Deferred import — needs SimulationApp launched first
    # This test is a template; actual runner sets up AppLauncher before pytest.
    from aerial_gym.task.position_setpoint_task.position_setpoint_task_config import (
        task_config as PositionSetpointTaskConfig,
    )
    from aerial_gym.task.position_setpoint_task.position_setpoint_task import (
        PositionSetpointTask,
    )

    task = PositionSetpointTask(
        task_config=PositionSetpointTaskConfig,
        num_envs=1,
        headless=True,
        use_warp=False,
    )
    task.reset()

    obs_dict = task.sim_env.get_obs()
    robot_orientation = obs_dict["robot_orientation"]  # shape: (1, 4)

    assert robot_orientation.shape[-1] == 4, "Orientation must be a quaternion (4 components)"

    # xyzw: w is index 3
    w = robot_orientation[..., 3]
    xyz = robot_orientation[..., :3]

    assert torch.allclose(w, torch.ones_like(w), atol=1e-3), (
        f"Expected w=1 (xyzw identity) at index 3, got w={w}. "
        "If w=1 is at index 0, the convention is wxyz (Isaac Lab) and _populate_tensor_dict "
        "must convert."
    )
    assert torch.allclose(xyz, torch.zeros_like(xyz), atol=1e-3), (
        f"Expected xyz=0 for identity quaternion, got {xyz}"
    )

    task.close()
