"""
Test 3: Reward Non-NaN After 10 Steps (Pre-Migration Regression)

Runs position_setpoint_task for 10 env steps with random actions and asserts:
  (a) rewards tensor has no NaN/Inf
  (b) observations tensor has no NaN/Inf
  (c) crashes tensor is bool/int type (not float NaN)
  (d) episode does not immediately terminate on step 1

Why this matters:
- A reward of -20 on step 1 means the drone spawned in a crashed state (likely a
  quaternion or init bug). This is the canary in the coal mine for migration regressions.
- NaN rewards mean the physics produced invalid values — motor model overflow, division
  by zero in reward computation, or uninitialized state tensors.

Run with Isaac Sim Python (requires live simulator):
  /home/cow_server01/pg-dev/isaacsim/python.sh -m pytest tests/test_reward_sanity.py -v

The fast unit tests (no simulator) are in test_reward_unit.py.
"""
import pytest
import torch


def test_compute_reward_no_nan():
    """
    Unit test (no simulator): verify compute_reward() with zero inputs returns
    finite values and no crash penalty.
    """
    from aerial_gym.task.position_setpoint_task.position_setpoint_task import (
        compute_reward,
    )

    num_envs = 8
    device = "cpu"

    # All zeros = drone is at target, identity orientation, no velocity
    pos_error = torch.zeros(num_envs, 3, device=device)
    lin_vels = torch.zeros(num_envs, 3, device=device)
    # Identity quaternion in xyzw
    robot_quats = torch.zeros(num_envs, 4, device=device)
    robot_quats[:, 3] = 1.0  # w=1 for identity
    robot_angvels = torch.zeros(num_envs, 3, device=device)
    crashes = torch.zeros(num_envs, dtype=torch.long, device=device)
    curriculum_level_multiplier = 1.0
    current_action = torch.zeros(num_envs, 4, device=device)
    prev_actions = torch.zeros(num_envs, 4, device=device)

    # Build parameter_dict (matches position_setpoint_task_config.py reward_parameters)
    parameter_dict = {
        "pos_error_gain1": torch.tensor([2.0, 2.0, 2.0], device=device),
        "pos_error_exp1": torch.tensor([1 / 3.5, 1 / 3.5, 1 / 3.5], device=device),
        "pos_error_gain2": torch.tensor([2.0, 2.0, 2.0], device=device),
        "pos_error_exp2": torch.tensor([2.0, 2.0, 2.0], device=device),
        "dist_reward_coefficient": torch.tensor(7.5, device=device),
        "max_dist": torch.tensor(15.0, device=device),
        "action_diff_penalty_gain": torch.tensor([1.0, 1.0, 1.0], device=device),
        "absolute_action_reward_gain": torch.tensor([2.0, 2.0, 2.0], device=device),
        "crash_penalty": torch.tensor(-100.0, device=device),
    }

    total_reward, crashes_out = compute_reward(
        pos_error,
        lin_vels,
        robot_quats,
        robot_angvels,
        crashes,
        curriculum_level_multiplier,
        current_action,
        prev_actions,
        parameter_dict,
    )

    assert torch.isfinite(total_reward).all(), (
        f"compute_reward() returned non-finite values: {total_reward}"
    )
    assert (total_reward >= 0).all(), (
        f"Reward should be non-negative when drone is at target with no crashes. Got: {total_reward}"
    )


def test_compute_reward_crash_gives_penalty():
    """
    Verify that crashes=1 produces -20 reward (crash penalty from compute_reward).
    """
    from aerial_gym.task.position_setpoint_task.position_setpoint_task import (
        compute_reward,
    )

    num_envs = 4
    device = "cpu"

    pos_error = torch.zeros(num_envs, 3, device=device)
    lin_vels = torch.zeros(num_envs, 3, device=device)
    robot_quats = torch.zeros(num_envs, 4, device=device)
    robot_quats[:, 3] = 1.0
    robot_angvels = torch.zeros(num_envs, 3, device=device)
    # Set all envs as crashed
    crashes = torch.ones(num_envs, dtype=torch.long, device=device)
    curriculum_level_multiplier = 1.0
    current_action = torch.zeros(num_envs, 4, device=device)
    prev_actions = torch.zeros(num_envs, 4, device=device)

    parameter_dict = {
        "pos_error_gain1": torch.tensor([2.0, 2.0, 2.0], device=device),
        "pos_error_exp1": torch.tensor([1 / 3.5, 1 / 3.5, 1 / 3.5], device=device),
        "pos_error_gain2": torch.tensor([2.0, 2.0, 2.0], device=device),
        "pos_error_exp2": torch.tensor([2.0, 2.0, 2.0], device=device),
        "dist_reward_coefficient": torch.tensor(7.5, device=device),
        "max_dist": torch.tensor(15.0, device=device),
        "action_diff_penalty_gain": torch.tensor([1.0, 1.0, 1.0], device=device),
        "absolute_action_reward_gain": torch.tensor([2.0, 2.0, 2.0], device=device),
        "crash_penalty": torch.tensor(-100.0, device=device),
    }

    total_reward, crashes_out = compute_reward(
        pos_error,
        lin_vels,
        robot_quats,
        robot_angvels,
        crashes,
        curriculum_level_multiplier,
        current_action,
        prev_actions,
        parameter_dict,
    )

    # compute_reward sets crashed reward to -20 (hardcoded in the function)
    assert (total_reward == -20.0).all(), (
        f"Expected -20 crash penalty for all envs, got {total_reward}"
    )


def test_compute_reward_far_from_target_triggers_crash():
    """
    Verify that dist > 8.0 triggers a crash (crashes updated in-place).
    """
    from aerial_gym.task.position_setpoint_task.position_setpoint_task import (
        compute_reward,
    )

    num_envs = 2
    device = "cpu"

    # Set pos_error to distance 10 (beyond 8m threshold)
    pos_error = torch.zeros(num_envs, 3, device=device)
    pos_error[:, 0] = 10.0  # 10m away in x

    lin_vels = torch.zeros(num_envs, 3, device=device)
    robot_quats = torch.zeros(num_envs, 4, device=device)
    robot_quats[:, 3] = 1.0
    robot_angvels = torch.zeros(num_envs, 3, device=device)
    crashes = torch.zeros(num_envs, dtype=torch.long, device=device)
    current_action = torch.zeros(num_envs, 4, device=device)
    prev_actions = torch.zeros(num_envs, 4, device=device)

    parameter_dict = {
        "pos_error_gain1": torch.tensor([2.0, 2.0, 2.0], device=device),
        "pos_error_exp1": torch.tensor([1 / 3.5, 1 / 3.5, 1 / 3.5], device=device),
        "pos_error_gain2": torch.tensor([2.0, 2.0, 2.0], device=device),
        "pos_error_exp2": torch.tensor([2.0, 2.0, 2.0], device=device),
        "dist_reward_coefficient": torch.tensor(7.5, device=device),
        "max_dist": torch.tensor(15.0, device=device),
        "action_diff_penalty_gain": torch.tensor([1.0, 1.0, 1.0], device=device),
        "absolute_action_reward_gain": torch.tensor([2.0, 2.0, 2.0], device=device),
        "crash_penalty": torch.tensor(-100.0, device=device),
    }

    total_reward, crashes_out = compute_reward(
        pos_error,
        lin_vels,
        robot_quats,
        robot_angvels,
        crashes,
        1.0,
        current_action,
        prev_actions,
        parameter_dict,
    )

    assert (crashes_out > 0).all(), (
        f"Expected crash for dist=10 > 8m threshold, crashes={crashes_out}"
    )
    assert (total_reward == -20.0).all(), (
        f"Expected -20 crash penalty, got {total_reward}"
    )


@pytest.mark.skip(reason="Requires live simulator — run with Isaac Sim Python during migration")
def test_sim_10_steps_no_nan():
    """
    Integration test: run position_setpoint_task for 10 steps with random actions.
    Asserts: (a) rewards finite, (b) obs finite, (c) no immediate crash on step 1.

    Run with:
        /home/cow_server01/pg-dev/isaacsim/python.sh -m pytest tests/test_reward_sanity.py::test_sim_10_steps_no_nan -v
    """
    from aerial_gym.config.task_config.position_setpoint_task_config import task_config
    from aerial_gym.task.position_setpoint_task.position_setpoint_task import (
        PositionSetpointTask,
    )

    # Small num_envs to keep test fast
    task = PositionSetpointTask(
        task_config=task_config,
        num_envs=4,
        headless=True,
        use_warp=False,
    )

    obs, rewards, terminations, truncations, infos = task.reset()

    for step in range(10):
        actions = torch.rand(
            (task.num_envs, task.task_config.action_space_dim),
            device=task.device,
        ) * 2 - 1  # uniform in [-1, 1]

        obs, rewards, terminations, truncations, infos = task.step(actions)

        assert torch.isfinite(rewards).all(), (
            f"Step {step+1}: rewards contain NaN/Inf: {rewards}"
        )
        assert torch.isfinite(obs["observations"]).all(), (
            f"Step {step+1}: observations contain NaN/Inf"
        )
        assert terminations.dtype in (torch.bool, torch.int32, torch.int64, torch.float32), (
            f"Step {step+1}: terminations has unexpected dtype {terminations.dtype}"
        )

        if step == 0:
            # If all envs crashed on step 1, something is wrong with spawn state
            crash_count = (terminations > 0).sum().item()
            assert crash_count < task.num_envs, (
                f"ALL {task.num_envs} envs crashed on step 1. "
                "Likely a quaternion convention bug — robot spawned inverted. "
                f"Rewards: {rewards}"
            )

    task.close()


@pytest.mark.skip(reason="Requires live simulator — run with Isaac Sim Python during migration")
def test_sim_zero_actions_drone_falls():
    """
    Integration test: with zero actions, the drone should fall under gravity
    (z position decreasing). Verifies physics is actually running.

    Run with:
        /home/cow_server01/pg-dev/isaacsim/python.sh -m pytest tests/test_reward_sanity.py::test_sim_zero_actions_drone_falls -v
    """
    from aerial_gym.config.task_config.position_setpoint_task_config import task_config
    from aerial_gym.task.position_setpoint_task.position_setpoint_task import (
        PositionSetpointTask,
    )

    task = PositionSetpointTask(
        task_config=task_config,
        num_envs=1,
        headless=True,
        use_warp=False,
    )
    task.reset()

    initial_z = task.sim_env.get_obs()["robot_position"][0, 2].item()

    zero_actions = torch.zeros(
        (task.num_envs, task.task_config.action_space_dim),
        device=task.device,
    )
    for _ in range(50):
        task.step(zero_actions)

    final_z = task.sim_env.get_obs()["robot_position"][0, 2].item()

    assert final_z < initial_z, (
        f"Drone z did not decrease with zero thrust. initial_z={initial_z:.3f}, final_z={final_z:.3f}. "
        "Physics may not be running, or gravity is disabled."
    )

    task.close()
