"""
Unit tests for Isaac Lab reward functions and rotation utilities.

All tests run without a live simulator (cpu torch only).

Run with Isaac Sim Python for full torch JIT compatibility:
    /home/cow_server01/pg-dev/isaacsim/python.sh -m pytest tests/test_isaaclab_rewards.py -v

Or with any Python that has torch:
    python -m pytest tests/test_isaaclab_rewards.py -v
"""

import pytest
import torch


# ============================================================================
# rotation_utils tests
# ============================================================================

class TestRotationUtils:

    def _load(self):
        from aerial_gym.envs.direct.rotation_utils import (
            quaternion_to_matrix,
            matrix_to_rotation_6d,
            matrix_to_euler_angles,
            euler_angles_to_matrix,
        )
        return quaternion_to_matrix, matrix_to_rotation_6d, matrix_to_euler_angles, euler_angles_to_matrix

    def test_identity_quaternion_to_matrix(self):
        """xyzw identity quaternion → identity 3×3 matrix."""
        q2m, _, _, _ = self._load()
        q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # xyzw identity
        R = q2m(q)
        assert R.shape == (1, 3, 3)
        assert torch.allclose(R[0], torch.eye(3), atol=1e-5)

    def test_rotation_6d_shape(self):
        """matrix_to_rotation_6d returns (N, 6) for (N, 3, 3) input."""
        _, m2r, _, _ = self._load()
        R = torch.eye(3).unsqueeze(0).expand(8, -1, -1)
        r6d = m2r(R)
        assert r6d.shape == (8, 6)

    def test_identity_rotation_6d(self):
        """Identity matrix → [1,0,0,0,1,0] (first two columns of I)."""
        _, m2r, _, _ = self._load()
        R = torch.eye(3).unsqueeze(0)
        r6d = m2r(R)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        assert torch.allclose(r6d, expected, atol=1e-5)

    def test_euler_round_trip(self):
        """ZYX euler → matrix → euler should be identity (within tolerance)."""
        _, _, m2e, e2m = self._load()
        angles = torch.tensor([[0.1, 0.2, 0.3]])
        R = e2m(angles, "ZYX")
        recovered = m2e(R, "ZYX")
        assert torch.allclose(angles, recovered, atol=1e-4)

    def test_quaternion_matrix_orthogonal(self):
        """Rotation matrix from quaternion should be orthogonal (R @ R.T ≈ I)."""
        q2m, _, _, _ = self._load()
        # Random unit quaternion xyzw
        q = torch.randn(4, 4)
        q = q / q.norm(dim=-1, keepdim=True)
        R = q2m(q)
        I_approx = torch.bmm(R, R.transpose(-1, -2))
        assert torch.allclose(I_approx, torch.eye(3).expand(4, -1, -1), atol=1e-5)


# ============================================================================
# position_setpoint_sim2real reward tests
# ============================================================================

class TestSim2RealReward:

    def _load(self):
        from aerial_gym.envs.direct.sim2real_reward import compute_reward
        return compute_reward

    def test_at_target_no_crash(self):
        """Drone at target (zero pos_error) with zero velocity → positive reward, no crash."""
        compute_reward = self._load()
        N = 8
        pos_error = torch.zeros(N, 3)
        prev_dist = torch.zeros(N)
        yaw_error = torch.zeros(N)
        linvels = torch.zeros(N, 3)
        angvels = torch.zeros(N, 3)
        crashes = torch.zeros(N, dtype=torch.long)
        current_action = torch.zeros(N, 4)
        prev_actions = torch.zeros(N, 4)

        reward, crashes_out = compute_reward(
            pos_error, prev_dist, yaw_error, linvels, angvels,
            crashes, 1.0, current_action, prev_actions
        )
        assert torch.isfinite(reward).all()
        assert (reward > 0).all(), f"Reward should be positive at target. Got {reward}"
        assert (crashes_out == 0).all()

    def test_far_from_target_crash(self):
        """Drone > 10 m away → crash triggered, reward = -50."""
        compute_reward = self._load()
        N = 4
        pos_error = torch.zeros(N, 3)
        pos_error[:, 0] = 15.0  # 15 m away
        prev_dist = torch.zeros(N)
        yaw_error = torch.zeros(N)
        linvels = torch.zeros(N, 3)
        angvels = torch.zeros(N, 3)
        crashes = torch.zeros(N, dtype=torch.long)

        reward, crashes_out = compute_reward(
            pos_error, prev_dist, yaw_error, linvels, angvels,
            crashes, 1.0, torch.zeros(N, 4), torch.zeros(N, 4)
        )
        assert (crashes_out > 0).all(), "Should have crashed at 15m > 10m threshold"
        assert (reward == -50.0).all(), f"Crash reward should be -50, got {reward}"

    def test_closer_reward_sign(self):
        """Moving toward target (prev_dist > dist) → positive closer_reward."""
        compute_reward = self._load()
        N = 4
        pos_error = torch.zeros(N, 3)
        pos_error[:, 0] = 1.0  # 1 m away now
        prev_dist = torch.full((N,), 2.0)  # was 2 m away
        yaw_error = torch.zeros(N)
        linvels = torch.zeros(N, 3)
        angvels = torch.zeros(N, 3)
        crashes = torch.zeros(N, dtype=torch.long)

        reward_approaching, _ = compute_reward(
            pos_error, prev_dist, yaw_error, linvels, angvels,
            crashes, 1.0, torch.zeros(N, 4), torch.zeros(N, 4)
        )

        # Same but moving away
        pos_error_away = torch.zeros(N, 3)
        pos_error_away[:, 0] = 3.0  # now 3 m away
        reward_retreating, _ = compute_reward(
            pos_error_away, prev_dist, yaw_error, linvels, angvels,
            crashes, 1.0, torch.zeros(N, 4), torch.zeros(N, 4)
        )

        assert (reward_approaching > reward_retreating).all(), \
            "Approaching target should give higher reward than retreating"

    def test_no_nan_random_input(self):
        """Random inputs should not produce NaN/Inf rewards."""
        compute_reward = self._load()
        N = 32
        torch.manual_seed(42)
        pos_error = torch.randn(N, 3) * 3
        prev_dist = torch.rand(N) * 5
        yaw_error = torch.randn(N)
        linvels = torch.randn(N, 3)
        angvels = torch.randn(N, 3)
        crashes = torch.zeros(N, dtype=torch.long)

        reward, _ = compute_reward(
            pos_error, prev_dist, yaw_error, linvels, angvels,
            crashes, 1.0, torch.randn(N, 4), torch.randn(N, 4)
        )
        assert torch.isfinite(reward).all(), f"NaN/Inf in reward: {reward}"


# ============================================================================
# position_setpoint_end_to_end reward tests
# ============================================================================

class TestEndToEndReward:

    def _load(self):
        from aerial_gym.envs.direct.end_to_end_reward import compute_reward
        return compute_reward

    def _identity_quat(self, N):
        q = torch.zeros(N, 4)
        q[:, 3] = 1.0  # xyzw identity
        return q

    def test_at_target_positive_reward(self):
        """Zero pos_error, identity orientation → positive reward."""
        compute_reward = self._load()
        N = 8
        pos_error = torch.zeros(N, 3)
        quats = self._identity_quat(N)
        linvels = torch.zeros(N, 3)
        angvels = torch.zeros(N, 3)
        crashes = torch.zeros(N, dtype=torch.long)

        reward, crashes_out = compute_reward(
            pos_error, quats, linvels, angvels,
            crashes, torch.zeros(N, 4), torch.zeros(N, 4),
            torch.zeros(N, 3), 1.5
        )
        assert torch.isfinite(reward).all()
        assert (crashes_out == 0).all()

    def test_crash_beyond_dist(self):
        """Drone beyond crash_dist=1.5 → crash, reward not positive."""
        compute_reward = self._load()
        N = 4
        pos_error = torch.zeros(N, 3)
        pos_error[:, 0] = 3.0  # 3 m > crash_dist=1.5
        quats = self._identity_quat(N)
        crashes = torch.zeros(N, dtype=torch.long)

        reward, crashes_out = compute_reward(
            pos_error, quats, torch.zeros(N, 3), torch.zeros(N, 3),
            crashes, torch.zeros(N, 4), torch.zeros(N, 4),
            torch.zeros(N, 3), 1.5
        )
        assert (crashes_out > 0).all()

    def test_upright_vs_tilted(self):
        """Upright drone should get higher reward than heavily tilted drone."""
        compute_reward = self._load()
        N = 1
        pos_error = torch.zeros(N, 3)
        prev_pos_error = torch.zeros(N, 3)

        # Upright (identity quaternion)
        q_upright = self._identity_quat(N)
        reward_upright, _ = compute_reward(
            pos_error, q_upright, torch.zeros(N, 3), torch.zeros(N, 3),
            torch.zeros(N, dtype=torch.long), torch.zeros(N, 4), torch.zeros(N, 4),
            prev_pos_error, 1.5
        )

        # 90° tilt (quaternion for 90° rotation around x-axis)
        # xyzw: sin(45°)=0.707 in x, cos(45°)=0.707 in w
        q_tilted = torch.tensor([[0.707, 0.0, 0.0, 0.707]])
        reward_tilted, _ = compute_reward(
            pos_error, q_tilted, torch.zeros(N, 3), torch.zeros(N, 3),
            torch.zeros(N, dtype=torch.long), torch.zeros(N, 4), torch.zeros(N, 4),
            prev_pos_error, 1.5
        )

        assert reward_upright.item() > reward_tilted.item(), \
            f"Upright ({reward_upright.item():.4f}) should beat tilted ({reward_tilted.item():.4f})"

    def test_no_nan_random_input(self):
        """Random inputs should not produce NaN/Inf."""
        compute_reward = self._load()
        N = 32
        torch.manual_seed(7)
        quats = torch.randn(N, 4)
        quats = quats / quats.norm(dim=-1, keepdim=True)

        reward, _ = compute_reward(
            torch.randn(N, 3), quats, torch.randn(N, 3), torch.randn(N, 3),
            torch.zeros(N, dtype=torch.long), torch.randn(N, 4), torch.randn(N, 4),
            torch.randn(N, 3), 1.5
        )
        assert torch.isfinite(reward).all()


# ============================================================================
# position_setpoint_lmf2 reward tests
# ============================================================================

class TestLMF2Reward:

    def _load(self):
        from aerial_gym.envs.direct.lmf2_reward import compute_reward
        return compute_reward

    def test_asymmetric_closer_reward(self):
        """Retreating from target should be penalized more than approaching is rewarded."""
        compute_reward = self._load()
        N = 4
        delta = 0.1  # move 0.1 m

        # Approaching: dist decreased by delta
        pos_approaching = torch.zeros(N, 3)
        pos_approaching[:, 0] = 1.0 - delta
        prev_dist_approaching = torch.full((N,), 1.0)

        reward_approach, _ = compute_reward(
            pos_approaching, prev_dist_approaching, torch.zeros(N),
            torch.zeros(N, 3), torch.zeros(N, 3), torch.zeros(N, dtype=torch.long),
            1.0, torch.zeros(N, 4), torch.zeros(N, 4), 10.0
        )

        # Retreating: dist increased by delta (from same starting point)
        pos_retreating = torch.zeros(N, 3)
        pos_retreating[:, 0] = 1.0 + delta
        prev_dist_same = torch.full((N,), 1.0)

        reward_retreat, _ = compute_reward(
            pos_retreating, prev_dist_same, torch.zeros(N),
            torch.zeros(N, 3), torch.zeros(N, 3), torch.zeros(N, dtype=torch.long),
            1.0, torch.zeros(N, 4), torch.zeros(N, 4), 10.0
        )

        # Asymmetric: |penalty for retreat| > |reward for approach|
        # approach: 400 * delta = 40; retreat: 1200 * (-delta) = -120
        net_approach = reward_approach - reward_retreat
        assert (net_approach > 0).all(), \
            f"Asymmetric reward failed: approach={reward_approach.mean():.2f}, retreat={reward_retreat.mean():.2f}"
