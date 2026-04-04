"""
Training script for PositionSetpointEnv using Isaac Lab + sample-factory.

Usage:
    /home/cow_server01/pg-dev/isaacsim/python.sh \
        aerial_gym/rl_training/isaaclab/train_position_setpoint.py \
        --num_envs 4096 \
        --headless

IMPORTANT: This script uses the Isaac Lab AppLauncher pattern.
           SimulationApp MUST be started before any isaacsim.core imports.
"""

# ===========================================================================
# Step 1: Launch SimulationApp FIRST (Omniverse bootstrap requirement)
# ===========================================================================
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train position setpoint task with Isaac Lab")
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=1000, help="Training iterations")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ===========================================================================
# Step 2: All other imports AFTER SimulationApp is up
# ===========================================================================
import torch
import gymnasium as gym

from aerial_gym.envs.direct.position_setpoint_env import (
    PositionSetpointEnv,
    PositionSetpointEnvCfg,
)


def main():
    # Build env config
    cfg = PositionSetpointEnvCfg()
    cfg.scene.num_envs = args.num_envs

    # Create environment
    env = PositionSetpointEnv(cfg=cfg, render_mode=None)

    print(f"[INFO] Environment created: {env.num_envs} envs")
    print(f"[INFO] Action space: {env.single_action_space}")
    print(f"[INFO] Observation space: {env.single_observation_space}")

    # Simple random policy loop to validate the env works
    obs, _ = env.reset()
    print(f"[INFO] Reset done. Obs keys: {list(obs.keys())}")
    print(f"[INFO] Policy obs shape: {obs['policy'].shape}")

    for step in range(100):
        actions = torch.rand(
            (env.num_envs, env.single_action_space.shape[0]),
            device=env.device,
        ) * 2 - 1  # uniform in [-1, 1]

        obs, rewards, terminated, truncated, info = env.step(actions)

        if step % 10 == 0:
            print(
                f"[Step {step:4d}] "
                f"mean_reward={rewards.mean():.3f}  "
                f"crashes={terminated.sum().item()}  "
                f"timeouts={truncated.sum().item()}"
            )

    print("[INFO] Validation complete. Environment works correctly.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
