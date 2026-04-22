"""
Train PositionSetpointSim2RealEnv with skrl PPO.

Usage:
    /home/cow_server01/pg-dev/isaacsim/python.sh \
        aerial_gym/rl_training/isaaclab/train_skrl_position_setpoint.py \
        --num_envs 4096 \
        --headless \
        --timesteps 500000

Expected: reward starts near -15 (random policy), converges toward positive values
within ~200k steps if physics is correct.

Resume training:
    ... train_skrl_position_setpoint.py --checkpoint logs/skrl/.../agent_*.pt

IMPORTANT: SimulationApp bootstrap must come first.
"""

# ===========================================================================
# Step 1: Stub isaacgym BEFORE any aerial_gym import (aerial_gym/__init__.py
# imports isaacgym unconditionally)
# ===========================================================================
import sys
from unittest.mock import MagicMock
for _m in ["isaacgym", "isaacgym.gymapi", "isaacgym.gymtorch", "isaacgym.gymutil",
           "isaacgym.torch_utils", "pytorch3d", "pytorch3d.transforms", "urdfpy"]:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

# ===========================================================================
# Step 2: Launch SimulationApp (must come before any isaacsim.core imports)
# ===========================================================================
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train sim2real position setpoint with skrl PPO")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--timesteps", type=int, default=500_000)
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint .pt file")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ===========================================================================
# Step 3: All other imports after SimulationApp
# ===========================================================================
import os
from datetime import datetime

import torch
import yaml

import skrl
from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from aerial_gym.envs.direct.position_setpoint_sim2real_env import (
    PositionSetpointSim2RealEnv,
    PositionSetpointSim2RealEnvCfg,
)


def main():
    # -----------------------------------------------------------------------
    # Build environment
    # -----------------------------------------------------------------------
    cfg = PositionSetpointSim2RealEnvCfg()
    cfg.scene.num_envs = args.num_envs

    env = PositionSetpointSim2RealEnv(cfg=cfg, render_mode=None)
    print(f"[INFO] Environment: {env.num_envs} envs | "
          f"obs={env.single_observation_space['policy'].shape} | "
          f"act={env.single_action_space.shape}", flush=True)

    # Wrap for skrl (handles the Isaac Lab ↔ skrl observation key mapping)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # -----------------------------------------------------------------------
    # Load agent config
    # -----------------------------------------------------------------------
    cfg_path = os.path.join(
        os.path.dirname(__file__), "agents", "sim2real_skrl_ppo.yaml"
    )
    with open(cfg_path) as f:
        agent_cfg = yaml.safe_load(f)

    if args.timesteps:
        agent_cfg["trainer"]["timesteps"] = args.timesteps

    # Set logging directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(
        "logs", "skrl", agent_cfg["agent"]["experiment"]["directory"],
        f"{timestamp}_ppo_torch"
    )
    os.makedirs(log_dir, exist_ok=True)
    agent_cfg["agent"]["experiment"]["directory"] = os.path.abspath(
        os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    )
    agent_cfg["agent"]["experiment"]["experiment_name"] = os.path.basename(log_dir)

    print(f"[INFO] Logging to: {log_dir}", flush=True)

    # -----------------------------------------------------------------------
    # Build and run runner
    # -----------------------------------------------------------------------
    runner = Runner(env, agent_cfg)

    if args.checkpoint:
        print(f"[INFO] Resuming from checkpoint: {args.checkpoint}", flush=True)
        runner.agent.load(args.checkpoint)

    runner.run()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[ERROR] Training failed: {e}", flush=True)
        traceback.print_exc()
    finally:
        simulation_app.close()
