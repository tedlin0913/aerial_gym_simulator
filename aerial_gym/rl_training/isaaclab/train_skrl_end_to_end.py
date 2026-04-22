"""
Train PositionSetpointEndToEndEnv with skrl PPO.

Uses 6D rotation representation (no quaternion singularities) with world-frame linear velocity.

Usage:
    /home/cow_server01/pg-dev/isaacsim/python.sh \
        aerial_gym/rl_training/isaaclab/train_skrl_end_to_end.py \
        --num_envs 1024 \
        --headless \
        --timesteps 1000000

Resume training:
    ... train_skrl_end_to_end.py --checkpoint logs/skrl/.../agent_*.pt

IMPORTANT: SimulationApp bootstrap must come first.
"""

# ===========================================================================
# Step 1: Stub isaacgym BEFORE any aerial_gym import
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

parser = argparse.ArgumentParser(description="Train end-to-end position setpoint with skrl PPO")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--timesteps", type=int, default=1_000_000)
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

from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from aerial_gym.envs.direct.position_setpoint_end_to_end_env import (
    PositionSetpointEndToEndEnv,
    PositionSetpointEndToEndEnvCfg,
)


def main():
    # -----------------------------------------------------------------------
    # Build environment
    # -----------------------------------------------------------------------
    cfg = PositionSetpointEndToEndEnvCfg()
    cfg.scene.num_envs = args.num_envs

    env = PositionSetpointEndToEndEnv(cfg=cfg, render_mode=None)
    print(f"[INFO] Environment: {env.num_envs} envs | "
          f"obs={env.single_observation_space['policy'].shape} | "
          f"act={env.single_action_space.shape}", flush=True)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # -----------------------------------------------------------------------
    # Load agent config
    # -----------------------------------------------------------------------
    cfg_path = os.path.join(
        os.path.dirname(__file__), "agents", "end_to_end_skrl_ppo.yaml"
    )
    with open(cfg_path) as f:
        agent_cfg = yaml.safe_load(f)

    if args.timesteps:
        agent_cfg["trainer"]["timesteps"] = args.timesteps

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
