"""
Train PositionSetpointLMF2Env with skrl PPO.

lmf2 quadrotor (1.24 kg, 10 N max thrust per motor, base_link force application).
Same 15-dim observation as end-to-end (rotation 6D, world-frame linvel).

Usage:
    /home/cow_server01/pg-dev/isaacsim/python.sh \
        aerial_gym/rl_training/isaaclab/train_skrl_lmf2.py \
        --num_envs 1024 \
        --headless \
        --timesteps 1000000

Resume training:
    ... train_skrl_lmf2.py --checkpoint logs/skrl/.../agent_*.pt

IMPORTANT: SimulationApp bootstrap must come first.
"""

import sys
from unittest.mock import MagicMock
for _m in ["isaacgym", "isaacgym.gymapi", "isaacgym.gymtorch", "isaacgym.gymutil",
           "isaacgym.torch_utils", "pytorch3d", "pytorch3d.transforms", "urdfpy"]:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train lmf2 position setpoint with skrl PPO")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--timesteps", type=int, default=1_000_000)
parser.add_argument("--checkpoint", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
from datetime import datetime

import torch
import yaml

from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from aerial_gym.envs.direct.position_setpoint_lmf2_env import (
    PositionSetpointLMF2Env,
    PositionSetpointLMF2EnvCfg,
)


def main():
    cfg = PositionSetpointLMF2EnvCfg()
    cfg.scene.num_envs = args.num_envs

    env = PositionSetpointLMF2Env(cfg=cfg, render_mode=None)
    print(f"[INFO] Environment: {env.num_envs} envs | "
          f"obs={env.single_observation_space['policy'].shape} | "
          f"act={env.single_action_space.shape}", flush=True)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    cfg_path = os.path.join(
        os.path.dirname(__file__), "agents", "lmf2_skrl_ppo.yaml"
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
