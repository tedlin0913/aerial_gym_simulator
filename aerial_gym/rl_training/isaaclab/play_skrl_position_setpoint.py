"""
Play (visualise) a trained PositionSetpointEnv policy in the Isaac Sim GUI.

Loads a checkpoint and runs the policy deterministically with no gradient
updates.  Keep --num_envs small (16–32) so the GUI stays responsive.

Usage:
    /home/cow_server01/pg-dev/isaacsim/python.sh \
        aerial_gym/rl_training/isaaclab/play_skrl_position_setpoint.py \
        --checkpoint logs/skrl/position_setpoint_isaaclab/<run>/checkpoints/best_agent.pt \
        --num_envs 16

IMPORTANT: SimulationApp bootstrap must come first (do NOT pass --headless).
"""

import sys
from unittest.mock import MagicMock
for _m in ["isaacgym", "isaacgym.gymapi", "isaacgym.gymtorch", "isaacgym.gymutil",
           "isaacgym.torch_utils", "pytorch3d", "pytorch3d.transforms", "urdfpy"]:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualise trained position-setpoint policy")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to checkpoint .pt file (e.g. .../best_agent.pt)")
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import torch
import yaml

from skrl.utils.runner.torch import Runner
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from aerial_gym.envs.direct.position_setpoint_env import (
    PositionSetpointEnv,
    PositionSetpointEnvCfg,
)


def main():
    cfg = PositionSetpointEnvCfg()
    cfg.scene.num_envs = args.num_envs

    env = PositionSetpointEnv(cfg=cfg, render_mode="rgb_array")
    print(f"[INFO] {env.num_envs} envs | "
          f"obs={env.single_observation_space['policy'].shape} | "
          f"act={env.single_action_space.shape}", flush=True)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # Build agent from same YAML as training (guarantees matching architecture)
    cfg_path = os.path.join(os.path.dirname(__file__), "agents",
                            "position_setpoint_skrl_ppo.yaml")
    with open(cfg_path) as f:
        agent_cfg = yaml.safe_load(f)

    # Zero timesteps so Runner doesn't train
    agent_cfg["trainer"]["timesteps"] = 0

    runner = Runner(env, agent_cfg)

    print(f"[INFO] Loading checkpoint: {args.checkpoint}", flush=True)
    runner.agent.load(args.checkpoint)
    runner.agent.set_running_mode("eval")

    # Inference loop — runs until the Isaac Sim window is closed
    # NOTE: pass obs directly (not wrapped in {"states": ...}) — the skrl wrapper
    # already formats observations the way agent.act() expects them.
    obs, _ = env.reset()
    step = 0
    print("[INFO] Running. Close the Isaac Sim window to exit.", flush=True)
    while simulation_app.is_running():
        with torch.no_grad():
            actions, _, _ = runner.agent.act(obs, timestep=0, timesteps=0)
        obs, rewards, terminated, truncated, _ = env.step(actions)
        step += 1
        if step % 200 == 0:
            print(f"[INFO] step={step}  mean_reward={rewards.mean().item():.3f}", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}", flush=True)
        traceback.print_exc()
    finally:
        simulation_app.close()
