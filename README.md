[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.1.0-green)](https://isaac-sim.github.io/IsaacLab/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5.0-green)](https://docs.isaacsim.omniverse.nvidia.com/)

# Aerial Gym Simulator — Isaac Lab Edition

High-fidelity GPU-accelerated reinforcement learning for autonomous aerial vehicles,
built on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/) (successor to Isaac Gym).

Train a hovering policy in minutes on a single GPU. Visualise the result live in the Isaac Sim GUI.

![Aerial Gym position control demo](./docs/gifs/Aerial%20Gym%20Position%20Control.gif)

> **Isaac Gym → Isaac Lab migration:** This branch has been fully ported to Isaac Lab.
> The original Isaac Gym codebase is preserved in git history.

---

## What's inside

| Component | Description |
|-----------|-------------|
| `aerial_gym/envs/direct/` | Isaac Lab `DirectRLEnv` environments |
| `aerial_gym/envs/assets/` | `ArticulationCfg` per robot (quad, x500, lmf2) |
| `aerial_gym/control/` | `ControlAllocator` — normalized actions → motor thrusts |
| `aerial_gym/rl_training/isaaclab/` | skrl PPO training + GUI play scripts |
| `aerial_gym/rl_training/isaaclab/agents/` | Per-task YAML hyperparameter configs |
| `tests/` | Unit tests for rewards + rotation utils (no SimApp needed) |

### Environments

| Environment | Obs dim | Robot | Notes |
|-------------|---------|-------|-------|
| `PositionSetpointEnv` | 13 | base quad | simplest hovering task |
| `PositionSetpointSim2RealEnv` | 17 | base quad | sensor noise, sim2real reward |
| `PositionSetpointEndToEndEnv` | 15 | base quad | rotation 6D obs |
| `PositionSetpointX500Env` | 15 | x500 (1.656 kg) | per-motor force application |
| `PositionSetpointLMF2Env` | 17 | lmf2 (1.24 kg) | base-link wrench |

---

## Requirements

| Software | Version |
|----------|---------|
| Ubuntu | 20.04 / 22.04 |
| NVIDIA driver | ≥ 525 |
| Isaac Sim | **4.5.0** |
| Isaac Lab | **2.1.0** |
| Python | 3.10 (bundled with Isaac Sim) |
| skrl | 1.4.3 |

> Isaac Gym is **not** required.

---

## Installation

### 1. Install Isaac Sim + Isaac Lab

Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).

Verify:
```bash
/path/to/isaacsim/python.sh -c "import isaaclab; print(isaaclab.__version__)"
# 0.40.5
```

### 2. Clone this repo

```bash
git clone https://github.com/tedlin0913/aerial_lab.git
cd aerial_lab
```

### 3. Install dependencies

Inside the Isaac Sim Python environment (no separate conda env needed):

```bash
/path/to/isaacsim/python.sh -m pip install -r requirements_isaaclab.txt
```

**Pinned versions used in development:**
```
skrl==1.4.3
warp-lang==1.0.0
numpy==1.26.0
scipy==1.10.1
matplotlib==3.8.4
networkx==3.4.2
pyyaml==6.0.1
```

### 4. Install aerial_gym

```bash
/path/to/isaacsim/python.sh -m pip install -e .
```

### 5. Verify

```bash
/path/to/isaacsim/python.sh -m pytest tests/test_isaaclab_rewards.py -v
# 14 passed
```

---

## Training

```bash
# Headless training — recommended for long runs
/path/to/isaacsim/python.sh \
    aerial_gym/rl_training/isaaclab/train_skrl_position_setpoint.py \
    --num_envs 4096 \
    --timesteps 5000000 \
    --headless

# Resume from checkpoint
/path/to/isaacsim/python.sh \
    aerial_gym/rl_training/isaaclab/train_skrl_position_setpoint.py \
    --checkpoint logs/skrl/position_setpoint_isaaclab/<run>/checkpoints/agent_1000000.pt \
    --num_envs 4096 --timesteps 5000000 --headless
```

Other training scripts: `train_skrl_sim2real.py`, `train_skrl_end_to_end.py`,
`train_skrl_x500.py`, `train_skrl_lmf2.py`.

---

## Visualising a trained policy

```bash
/path/to/isaacsim/python.sh \
    aerial_gym/rl_training/isaaclab/play_skrl_position_setpoint.py \
    --checkpoint logs/skrl/position_setpoint_isaaclab/<run>/checkpoints/best_agent.pt \
    --num_envs 16
```

Keep `--num_envs` at 16–32 so the GUI stays responsive. The Isaac Sim window takes
2–3 minutes to open on first launch (shader compilation).

---

## Running tests

```bash
# No simulator needed — pure PyTorch
/path/to/isaacsim/python.sh -m pytest tests/test_isaaclab_rewards.py -v

# Full test suite
/path/to/isaacsim/python.sh -m pytest tests/ -v
```

---

## Project structure

```
aerial_gym_simulator/
├── aerial_gym/
│   ├── envs/
│   │   ├── direct/                   # Isaac Lab DirectRLEnv environments
│   │   │   ├── aerial_gym_base_env.py
│   │   │   ├── position_setpoint_env.py
│   │   │   ├── position_setpoint_sim2real_env.py
│   │   │   ├── position_setpoint_end_to_end_env.py
│   │   │   ├── position_setpoint_x500_env.py
│   │   │   ├── position_setpoint_lmf2_env.py
│   │   │   ├── rotation_utils.py     # quaternion↔matrix, rotation 6D
│   │   │   ├── sim2real_reward.py    # standalone JIT reward (testable w/o SimApp)
│   │   │   ├── end_to_end_reward.py
│   │   │   └── lmf2_reward.py
│   │   └── assets/                   # ArticulationCfg per robot
│   │       ├── base_quad_asset.py
│   │       ├── x500_asset.py
│   │       └── lmf2_asset.py
│   ├── control/
│   │   └── control_allocation.py     # ControlAllocator
│   ├── config/robot_config/          # Physical parameters per robot
│   └── rl_training/isaaclab/
│       ├── train_skrl_*.py           # Training entry points
│       ├── play_skrl_*.py            # GUI inference entry points
│       └── agents/*.yaml             # PPO hyperparameter configs
├── resources/robots/                 # URDF models
│   ├── quad/model.urdf
│   ├── x500/model.urdf
│   └── lmf2/model.urdf
├── tests/
│   ├── conftest.py                   # Import hook — stubs omni.* w/o SimApp
│   └── test_isaaclab_rewards.py      # 14 unit tests
└── docs/                             # MkDocs documentation
```

---

## Key design notes

**Force application modes**

- `motor_link` — thrust applied per-motor at individual prop links (base quad, x500)
- `base_link` — full 6D wrench at root body (lmf2, for acceleration-control)

**Rotation representation**

End-to-end and x500 envs use 6D continuous rotation (first two columns of the rotation matrix) instead of quaternions, avoiding sign ambiguity and gimbal lock.

**Observation noise (sim2real / lmf2 envs)**

| Signal | Noise std |
|--------|-----------|
| Position error | 0.03 m |
| Orientation | 0.02 rad (Euler round-trip) |
| Linear velocity | 0.02 m/s |
| Angular velocity | 0.02 rad/s |

**Reward functions** are extracted into standalone torch-JIT modules (`*_reward.py`) so they can be imported and unit-tested without launching Isaac Sim.

---

## Acknowledgements

This project is a fork of the [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator) by Mihir Kulkarni, Welf Rehberg, and Kostas Alexis at the [Autonomous Robots Lab, NTNU](https://www.autonomousrobotslab.com). If you use this work in your research, please cite the original paper:

```bibtex
@ARTICLE{kulkarni2025aerial,
  author={Kulkarni, Mihir and Rehberg, Welf and Alexis, Kostas},
  journal={IEEE Robotics and Automation Letters},
  title={Aerial Gym Simulator: A Framework for Highly Parallelized Simulation of Aerial Robots},
  year={2025},
  volume={10},
  number={4},
  pages={4093-4100},
  doi={10.1109/LRA.2025.3548507}
}
```

## License

BSD 3-Clause — see [LICENSE](LICENSE).
