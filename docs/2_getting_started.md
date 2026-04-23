# Getting Started

> **Isaac Lab edition** — This page covers installation for the Isaac Lab backend.
> The original Isaac Gym instructions are preserved in git history.

## System requirements

- Ubuntu 20.04 or 22.04
- NVIDIA GPU with driver ≥ 525
- Isaac Sim **4.5.0** ([download](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html))
- Isaac Lab **2.1.0** (installed alongside Isaac Sim)
- Python 3.10 (bundled with Isaac Sim — no separate conda env needed)

## Installation

### 1. Install Isaac Sim + Isaac Lab

Follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).

After installation, verify both are working:

```bash
/path/to/isaacsim/python.sh -c "import isaaclab; print(isaaclab.__version__)"
# Expected: 0.40.5
```

### 2. Clone the repository

```bash
git clone https://github.com/ntnu-arl/aerial_gym_simulator.git
cd aerial_gym_simulator
```

### 3. Install Python dependencies

All packages are installed into the Isaac Sim Python environment:

```bash
/path/to/isaacsim/python.sh -m pip install -r requirements_isaaclab.txt
```

The pinned dependency file (`requirements_isaaclab.txt`):

```
skrl==1.4.3
warp-lang==1.0.0
numpy==1.26.0
scipy==1.10.1
matplotlib==3.8.4
networkx==3.4.2
pyyaml==6.0.1
```

### 4. Install aerial_gym as an editable package

```bash
/path/to/isaacsim/python.sh -m pip install -e .
```

### 5. Verify the installation

Run the unit tests — these work without a GPU or a running simulator:

```bash
/path/to/isaacsim/python.sh -m pytest tests/test_isaaclab_rewards.py -v
```

Expected output:

```
14 passed in 1.55s
```

---

## Quickstart: train a hovering policy

### Train headless (recommended)

```bash
/path/to/isaacsim/python.sh \
    aerial_gym/rl_training/isaaclab/train_skrl_position_setpoint.py \
    --num_envs 4096 \
    --timesteps 5000000 \
    --headless
```

- `--num_envs 4096` — more parallel environments = faster wall-clock convergence
- `--timesteps 5000000` — 5M environment steps; expect 2–4 h on a modern GPU
- `--headless` — no GUI, full GPU to physics + training

Checkpoints are written to `logs/skrl/position_setpoint_isaaclab/<timestamp>/checkpoints/`.

### Resume training from a checkpoint

```bash
/path/to/isaacsim/python.sh \
    aerial_gym/rl_training/isaaclab/train_skrl_position_setpoint.py \
    --checkpoint logs/skrl/position_setpoint_isaaclab/<run>/checkpoints/agent_1000000.pt \
    --num_envs 4096 --timesteps 5000000 --headless
```

### Visualise the trained policy

```bash
/path/to/isaacsim/python.sh \
    aerial_gym/rl_training/isaaclab/play_skrl_position_setpoint.py \
    --checkpoint logs/skrl/position_setpoint_isaaclab/<run>/checkpoints/best_agent.pt \
    --num_envs 16
```

Keep `--num_envs` at 16–32. The Isaac Sim window takes 2–3 minutes to open on first
launch due to shader compilation — this is normal.

---

## Training other environments

| Script | Environment | Robot |
|--------|-------------|-------|
| `train_skrl_position_setpoint.py` | `PositionSetpointEnv` | base quad, 13-dim obs |
| `train_skrl_sim2real.py` | `PositionSetpointSim2RealEnv` | base quad, sensor noise |
| `train_skrl_end_to_end.py` | `PositionSetpointEndToEndEnv` | base quad, rotation 6D |
| `train_skrl_x500.py` | `PositionSetpointX500Env` | x500, 1.656 kg |
| `train_skrl_lmf2.py` | `PositionSetpointLMF2Env` | lmf2, 1.24 kg, base-link wrench |

All scripts accept the same `--num_envs`, `--timesteps`, `--headless`, `--checkpoint` flags.

---

## Common issues

### Isaac Sim GUI is unresponsive to mouse

The training loop is starving the GUI event loop. Use fewer environments:

```bash
--num_envs 32   # instead of 256 or 4096
```

For long training runs, always use `--headless`.

### First launch is very slow

Isaac Sim compiles shaders on first launch (~5–10 minutes). Subsequent launches are
faster because the cache is reused.

### `No module named 'omni.kit.commands'`

Isaac Lab modules that import the full simulation stack (e.g. `isaaclab.sim`) cannot
be imported without a running SimulationApp. The unit tests in `tests/` use a custom
import hook (`conftest.py`) to stub these modules automatically.

### Training not converging

- Increase `--num_envs` (4096 works well on a 24 GB GPU)
- Increase `--timesteps` (5M is a good starting point)
- Check reward progression in TensorBoard:
  ```bash
  tensorboard --logdir logs/skrl/position_setpoint_isaaclab --port 6006
  ```
