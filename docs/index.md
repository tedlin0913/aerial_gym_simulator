[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.1.0-green)](https://isaac-sim.github.io/IsaacLab/) [![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5.0-green)](https://docs.isaacsim.omniverse.nvidia.com/)

# [:arl-arl-logo: Aerial Gym Simulator — Isaac Lab Edition](index.md)

Welcome to the documentation of the Aerial Gym Simulator (Isaac Lab Edition) &nbsp;&nbsp; [:fontawesome-brands-github:](https://www.github.com/tedlin0913/aerial_lab)

The Aerial Gym Simulator is a high-fidelity, GPU-accelerated simulator for training Micro Aerial Vehicle (MAV) platforms to learn to fly using reinforcement learning. This fork has been **fully ported to [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/)** (Isaac Sim 4.5.0), replacing the deprecated Isaac Gym backend. We offer aerial robot models for standard quadrotor platforms (base quad, x500, lmf2), each with a dedicated reinforcement learning environment and reward function. Policies train in minutes on a single GPU and can be visualized live in the Isaac Sim GUI.

This edition is open-source and released under the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause).


Train a hovering policy in minutes and watch it converge in the Isaac Sim GUI:

![Aerial Gym Position Control](./gifs/Aerial%20Gym%20Position%20Control.gif)

GPU-parallelized training across thousands of environments simultaneously:

![RL Training](./gifs/rl_for_position.gif)


## Features

- ??? success "**Isaac Lab Native**"
      Fully ported to [Isaac Lab 2.1.0](https://isaac-sim.github.io/IsaacLab/) — no Isaac Gym required. Uses `DirectRLEnv` for clean, minimal environment definitions with full Isaac Sim 4.5.0 compatibility.
- ??? note "**5 Ready-to-Train Environments**"
      `PositionSetpointEnv`, `PositionSetpointSim2RealEnv`, `PositionSetpointEndToEndEnv`, `PositionSetpointX500Env`, `PositionSetpointLMF2Env` — each with a dedicated PPO config and play script.
- ??? note "**skrl PPO Training**"
      Uses [skrl 1.4.3](https://skrl.readthedocs.io/) with `RunningStandardScaler` observation normalization, fixed learning rate, and deterministic eval mode. Train headless with `--num_envs 4096` for fastest convergence.
- ??? note "**Testable Reward Functions**"
      Reward functions are extracted into standalone `@torch.jit.script` modules (`sim2real_reward.py`, `end_to_end_reward.py`, `lmf2_reward.py`) that can be imported and unit-tested without launching Isaac Sim.
- ??? note "**Rotation Utilities**"
      `rotation_utils.py` provides quaternion ↔ rotation matrix ↔ ZYX Euler conversions, 6D continuous rotation representation for end-to-end policies, and small-angle wrapping — all JIT-compiled.
- ??? note "**Sim2Real Ready**"
      Observation noise (position ±0.03 m, orientation ±0.02 rad, velocity ±0.02 m/s) baked into `PositionSetpointSim2RealEnv` and lmf2. Asymmetric `closer_reward` in lmf2 penalizes retreating 3× harder than approaching.
- ??? note "**Unit Tests — No SimApp Needed**"
      14 tests covering all reward functions and rotation utilities. Run in ~1.5 s on CPU with no GPU and no running simulator, using a custom import hook that stubs `omni.*` and `isaaclab.sim` automatically.


## Why Aerial Gym Simulator — Isaac Lab Edition?

Isaac Gym was deprecated by NVIDIA. This edition migrates the full simulator to Isaac Lab, the supported successor, while keeping the same simple training interface. You get:

- **One command to train** — `train_skrl_position_setpoint.py --num_envs 4096 --headless`
- **One command to visualize** — `play_skrl_position_setpoint.py --checkpoint best_agent.pt --num_envs 16`
- **Fast iteration** — reward functions test in 1.5 s without launching the simulator
- **Three robot models** — base quad, x500 (1.656 kg), lmf2 (1.24 kg, base-link wrench control)
- **Checkpoint resume** — training interrupted? Resume from any saved checkpoint

Policies converge in 5M environment steps (~2–4 h on a modern 24 GB GPU with 4096 envs).


## Environments

| Environment | Obs | Robot | Notes |
|-------------|-----|-------|-------|
| `PositionSetpointEnv` | 13 | base quad | simplest hovering task |
| `PositionSetpointSim2RealEnv` | 17 | base quad | sensor noise, sim2real reward |
| `PositionSetpointEndToEndEnv` | 15 | base quad | rotation 6D obs |
| `PositionSetpointX500Env` | 15 | x500 (1.656 kg) | per-motor force, 6D rotation |
| `PositionSetpointLMF2Env` | 17 | lmf2 (1.24 kg) | base-link wrench, asymmetric reward |


## Quick Links

- [Installation](./2_getting_started.md/#installation)
- [Train a hovering policy](./2_getting_started.md/#quickstart-train-a-hovering-policy)
- [All environments and training scripts](./6_rl_training.md/#available-environments)
- [Reward function design](./6_rl_training.md/#reward-functions)
- [Unit tests](./6_rl_training.md/#unit-tests)
- [FAQ & Troubleshooting](./7_FAQ_and_troubleshooting.md)


## Citing

When referencing the Aerial Gym Simulator in your research, please cite the original paper:

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


## Contact

Ted Lin &nbsp;&nbsp;&nbsp;&nbsp; [GitHub](https://github.com/tedlin0913)

For bugs and feature requests, please open an [Issue](https://github.com/tedlin0913/aerial_lab/issues) on GitHub.

The original Aerial Gym Simulator was developed at the [Autonomous Robots Lab](https://www.autonomousrobotslab.com), [Norwegian University of Science and Technology (NTNU)](https://www.ntnu.no).


## Acknowledgements

This Isaac Lab edition builds on the original [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator) by Mihir Kulkarni, Welf Rehberg, and Kostas Alexis (NTNU ARL). Original work supported by RESNAV (AFOSR Award No. FA8655-21-1-7033) and SPEAR (Horizon Europe Grant Agreement No. 101119774).


## FAQs and Troubleshooting

See the [FAQ page](./7_FAQ_and_troubleshooting.md) or open an [Issue](https://github.com/tedlin0913/aerial_lab/issues) on GitHub.
