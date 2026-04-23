[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.1.0-green)](https://isaac-sim.github.io/IsaacLab/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5.0-green)](https://docs.isaacsim.omniverse.nvidia.com/)

# Aerial Lab

GPU-accelerated reinforcement learning for autonomous aerial vehicles, built on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

![Aerial Gym position control demo](./docs/gifs/Aerial%20Gym%20Position%20Control.gif)

---

## Quickstart

**1. Install Isaac Sim + Isaac Lab** — follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).

**2. Clone and install**

```bash
git clone https://github.com/tedlin0913/aerial_lab.git
cd aerial_lab
/path/to/isaacsim/python.sh -m pip install -r requirements_isaaclab.txt
/path/to/isaacsim/python.sh -m pip install -e .
```

**3. Train**

```bash
/path/to/isaacsim/python.sh \
    aerial_gym/rl_training/isaaclab/train_skrl_position_setpoint.py \
    --num_envs 4096 --timesteps 5000000 --headless
```

**4. Visualise**

```bash
/path/to/isaacsim/python.sh \
    aerial_gym/rl_training/isaaclab/play_skrl_position_setpoint.py \
    --checkpoint logs/skrl/position_setpoint_isaaclab/<run>/checkpoints/best_agent.pt \
    --num_envs 16
```

For full documentation, visit the [project website](https://tedlin0913.github.io/aerial_lab/).

---

## Acknowledgements

This project is a fork of the [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator) by Mihir Kulkarni, Welf Rehberg, and Kostas Alexis at the [Autonomous Robots Lab, NTNU](https://www.autonomousrobotslab.com), ported to Isaac Lab.

If you use this work in your research, please cite the original paper:

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
