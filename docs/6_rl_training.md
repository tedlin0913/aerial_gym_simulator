# Reinforcement Learning

## Isaac Lab training (current)

The Isaac Lab backend uses [skrl](https://skrl.readthedocs.io/) with PPO.
All training scripts live in `aerial_gym/rl_training/isaaclab/`.

### Available environments

| Script | Environment | Obs | Robot |
|--------|-------------|-----|-------|
| `train_skrl_position_setpoint.py` | `PositionSetpointEnv` | 13 | base quad |
| `train_skrl_sim2real.py` | `PositionSetpointSim2RealEnv` | 17 | base quad + sensor noise |
| `train_skrl_end_to_end.py` | `PositionSetpointEndToEndEnv` | 15 | base quad, rot6D obs |
| `train_skrl_x500.py` | `PositionSetpointX500Env` | 15 | x500 (1.656 kg) |
| `train_skrl_lmf2.py` | `PositionSetpointLMF2Env` | 17 | lmf2 (1.24 kg) |

### Train

```bash
/path/to/isaacsim/python.sh \
    aerial_gym/rl_training/isaaclab/train_skrl_position_setpoint.py \
    --num_envs 4096 \
    --timesteps 5000000 \
    --headless
```

Checkpoints are saved to:
```
logs/skrl/<experiment_name>/<timestamp>_ppo_torch/checkpoints/
    agent_100000.pt
    agent_200000.pt
    ...
    best_agent.pt      ← highest reward seen during training
```

### Resume training

```bash
/path/to/isaacsim/python.sh \
    aerial_gym/rl_training/isaaclab/train_skrl_position_setpoint.py \
    --checkpoint logs/skrl/position_setpoint_isaaclab/<run>/checkpoints/agent_1000000.pt \
    --num_envs 4096 --timesteps 5000000 --headless
```

### Visualise in the GUI

```bash
/path/to/isaacsim/python.sh \
    aerial_gym/rl_training/isaaclab/play_skrl_position_setpoint.py \
    --checkpoint logs/skrl/position_setpoint_isaaclab/<run>/checkpoints/best_agent.pt \
    --num_envs 16
```

Use 16–32 envs for a responsive GUI. The play script:

1. Builds the same network architecture as training (from the YAML config)
2. Runs a warmup forward pass to initialize `LazyLinear` input shapes
3. Loads the checkpoint (weights + RunningStandardScaler statistics)
4. Runs the policy in eval mode (deterministic mean action)

### PPO hyperparameters

Configs are in `aerial_gym/rl_training/isaaclab/agents/`.
Key settings (same across all envs):

```yaml
learning_rate: 3.0e-04          # fixed LR (no adaptive scheduler)
rollouts: 32                    # steps per env per rollout
learning_epochs: 4
mini_batches: 4
discount_factor: 0.99
ratio_clip: 0.2
state_preprocessor: RunningStandardScaler
rewards_shaper_scale: 0.01      # scales rewards before value estimation
```

> **Note on LR schedulers:** `KLAdaptiveLR` was removed — it caused the learning rate
> to collapse to 0 within the first epoch due to KL oscillation. Use fixed LR.

---

## Reward functions

Each environment has a standalone reward module that can be imported and tested
without launching Isaac Sim:

| Module | Used by |
|--------|---------|
| `sim2real_reward.py` | `PositionSetpointSim2RealEnv`, `PositionSetpointLMF2Env` (base) |
| `end_to_end_reward.py` | `PositionSetpointEndToEndEnv`, `PositionSetpointX500Env` |
| `lmf2_reward.py` | `PositionSetpointLMF2Env` |

All reward functions are `@torch.jit.script` decorated for GPU performance.

### sim2real reward structure

```
total = pos_reward + dist_reward
      + pos_reward × (speed_reward + action_penalty + closer_reward/10)
      + action_penalty + action_difference_penalty
      + closer_reward + yaw_error_reward
```

Crash condition: `dist > 10 m` → `reward = -50`, episode terminates.

### lmf2 reward (acceleration-tuned)

Same structure but with **asymmetric `closer_reward`**:

```python
closer_reward = where(dist < prev_dist,
    400 × (prev_dist - dist),    # approaching: +reward
    1200 × (prev_dist - dist),   # retreating:  -penalty (3× heavier)
)
```

---

## Unit tests

The reward functions and `rotation_utils` are covered by 14 unit tests:

```bash
/path/to/isaacsim/python.sh -m pytest tests/test_isaaclab_rewards.py -v
```

These run in ~1.5 s with no GPU and no running simulator.

---

## Isaac Gym training (legacy)

The original Isaac Gym RL training is preserved in git history but is no longer
maintained. NVIDIA has deprecated Isaac Gym in favour of Isaac Lab.

For historical reference: the navigation policy from
[Reinforcement Learning for Collision-free Flight Exploiting Deep Collision Encoding](https://arxiv.org/abs/2402.03947)
was trained using the Isaac Gym backend:

```bash
# Legacy only — requires Isaac Gym
cd examples/dce_rl_navigation
bash run_trained_navigation_policy.sh
```

![RL for Navigation](./gifs/rl_for_navigation_example.gif)
