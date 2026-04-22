"""
Isaac Lab ArticulationCfg for the lmf2 quadrotor.

Robot structure (5 links, all joints fixed):
    base_link (1.2 kg) — root
    front_left_prop  (0.01 kg) — child of base_link
    front_right_prop (0.01 kg) — child of base_link
    back_left_prop   (0.01 kg) — child of base_link
    back_right_prop  (0.01 kg) — child of base_link

Total mass: 1.24 kg
Hover thrust per motor: 1.24 × 9.81 / 4 ≈ 3.04 N
Max thrust per motor: 10 N (3.3× headroom)

Uses "base_link" force application — total wrench applied at root, not per-motor links.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

_AERIAL_GYM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
_LMF2_URDF = os.path.join(_AERIAL_GYM_DIR, "resources", "robots", "lmf2", "model.urdf")


LMF2_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_LMF2_URDF,
        fix_base=False,
        merge_fixed_joints=False,
        joint_drive=None,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={},
        joint_vel={},
    ),
    actuators={},
)
"""ArticulationCfg for lmf2 quadrotor (resources/robots/lmf2/model.urdf).

Forces are applied at the root body (base_link) via "base_link" mode in
Lmf2Cfg.control_allocator_config (not per-motor link).
"""
