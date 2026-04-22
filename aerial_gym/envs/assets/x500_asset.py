"""
Isaac Lab ArticulationCfg for the x500 quadrotor.

Robot structure (5 links, all joints fixed):
    base_link (1.4 kg) — root
    front_right_prop (0.064 kg) — child of base_link
    back_right_prop  (0.064 kg) — child of base_link
    back_left_prop   (0.064 kg) — child of base_link
    front_left_prop  (0.064 kg) — child of base_link

Total mass: 1.656 kg
Hover thrust per motor: 1.656 × 9.81 / 4 ≈ 4.06 N
Max thrust per motor: 20 N (5× headroom)

Isaac Lab breadth-first body order:
    body 0: base_link
    body 1: front_right_prop
    body 2: back_right_prop
    body 3: back_left_prop
    body 4: front_left_prop

Motor application order (matches X500Cfg.control_allocator_config.application_mask):
    [front_left_prop, front_right_prop, back_left_prop, back_right_prop]
    → body IDs [4, 1, 3, 2]
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

_AERIAL_GYM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
_X500_URDF = os.path.join(_AERIAL_GYM_DIR, "resources", "robots", "x500", "model.urdf")


X500_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_X500_URDF,
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
        rot=(1.0, 0.0, 0.0, 0.0),  # wxyz identity
        joint_pos={},
        joint_vel={},
    ),
    actuators={},
)
"""ArticulationCfg for x500 quadrotor (resources/robots/x500/model.urdf).

Use with AerialGymBaseEnv subclasses that set:
    _motor_body_names = ["front_left_prop", "front_right_prop",
                         "back_left_prop", "back_right_prop"]
"""
