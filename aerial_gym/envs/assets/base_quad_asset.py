"""
Isaac Lab ArticulationCfg for aerial_gym's base_quadrotor.

Replaces:
    gym.load_asset(sim, asset_root, asset_file, asset_options)  # Isaac Gym

With:
    BASE_QUAD_CFG = ArticulationCfg(spawn=sim_utils.UrdfFileCfg(...))  # Isaac Lab

NOTE: Isaac Lab converts the URDF to a USD file on first run (cached in /tmp).
      The conversion preserves the link/joint hierarchy but may reorder bodies
      breadth-first instead of depth-first.

      Run tests/test_joint_ordering.py::test_isaac_lab_body_order_for_permutation
      after first launch to capture the actual body index order.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Absolute path to the quad URDF (avoids CWD dependency)
_AERIAL_GYM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
_QUAD_URDF = os.path.join(_AERIAL_GYM_DIR, "resources", "robots", "quad", "quad.urdf")


BASE_QUAD_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_QUAD_URDF,
        # Preserve fixed joints so individual motor links remain addressable
        # for per-link force application (force_application_level = "motor_link")
        fix_base=False,
        merge_fixed_joints=False,
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
        rot=(1.0, 0.0, 0.0, 0.0),  # wxyz identity (Isaac Lab convention)
    ),
    actuators={
        # All joints in quad.urdf are fixed — no DOF actuators needed.
        # Forces are applied externally via set_external_force_and_torque().
        # ImplicitActuatorCfg is required by ArticulationCfg but has no effect
        # when stiffness=damping=0 on fixed joints.
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""ArticulationCfg for aerial_gym base_quadrotor (quad.urdf).

Body structure (expected after URDF breadth-first parse in Isaac Lab):
    body 0: base_link
    body 1: arm_motor_0
    body 2: arm_motor_1
    body 3: arm_motor_2
    body 4: arm_motor_3
    body 5: motor_0
    body 6: motor_1
    body 7: motor_2
    body 8: motor_3

Isaac Gym depth-first order (pre-migration reference):
    body 0: base_link
    body 1: arm_motor_0, body 2: motor_0
    body 3: arm_motor_1, body 4: motor_1
    body 5: arm_motor_2, body 6: motor_2
    body 7: arm_motor_3, body 8: motor_3
    application_mask = [5, 6, 7, 8] → motors at every-other index

    With breadth-first (Isaac Lab), application_mask for motor_0..3 changes.
    Use robot.find_bodies("motor_.*") to get the actual indices at runtime.
    See AerialGymBaseEnv._find_motor_body_ids().
"""
