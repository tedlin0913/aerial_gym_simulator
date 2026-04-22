"""
AerialGymBaseEnv — Isaac Lab DirectRLEnv base class for aerial_gym tasks.

Replaces:
    EnvManager + IGE_env_manager + RobotManagerIGE  (Isaac Gym)

With:
    AerialGymBaseEnv(DirectRLEnv)                   (Isaac Lab)

Architecture:
    - Spawns the robot as an Isaac Lab Articulation
    - Populates global_tensor_dict with physics state in xyzw convention
      (all 8 existing task reward functions expect xyzw)
    - Delegates control: existing ControlAllocator + motor model handles
      action → motor thrust → force/torque conversion
    - Applies forces via articulation.set_external_force_and_torque()
    - Exposes get_obs() → global_tensor_dict (same interface as EnvManager)

Quaternion boundary:
    Isaac Lab stores root_quat_w in wxyz format (w first).
    aerial_gym math.py uses xyzw (w last, index 3).
    Conversion is done ONCE in _populate_tensor_dict() — nowhere else.

    wxyz → xyzw:  q_xyzw = torch.roll(q_wxyz, -1, dims=-1)
    xyzw → wxyz:  q_wxyz = torch.roll(q_xyzw,  1, dims=-1)
"""

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from dataclasses import MISSING

from isaaclab.utils import configclass

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("aerial_gym_base_env")


@configclass
class AerialGymBaseEnvCfg(DirectRLEnvCfg):
    """Base configuration shared by all aerial_gym Isaac Lab environments."""

    # Robot asset — must be provided by subclass cfg
    robot: ArticulationCfg = MISSING

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,          # 100 Hz physics (matches aerial_gym default sim dt)
        render_interval=2,   # decimation=2, so 50 Hz RL control rate
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Scene: overridden by subclasses for their num_envs / env_spacing
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=5.0,
        replicate_physics=True,
    )

    # RL episode length (overridden per task)
    episode_length_s: float = 5.0   # 500 steps × 0.01s = 5s (matches task_config)
    decimation: int = 2

    # Action/observation dims — overridden by subclasses
    action_space: int = 4
    observation_space: int = 13
    state_space: int = 0


class AerialGymBaseEnv(DirectRLEnv):
    """
    Base Isaac Lab environment for aerial_gym tasks.

    Subclasses implement:
        _setup_robot_cfg() → ArticulationCfg for the robot
        _setup_control_allocator() → ControlAllocator instance
        _get_rewards() → task reward (reads from self.obs_dict)
        _get_dones() → (terminated, time_out) tuple
        _build_observation_tensor() → fills self._obs_buf
    """

    cfg: AerialGymBaseEnvCfg

    def __init__(
        self,
        cfg: AerialGymBaseEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        # cfg.robot must already be set by the subclass cfg (it's a @configclass field)
        super().__init__(cfg, render_mode, **kwargs)

        self._init_global_tensor_dict()
        self._control_allocator = self._setup_control_allocator()
        self._find_motor_body_ids()

        # Action buffer
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)

        # Curriculum / episode trackers
        self.obs_dict["curriculum_level"] = torch.zeros(self.num_envs, device=self.device)
        self.obs_dict["curriculum_level_multiplier"] = torch.ones(self.num_envs, device=self.device)

        # Drag coefficient defaults — all zero (no drag) until _setup_drag_coefficients() called
        N, dev = self.num_envs, self.device
        _zeros = torch.zeros(N, 3, device=dev)
        self._linvel_lin_damp  = _zeros
        self._linvel_quad_damp = _zeros
        self._angvel_lin_damp  = _zeros
        self._angvel_quad_damp = _zeros
        # Drag force/torque buffers — (N, 1, 3); index 0 = root body slot in combined buffer
        self._root_drag_force  = torch.zeros(N, 1, 3, device=dev)
        self._root_drag_torque = torch.zeros(N, 1, 3, device=dev)

        # Combined force buffer: root body (drag) + motor bodies (thrust) in one array.
        # IMPORTANT: Isaac Lab's set_external_force_and_torque() sets has_external_wrench=False
        # when the passed tensor is all-zeros, even if the internal buffer still holds non-zero
        # values from a prior call. A second all-zero call (e.g. zero drag) would silently
        # disable all previously set motor forces. Combining into one call avoids this.
        # Layout: slot 0 = root body (drag), slots 1..4 = motors.
        num_force_bodies = 1 + len(self._motor_body_ids)
        self._all_force_body_ids = [0] + list(self._motor_body_ids)
        self._combined_forces  = torch.zeros(N, num_force_bodies, 3, device=dev)
        self._combined_torques = torch.zeros(N, num_force_bodies, 3, device=dev)

        logger.info(
            f"AerialGymBaseEnv: {self.num_envs} envs, "
            f"motor body IDs={self._motor_body_ids}, "
            f"device={self.device}"
        )

    # =========================================================================
    # Override in subclasses
    # =========================================================================

    def _setup_control_allocator(self):
        """Return a ControlAllocator instance. Called after super().__init__."""
        raise NotImplementedError("Subclass must implement _setup_control_allocator()")

    def _build_observation_tensor(self):
        """Fill self._obs_buf with the current observation. Called from _get_observations()."""
        raise NotImplementedError("Subclass must implement _build_observation_tensor()")

    # =========================================================================
    # Isaac Lab lifecycle
    # =========================================================================

    def _setup_scene(self):
        """Called by DirectRLEnv.__init__ — set up robot, ground, lights."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Store actions, run control allocation and drag simulation.
        Called once per RL step, BEFORE physics substeps.
        """
        self._prev_actions[:] = self._actions
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Update action tracking in obs_dict
        self.obs_dict["robot_actions"][:] = self._actions
        self.obs_dict["robot_prev_actions"][:] = self._prev_actions

        # Control allocation: action → motor thrusts → per-motor forces/torques
        # ControlAllocator uses its internal motor model (dynamics, saturation)
        self._compute_forces_from_actions(self._actions)

    def _apply_action(self) -> None:
        """
        Apply per-motor forces/torques to the articulation.
        Called once per physics substep (decimation times per RL step).
        """
        # Combine drag (root body slot 0) and motor thrust (slots 1..) into one call.
        # A single set_external_force_and_torque() call is mandatory: calling it twice
        # would overwrite has_external_wrench based on the second call's tensor (which
        # may be all-zeros for zero-drag robots), silently disabling the first call's forces.
        self._combined_forces[:, 0:1]  = self._root_drag_force   # (N, 1, 3)
        self._combined_forces[:, 1:]   = self._motor_forces       # (N, 4, 3)
        self._combined_torques[:, 0:1] = self._root_drag_torque
        self._combined_torques[:, 1:]  = self._motor_torques
        self._robot.set_external_force_and_torque(
            self._combined_forces,   # (N, 5, 3) — local body frame
            self._combined_torques,  # (N, 5, 3) — local body frame
            body_ids=self._all_force_body_ids,
            env_ids=None,
        )

    def _get_observations(self) -> dict:
        """
        Read Isaac Lab physics state, populate global_tensor_dict (xyzw convention),
        compute derived state (body-frame velocities), return observations.
        """
        self._populate_tensor_dict()
        self._update_derived_state()
        self._build_observation_tensor()
        return {"policy": self._obs_buf}

    def _get_rewards(self) -> torch.Tensor:
        """Override in subclass to compute task reward from self.obs_dict."""
        raise NotImplementedError

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Override in subclass. Return (terminated, time_out)."""
        raise NotImplementedError

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset specified environments to initial state."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Sample initial state: spawn above env origin with identity orientation
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # Small random yaw for training diversity, zero roll/pitch
        num_reset = len(env_ids)
        yaw = (torch.rand(num_reset, device=self.device) - 0.5) * (math.pi / 3)  # ±30 deg
        # Build wxyz quaternion for yaw-only rotation: q = [cos(yaw/2), 0, 0, sin(yaw/2)]
        half_yaw = yaw * 0.5
        qw = torch.cos(half_yaw)
        qx = torch.zeros_like(half_yaw)
        qy = torch.zeros_like(half_yaw)
        qz = torch.sin(half_yaw)
        default_root_state[:, 3:7] = torch.stack([qw, qx, qy, qz], dim=-1)  # wxyz

        # Spawn at ~1.5m height above env origin
        default_root_state[:, 2] = 1.5

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:] * 0.0, env_ids)

        # Reset motor model state for affected envs
        if hasattr(self, "_control_allocator"):
            self._control_allocator.reset_idx(env_ids)

        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        # Reset episode-level obs_dict buffers
        self.obs_dict["crashes"][env_ids] = 0
        self.obs_dict["truncations"][env_ids] = 0
        self.obs_dict["robot_actions"][env_ids] = 0.0
        self.obs_dict["robot_prev_actions"][env_ids] = 0.0

    # =========================================================================
    # EnvManager-compatible interface (existing tasks call these)
    # =========================================================================

    def get_obs(self) -> dict:
        """Return the global tensor dict (same interface as EnvManager.get_obs())."""
        return self.obs_dict

    @property
    def sim_steps(self) -> torch.Tensor:
        """Episode step count per env (used by existing tasks for truncation check)."""
        return self.episode_length_buf

    def post_reward_calculation_step(self) -> None:
        """
        Called by existing tasks after reward calculation.
        In DirectRLEnv, resets are handled automatically — this is a no-op bridge.
        """
        pass

    def delete_env(self) -> None:
        """Cleanup (called by task.close())."""
        pass

    # =========================================================================
    # Internal: state population and force computation
    # =========================================================================

    def _init_global_tensor_dict(self) -> None:
        """
        Allocate the global_tensor_dict tensors.
        These are standalone tensors (not views into Isaac Gym state buffers).
        They are updated each step by _populate_tensor_dict().
        """
        N = self.num_envs
        dev = self.device

        self.obs_dict = {
            # Robot state — filled by _populate_tensor_dict()
            "robot_position":          torch.zeros(N, 3,  device=dev),
            "robot_orientation":       torch.zeros(N, 4,  device=dev),  # xyzw convention
            "robot_linvel":            torch.zeros(N, 3,  device=dev),  # world frame
            "robot_angvel":            torch.zeros(N, 3,  device=dev),  # world frame
            # Derived state — filled by _update_derived_state()
            "robot_body_linvel":       torch.zeros(N, 3,  device=dev),  # body frame
            "robot_body_angvel":       torch.zeros(N, 3,  device=dev),  # body frame
            "robot_euler_angles":      torch.zeros(N, 3,  device=dev),
            "robot_vehicle_orientation": torch.zeros(N, 4, device=dev),  # xyzw
            "robot_vehicle_linvel":    torch.zeros(N, 3,  device=dev),
            # Actions
            "robot_actions":           torch.zeros(N, self.cfg.action_space, device=dev),
            "robot_prev_actions":      torch.zeros(N, self.cfg.action_space, device=dev),
            # Episode management
            "crashes":                 torch.zeros(N, dtype=torch.long, device=dev),
            "truncations":             torch.zeros(N, dtype=torch.long, device=dev),
            # Force/torque tensors — for compatibility (not used for apply in Isaac Lab path)
            "robot_force_tensor":      torch.zeros(N, 9, 3, device=dev),
            "robot_torque_tensor":     torch.zeros(N, 9, 3, device=dev),
            # Warp mesh list (navigation_task compatibility — None = no warp env)
            "CONST_WARP_MESH_ID_LIST": None,
        }

    def _populate_tensor_dict(self) -> None:
        """
        Copy current physics state from Isaac Lab ArticulationData into obs_dict.

        QUATERNION CONVENTION:
            Isaac Lab: root_quat_w = [w, x, y, z]  (wxyz, w is first)
            aerial_gym: robot_orientation = [x, y, z, w]  (xyzw, w is last)

            Conversion: xyzw = torch.roll(wxyz, -1, dims=-1)
        """
        data = self._robot.data

        # Position: world frame (no convention change needed)
        self.obs_dict["robot_position"][:] = data.root_pos_w

        # Orientation: wxyz → xyzw
        self.obs_dict["robot_orientation"][:] = torch.roll(data.root_quat_w, -1, dims=-1)

        # Linear velocity: world frame
        self.obs_dict["robot_linvel"][:] = data.root_lin_vel_w

        # Angular velocity: world frame
        self.obs_dict["robot_angvel"][:] = data.root_ang_vel_w

    def _update_derived_state(self) -> None:
        """
        Compute body-frame velocities, euler angles, vehicle orientation.
        Uses existing aerial_gym math functions (all expect xyzw convention).
        Called after _populate_tensor_dict().
        """
        from aerial_gym.utils.math import (
            quat_rotate_inverse,
            get_euler_xyz_tensor,
            vehicle_frame_quat_from_quat,
            quat_from_euler_xyz_tensor,
            ssa,
        )

        q = self.obs_dict["robot_orientation"]       # xyzw
        linvel_w = self.obs_dict["robot_linvel"]     # world frame
        angvel_w = self.obs_dict["robot_angvel"]     # world frame

        # Body-frame velocities: rotate world-frame vel by inverse of body orientation
        self.obs_dict["robot_body_linvel"][:] = quat_rotate_inverse(q, linvel_w)
        self.obs_dict["robot_body_angvel"][:] = quat_rotate_inverse(q, angvel_w)

        # Euler angles (used by some reward functions for tilt detection)
        self.obs_dict["robot_euler_angles"][:] = ssa(get_euler_xyz_tensor(q))

        # Vehicle frame: yaw-only rotation (no roll/pitch)
        vehicle_q = vehicle_frame_quat_from_quat(q)  # xyzw
        self.obs_dict["robot_vehicle_orientation"][:] = vehicle_q
        self.obs_dict["robot_vehicle_linvel"][:] = quat_rotate_inverse(vehicle_q, linvel_w)

    def _compute_forces_from_actions(self, actions: torch.Tensor) -> None:
        """
        Run control allocation to convert RL actions → per-motor forces/torques.
        Then add aerodynamic drag at the root body.
        Stores results in self._motor_forces, self._motor_torques,
        self._root_drag_force, self._root_drag_torque.
        """
        # Motor thrust: action → motor thrusts → per-motor body-frame forces
        motor_forces, motor_torques = self._control_allocator.allocate_output(
            actions, output_mode="forces"
        )
        # motor_forces: (num_envs, 4, 3) — local body frame
        self._motor_forces = motor_forces
        self._motor_torques = motor_torques

        # Aerodynamic drag at root body (body 0 / base_link)
        # From base_multirotor.simulate_drag() — uses body-frame velocities
        # populated by _update_derived_state() in the previous _get_observations().
        body_linvel = self.obs_dict["robot_body_linvel"]   # (N, 3) body frame
        body_angvel = self.obs_dict["robot_body_angvel"]   # (N, 3) body frame

        drag_force = (
            -self._linvel_lin_damp * body_linvel
            - self._linvel_quad_damp * torch.norm(body_linvel, dim=-1, keepdim=True) * body_linvel
        )
        drag_torque = (
            -self._angvel_lin_damp * body_angvel
            - self._angvel_quad_damp * body_angvel.abs() * body_angvel
        )

        # Shape: (N, 1, 3) — one root body per env
        self._root_drag_force = drag_force.unsqueeze(1)
        self._root_drag_torque = drag_torque.unsqueeze(1)

    def _setup_drag_coefficients(self, robot_cfg) -> None:
        """
        Initialize drag coefficient tensors from robot damping config.
        Must be called from subclass __init__ after super().__init__().

        Pass the robot config class (e.g., BaseQuadCfg).
        If damping config is absent or all-zeros, drag is a no-op.
        """
        N, dev = self.num_envs, self.device
        damp = getattr(robot_cfg, "damping", None)

        def _coeff(attr, default):
            val = getattr(damp, attr, default) if damp else default
            return torch.tensor(val, device=dev, dtype=torch.float32).expand(N, 3)

        self._linvel_lin_damp  = _coeff("linvel_linear_damping_coefficient",  [0.0, 0.0, 0.0])
        self._linvel_quad_damp = _coeff("linvel_quadratic_damping_coefficient", [0.0, 0.0, 0.0])
        self._angvel_lin_damp  = _coeff("angular_linear_damping_coefficient",  [0.0, 0.0, 0.0])
        self._angvel_quad_damp = _coeff("angular_quadratic_damping_coefficient",[0.0, 0.0, 0.0])

    def _find_motor_body_ids(self) -> None:
        """
        Find Isaac Lab body indices for motor_0..motor_3.
        Isaac Lab uses breadth-first body ordering (may differ from Isaac Gym's depth-first).
        """
        motor_body_ids, motor_body_names = self._robot.find_bodies(
            ["motor_0", "motor_1", "motor_2", "motor_3"]
        )
        self._motor_body_ids = motor_body_ids
        logger.info(f"Motor body IDs (Isaac Lab breadth-first order): {motor_body_ids}")
        logger.info(f"Motor body names: {motor_body_names}")
