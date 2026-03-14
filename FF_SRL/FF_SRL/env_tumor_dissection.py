"""
Gymnasium environment for tumor dissection with adhesion bonds.

Task: Control dissector tip to push along tumor-bone interface,
      breaking adhesion bonds by mechanical dissection.

Unlike peeling (grasp & pull), the dissector pushes tissue away from bone,
locally separating adhesion bonds — matching real surgical debulking.

Action:  3D dissector movement (dx, dy, dz), continuous [-1, 1]
Obs:     [tip_x, tip_y, tip_z, bonds_alive_ratio, max_deformation,
          local_broken_ratio, tip_y_above_bone]
Reward:  +new_broken_bonds - λ * deformation - penalty_bone_contact
Done:    bonds_broken >= target_ratio OR max_steps reached
"""

import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

import warp as wp
from pxr import Usd
import FF_SRL as dk


class TumorDissectionEnv(gym.Env):
    """Single-env tumor dissection environment for stable-baselines3."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        device="cuda:0",
        usd_path=None,
        # Simulation
        sim_substeps=16,
        sim_fps=30,
        # Action
        action_strength=0.04,  # cm per step
        # Adhesion
        bond_distance=0.25,    # cm
        adhesion_d_contact=0.03,
        adhesion_d_rest=0.32,
        adhesion_d_neutral_start=0.5,
        adhesion_break_ratio=1.27,
        adhesion_stretch_abs_min=0.5,
        adhesion_alpha=1e-6,
        # Push
        push_radius=0.6,      # cm - radius of push influence
        push_strength=0.08,   # cm - max displacement per constraint solve
        # Task
        target_break_ratio=0.6,
        max_steps=200,
        # Reward
        reward_break_weight=1.0,
        reward_deform_penalty=0.01,
        reward_bone_contact_penalty=1.0,
    ):
        super().__init__()

        self.device = device
        self.max_steps = max_steps
        self.target_break_ratio = target_break_ratio
        self.action_strength = action_strength
        self.push_radius = push_radius
        self.push_strength = push_strength
        self.reward_break_weight = reward_break_weight
        self.reward_deform_penalty = reward_deform_penalty
        self.reward_bone_contact_penalty = reward_bone_contact_penalty

        self._adhesion_params = dict(
            globalAdhesionDContact=adhesion_d_contact,
            globalAdhesionDRest=adhesion_d_rest,
            globalAdhesionDNeutralStart=adhesion_d_neutral_start,
            globalAdhesionBreakRatio=adhesion_break_ratio,
            globalAdhesionStretchAbsMin=adhesion_stretch_abs_min,
            globalAdhesionAlpha=adhesion_alpha,
        )
        self._bond_distance = bond_distance
        self._sim_substeps = sim_substeps
        self._sim_fps = sim_fps

        if usd_path is None:
            usd_path = os.path.join(
                os.path.dirname(__file__), 'scenes', 'tumorBone.usd')
        self._usd_path = usd_path

        # Action space: 3D dissector tip movement
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Obs: [tip_x, tip_y, tip_z, bonds_alive_ratio, max_deformation,
        #       local_broken_ratio, tip_y_above_bone]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        self._init_sim()

    def _init_sim(self):
        """Create simulation model, integrator, adhesion bonds, push arrays."""
        self._stage = Usd.Stage.Open(self._usd_path)

        self.simModel = dk.SimModelDO(
            self._stage,
            numEnvs=1,
            device=self.device,
            simSubsteps=self._sim_substeps,
            simFrameRate=self._sim_fps,
            simConstraintsSteps=5,
            globalKsDistance=1.0,
            globalKsVolume=1.0,
            globalKsDrag=1.0,
            globalLaparoscopeDragLookupRadius=0.5,
            environmentGroundLevel=-100.0,
            **self._adhesion_params,
        )

        # Physics stiffness
        self.simModel.globalMu = 1e7
        self.simModel.globalLambda = 5e7
        self.simModel.globalDistanceCompliance = 0.1
        self.simModel.globalVolumeCompliance = 0.1
        self.simModel.velocityDamping = 0.9

        # Create adhesion bonds
        self.simModel.createRigidAdhesionFromMeshes(
            bondDistance=self._bond_distance)
        self.total_bonds = self.simModel.numRigidAdhesionBonds

        # Store bond rigid points (bone-side anchor) for local observation
        if self.simModel.rigidAdhesionRigidPoint is not None:
            self._bond_rigid_points = self.simModel.rigidAdhesionRigidPoint.numpy().copy()
        else:
            self._bond_rigid_points = np.zeros((0, 3), dtype=np.float32)

        # Integrator
        self.simIntegrator = dk.SimIntegratorDO(self.device)

        # Store initial vertex positions
        self._initial_verts = self.simModel.initialVertex.numpy().copy()

        # Compute bone surface Y (top of bone = bottom of interface)
        self._bone_top_y = self._compute_bone_top_y()

        # Find starting position: edge of tumor-bone interface
        self._start_pos = self._find_interface_edge()

        # Setup push mechanism arrays on simModel
        self._setup_push_arrays()

        # Position dissector at start
        self._position_dissector(self._start_pos)

    def _compute_bone_top_y(self):
        """Get the top Y coordinate of the bone surface."""
        rigidVerts = []
        for simRigid in self.simModel.simEnvironment.simRigids:
            rigidVerts.append(np.array(simRigid.vertex))
        if len(rigidVerts) == 0:
            return 0.0
        rigidVerts = np.vstack(rigidVerts)
        return float(rigidVerts[:, 1].max())

    def _find_interface_edge(self):
        """Find starting position at the edge of tumor-bone adhesion interface.

        Strategy: find the bond with the largest X (or Z) coordinate —
        this is at the periphery of the adhesion zone. Start the dissector
        slightly outside and above it.
        """
        if len(self._bond_rigid_points) == 0:
            # Fallback: top of tumor
            verts = self.simModel.vertex.numpy()
            inv_mass = self.simModel.inverseMass.numpy()
            free_idx = np.where(inv_mass > 0.0)[0]
            top_idx = free_idx[np.argmax(verts[free_idx, 1])]
            return verts[top_idx].copy()

        # Bond rigid points are on the bone surface
        bond_pts = self._bond_rigid_points[:self.simModel.numRigidAdhesionBondsPerEnv]
        centroid = bond_pts.mean(axis=0)

        # Find the bond furthest from centroid in XZ plane (edge of interface)
        xz_dist = np.sqrt((bond_pts[:, 0] - centroid[0])**2 +
                          (bond_pts[:, 2] - centroid[2])**2)
        edge_idx = np.argmax(xz_dist)
        edge_pt = bond_pts[edge_idx].copy()

        # Start slightly outside the interface edge and above bone
        direction_xz = edge_pt - centroid
        direction_xz[1] = 0.0
        norm = np.linalg.norm(direction_xz)
        if norm > 1e-6:
            direction_xz /= norm
        else:
            direction_xz = np.array([1.0, 0.0, 0.0])

        # Position: edge + small offset outward, and above bone surface
        start = edge_pt + direction_xz * 0.3  # 0.3 cm outside edge
        start[1] = self._bone_top_y + 0.2     # slightly above bone

        print(f"  Interface edge: {edge_pt}")
        print(f"  Dissector start: {start}")
        print(f"  Bone top Y: {self._bone_top_y:.2f}")

        return start

    def _setup_push_arrays(self):
        """Create warp arrays for the push constraint on simModel."""
        # Single vec3 for the tip position
        self.simModel.pushTipPos = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self.simModel.pushRadius = self.push_radius
        self.simModel.pushStrength = self.push_strength

        # Track tip position in numpy for obs
        self._tip_pos = self._start_pos.copy()

    def _position_dissector(self, pos):
        """Move the laparoscope to the given position (used for init)."""
        target = torch.tensor(
            [pos[0], pos[1], pos[2]],
            dtype=torch.float32, device=self.device)
        lap_pos = self.simModel.getLaparoscopePositionsTensor()
        delta = target - lap_pos[0]
        action = torch.tensor(
            [delta[0].item(), delta[1].item(), delta[2].item()],
            dtype=torch.float32, device=self.device)
        self.simModel.applyCartesianActions(wp.from_torch(action))

        # Update push tip position
        self._tip_pos = pos.copy()
        self._update_push_tip()

    def _update_push_tip(self):
        """Write current tip position to the GPU push array."""
        tip_np = np.array([self._tip_pos], dtype=np.float32).reshape(1, 3)
        self.simModel.pushTipPos = wp.array(
            [wp.vec3(self._tip_pos[0], self._tip_pos[1], self._tip_pos[2])],
            dtype=wp.vec3, device=self.device)

    def _get_local_broken_ratio(self, radius=1.0):
        """Fraction of bonds broken near the current tip position.

        Looks at bonds whose rigid anchor is within `radius` cm of tip.
        """
        if self.total_bonds == 0:
            return 0.0

        n_per_env = self.simModel.numRigidAdhesionBondsPerEnv
        bond_pts = self._bond_rigid_points[:n_per_env]
        active = self.simModel.rigidAdhesionActive.numpy()[:n_per_env]

        # Distance from tip to each bond anchor
        dists = np.linalg.norm(bond_pts - self._tip_pos, axis=1)
        nearby = dists < radius

        n_nearby = int(np.sum(nearby))
        if n_nearby == 0:
            return 0.0

        n_broken_nearby = int(np.sum((nearby) & (active < 0.5)))
        return n_broken_nearby / n_nearby

    def _get_obs(self):
        """Get observation vector."""
        # Tip position
        tip = self._tip_pos.copy()

        # Global bonds alive ratio
        active = self.simModel.rigidAdhesionActive.numpy()
        alive_ratio = float(np.sum(active > 0.5)) / max(self.total_bonds, 1)

        # Max deformation (cm)
        verts = self.simModel.vertex.numpy()
        max_deform = float(np.max(np.linalg.norm(
            verts - self._initial_verts, axis=1)))

        # Local broken ratio near tip
        local_broken = self._get_local_broken_ratio(radius=1.0)

        # Tip height above bone surface (negative = below bone = bad)
        tip_above_bone = tip[1] - self._bone_top_y

        obs = np.array([
            tip[0], tip[1], tip[2],
            alive_ratio,
            max_deform,
            local_broken,
            tip_above_bone,
        ], dtype=np.float32)

        return obs

    def _get_broken_count(self):
        active = self.simModel.rigidAdhesionActive.numpy()
        return self.total_bonds - int(np.sum(active > 0.5))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation state
        self.simModel.resetFixedModel([0])

        # Disable push during settle
        self.simModel.pushTipPos = None

        # Position dissector at start
        self._position_dissector(self._start_pos)

        # Settle with aggressive damping
        original_damping = self.simModel.velocityDamping
        self.simModel.velocityDamping = 0.5
        for _ in range(50):
            self.simModel.resetCollisionInfo()
            self.simIntegrator.stepModel(self.simModel)
        self.simModel.velocityDamping = original_damping

        # Re-enable push
        self._setup_push_arrays()
        self._update_push_tip()

        self._step_count = 0
        self._prev_broken = self._get_broken_count()
        self._prev_max_deform = 0.0

        obs = self._get_obs()
        info = {"broken": self._prev_broken, "total_bonds": self.total_bonds}

        return obs, info

    def step(self, action):
        self._step_count += 1

        # Scale and apply action to laparoscope
        action = np.clip(action, -1.0, 1.0)
        scaled = action * self.action_strength
        action_tensor = torch.tensor(scaled, dtype=torch.float32,
                                     device=self.device)
        self.simModel.applyCartesianActions(wp.from_torch(action_tensor))

        # Update tip position from laparoscope
        lap_pos = self.simModel.getLaparoscopePositionsTensor()
        tip_torch = lap_pos[0].cpu().numpy()
        self._tip_pos = tip_torch.copy()
        self._update_push_tip()

        # Step physics (push kernel runs inside stepModel)
        self.simModel.resetCollisionInfo()
        self.simIntegrator.stepModel(self.simModel)

        # Get new state
        obs = self._get_obs()
        broken_now = self._get_broken_count()
        new_breaks = broken_now - self._prev_broken
        max_deform = obs[4]
        deform_increase = max(0.0, max_deform - self._prev_max_deform)

        # Tip height above bone
        tip_above_bone = obs[6]  # can be negative if below bone

        # Reward
        reward = self.reward_break_weight * new_breaks

        # Penalize deformation
        reward -= self.reward_deform_penalty * deform_increase

        # Penalize going below bone surface
        if tip_above_bone < 0:
            reward -= self.reward_bone_contact_penalty * abs(tip_above_bone)

        # Check NaN
        verts = self.simModel.vertex.numpy()
        has_nan = np.any(np.isnan(verts))

        # Done conditions
        break_ratio = broken_now / max(self.total_bonds, 1)
        terminated = break_ratio >= self.target_break_ratio
        truncated = (self._step_count >= self.max_steps) or has_nan

        if terminated:
            reward += 50.0  # success bonus

        # Update state
        self._prev_broken = broken_now
        self._prev_max_deform = max_deform

        info = {
            "broken": broken_now,
            "break_ratio": break_ratio,
            "max_deformation_mm": max_deform * 10,
            "new_breaks": new_breaks,
            "step": self._step_count,
            "tip_above_bone": tip_above_bone,
        }

        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        pass
