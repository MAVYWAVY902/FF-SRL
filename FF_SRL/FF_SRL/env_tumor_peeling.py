"""
Gymnasium environment for tumor peeling with adhesion bonds.

Task: Control laparoscope pull direction to peel tumor from bone
      by breaking adhesion bonds, while minimizing tissue damage.

Action:  3D pull direction (dx, dy, dz), continuous [-1, 1]
Obs:     [tip_x, tip_y, tip_z, bonds_ratio, max_deformation]
Reward:  +new_broken_bonds - lambda * deformation_increase
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


class TumorPeelingEnv(gym.Env):
    """Single-env tumor peeling environment for stable-baselines3."""

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
        bond_distance=0.25,    # cm - bond contacting surface
        adhesion_d_contact=0.03,
        adhesion_d_rest=0.32,
        adhesion_d_neutral_start=0.5,
        adhesion_break_ratio=1.27,
        adhesion_stretch_abs_min=0.5,
        adhesion_alpha=1e-6,
        # Grab
        grab_radius=0.5,       # cm
        # Task
        target_break_ratio=0.6,
        max_steps=100,
        # Reward
        reward_break_weight=1.0,
        reward_deform_penalty=0.01,
    ):
        super().__init__()

        self.device = device
        self.max_steps = max_steps
        self.target_break_ratio = target_break_ratio
        self.action_strength = action_strength
        self.grab_radius = grab_radius
        self.reward_break_weight = reward_break_weight
        self.reward_deform_penalty = reward_deform_penalty

        # Store adhesion params for model creation
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

        # Action space: 3D direction (normalized internally)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Obs: [tip_x, tip_y, tip_z, bonds_alive_ratio, max_deformation_cm]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        # Initialize simulation
        self._init_sim()

    def _init_sim(self):
        """Create simulation model, integrator, adhesion bonds."""
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

        # Fix physics stiffness to match C++ codebase (E=8e4 Pa, ν=0.45)
        # Stable Neo-Hookean constraints (per-element α = 1/(μ*V₀*dt²)):
        # FF-SRL uses cm units with unit masses, so μ/λ need to be scaled up.
        # C++ uses μ≈27.6kPa, λ≈248kPa in meters. In cm with unit mass, we scale.
        self.simModel.globalMu = 1e7
        self.simModel.globalLambda = 5e7
        # Re-enable distance constraints as edge-level stretch protection
        self.simModel.globalDistanceCompliance = 0.1
        self.simModel.globalVolumeCompliance = 0.1
        # Velocity damping per substep (matches C++ damping-multiplier behavior)
        self.simModel.velocityDamping = 0.9

        # Create adhesion bonds
        self.simModel.createRigidAdhesionFromMeshes(
            bondDistance=self._bond_distance)
        self.total_bonds = self.simModel.numRigidAdhesionBonds

        # Integrator
        self.simIntegrator = dk.SimIntegratorDO(self.device)

        # Find grab point: topmost free vertex
        verts = self.simModel.vertex.numpy()
        inv_mass = self.simModel.inverseMass.numpy()
        free_indices = np.where(inv_mass > 0.0)[0]
        free_verts = verts[free_indices]
        top_local = np.argmax(free_verts[:, 1])
        self._grab_vertex_idx = int(free_indices[top_local])
        self._grab_pos = verts[self._grab_vertex_idx].copy()

        # Store initial vertex positions for deformation calculation
        self._initial_verts = self.simModel.initialVertex.numpy().copy()

        # Move laparoscope to grab point and grab
        self._position_and_grab()

        # Save post-grab state as the "initial" state for resets
        # (store the warp arrays we need to restore)
        self._post_grab_vertex = self.simModel.vertex.numpy().copy()

    def _position_and_grab(self):
        """Move laparoscope tip to grab point and clamp region."""
        target = torch.tensor(
            [self._grab_pos[0], self._grab_pos[1] - 0.5, self._grab_pos[2]],
            dtype=torch.float32, device=self.device)
        lap_pos = self.simModel.getLaparoscopePositionsTensor()
        delta = target - lap_pos[0]
        action = torch.tensor(
            [delta[0].item(), delta[1].item(), delta[2].item()],
            dtype=torch.float32, device=self.device)
        self.simModel.applyCartesianActions(wp.from_torch(action))

        envs = torch.ones(1, dtype=torch.float32, device=self.device)
        self.simModel.forceLaparoscopeClampRegion(
            self._grab_vertex_idx, self.grab_radius, envs,
            on=1.0, animate=True)

    def _get_obs(self):
        """Get observation: [tip_xyz, bonds_ratio, max_deformation]."""
        # Tip position
        lap_pos = self.simModel.getLaparoscopePositionsTensor()
        tip = lap_pos[0].cpu().numpy()  # shape (3,)

        # Bonds alive ratio
        active = self.simModel.rigidAdhesionActive.numpy()
        alive_ratio = float(np.sum(active > 0.5)) / max(self.total_bonds, 1)

        # Max deformation (cm)
        verts = self.simModel.vertex.numpy()
        max_deform = float(np.max(np.linalg.norm(
            verts - self._initial_verts, axis=1)))

        obs = np.array([
            tip[0], tip[1], tip[2],
            alive_ratio,
            max_deform,
        ], dtype=np.float32)

        return obs

    def _get_broken_count(self):
        active = self.simModel.rigidAdhesionActive.numpy()
        return self.total_bonds - int(np.sum(active > 0.5))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation state
        self.simModel.resetFixedModel([0])

        # Re-grab
        self._position_and_grab()

        # Settle with aggressive damping to suppress oscillation
        original_damping = self.simModel.velocityDamping
        self.simModel.velocityDamping = 0.5  # strong damping for settle
        for _ in range(50):
            self.simModel.resetCollisionInfo()
            self.simIntegrator.stepModel(self.simModel)
        self.simModel.velocityDamping = original_damping

        self._step_count = 0
        self._prev_broken = self._get_broken_count()
        self._prev_max_deform = 0.0

        obs = self._get_obs()
        self._initial_tip_y = obs[1]  # remember initial tip Y for penetration penalty
        info = {"broken": self._prev_broken, "total_bonds": self.total_bonds}

        return obs, info

    def step(self, action):
        self._step_count += 1

        # Scale and apply action
        action = np.clip(action, -1.0, 1.0)
        scaled = action * self.action_strength
        action_tensor = torch.tensor(scaled, dtype=torch.float32,
                                     device=self.device)
        self.simModel.applyCartesianActions(wp.from_torch(action_tensor))

        # Step physics
        self.simModel.resetCollisionInfo()
        self.simIntegrator.stepModel(self.simModel)

        # Get new state
        obs = self._get_obs()
        broken_now = self._get_broken_count()
        new_breaks = broken_now - self._prev_broken
        max_deform = obs[4]  # max_deformation from obs
        deform_increase = max(0.0, max_deform - self._prev_max_deform)

        # Penalize downward movement (pushing into bone)
        verts = self.simModel.vertex.numpy()
        tip_y = obs[1]  # current tip Y
        tip_y_drop = max(0.0, self._initial_tip_y - tip_y)  # how far below start

        # Reward: break bonds (+) but penalize deformation and downward push
        reward = (self.reward_break_weight * new_breaks
                  - self.reward_deform_penalty * deform_increase
                  - 2.0 * tip_y_drop)  # penalize pushing into bone

        # Check NaN
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
        }

        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        pass
