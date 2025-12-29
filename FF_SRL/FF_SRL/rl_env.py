"""
RL Environment Wrapper for FF-SRL Surgical Simulation

This module provides a Gym-compatible environment interface for the FF-SRL simulator,
enabling reinforcement learning training for surgical robot manipulation tasks.

Based on the paper: "FF-SRL: High Performance GPU-Based Surgical Simulation For Robot Learning"
"""

import torch
import warp as wp
wp.init()
import numpy as np
from typing import Optional, Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces

import FF_SRL as dk
from pxr import Usd, UsdGeom


class FFSRLEnv(gym.Env):
    """
    Gymnasium environment for FF-SRL surgical simulation.
    
    This environment implements the tissue retraction task where a surgical instrument
    must reach a target point on deformable tissue.
    
    Observation Space:
        - Instrument end-effector position (3D)
        - Target point position (3D)
        - Optionally: tissue state, velocities, etc.
    
    Action Space:
        - 3D Cartesian movements (dx, dy, dz) for the instrument
        - Optional: gripper open/close action
    
    Reward:
        R = wl * l + wd * d + ws * s
        where:
            l: distance between end-effector and target
            d: change in distance from previous step
            s: success flag (1 if l < threshold, 0 otherwise)
            wl=-1, wd=-10, ws=100 (default weights from paper)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        scene_path: str = None,
        num_envs: int = 1,
        vectorized: bool = False,
        device: str = "cuda:0",
        max_episode_steps: int = 200,
        success_threshold: float = 0.3,  # 3mm = 0.3cm (workspace units are in cm)
        reward_weights: Tuple[float, float, float] = (-5.0, -10.0, 100.0),
        fps: int = 30,
        sim_substeps: int = 8,
        action_strength: float = 0.1,
        workspace_low: Optional[np.ndarray] = None,
        workspace_high: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        visualize: bool = False,
        use_graph: bool = True,
        randomize_target_each_reset: bool = False,
        **kwargs
    ):
        """
        Initialize the FF-SRL RL environment.
        
        Args:
            scene_path: Path to USD scene file
            num_envs: Number of parallel environments
            device: CUDA device
            max_episode_steps: Maximum steps per episode
            success_threshold: Distance threshold for success (meters)
            reward_weights: (wl, wd, ws) reward function weights
            fps: Simulation frame rate
            sim_substeps: Number of simulation substeps per frame
            action_strength: Scaling factor for actions
            workspace_low: Lower bounds of workspace
            workspace_high: Upper bounds of workspace
            target_position: Fixed target position (if None, randomized)
            visualize: Enable visualization
            use_graph: Use CUDA graphs for optimization
        """
        super().__init__()
        
        # Handle device compatibility between PyTorch and Warp
        if isinstance(device, torch.device):
            self.torch_device = device
            self.device = str(device)  # Warp needs string
        else:
            self.device = device
            self.torch_device = torch.device(device)  # PyTorch needs torch.device
        
        self.num_envs = num_envs
        self.vectorized = vectorized
        self.max_episode_steps = max_episode_steps
        self.success_threshold = success_threshold
        self.reward_weights = reward_weights
        self.fps = fps
        self.sim_substeps = sim_substeps
        self.action_strength = action_strength
        self.visualize = visualize
        self.use_graph = use_graph
        self.randomize_target_each_reset = randomize_target_each_reset
        
        # Set default scene path if not provided
        if scene_path is None:
            import os
            scene_path = os.path.join(
                os.path.dirname(__file__), 
                "scenes", 
                "liverRetraction.usd"
            )
        self.scene_path = scene_path
        
        # Workspace bounds (default values from testLiverRetraction.py)
        # IMPORTANT: Upper Y bound must provide sufficient margin around target
        # Target Y≈11.66, need at least 3.0 margin for success threshold 0.3
        if workspace_low is None:
            workspace_low = np.array([-7.0, 0.7, -10.6], dtype=np.float32)
        if workspace_high is None:
            workspace_high = np.array([7.0, 15.0, 7.4], dtype=np.float32)  # Y: 12.0→15.0 for target reachability

        # Keep CPU copies for passing into Warp/SimModelDO (avoid .tolist() on CUDA tensors).
        self.workspace_low_np = np.array(workspace_low, dtype=np.float32)
        self.workspace_high_np = np.array(workspace_high, dtype=np.float32)

        self.workspace_low = torch.tensor(self.workspace_low_np, dtype=torch.float32, device=self.torch_device)
        self.workspace_high = torch.tensor(self.workspace_high_np, dtype=torch.float32, device=self.torch_device)
        
        # Target position (fixed or will be randomized in reset)
        # Target position
        # Paper: fixed target point on tissue surface.
        # TEMPORARY: Use reachable target [0.31, 10.11, 1.61] found via mesh analysis
        self.fixed_target = target_position is not None
        if target_position is not None:
            self.target_position = torch.tensor(target_position, dtype=torch.float32, device=self.torch_device).repeat(num_envs, 1)
        else:
            # Use analyzed reachable target (distance 2.41 from initial pos)
            reachable_target = torch.tensor([0.31, 10.11, 1.61], dtype=torch.float32, device=self.torch_device)
            self.target_position = reachable_target.repeat(num_envs, 1)
            # Original inference (distance 6.80, too far):
            # inferred = self._infer_fixed_target_from_stage()
            # self.target_position = inferred.repeat(num_envs, 1)
        
        # Episode tracking
        self.current_step = torch.zeros(num_envs, dtype=torch.int32, device=self.torch_device)
        self.episode_lengths = torch.zeros(num_envs, dtype=torch.int32, device=self.torch_device)
        self.episode_rewards = torch.zeros(num_envs, dtype=torch.float32, device=self.torch_device)
        
        # Previous distance for reward calculation
        self.prev_distance = torch.zeros(num_envs, dtype=torch.float32, device=self.torch_device)
        
        # Define observation and action spaces
        # Observation: [effector_x, effector_y, effector_z, target_x, target_y, target_z]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),  # 3 for effector pos + 3 for target pos
            dtype=np.float32
        )
        
        # Action: [dx, dy, dz] continuous Cartesian movements
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Cached tensors for action clipping (policy outputs may be unbounded).
        self._action_low_t = torch.tensor(self.action_space.low, dtype=torch.float32, device=self.torch_device)
        self._action_high_t = torch.tensor(self.action_space.high, dtype=torch.float32, device=self.torch_device)
        
        # Initialize simulation
        self._init_simulation()

        # By default, this class behaves like a single-env Gym env.
        # For batched training, `VectorizedFFSRLEnv` constructs this with `vectorized=True`.
        if self.num_envs != 1 and not self.vectorized:
            raise ValueError(
                "FFSRLEnv is single-env by default (num_envs=1). "
                "Use VectorizedFFSRLEnv for num_envs>1."
            )
        
    def _init_simulation(self):
        """Initialize the FF-SRL simulation model."""
        self.dt = 1.0 / float(self.fps)
        self.sim_dt = self.dt / float(self.sim_substeps)
        
        # Load USD stage
        self.stage = Usd.Stage.Open(self.scene_path)
        
        # Create simulation model
        self.simModel = dk.SimModelDO(
            self.stage,
            self.num_envs,
            self.device,
            globalKsDrag=1.0,
            simFrameRate=self.fps,
            simConstraintsSteps=1,
            simSubsteps=self.sim_substeps,
            globalLaparoscopeDragLookupRadius=1.0,
            globalKsDistance=1.0,
            globalKsVolume=1.0
            ,workspaceLow=self.workspace_low_np.tolist()
            ,workspaceHigh=self.workspace_high_np.tolist()
        )
        
        # Create BVH for collision detection
        self.simBVH = dk.SimBVH(
            self.device,
            self.simModel,
            binsNumber=8,
            load=True,
            save=False,
            path=self.scene_path.replace(".usd", ".npz")
        )
        
        # Create integrator
        self.simIntegrator = dk.SimIntegratorDO(self.device)
        
        # Setup remote center of motion (RCM) constraint
        self.remoteCenterOfMotionPos = torch.tensor(
            [9.036183, 35.260103, 1.567807],
            dtype=torch.float32,
            device=self.torch_device
        ).repeat(self.num_envs, 1)
        
        self.remoteCenterOfMotionRot = torch.tensor(
            [0.0, 0.0, 1.0],
            dtype=torch.float32,
            device=self.torch_device
        ).repeat(self.num_envs, 1)
        
        self.simModel.setRotationCenter(
            wp.from_torch(self.remoteCenterOfMotionPos, dtype=wp.vec3),
            wp.from_torch(self.remoteCenterOfMotionRot, dtype=wp.vec3)
        )
        
        # Set initial effector position to [0, 10, 4] for RL training (paper spec)
        # The USD scene has the tool at a far port position; we need it in the workspace
        desired_initial_pos = torch.tensor([0.0, 10.0, 4.0], dtype=torch.float32, device=self.torch_device)
        current_pos = self.simModel.getLaparoscopePositionsTensor()[0]  # All envs start at same pos
        shift_to_desired = desired_initial_pos - current_pos
        self.simModel.setEffectorInitialPosition(wp.from_torch(shift_to_desired, dtype=wp.float32))
        
        # Initialize renderer if visualization enabled
        if self.visualize:
            self.renderer = dk.render.WarpRaycastRendererDO(
                device=self.device,
                simModel=self.simModel,
                resolution=256,
                cameraPos=[0.0, 50.0, 50.0],
                cameraRot=[0.0, 0.0, 0.0],
                lightPos=[0.0, 35.0, 50.0],
                lightIntensity=0.03,
                mode="gpu"
            )
            self.renderer.lookAt([0.0, 10, 0.0])
            self.renderGraph = None
        
        # CUDA graph for optimization
        self.capturedGraph = None
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        self._reset_envs(list(range(self.num_envs)))

        # Get initial observation
        obs = self._get_obs()
        
        info = self._get_info()
        
        return obs[0].cpu().numpy(), info

    def _reset_envs(self, env_ids) -> None:
        """Reset a subset of environments in-place.

        This is used by the vectorized wrapper to ensure environments don't remain
        permanently truncated after reaching `max_episode_steps`.
        """
        if not env_ids:
            return

        idx = torch.tensor(env_ids, dtype=torch.int64, device=self.torch_device)

        # Reset episode tracking for selected envs
        self.current_step.index_fill_(0, idx, 0)
        self.episode_lengths.index_fill_(0, idx, 0)
        self.episode_rewards.index_fill_(0, idx, 0.0)

        # Target handling
        if (not self.fixed_target) and self.randomize_target_each_reset:
            new_targets = self._sample_target_positions()
            self.target_position.index_copy_(0, idx, new_targets.index_select(0, idx))

        # Reset simulation state (uses initial position set in _init_simulation)
        self.simModel.resetFixedModel(env_ids)

        # Initialize prev_distance for selected envs
        effector_pos = self._get_effector_positions()
        prev = torch.norm(effector_pos - self.target_position, dim=1)
        self.prev_distance.index_copy_(0, idx, prev.index_select(0, idx))
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute (shape: [3] for single env)
        
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended (success)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.torch_device)
        action_tensor = torch.clamp(action_tensor, self._action_low_t, self._action_high_t)
        scaled_action = (action_tensor * self.action_strength).reshape(-1)
        
        # Apply action to simulation
        self.simModel.resetCollisionInfo()
        self.simModel.applyCartesianActionsInWorkspace(
            wp.from_torch(scaled_action, dtype=wp.float32)
        )
        
        # Step simulation
        if self.use_graph and self.capturedGraph is None:
            wp.capture_begin()
            self.simIntegrator.stepModel(self.simModel)
            self.capturedGraph = wp.capture_end()
        
        if self.use_graph:
            wp.capture_launch(self.capturedGraph)
        else:
            self.simIntegrator.stepModel(self.simModel)
        
        # Get new observation
        obs = self._get_obs()
        
        # Calculate reward
        reward, info = self._compute_reward()
        
        # Update step counter
        self.current_step += 1
        self.episode_rewards += reward
        
        # Check termination conditions
        terminated = info["success"][0]
        
        # No early stopping - let episodes run full length
        # Early stopping was causing reward hacking (agent learned to flee quickly)
        truncated = (self.current_step >= self.max_episode_steps)[0]
        
        return (
            obs[0].cpu().numpy(),
            reward[0].item(),
            bool(terminated.item()),
            bool(truncated.item()),
            info
        )
    
    def _get_obs(self) -> torch.Tensor:
        """Get current observation for all environments."""
        # Get effector positions (shape: [num_envs, 3])
        effector_pos = self._get_effector_positions()
        
        # Concatenate effector and target positions
        obs = torch.cat([effector_pos, self.target_position], dim=1)
        
        return obs
    
    def _get_effector_positions(self) -> torch.Tensor:
        """Extract end-effector positions from simulation."""
        # Get laparoscope observations which include base and tip positions
        laparoscope_obs = self.simModel.getModelObservationsTensor()
        
        # Reshape to [num_envs, num_laparoscopes, 2, 3]
        # Assuming 1 laparoscope per env: [num_envs, 6] -> [num_envs, 2, 3]
        reshaped = laparoscope_obs.view(self.num_envs, 2, 3)
        
        # Extract tip positions (index 1)
        effector_pos = reshaped[:, 1, :]
        
        return effector_pos
    
    def _compute_reward(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute reward based on distance to target.
        
        NEW APPROACH: Dense shaping reward to prevent reward hacking.
        Instead of penalty-based reward that encourages early termination,
        use potential-based shaping: reward = -distance^2 + large_bonus_on_success
        
        This makes getting closer ALWAYS better than staying far or fleeing.
        """
        # Get current effector position
        effector_pos = self._get_effector_positions()
        
        # Calculate distance to target
        distance = torch.norm(effector_pos - self.target_position, dim=1)
        
        # Check success
        success = distance < self.success_threshold
        success_flag = success.float()
        
        # Dense shaping reward: -distance^2 heavily penalizes being far
        # Squared distance makes fleeing MUCH more expensive than staying close
        # Range: [-2.41^2, 0] = [-5.8, 0] normally, worse if fleeing
        distance_penalty = -(distance ** 2)
        
        # Large success bonus (needs to overcome max possible distance penalty)
        # If agent flees to workspace boundary (~15 units), penalty is -225
        # Success bonus must dominate: set to +300
        success_bonus = 300.0 * success_flag
        
        # Continuous progress bonus: extra reward for getting closer
        distance_change = distance - self.prev_distance
        progress_bonus = torch.where(
            distance_change < -0.01,  # Improved by >1cm
            torch.tensor(10.0, device=distance.device),  # Strong positive
            torch.where(
                distance_change > 0.01,  # Got worse by >1cm  
                torch.tensor(-10.0, device=distance.device),  # Strong negative
                torch.tensor(0.0, device=distance.device)  # No change
            )
        )
        
        # Final reward: dense shaping + success bonus + progress bonus
        reward = distance_penalty + success_bonus + progress_bonus
        
        # Update previous distance
        self.prev_distance = distance.clone()
        
        info = {
            "success": success,
            "distance_to_target": distance,
            "distance_change": distance_change,
            "effector_position": effector_pos,
        }
        
        return reward, info

    def _infer_fixed_target_from_stage(self) -> torch.Tensor:
        """Infer a deterministic fixed target point from the USD stage.

        The paper uses a fixed target point on the tissue surface. The provided scenes
        don't expose an explicit 'target' prim, so we infer it from the tissue mesh:
        pick a point on the upper surface near the mesh center.

        Returns:
            Tensor [3] on self.device.
        """
        stage = Usd.Stage.Open(self.scene_path)
        mesh_prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        if not mesh_prims:
            return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.torch_device)

        def is_tool_mesh(prim) -> bool:
            path = str(prim.GetPath()).lower()
            return "laparoscope" in path or "instrument" in path or "dvrk" in path

        # Prefer a mesh under a Tissue hierarchy; otherwise pick the largest non-tool mesh.
        tissue_candidates = [
            p for p in mesh_prims
            if "/tissue" in str(p.GetPath()).lower() and not is_tool_mesh(p)
        ]
        candidates = tissue_candidates if tissue_candidates else [p for p in mesh_prims if not is_tool_mesh(p)]
        if not candidates:
            candidates = mesh_prims

        def num_points(prim) -> int:
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
            return len(pts)

        tissue_prim = max(candidates, key=num_points)
        mesh = UsdGeom.Mesh(tissue_prim)
        pts = mesh.GetPointsAttr().Get() or []
        if not pts:
            return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.torch_device)

        xf = UsdGeom.Xformable(tissue_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world_pts = np.array([xf.Transform(p) for p in pts], dtype=np.float32)

        center_xz = world_pts[:, [0, 2]].mean(axis=0)
        y = world_pts[:, 1]
        top_k = min(50, len(world_pts))
        top_ids = np.argpartition(y, -top_k)[-top_k:]
        xz = world_pts[top_ids][:, [0, 2]]
        best_local = int(np.argmin(np.linalg.norm(xz - center_xz[None, :], axis=1)))
        target = world_pts[top_ids[best_local]]

        return torch.tensor(target, dtype=torch.float32, device=self.torch_device)
    
    def _sample_target_positions(self) -> torch.Tensor:
        """Sample random target positions within workspace."""
        # Sample uniformly within workspace bounds
        target = torch.rand(self.num_envs, 3, device=self.torch_device)
        target = self.workspace_low + target * (self.workspace_high - self.workspace_low)
        
        # Optionally: constrain to tissue surface
        # For now, keep it simple with workspace sampling
        
        return target
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        return {
            "episode_step": self.current_step[0].item(),
            "episode_reward": self.episode_rewards[0].item(),
        }
    
    def render(self):
        """Render the environment."""
        if not self.visualize:
            return None
        
        if self.renderGraph is None:
            wp.capture_begin()
            self.renderer.renderNew(self.simBVH, self.simModel)
            self.renderGraph = wp.capture_end()
        else:
            wp.capture_launch(self.renderGraph)
        
        # Return rendered image if needed
        return None
    
    def close(self):
        """Clean up resources."""
        pass


class VectorizedFFSRLEnv(gym.Env):
    """
    Vectorized version of FF-SRL environment for efficient parallel training.
    
    This wrapper treats the num_envs dimension as proper vectorization,
    compatible with rl_games and other RL frameworks expecting vectorized envs.
    """
    
    def __init__(self, **kwargs):
        """Initialize vectorized environment."""
        num_envs = kwargs.get("num_envs", 1)
        self.num_envs = num_envs
        
        # Create base environment (multi-env simulator core)
        kwargs["vectorized"] = True
        self.env = FFSRLEnv(**kwargs)
        
        # Copy spaces from base env (but they're already vectorized internally)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self, seed=None, options=None):
        """Reset all environments."""
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Get observations for all environments
        all_obs = self.env._get_obs()
        
        return all_obs.cpu().numpy(), info
    
    def step(self, actions):
        """
        Step all environments with vectorized actions.
        
        Args:
            actions: Array of shape [num_envs, action_dim]
        """
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device=self.env.device)
        actions_tensor = torch.clamp(actions_tensor, self.env._action_low_t, self.env._action_high_t)
        scaled_actions = (actions_tensor * self.env.action_strength).reshape(-1)

        # SimModelDO expects a flat [3*num_envs] cartesian action array.
        self.env.simModel.resetCollisionInfo()
        self.env.simModel.applyCartesianActionsInWorkspace(wp.from_torch(scaled_actions, dtype=wp.float32))
        
        # Step simulation
        if self.env.use_graph and self.env.capturedGraph is None:
            wp.capture_begin()
            self.env.simIntegrator.stepModel(self.env.simModel)
            self.env.capturedGraph = wp.capture_end()
        
        if self.env.use_graph:
            wp.capture_launch(self.env.capturedGraph)
        else:
            self.env.simIntegrator.stepModel(self.env.simModel)
        
        # Get observations for all envs
        obs = self.env._get_obs()
        
        # Compute rewards for all envs
        rewards, infos = self.env._compute_reward()
        
        # Update counters
        self.env.current_step += 1
        self.env.episode_rewards += rewards
        
        # Check termination
        terminated = infos["success"]
        
        # No early stopping - let episodes run full length
        truncated = (self.env.current_step >= self.env.max_episode_steps)

        # Auto-reset done envs so training stays well-defined.
        done = torch.logical_or(terminated, truncated)
        if bool(done.any().item()):
            done_ids = done.nonzero(as_tuple=False).squeeze(-1).tolist()
            self.env._reset_envs(done_ids)
            obs = self.env._get_obs()
        
        # Convert to numpy
        obs_np = obs.cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        terminated_np = terminated.cpu().numpy()
        truncated_np = truncated.cpu().numpy()
        
        return obs_np, rewards_np, terminated_np, truncated_np, infos
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
