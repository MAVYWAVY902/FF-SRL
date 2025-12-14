"""
Tool Controller for FF-SRL
Handles surgical instrument manipulation:
- Keyboard control (joint-level)
- Mouse control (Cartesian space with clutch like xpbd-tissue-sim)
- Gamepad control (analog sticks)
- Haptic device ready (placeholder for future integration)
"""

import torch
import warp as wp
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from .input_handler import InputHandler, Key, MouseButton, InputDevice


@dataclass
class ToolControlParams:
    """Tool control parameters"""
    # Keyboard control speeds
    translation_speed: float = 0.1    # Units per frame
    rotation_speed: float = 0.1       # Radians per frame
    
    # Mouse control
    mouse_sensitivity: float = 0.00005  # Like xpbd-tissue-sim VirtuosoSimulation
    scroll_speed: float = 0.5           # Z-axis movement per scroll unit
    
    # Gamepad control
    gamepad_translation_speed: float = 0.05
    gamepad_rotation_speed: float = 0.05
    
    # Workspace limits
    workspace_min: torch.Tensor = None
    workspace_max: torch.Tensor = None
    
    # Grasping
    grasp_toggle: bool = False
    grasp_radius: float = 1.0
    grasp_radius_min: float = 0.1
    grasp_radius_max: float = 5.0
    grasp_radius_increment: float = 0.1


class ToolController:
    """
    Surgical tool controller
    Provides multiple control schemes for laparoscopic instrument
    """
    
    def __init__(self, sim_model, device: str = "cuda:0"):
        """
        Initialize tool controller
        
        Args:
            sim_model: SimModelDO instance
            device: CUDA device
        """
        self.sim_model = sim_model
        self.device = device
        
        self.params = ToolControlParams()
        
        # Set workspace limits from model if available
        if hasattr(sim_model, 'workspaceHigh'):
            self.params.workspace_max = sim_model.workspaceHigh
        if hasattr(sim_model, 'workspaceLow'):
            self.params.workspace_min = sim_model.workspaceLow
        
        # Control mode
        self.control_mode = InputDevice.KEYBOARD
        self.clutch_active = False  # Space bar clutch
        
        # Current actions (accumulated per frame)
        self.cartesian_actions = torch.zeros(3, dtype=torch.float32, device=device)
        
        # Mouse tracking for clutch mode
        self.last_mouse_pos = np.array([0.0, 0.0])
        
        # Grasping state
        self.is_grasping = False
        
    def has_input(self) -> bool:
        """
        Check if there are any pending actions
        Returns True if cartesian_actions is non-zero
        """
        return torch.any(self.cartesian_actions != 0.0).item()
        
    def update_from_keyboard(self, input_handler: InputHandler):
        """
        Update tool actions from keyboard input
        Like xpbd-tissue-sim keyboard mode: Q/A, W/S, E/D, R/F
        
        Args:
            input_handler: InputHandler instance
        """
        actions = torch.zeros(3, dtype=torch.float32, device=self.device)
        
        # Translation controls (J/L = X, U/O = Y, I/K = Z)
        if input_handler.is_key_pressed(Key.J):
            actions[0] = -self.params.translation_speed
        if input_handler.is_key_pressed(Key.L):
            actions[0] = self.params.translation_speed
        if input_handler.is_key_pressed(Key.U):
            actions[1] = self.params.translation_speed
        if input_handler.is_key_pressed(Key.O):
            actions[1] = -self.params.translation_speed
        if input_handler.is_key_pressed(Key.I):
            actions[2] = -self.params.translation_speed
        if input_handler.is_key_pressed(Key.K):
            actions[2] = self.params.translation_speed
        
        # Alternative WASD + QE controls (more intuitive)
        if input_handler.is_key_pressed(Key.A):
            actions[0] = -self.params.translation_speed
        if input_handler.is_key_pressed(Key.D):
            actions[0] = self.params.translation_speed
        if input_handler.is_key_pressed(Key.W):
            actions[2] = -self.params.translation_speed  # Forward
        if input_handler.is_key_pressed(Key.S):
            actions[2] = self.params.translation_speed   # Backward
        if input_handler.is_key_pressed(Key.Q):
            actions[1] = self.params.translation_speed   # Up
        if input_handler.is_key_pressed(Key.E):
            actions[1] = -self.params.translation_speed  # Down
        
        self.cartesian_actions = actions
    
    def update_from_mouse(self, input_handler: InputHandler, camera_controller):
        """
        Update tool from mouse input (like xpbd-tissue-sim mouse mode)
        Hold Space + move mouse = translate in camera plane
        Hold Space + scroll = translate along camera Z
        
        Args:
            input_handler: InputHandler instance
            camera_controller: CameraController for getting camera axes
        """
        actions = torch.zeros(3, dtype=torch.float32, device=self.device)
        
        # Check if clutch (spacebar) is held
        space_held = input_handler.is_key_pressed(Key.SPACE)
        
        if space_held:
            # Get mouse delta
            dx, dy = input_handler.get_mouse_delta()
            
            if abs(dx) > 0.01 or abs(dy) > 0.01:  # Threshold for noise
                # Get camera axes
                right, up, forward = camera_controller._get_camera_axes()
                
                # Compute offset in camera plane
                offset = right * dx + up * dy
                offset_torch = torch.from_numpy(offset.astype(np.float32)).to(self.device)
                
                actions = offset_torch * self.params.mouse_sensitivity
            
            # Handle scroll for Z movement
            scroll_dx, scroll_dy = input_handler.get_scroll_delta()
            if abs(scroll_dy) > 0.01:
                forward_torch = torch.from_numpy(forward.astype(np.float32)).to(self.device)
                actions += forward_torch * scroll_dy * self.params.scroll_speed
        
        self.cartesian_actions = actions
        self.clutch_active = space_held
    
    def update_from_gamepad(self, input_handler: InputHandler):
        """
        Update tool from gamepad input
        Left stick = XY translation
        Right stick = Z translation + rotation
        
        Args:
            input_handler: InputHandler instance
        """
        gamepad = input_handler.gamepad_state
        actions = torch.zeros(3, dtype=torch.float32, device=self.device)
        
        # Left stick for XY
        actions[0] = gamepad.left_stick_x * self.params.gamepad_translation_speed
        actions[1] = gamepad.left_stick_y * self.params.gamepad_translation_speed
        
        # Right stick Y for Z
        actions[2] = gamepad.right_stick_y * self.params.gamepad_translation_speed
        
        # Triggers for grasp radius
        if gamepad.right_trigger > 0.1:
            self.params.grasp_radius += self.params.grasp_radius_increment
        if gamepad.left_trigger > 0.1:
            self.params.grasp_radius -= self.params.grasp_radius_increment
        
        self.params.grasp_radius = np.clip(
            self.params.grasp_radius,
            self.params.grasp_radius_min,
            self.params.grasp_radius_max
        )
        
        # Button A for grasp toggle
        if gamepad.button_a and not self.is_grasping:
            self.toggle_grasp()
        
        self.cartesian_actions = actions
    
    def toggle_grasp(self):
        """Toggle grasping on/off (like xpbd-tissue-sim left click)"""
        self.is_grasping = not self.is_grasping
        print(f"Grasping: {'ON' if self.is_grasping else 'OFF'}")
        
        if not self.is_grasping:
            # Release all vertices when grasping is turned off
            self._release_grasp()
    
    def _release_grasp(self):
        """Release all grasped vertices"""
        # Set all drag constraints to 0
        if hasattr(self.sim_model, 'activeDragConstraint'):
            import warp as wp
            zeros = wp.zeros_like(self.sim_model.activeDragConstraint)
            wp.copy(zeros, self.sim_model.activeDragConstraint)
    
    def apply_actions(self, num_envs: int = 1):
        """
        Apply accumulated actions to simulation model
        
        Args:
            num_envs: Number of environments
        """
        # If grasping is active, trigger clamp check every frame
        if self.is_grasping:
            envs = torch.tensor(
                [1] * self.sim_model.numEnvs,
                dtype=torch.int32,
                device=self.device
            )
            self.sim_model.forceLaparoscopeClamp(envs)
        
        if torch.any(self.cartesian_actions != 0.0):
            # Clamp to workspace limits if defined
            if self.params.workspace_min is not None and self.params.workspace_max is not None:
                current_pos = self.sim_model.getLaparoscopePositionsTensor()
                new_pos = current_pos + self.cartesian_actions
                
                # Check bounds
                new_pos = torch.clamp(
                    new_pos,
                    self.params.workspace_min,
                    self.params.workspace_max
                )
                self.cartesian_actions = new_pos - current_pos
            
            # cartesian_actions is shape [3] (x, y, z)
            # For multiple environments, need to repeat to get [3*num_envs] in 1D
            # e.g., [x, y, z] -> [x, y, z, x, y, z] for 2 envs
            if num_envs > 1:
                # Stack num_envs copies and flatten to 1D
                actions_to_apply = torch.cat([self.cartesian_actions] * num_envs)
            else:
                # Single environment: just use [3] directly
                actions_to_apply = self.cartesian_actions
            
            # Ensure it's 1D before converting to Warp
            actions_to_apply = actions_to_apply.flatten()
            
            # Apply to model (must be 1D array)
            self.sim_model.applyCartesianActionsInWorkspace(
                wp.from_torch(actions_to_apply)
            )
        
        # Reset actions for next frame
        self.cartesian_actions = torch.zeros(3, dtype=torch.float32, device=self.device)
    
    def reset_tool(self):
        """Reset tool to initial position"""
        envs = list(range(0, self.sim_model.numEnvs))
        self.sim_model.resetFixedModel(envs)
        self.is_grasping = False
    
    def get_current_position(self) -> torch.Tensor:
        """Get current tool tip position"""
        return self.sim_model.getLaparoscopePositionsTensor()
    
    def set_position(self, position: torch.Tensor):
        """
        Set absolute tool position
        
        Args:
            position: [x, y, z] target position
        """
        current_pos = self.get_current_position()
        shift = position - current_pos
        self.sim_model.applyCartesianActions(wp.from_torch(shift[0], dtype=wp.float32))
