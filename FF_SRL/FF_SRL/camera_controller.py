"""
Camera Controller for FF-SRL
Provides Easy3D-style camera controls: orbit, pan, zoom
Integrates with WarpRaycastRenderer
"""

import numpy as np
import warp as wp
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CameraParams:
    """Camera parameters matching render.py Camera class"""
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # Euler angles [rx, ry, rz] in degrees
    target: np.ndarray    # Look-at target point
    distance: float       # Distance from target
    
    # Field of view
    horizontal_aperture: float = 45.0
    vertical_aperture: float = 45.0
    
    # Movement constraints
    min_distance: float = 1.0
    max_distance: float = 200.0
    
    # Sensitivity
    orbit_sensitivity: float = 0.3
    pan_sensitivity: float = 0.01
    zoom_sensitivity: float = 1.0


class CameraController:
    """
    Easy3D-style camera controller
    Supports orbit (rotate around target), pan, and zoom
    """
    
    def __init__(self, renderer):
        """
        Initialize camera controller
        
        Args:
            renderer: WarpRaycastRendererDO instance
        """
        self.renderer = renderer
        
        # Extract initial camera state
        initial_pos = np.array(renderer.camera.pos.numpy()[0])
        initial_rot = np.array([0.0, 0.0, 0.0])  # Start with zero rotation
        
        self.params = CameraParams(
            position=initial_pos.copy(),
            rotation=initial_rot.copy(),
            target=np.array([0.0, 10.0, 0.0]),  # Default target
            distance=np.linalg.norm(initial_pos - np.array([0.0, 10.0, 0.0]))
        )
        
        # Compute initial target from lookAt if available
        # For now, use default
        
    def orbit(self, delta_azimuth: float, delta_elevation: float):
        """
        Orbit camera around target (like Easy3D mouse drag)
        
        Args:
            delta_azimuth: Horizontal rotation in degrees
            delta_elevation: Vertical rotation in degrees
        """
        # Update rotation angles
        self.params.rotation[1] += delta_azimuth * self.params.orbit_sensitivity
        self.params.rotation[0] += delta_elevation * self.params.orbit_sensitivity
        
        # Clamp elevation to prevent gimbal lock
        self.params.rotation[0] = np.clip(self.params.rotation[0], -89.0, 89.0)
        
        # Compute new camera position
        self._update_position_from_orbit()
        self._apply_to_renderer()
    
    def pan(self, delta_x: float, delta_y: float, camera_plane: bool = True):
        """
        Pan camera (translate target point)
        
        Args:
            delta_x: Horizontal pan
            delta_y: Vertical pan
            camera_plane: If True, pan in camera's view plane. If False, world XY
        """
        if camera_plane:
            # Get camera right and up vectors
            right, up, forward = self._get_camera_axes()
            
            # Move target along camera right and up
            pan_vector = (right * delta_x + up * delta_y) * self.params.pan_sensitivity * self.params.distance
            self.params.target += pan_vector
            self.params.position += pan_vector
        else:
            # World-space pan
            self.params.target[0] += delta_x * self.params.pan_sensitivity
            self.params.target[2] += delta_y * self.params.pan_sensitivity
            self._update_position_from_orbit()
        
        self._apply_to_renderer()
    
    def zoom(self, delta: float, towards_cursor: bool = False):
        """
        Zoom camera (change distance to target)
        
        Args:
            delta: Zoom amount (positive = zoom in)
            towards_cursor: If True, zoom towards mouse cursor (not implemented yet)
        """
        # Update distance
        self.params.distance -= delta * self.params.zoom_sensitivity
        self.params.distance = np.clip(
            self.params.distance,
            self.params.min_distance,
            self.params.max_distance
        )
        
        # Update position
        self._update_position_from_orbit()
        self._apply_to_renderer()
    
    def look_at(self, target: np.ndarray, distance: Optional[float] = None):
        """
        Point camera at specific target
        
        Args:
            target: [x, y, z] target point
            distance: Optional distance from target (keeps current if None)
        """
        self.params.target = np.array(target)
        
        if distance is not None:
            self.params.distance = distance
        
        self._update_position_from_orbit()
        self._apply_to_renderer()
    
    def set_position(self, position: np.ndarray, target: Optional[np.ndarray] = None):
        """
        Set absolute camera position
        
        Args:
            position: [x, y, z] camera position
            target: Optional [x, y, z] target (computes rotation if provided)
        """
        self.params.position = np.array(position)
        
        if target is not None:
            self.params.target = np.array(target)
            # Compute rotation from position to target
            direction = self.params.target - self.params.position
            self.params.distance = np.linalg.norm(direction)
            direction_norm = direction / self.params.distance
            
            # Compute azimuth and elevation
            self.params.rotation[1] = np.degrees(np.arctan2(direction_norm[0], direction_norm[2]))
            self.params.rotation[0] = np.degrees(np.arcsin(-direction_norm[1]))
        
        self._apply_to_renderer()
    
    def frame_scene(self, center: np.ndarray, radius: float):
        """
        Frame camera to view entire scene
        
        Args:
            center: [x, y, z] scene center
            radius: Scene bounding sphere radius
        """
        # Set target to scene center
        self.params.target = np.array(center)
        
        # Compute distance to frame scene
        # Use vertical FOV to compute required distance
        fov_rad = np.radians(self.params.vertical_aperture)
        self.params.distance = radius / np.tan(fov_rad / 2.0) * 1.5  # 1.5x for margin
        
        self._update_position_from_orbit()
        self._apply_to_renderer()
    
    def _update_position_from_orbit(self):
        """Compute camera position from target, distance, and rotation"""
        # Convert rotation to radians
        azimuth = np.radians(self.params.rotation[1])
        elevation = np.radians(self.params.rotation[0])
        
        # Spherical to Cartesian
        x = self.params.distance * np.cos(elevation) * np.sin(azimuth)
        y = self.params.distance * np.sin(elevation)
        z = self.params.distance * np.cos(elevation) * np.cos(azimuth)
        
        # Position = target + offset
        self.params.position = self.params.target + np.array([x, y, z])
    
    def _get_camera_axes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get camera right, up, and forward vectors"""
        # Forward: from camera to target
        forward = self.params.target - self.params.position
        forward = forward / np.linalg.norm(forward)
        
        # Right: cross(forward, world_up)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        
        # Up: cross(right, forward)
        up = np.cross(right, forward)
        
        return right, up, forward
    
    def _apply_to_renderer(self):
        """Apply current camera params to renderer"""
        # Update renderer camera position
        self.renderer.camera.pos = wp.from_numpy(
            self.params.position.reshape(1, 3),
            dtype=wp.vec3,
            device=self.renderer.device
        )
        
        # Update lookAt (if renderer has this method)
        if hasattr(self.renderer, 'lookAt'):
            self.renderer.lookAt(self.params.target.tolist())
    
    def handle_mouse_orbit(self, dx: float, dy: float, button_pressed: bool):
        """
        Handle mouse drag for orbit (like Easy3D)
        
        Args:
            dx: Mouse delta X
            dy: Mouse delta Y
            button_pressed: True if mouse button is held
        """
        if button_pressed:
            self.orbit(dx, -dy)  # Invert Y for natural mouse movement
    
    def handle_mouse_pan(self, dx: float, dy: float, button_pressed: bool):
        """
        Handle mouse drag for pan
        
        Args:
            dx: Mouse delta X
            dy: Mouse delta Y
            button_pressed: True if middle mouse button is held
        """
        if button_pressed:
            self.pan(-dx, dy)  # Invert X for natural movement
    
    def handle_mouse_zoom(self, scroll_delta: float):
        """
        Handle mouse scroll for zoom
        
        Args:
            scroll_delta: Scroll wheel delta (positive = zoom in)
        """
        self.zoom(scroll_delta)
