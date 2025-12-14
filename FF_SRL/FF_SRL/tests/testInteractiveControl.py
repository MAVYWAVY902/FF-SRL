"""
Example: Interactive Liver Retraction with Full Input Control
Demonstrates:
- Camera control (orbit, pan, zoom with mouse)
- Tool control (keyboard WASD/QE or mouse with Space clutch)
- Gamepad support (optional)
- Easy3D-style interaction
"""

import torch
from pynput import keyboard, mouse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import numpy as np
import FF_SRL as dk

# Import new interaction modules
from FF_SRL.input_handler import InputHandler, PynputAdapter, Key, MouseButton
from FF_SRL.camera_controller import CameraController
from FF_SRL.tool_controller import ToolController

wp.init()

try:
    from pxr import Usd
except ModuleNotFoundError:
    print("No pxr package")


class InteractiveSimulation:
    """Interactive surgical simulation with full input control"""
    
    def __init__(self, num_envs=1):
        self.device = "cuda:0"
        self.num_envs = num_envs
        
        # Simulation parameters
        self.fps = 20
        self.dt = 1.0 / float(self.fps)
        self.sim_substeps = 20
        self.constraint_steps = 10
        self.sim_dt = self.dt / float(self.sim_substeps)
        
        # Workspace limits
        self.workspace_high = torch.tensor([7.0, 9.0, 7.4], dtype=torch.float, device=self.device)
        self.workspace_low = torch.tensor([-7.0, 0.7, -10.6], dtype=torch.float, device=self.device)
        
        # Initialize simulation model
        print("Loading USD scene...")
        self.stage = Usd.Stage.Open("../scenes/liverRetractionTexture.usd")
        self.sim_model = dk.SimModelDO(
            self.stage, self.num_envs, self.device,
            globalKsDrag=1.0,
            simFrameRate=self.fps,
            simConstraintsSteps=self.constraint_steps,
            simSubsteps=self.sim_substeps,
            globalLaparoscopeDragLookupRadius=1.0,
            globalKsDistance=1.0,
            globalKsVolume=1.0
        )
        
        print("Building BVH...")
        self.sim_bvh = dk.SimBVH(
            self.device, self.sim_model,
            binsNumber=8, load=True, save=False,
            path="../scenes/liverRetractionTexture.npz"
        )
        
        print("Initializing renderer...")
        self.renderer = dk.render.WarpRaycastRendererDO(
            device=self.device,
            simModel=self.sim_model,
            resolution=512,  # Higher resolution for better visuals
            cameraPos=[2.0, 22.0, 40.0],
            cameraRot=[0.0, 0.0, 0.0],
            lightPos=[0.0, 15.0, 80.0],
            lightIntensity=0.03,
            mode="gpu",
            horizontalAperture=45,
            verticalAperture=45
        )
        self.renderer.lookAt([-2.75, 22.5, -4.0])
        
        self.sim_integrator = dk.SimIntegratorDO(self.device)
        
        # Set remote center of motion
        rcm_pos = torch.tensor(
            [9.036183, 40.260103, 5.567807],
            dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        rcm_rot = torch.tensor(
            [0.0, 0.0, 0.0],
            dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        self.sim_model.setRotationCenter(
            wp.from_torch(rcm_pos, dtype=wp.vec3),
            wp.from_torch(rcm_rot, dtype=wp.vec3)
        )
        
        # Initialize tool position
        self.initial_tool_pos = torch.tensor(
            [0.0, 23.0, -2.0],
            dtype=torch.float, device=self.device
        )
        self._set_initial_tool_position()
        
        # === New Input System ===
        print("Setting up input handlers...")
        self.input_handler = InputHandler()
        self.camera_controller = CameraController(self.renderer)
        self.tool_controller = ToolController(self.sim_model, self.device)
        
        # Set workspace limits
        self.tool_controller.params.workspace_min = self.workspace_low
        self.tool_controller.params.workspace_max = self.workspace_high
        
        # Setup input callbacks
        self._setup_input_callbacks()
        
        # Control state
        self.running = True
        self.paused = False
        self.show_help = False
        self.control_mode = "keyboard"  # or "mouse" or "gamepad"
        
        # CUDA graph optimization (only for simulation, renderer has its own)
        self.sim_graph = None
        
        # Frame rate control
        self.target_fps = 60
        self.frame_time = 1.0 / self.target_fps
        self.last_frame_time = time.time()
        
        print("Initialization complete!")
        self._print_controls()
    
    def _set_initial_tool_position(self):
        """Set initial tool position"""
        current_pos = self.sim_model.getLaparoscopePositionsTensor()
        shift = self.initial_tool_pos - current_pos
        self.sim_model.applyCartesianActions(wp.from_torch(shift[0], dtype=wp.float32))
    
    def _setup_input_callbacks(self):
        """Setup keyboard shortcuts and callbacks"""
        # Quit
        self.input_handler.register_key_callback(Key.ESC, self.quit)
        
        # Pause/Resume
        self.input_handler.register_key_callback(Key.P, self.toggle_pause)
        
        # Reset
        self.input_handler.register_key_callback(Key.R, self.reset_simulation)
        
        # Grasp toggle
        self.input_handler.register_key_callback(Key.F, self.toggle_grasp)
        
        # Control mode switching
        self.input_handler.register_key_callback(Key.NUM_1, lambda: self.set_control_mode("keyboard"))
        self.input_handler.register_key_callback(Key.NUM_2, lambda: self.set_control_mode("mouse"))
        self.input_handler.register_key_callback(Key.NUM_3, lambda: self.set_control_mode("gamepad"))
        
        # Help
        self.input_handler.register_key_callback(Key.H, self.toggle_help)
        
        # Camera presets
        self.input_handler.register_key_callback(Key.NUM_7, self.camera_front_view)
        self.input_handler.register_key_callback(Key.NUM_8, self.camera_top_view)
        self.input_handler.register_key_callback(Key.NUM_9, self.camera_side_view)
        
        # Grasp radius adjustment
        self.input_handler.register_key_callback(Key.Z, self.decrease_grasp_radius)
        self.input_handler.register_key_callback(Key.X, self.increase_grasp_radius)
    
    def set_control_mode(self, mode: str):
        """Switch control mode"""
        self.control_mode = mode
        print(f"Control mode: {mode}")
    
    def toggle_pause(self):
        """Toggle simulation pause"""
        self.paused = not self.paused
        print(f"Simulation {'paused' if self.paused else 'resumed'}")
    
    def quit(self):
        """Quit simulation"""
        print("Exiting...")
        self.running = False
    
    def reset_simulation(self):
        """Reset to initial state"""
        print("Resetting simulation...")
        self.tool_controller.reset_tool()
    
    def toggle_grasp(self):
        """Toggle grasping"""
        self.tool_controller.toggle_grasp()
        # Print statement is now in tool_controller.toggle_grasp()
    
    def increase_grasp_radius(self):
        """Increase grasp radius"""
        self.tool_controller.params.grasp_radius += 0.1
        print(f"Grasp radius: {self.tool_controller.params.grasp_radius:.2f}")
    
    def decrease_grasp_radius(self):
        """Decrease grasp radius"""
        self.tool_controller.params.grasp_radius = max(
            0.1, self.tool_controller.params.grasp_radius - 0.1
        )
        print(f"Grasp radius: {self.tool_controller.params.grasp_radius:.2f}")
    
    def toggle_help(self):
        """Toggle help display"""
        self.show_help = not self.show_help
        if self.show_help:
            self._print_controls()
    
    def camera_front_view(self):
        """Set camera to front view"""
        self.camera_controller.set_position(
            np.array([0.0, 20.0, 40.0]),
            np.array([0.0, 20.0, 0.0])
        )
        print("Camera: Front view")
    
    def camera_top_view(self):
        """Set camera to top view"""
        self.camera_controller.set_position(
            np.array([0.0, 50.0, 0.0]),
            np.array([0.0, 20.0, 0.0])
        )
        print("Camera: Top view")
    
    def camera_side_view(self):
        """Set camera to side view"""
        self.camera_controller.set_position(
            np.array([40.0, 20.0, 0.0]),
            np.array([0.0, 20.0, 0.0])
        )
        print("Camera: Side view")
    
    def _print_controls(self):
        """Print control instructions"""
        print("\n" + "="*70)
        print("INTERACTIVE SIMULATION CONTROLS")
        print("="*70)
        print("\n[TOOL CONTROL]")
        print("  Keyboard Mode (1):")
        print("    WASD + QE  - Move tool (W=forward, S=back, A=left, D=right, Q=up, E=down)")
        print("    F          - Toggle grasp")
        print("    Z/X        - Decrease/increase grasp radius")
        print("\n  Mouse Mode (2):")
        print("    Hold SPACE + Move Mouse  - Translate tool in camera plane")
        print("    Hold SPACE + Scroll      - Move tool forward/backward")
        print("    Left Click               - Toggle grasp (not implemented in this demo)")
        print("\n[CAMERA CONTROL]")
        print("  Left Mouse Drag    - Orbit around target")
        print("  Middle Mouse Drag  - Pan camera")
        print("  Scroll Wheel       - Zoom in/out")
        print("  7/8/9              - Camera presets (front/top/side)")
        print("\n[SIMULATION]")
        print("  P       - Pause/Resume")
        print("  R       - Reset")
        print("  H       - Toggle this help")
        print("  ESC     - Quit")
        print("="*70 + "\n")
    
    def update_input(self):
        """Update input state (call once per frame)"""
        # Update tool control based on mode
        if self.control_mode == "keyboard":
            self.tool_controller.update_from_keyboard(self.input_handler)
        elif self.control_mode == "mouse":
            self.tool_controller.update_from_mouse(self.input_handler, self.camera_controller)
        elif self.control_mode == "gamepad":
            self.tool_controller.update_from_gamepad(self.input_handler)
        
        # Update camera from mouse ONLY if mouse actually moved/scrolled
        # (Event-driven like Easy3D, not polling every frame)
        dx, dy = self.input_handler.get_mouse_delta()
        scroll_dx, scroll_dy = self.input_handler.get_scroll_delta()
        
        # Only update camera if there's actual input
        if abs(dx) > 0.01 or abs(dy) > 0.01 or abs(scroll_dy) > 0.01:
            left_pressed = self.input_handler.is_mouse_button_pressed(MouseButton.LEFT)
            middle_pressed = self.input_handler.is_mouse_button_pressed(MouseButton.MIDDLE)
            
            # Only orbit if not in mouse control mode (to avoid conflicts)
            if left_pressed and self.control_mode != "mouse" and (abs(dx) > 0.01 or abs(dy) > 0.01):
                self.camera_controller.handle_mouse_orbit(dx, dy, True)
            
            if middle_pressed and (abs(dx) > 0.01 or abs(dy) > 0.01):
                self.camera_controller.handle_mouse_pan(dx, dy, True)
            
            # Zoom from scroll
            if abs(scroll_dy) > 0.01:
                if not (self.control_mode == "mouse" and self.input_handler.is_key_pressed(Key.SPACE)):
                    self.camera_controller.handle_mouse_zoom(scroll_dy)
        
        # Reset input deltas
        self.input_handler.reset_deltas()
    
    def step(self):
        """Simulation step"""
        if not self.paused:
            # Apply tool actions only if there's been input or grasping is active
            if (self.tool_controller.has_input() or 
                self.tool_controller.is_grasping):
                self.tool_controller.apply_actions(self.num_envs)
            
            # Physics step - use Reduce version for single environment (faster)
            if self.sim_graph is None:
                wp.capture_begin()
                if self.sim_model.reduce:
                    self.sim_integrator.stepModelReduce(self.sim_model)
                else:
                    self.sim_integrator.stepModel(self.sim_model)
                self.sim_graph = wp.capture_end()
            else:
                wp.capture_launch(self.sim_graph)
    
    def render(self):
        """Render frame"""
        self.sim_bvh.refitBVH(useGraph=True)
        
        # renderNew() handles graph capture internally - just call it
        # (it creates its own graph inside, we don't need to capture here)
        self.renderer.renderNew(self.sim_bvh, self.sim_model)
    
    def run(self, max_iterations=10000):
        """Main simulation loop"""
        print("\nStarting simulation loop...")
        
        # Start pynput listeners
        kb_listener = keyboard.Listener(
            on_press=lambda key: PynputAdapter.on_press(key, self.input_handler),
            on_release=lambda key: PynputAdapter.on_release(key, self.input_handler)
        )
        kb_listener.start()
        
        # Mouse listener for position and buttons
        mouse_listener = mouse.Listener(
            on_move=lambda x, y: self.input_handler.update_mouse_position(x, y),
            on_click=lambda x, y, button, pressed: self.input_handler.update_mouse_button(
                MouseButton(button.value), pressed
            ),
            on_scroll=lambda x, y, dx, dy: self.input_handler.update_mouse_scroll(dx, dy)
        )
        mouse_listener.start()
        
        # Main loop
        iteration = 0
        try:
            while self.running and iteration < max_iterations:
                # Frame timing
                current_time = time.time()
                delta_time = current_time - self.last_frame_time
                
                # Limit frame rate
                if delta_time < self.frame_time:
                    time.sleep(self.frame_time - delta_time)
                    current_time = time.time()
                    delta_time = current_time - self.last_frame_time
                
                self.last_frame_time = current_time
                
                # Update input
                self.update_input()
                
                # Simulation step
                self.step()
                
                # Render
                self.render()
                
                iteration += 1
                
                if iteration % 100 == 0:
                    print(f"Iteration {iteration}/{max_iterations}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            kb_listener.stop()
            mouse_listener.stop()
            print("Simulation ended")


if __name__ == "__main__":
    # Parse command line args
    num_envs = 1
    if len(sys.argv) > 1:
        num_envs = int(sys.argv[1])
    
    # Run simulation
    sim = InteractiveSimulation(num_envs=num_envs)
    sim.run(max_iterations=10000)
