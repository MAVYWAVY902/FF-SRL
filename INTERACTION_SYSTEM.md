# FF-SRL Interactive Control System

Complete input handling system for FF-SRL surgical simulation, inspired by Easy3D and xpbd-tissue-sim interaction patterns.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                 User Input Devices                  │
│  Keyboard  │  Mouse  │  Gamepad  │  Haptic (future) │
└────────────┬────────────┬──────────┬─────────────────┘
             │            │          │
             ▼            ▼          ▼
┌────────────────────────────────────────────────────┐
│              InputHandler (Unified API)            │
│  • Key states        • Mouse state                 │
│  • Callbacks         • Gamepad state               │
│  • Event routing     • Modifier tracking           │
└────────────┬───────────────────────────────────────┘
             │
        ┌────┴────┐
        ▼         ▼
┌──────────────┐  ┌──────────────┐
│   Camera     │  │     Tool     │
│  Controller  │  │  Controller  │
│              │  │              │
│ • Orbit      │  │ • Keyboard   │
│ • Pan        │  │ • Mouse      │
│ • Zoom       │  │ • Gamepad    │
│ • Presets    │  │ • Grasping   │
└──────┬───────┘  └───────┬──────┘
       │                  │
       ▼                  ▼
┌────────────────────────────────┐
│     WarpRaycastRenderer +      │
│         SimModelDO             │
└────────────────────────────────┘
```

## Components

### 1. InputHandler (`input_handler.py`)

**Purpose**: Unified input abstraction layer

**Features**:
- Device-agnostic key/button enumeration
- State tracking (pressed, released, held)
- Callback system for event-driven control
- Modifier key support (Shift, Ctrl, Alt)
- Mouse delta tracking
- Scroll wheel support

**Example**:
```python
from FF_SRL.input_handler import InputHandler, Key, MouseButton

handler = InputHandler()

# Check key state
if handler.is_key_pressed(Key.W):
    move_forward()

# Register callback
handler.register_key_callback(Key.ESC, quit_simulation)
```

### 2. CameraController (`camera_controller.py`)

**Purpose**: Easy3D-style camera manipulation

**Features**:
- **Orbit**: Rotate camera around target point
- **Pan**: Translate camera parallel to view plane
- **Zoom**: Move camera closer/farther from target
- **Look-at**: Point camera at specific target
- **Presets**: Front, top, side views
- **Frame scene**: Auto-fit camera to bounding sphere

**Mouse Controls** (Easy3D standard):
- Left drag → Orbit
- Middle drag → Pan
- Scroll wheel → Zoom

**Example**:
```python
from FF_SRL.camera_controller import CameraController

cam = CameraController(renderer)

# Orbit (like Easy3D mouse drag)
cam.orbit(delta_azimuth=5.0, delta_elevation=2.0)

# Look at liver
cam.look_at(target=[0, 20, 0], distance=30.0)

# Frame entire scene
cam.frame_scene(center=[0, 20, 0], radius=15.0)
```

### 3. ToolController (`tool_controller.py`)

**Purpose**: Surgical instrument manipulation

**Features**:
- **Keyboard control**: WASD/QE for Cartesian motion
- **Mouse control**: Space clutch + mouse (like xpbd-tissue-sim)
- **Gamepad control**: Analog sticks for smooth motion
- **Grasping**: Toggle clamp with collision detection
- **Workspace limits**: Automatic clamping to safe region
- **Haptic-ready**: Architecture supports future haptic integration

**Control Modes**:

#### Keyboard Mode
```
W/S - Forward/Backward (Z-axis)
A/D - Left/Right (X-axis)
Q/E - Up/Down (Y-axis)
F   - Toggle grasp
```

#### Mouse Mode (xpbd-tissue-sim style)
```
Hold SPACE + Move Mouse  → Translate in camera plane
Hold SPACE + Scroll      → Forward/backward
Left Click               → Toggle grasp
```

#### Gamepad Mode
```
Left Stick  → XY translation
Right Stick → Z translation
Triggers    → Grasp radius
Button A    → Toggle grasp
```

**Example**:
```python
from FF_SRL.tool_controller import ToolController

tool = ToolController(sim_model, device="cuda:0")

# Keyboard update (call per frame)
tool.update_from_keyboard(input_handler)

# Mouse update (call per frame)
tool.update_from_mouse(input_handler, camera_controller)

# Apply accumulated actions
tool.apply_actions(num_envs=1)

# Toggle grasping
tool.toggle_grasp()
```

## Usage

### Basic Example

```python
from FF_SRL.input_handler import InputHandler, PynputAdapter
from FF_SRL.camera_controller import CameraController
from FF_SRL.tool_controller import ToolController
from pynput import keyboard, mouse

# 1. Setup
input_handler = InputHandler()
camera_controller = CameraController(renderer)
tool_controller = ToolController(sim_model)

# 2. Register callbacks
input_handler.register_key_callback(Key.ESC, quit)
input_handler.register_key_callback(Key.F, tool_controller.toggle_grasp)

# 3. Start pynput listeners
kb_listener = keyboard.Listener(
    on_press=lambda key: PynputAdapter.on_press(key, input_handler),
    on_release=lambda key: PynputAdapter.on_release(key, input_handler)
)
kb_listener.start()

mouse_listener = mouse.Listener(
    on_move=lambda x, y: input_handler.update_mouse_position(x, y),
    on_scroll=lambda x, y, dx, dy: input_handler.update_mouse_scroll(dx, dy)
)
mouse_listener.start()

# 4. Main loop
while running:
    # Update tool from keyboard
    tool_controller.update_from_keyboard(input_handler)
    
    # Update camera from mouse
    dx, dy = input_handler.get_mouse_delta()
    if input_handler.is_mouse_button_pressed(MouseButton.LEFT):
        camera_controller.handle_mouse_orbit(dx, dy, True)
    
    # Apply actions
    tool_controller.apply_actions()
    
    # Physics + Render
    sim_integrator.stepModel(sim_model)
    renderer.renderNew(sim_bvh, sim_model)
    
    # Reset deltas
    input_handler.reset_deltas()
```

### Full Interactive Demo

See `tests/testInteractiveControl.py` for complete working example with:
- All control modes
- Camera manipulation
- Pause/reset functionality
- Help display
- CUDA graph optimization

Run:
```bash
cd FF_SRL/FF_SRL/tests
python testInteractiveControl.py
```

## Comparison with xpbd-tissue-sim

| Feature | xpbd-tissue-sim (C++) | FF-SRL (Python) |
|---------|----------------------|-----------------|
| **Camera** | Easy3D Viewer | Custom CameraController |
| **Orbit** | Left drag | Left drag |
| **Pan** | Middle drag | Middle drag |
| **Zoom** | Scroll | Scroll |
| **Tool - Keyboard** | QWER/ASDF | WASD/QE |
| **Tool - Mouse** | Space + mouse | Space + mouse |
| **Clutch** | Space bar | Space bar |
| **Grasp** | Left click | F key (or left click) |
| **Haptic** | Virtuoso device | Architecture ready |
| **Gamepad** | ❌ | ✅ Xbox/generic |

## Input Device Support

### ✅ Keyboard (pynput)
- All alphanumeric keys
- Function keys (F1-F12)
- Modifiers (Shift, Ctrl, Alt)
- Arrow keys, Space, Enter, Esc

**Installation**: `pip install pynput` (already in requirements)

### ✅ Mouse (pynput)
- Position tracking
- Button states (left, right, middle)
- Delta movement
- Scroll wheel

**Installation**: Included with keyboard (pynput)

### ✅ Gamepad (pygame - optional)
- Xbox controller layout
- Generic joystick support
- Analog sticks (4-axis)
- Triggers (2-axis)
- Buttons (A/B/X/Y + bumpers)
- D-pad

**Installation**: `pip install pygame`

**Usage**:
```python
from FF_SRL.input_handler import PygameAdapter

# In main loop
PygameAdapter.update_gamepad(0, input_handler)
tool_controller.update_from_gamepad(input_handler)
```

### 🔄 Haptic Device (future)
- Architecture supports haptic integration
- Placeholder in `InputDevice.HAPTIC`
- Compatible with Virtuoso/Phantom/ForceDimension APIs

**To add haptic support**:
1. Create adapter in `input_handler.py` (like `PygameAdapter`)
2. Map haptic position → tool position
3. Map tool forces → haptic feedback
4. Use `InputDevice.HAPTIC` mode in `ToolController`

## Advanced Features

### Custom Key Bindings

```python
# Override default controls
input_handler.register_key_callback(Key.G, tool_controller.toggle_grasp)
input_handler.register_key_callback(Key.T, lambda: print("Custom action!"))
```

### Camera Smoothing

```python
# Add inertia to camera movement
class SmoothCamera(CameraController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity = np.zeros(3)
        self.damping = 0.9
    
    def orbit(self, dx, dy):
        self.velocity[0] = dx * 0.1
        self.velocity[1] = dy * 0.1
        super().orbit(self.velocity[0], self.velocity[1])
        self.velocity *= self.damping
```

### Multi-Environment Control

```python
# Control different tools in different environments
tool_controllers = [
    ToolController(sim_model, device="cuda:0")
    for _ in range(num_envs)
]

# Update all
for i, tc in enumerate(tool_controllers):
    tc.update_from_keyboard(input_handler)
    tc.apply_actions(num_envs=1)
```

### Recording and Playback

```python
# Record user actions
action_history = []

while recording:
    tool_controller.update_from_keyboard(input_handler)
    action_history.append(tool_controller.cartesian_actions.clone())
    tool_controller.apply_actions()

# Replay
for actions in action_history:
    tool_controller.cartesian_actions = actions
    tool_controller.apply_actions()
```

## Performance Considerations

### Input Polling vs Events
- **Keyboard/Mouse**: Event-driven (pynput callbacks)
- **Gamepad**: Polling (call per frame)
- **Overhead**: Negligible (<0.1ms per frame)

### CUDA Graph Compatibility
- Input handling happens on CPU
- Actions applied before graph capture
- ✅ Compatible with CUDA graphs

### Multi-Threading
- pynput runs listeners in separate threads
- InputHandler is **not** thread-safe by default
- Use locks if accessing from multiple threads:
  ```python
  import threading
  input_handler._lock = threading.Lock()
  ```

## Troubleshooting

### Problem: Keys not responding
**Solution**: Check pynput listener is started:
```python
kb_listener.start()
# Not: kb_listener.run() which blocks
```

### Problem: Mouse delta always zero
**Solution**: Call `reset_deltas()` at end of frame:
```python
input_handler.reset_deltas()
```

### Problem: Camera jumps when clicking
**Solution**: Check for conflicting mouse modes:
```python
# Don't orbit if in mouse tool control mode
if left_pressed and control_mode != "mouse":
    camera_controller.orbit(dx, dy)
```

### Problem: Gamepad not detected
**Solution**: 
1. Install pygame: `pip install pygame`
2. Initialize pygame joystick:
   ```python
   import pygame
   pygame.init()
   pygame.joystick.init()
   ```

### Problem: Tool moves too fast/slow
**Solution**: Adjust sensitivity parameters:
```python
tool_controller.params.translation_speed = 0.05  # Slower
tool_controller.params.mouse_sensitivity = 0.0001  # Less sensitive
```

## Future Enhancements

- [ ] Haptic device integration (Virtuoso, Phantom, etc.)
- [ ] VR controller support (Oculus Touch, Vive)
- [ ] Touch input for tablets
- [ ] Network input (remote control over TCP/IP)
- [ ] Action recording/playback system
- [ ] Custom gesture recognition
- [ ] Multi-hand bimanual control
- [ ] Force feedback visualization

## References

- Easy3D: https://github.com/LiangliangNan/Easy3D
- xpbd-tissue-sim: Your C++ project
- pynput: https://pynput.readthedocs.io/
- pygame joystick: https://www.pygame.org/docs/ref/joystick.html
