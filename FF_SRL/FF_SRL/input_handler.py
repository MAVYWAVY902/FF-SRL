"""
Input Handler Module for FF-SRL
Provides unified interface for keyboard, mouse, gamepad, and haptic device input
Inspired by Easy3D and xpbd-tissue-sim interaction patterns
"""

import numpy as np
from enum import IntEnum, auto
from typing import Callable, Dict, Optional, Tuple
from dataclasses import dataclass, field


class Key(IntEnum):
    """Keyboard key enumeration"""
    # Letters
    A = auto(); B = auto(); C = auto(); D = auto(); E = auto()
    F = auto(); G = auto(); H = auto(); I = auto(); J = auto()
    K = auto(); L = auto(); M = auto(); N = auto(); O = auto()
    P = auto(); Q = auto(); R = auto(); S = auto(); T = auto()
    U = auto(); V = auto(); W = auto(); X = auto(); Y = auto(); Z = auto()
    
    # Numbers
    NUM_0 = auto(); NUM_1 = auto(); NUM_2 = auto(); NUM_3 = auto(); NUM_4 = auto()
    NUM_5 = auto(); NUM_6 = auto(); NUM_7 = auto(); NUM_8 = auto(); NUM_9 = auto()
    
    # Function keys
    F1 = auto(); F2 = auto(); F3 = auto(); F4 = auto(); F5 = auto()
    F6 = auto(); F7 = auto(); F8 = auto(); F9 = auto(); F10 = auto()
    F11 = auto(); F12 = auto()
    
    # Special keys
    SPACE = auto(); ENTER = auto(); TAB = auto(); ESC = auto()
    BACKSPACE = auto(); DELETE = auto()
    LEFT = auto(); RIGHT = auto(); UP = auto(); DOWN = auto()
    SHIFT = auto(); CTRL = auto(); ALT = auto()
    
    UNKNOWN = auto()


class MouseButton(IntEnum):
    """Mouse button enumeration"""
    LEFT = 0
    RIGHT = 1
    MIDDLE = 2


class KeyAction(IntEnum):
    """Keyboard/mouse action type"""
    PRESS = auto()
    RELEASE = auto()
    HOLD = auto()


class InputDevice(IntEnum):
    """Input device types"""
    KEYBOARD = auto()
    MOUSE = auto()
    GAMEPAD = auto()
    HAPTIC = auto()


@dataclass
class MouseState:
    """Current mouse state"""
    x: float = 0.0
    y: float = 0.0
    dx: float = 0.0  # Delta from last frame
    dy: float = 0.0
    scroll_dx: float = 0.0
    scroll_dy: float = 0.0
    buttons: Dict[MouseButton, bool] = field(default_factory=lambda: {
        MouseButton.LEFT: False,
        MouseButton.RIGHT: False,
        MouseButton.MIDDLE: False
    })
    last_x: float = 0.0
    last_y: float = 0.0


@dataclass
class GamepadState:
    """Xbox/generic gamepad state"""
    left_stick_x: float = 0.0
    left_stick_y: float = 0.0
    right_stick_x: float = 0.0
    right_stick_y: float = 0.0
    left_trigger: float = 0.0   # 0.0 to 1.0
    right_trigger: float = 0.0  # 0.0 to 1.0
    button_a: bool = False
    button_b: bool = False
    button_x: bool = False
    button_y: bool = False
    dpad_up: bool = False
    dpad_down: bool = False
    dpad_left: bool = False
    dpad_right: bool = False
    left_bumper: bool = False
    right_bumper: bool = False
    start: bool = False
    back: bool = False


class InputHandler:
    """
    Unified input handling system
    Manages keyboard, mouse, gamepad, and haptic device inputs
    """
    
    def __init__(self):
        self.keyboard_state: Dict[Key, bool] = {}
        self.mouse_state = MouseState()
        self.gamepad_state = GamepadState()
        
        # Callback system
        self.key_callbacks: Dict[Key, Callable] = {}
        self.mouse_button_callbacks: Dict[MouseButton, Callable] = {}
        self.mouse_move_callback: Optional[Callable] = None
        self.mouse_scroll_callback: Optional[Callable] = None
        
        # Modifier keys
        self.shift_held = False
        self.ctrl_held = False
        self.alt_held = False
        
        # Input mode
        self.active_device = InputDevice.KEYBOARD
        self.enable_mouse_camera = True  # Like Easy3D viewer control
        
    def is_key_pressed(self, key: Key) -> bool:
        """Check if key is currently pressed"""
        return self.keyboard_state.get(key, False)
    
    def is_mouse_button_pressed(self, button: MouseButton) -> bool:
        """Check if mouse button is currently pressed"""
        return self.mouse_state.buttons[button]
    
    def get_mouse_delta(self) -> Tuple[float, float]:
        """Get mouse movement delta since last frame"""
        return (self.mouse_state.dx, self.mouse_state.dy)
    
    def get_mouse_position(self) -> Tuple[float, float]:
        """Get current mouse position"""
        return (self.mouse_state.x, self.mouse_state.y)
    
    def get_scroll_delta(self) -> Tuple[float, float]:
        """Get scroll wheel delta"""
        return (self.mouse_state.scroll_dx, self.mouse_state.scroll_dy)
    
    def update_key_state(self, key: Key, pressed: bool):
        """Update keyboard state"""
        self.keyboard_state[key] = pressed
        
        # Update modifiers
        if key == Key.SHIFT:
            self.shift_held = pressed
        elif key == Key.CTRL:
            self.ctrl_held = pressed
        elif key == Key.ALT:
            self.alt_held = pressed
        
        # Trigger callback if registered
        if pressed and key in self.key_callbacks:
            self.key_callbacks[key]()
    
    def update_mouse_button(self, button: MouseButton, pressed: bool):
        """Update mouse button state"""
        self.mouse_state.buttons[button] = pressed
        
        if button in self.mouse_button_callbacks:
            self.mouse_button_callbacks[button](pressed)
    
    def update_mouse_position(self, x: float, y: float):
        """Update mouse position and compute delta"""
        self.mouse_state.dx = x - self.mouse_state.last_x
        self.mouse_state.dy = y - self.mouse_state.last_y
        self.mouse_state.x = x
        self.mouse_state.y = y
        self.mouse_state.last_x = x
        self.mouse_state.last_y = y
        
        if self.mouse_move_callback:
            self.mouse_move_callback(x, y, self.mouse_state.dx, self.mouse_state.dy)
    
    def update_mouse_scroll(self, dx: float, dy: float):
        """Update scroll wheel delta"""
        self.mouse_state.scroll_dx = dx
        self.mouse_state.scroll_dy = dy
        
        if self.mouse_scroll_callback:
            self.mouse_scroll_callback(dx, dy)
    
    def reset_deltas(self):
        """Reset delta values (call at end of frame)"""
        self.mouse_state.dx = 0.0
        self.mouse_state.dy = 0.0
        self.mouse_state.scroll_dx = 0.0
        self.mouse_state.scroll_dy = 0.0
    
    def register_key_callback(self, key: Key, callback: Callable):
        """Register callback for specific key press"""
        self.key_callbacks[key] = callback
    
    def register_mouse_button_callback(self, button: MouseButton, callback: Callable):
        """Register callback for mouse button"""
        self.mouse_button_callbacks[button] = callback
    
    def register_mouse_move_callback(self, callback: Callable):
        """Register callback for mouse movement"""
        self.mouse_move_callback = callback
    
    def register_mouse_scroll_callback(self, callback: Callable):
        """Register callback for scroll wheel"""
        self.mouse_scroll_callback = callback


class PynputAdapter:
    """
    Adapter for pynput library (already used in testLiverRetractionTexture.py)
    Converts pynput events to InputHandler
    """
    
    # Pynput key to our Key enum mapping
    KEY_MAP = {
        'a': Key.A, 'b': Key.B, 'c': Key.C, 'd': Key.D, 'e': Key.E,
        'f': Key.F, 'g': Key.G, 'h': Key.H, 'i': Key.I, 'j': Key.J,
        'k': Key.K, 'l': Key.L, 'm': Key.M, 'n': Key.N, 'o': Key.O,
        'p': Key.P, 'q': Key.Q, 'r': Key.R, 's': Key.S, 't': Key.T,
        'u': Key.U, 'v': Key.V, 'w': Key.W, 'x': Key.X, 'y': Key.Y,
        'z': Key.Z,
        '0': Key.NUM_0, '1': Key.NUM_1, '2': Key.NUM_2, '3': Key.NUM_3,
        '4': Key.NUM_4, '5': Key.NUM_5, '6': Key.NUM_6, '7': Key.NUM_7,
        '8': Key.NUM_8, '9': Key.NUM_9,
    }
    
    @staticmethod
    def on_press(key, input_handler: InputHandler):
        """Pynput key press event handler"""
        from pynput import keyboard as kb
        
        mapped_key = Key.UNKNOWN
        
        # Handle character keys
        if hasattr(key, 'char') and key.char:
            char_lower = key.char.lower()
            mapped_key = PynputAdapter.KEY_MAP.get(char_lower, Key.UNKNOWN)
        
        # Handle special keys
        elif key == kb.Key.space:
            mapped_key = Key.SPACE
        elif key == kb.Key.enter:
            mapped_key = Key.ENTER
        elif key == kb.Key.esc:
            mapped_key = Key.ESC
        elif key == kb.Key.tab:
            mapped_key = Key.TAB
        elif key == kb.Key.shift or key == kb.Key.shift_r:
            mapped_key = Key.SHIFT
        elif key == kb.Key.ctrl or key == kb.Key.ctrl_r:
            mapped_key = Key.CTRL
        elif key == kb.Key.alt or key == kb.Key.alt_r:
            mapped_key = Key.ALT
        elif key == kb.Key.left:
            mapped_key = Key.LEFT
        elif key == kb.Key.right:
            mapped_key = Key.RIGHT
        elif key == kb.Key.up:
            mapped_key = Key.UP
        elif key == kb.Key.down:
            mapped_key = Key.DOWN
        
        input_handler.update_key_state(mapped_key, True)
    
    @staticmethod
    def on_release(key, input_handler: InputHandler):
        """Pynput key release event handler"""
        from pynput import keyboard as kb
        
        mapped_key = Key.UNKNOWN
        
        # Same mapping logic as on_press
        if hasattr(key, 'char') and key.char:
            char_lower = key.char.lower()
            mapped_key = PynputAdapter.KEY_MAP.get(char_lower, Key.UNKNOWN)
        elif key == kb.Key.space:
            mapped_key = Key.SPACE
        elif key == kb.Key.shift or key == kb.Key.shift_r:
            mapped_key = Key.SHIFT
        elif key == kb.Key.ctrl or key == kb.Key.ctrl_r:
            mapped_key = Key.CTRL
        elif key == kb.Key.alt or key == kb.Key.alt_r:
            mapped_key = Key.ALT
        
        input_handler.update_key_state(mapped_key, False)


class PygameAdapter:
    """
    Adapter for pygame (useful for gamepad/joystick support)
    Optional: only use if pygame is installed
    """
    
    @staticmethod
    def update_gamepad(gamepad_index: int, input_handler: InputHandler):
        """Update gamepad state from pygame joystick"""
        try:
            import pygame
            
            if not pygame.joystick.get_init():
                pygame.joystick.init()
            
            if pygame.joystick.get_count() <= gamepad_index:
                return
            
            joystick = pygame.joystick.Joystick(gamepad_index)
            if not joystick.get_init():
                joystick.init()
            
            gamepad = input_handler.gamepad_state
            
            # Analog sticks (typically axes 0-3)
            gamepad.left_stick_x = joystick.get_axis(0)
            gamepad.left_stick_y = -joystick.get_axis(1)  # Invert Y
            gamepad.right_stick_x = joystick.get_axis(2)
            gamepad.right_stick_y = -joystick.get_axis(3)
            
            # Triggers (axes 4-5 on Xbox controller)
            if joystick.get_numaxes() > 4:
                gamepad.left_trigger = (joystick.get_axis(4) + 1.0) / 2.0
            if joystick.get_numaxes() > 5:
                gamepad.right_trigger = (joystick.get_axis(5) + 1.0) / 2.0
            
            # Buttons (Xbox layout)
            if joystick.get_numbuttons() >= 11:
                gamepad.button_a = joystick.get_button(0)
                gamepad.button_b = joystick.get_button(1)
                gamepad.button_x = joystick.get_button(2)
                gamepad.button_y = joystick.get_button(3)
                gamepad.left_bumper = joystick.get_button(4)
                gamepad.right_bumper = joystick.get_button(5)
                gamepad.back = joystick.get_button(6)
                gamepad.start = joystick.get_button(7)
            
            # D-pad (typically hat 0)
            if joystick.get_numhats() > 0:
                hat = joystick.get_hat(0)
                gamepad.dpad_left = hat[0] < 0
                gamepad.dpad_right = hat[0] > 0
                gamepad.dpad_down = hat[1] < 0
                gamepad.dpad_up = hat[1] > 0
            
        except ImportError:
            print("Warning: pygame not installed, gamepad support disabled")
