"""Keyboard teleop for the laptop client (global keys via pynput, like LeKiwi)."""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Any

from .protocol import RobotCommand

logger = logging.getLogger(__name__)

PYNPUT_AVAILABLE = False
_pynput_keyboard: Any = None

try:
    from pynput import keyboard as pynput_keyboard

    PYNPUT_AVAILABLE = True
    _pynput_keyboard = pynput_keyboard
except ImportError:
    pass

try:
    import termios
    import tty

    TERMIOS_AVAILABLE = True
except ImportError:
    TERMIOS_AVAILABLE = False


def _movement_from_pressed(pressed: set[str]) -> str:
    if pressed & {"w", "up"}:
        return "forward"
    if pressed & {"s", "down"}:
        return "backward"
    if pressed & {"a", "left"}:
        return "turn_left"
    if pressed & {"d", "right"}:
        return "turn_right"
    return "stop"


class PynputKeyboardTeleop:
    """Global keyboard listener — works while Rerun or any other app is focused."""

    def __init__(self) -> None:
        if not PYNPUT_AVAILABLE:
            raise RuntimeError(
                "pynput is required for global keyboard teleop. "
                "Install with: uv sync  (or pip install pynput)"
            )

        self.speed = 100
        self.min_speed = 30
        self.max_speed = 200
        self.speed_step = 20

        self.pan_angle = 90
        self.tilt_angle = 90
        self.servo_step = 10

        self.red_led_on = False
        self.blue_led_on = False

        self.quit_requested = False
        self.running = True

        self._pressed: set[str] = set()
        self._pending_beep: dict[str, float] | None = None
        self._center_servos = False
        self._servo_dirty = False
        self._lock = threading.Lock()
        self._listener = None

    def start(self) -> None:
        self._listener = _pynput_keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        logger.info("Global keyboard listener started (pynput)")

    def get_command(self) -> RobotCommand:
        with self._lock:
            beep = self._pending_beep
            self._pending_beep = None
            center_servos = self._center_servos
            self._center_servos = False
            servo_dirty = self._servo_dirty
            self._servo_dirty = False
            movement = _movement_from_pressed(set(self._pressed))
            return RobotCommand(
                movement=movement,
                speed=self.speed,
                pan=self.pan_angle,
                tilt=self.tilt_angle,
                red_led=self.red_led_on,
                blue_led=self.blue_led_on,
                beep=beep,
                center_servos=center_servos,
                servo_dirty=servo_dirty,
                quit=self.quit_requested,
            )

    def _on_press(self, key: Any) -> None:
        name = self._key_name(key)
        if name is None:
            return

        with self._lock:
            if name in {"w", "a", "s", "d", "up", "down", "left", "right"}:
                self._pressed.add(name)
                return

            self._handle_action_key(name)

    def _on_release(self, key: Any) -> None:
        name = self._key_name(key)
        if name is None:
            return

        if key == _pynput_keyboard.Key.esc:
            with self._lock:
                self.quit_requested = True
                self.running = False
            return False  # stop listener

        with self._lock:
            self._pressed.discard(name)

    def _key_name(self, key: Any) -> str | None:
        if hasattr(key, "char") and key.char is not None:
            return key.char.lower()
        mapping = {
            _pynput_keyboard.Key.up: "up",
            _pynput_keyboard.Key.down: "down",
            _pynput_keyboard.Key.left: "left",
            _pynput_keyboard.Key.right: "right",
        }
        return mapping.get(key)

    def _handle_action_key(self, key_lower: str) -> None:
        if key_lower in {"+", "="}:
            self.speed = min(self.max_speed, self.speed + self.speed_step)
        elif key_lower in {"-", "_"}:
            self.speed = max(self.min_speed, self.speed - self.speed_step)
        elif key_lower.isdigit() and key_lower != "0":
            self.speed = int(key_lower) * 22
        elif key_lower == "i":
            self.tilt_angle = max(0, self.tilt_angle - self.servo_step)
            self._servo_dirty = True
        elif key_lower == "k":
            self.tilt_angle = min(180, self.tilt_angle + self.servo_step)
            self._servo_dirty = True
        elif key_lower == "j":
            self.pan_angle = min(180, self.pan_angle + self.servo_step)
            self._servo_dirty = True
        elif key_lower == "l":
            self.pan_angle = max(0, self.pan_angle - self.servo_step)
            self._servo_dirty = True
        elif key_lower == "o":
            self.pan_angle = 90
            self.tilt_angle = 90
            self._center_servos = True
            self._servo_dirty = True
        elif key_lower == "b":
            self._pending_beep = {"duration": 0.1, "frequency": 440}
        elif key_lower == "h":
            self._pending_beep = {"duration": 0.5, "frequency": 600}
        elif key_lower == "r":
            self.red_led_on = not self.red_led_on
        elif key_lower == "e":
            self.blue_led_on = not self.blue_led_on
        elif key_lower == "x":
            self.red_led_on = False
            self.blue_led_on = False
        elif key_lower == "q":
            self.quit_requested = True
            self.running = False

    def stop(self) -> None:
        self.running = False
        if self._listener is not None:
            self._listener.stop()


class TermiosKeyboardTeleop:
    """Terminal-only keyboard (requires terminal window focus)."""

    def __init__(self) -> None:
        if not TERMIOS_AVAILABLE:
            raise RuntimeError("termios is required for terminal keyboard teleop")

        self.speed = 100
        self.min_speed = 30
        self.max_speed = 200
        self.speed_step = 20
        self.pan_angle = 90
        self.tilt_angle = 90
        self.servo_step = 10
        self.red_led_on = False
        self.blue_led_on = False
        self.movement = "stop"
        self.running = True
        self.quit_requested = False
        self._movement_active = False
        self._last_key_time = 0.0
        self._key_timeout = 0.6
        self._pending_beep: dict[str, float] | None = None
        self._center_servos = False
        self._servo_dirty = False
        self._lock = threading.Lock()

    def start(self) -> None:
        threading.Thread(target=self._stop_on_release, daemon=True).start()
        threading.Thread(target=self._read_loop, daemon=True).start()

    def get_command(self) -> RobotCommand:
        with self._lock:
            beep = self._pending_beep
            self._pending_beep = None
            center_servos = self._center_servos
            self._center_servos = False
            servo_dirty = self._servo_dirty
            self._servo_dirty = False
            return RobotCommand(
                movement=self.movement,
                speed=self.speed,
                pan=self.pan_angle,
                tilt=self.tilt_angle,
                red_led=self.red_led_on,
                blue_led=self.blue_led_on,
                beep=beep,
                center_servos=center_servos,
                servo_dirty=servo_dirty,
                quit=self.quit_requested,
            )

    def _stop_on_release(self) -> None:
        while self.running:
            time.sleep(0.05)
            with self._lock:
                if self._movement_active and time.time() - self._last_key_time > self._key_timeout:
                    self.movement = "stop"
                    self._movement_active = False

    def _read_loop(self) -> None:
        while self.running:
            self._handle_key(self._read_key())

    def _read_key(self) -> str:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    return f"\x1b[{ch3}"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, key: str) -> None:
        with self._lock:
            self._last_key_time = time.time()
            key_lower = key.lower()
            if key in ["w", "W"] or key == "\x1b[A":
                self.movement = "forward"
                self._movement_active = True
            elif key in ["s", "S"] or key == "\x1b[B":
                self.movement = "backward"
                self._movement_active = True
            elif key in ["a", "A"] or key == "\x1b[D":
                self.movement = "turn_left"
                self._movement_active = True
            elif key in ["d", "D"] or key == "\x1b[C":
                self.movement = "turn_right"
                self._movement_active = True
            elif key_lower in {"+", "="}:
                self.speed = min(self.max_speed, self.speed + self.speed_step)
            elif key_lower in {"-", "_"}:
                self.speed = max(self.min_speed, self.speed - self.speed_step)
            elif key.isdigit() and key != "0":
                self.speed = int(key) * 22
            elif key_lower == "i":
                self.tilt_angle = max(0, self.tilt_angle - self.servo_step)
                self._servo_dirty = True
            elif key_lower == "k":
                self.tilt_angle = min(180, self.tilt_angle + self.servo_step)
                self._servo_dirty = True
            elif key_lower == "j":
                self.pan_angle = min(180, self.pan_angle + self.servo_step)
                self._servo_dirty = True
            elif key_lower == "l":
                self.pan_angle = max(0, self.pan_angle - self.servo_step)
                self._servo_dirty = True
            elif key_lower == "o":
                self.pan_angle = 90
                self.tilt_angle = 90
                self._center_servos = True
                self._servo_dirty = True
            elif key_lower == "b":
                self._pending_beep = {"duration": 0.1, "frequency": 440}
            elif key_lower == "h":
                self._pending_beep = {"duration": 0.5, "frequency": 600}
            elif key_lower == "r":
                self.red_led_on = not self.red_led_on
            elif key_lower == "e":
                self.blue_led_on = not self.blue_led_on
            elif key_lower == "x":
                self.red_led_on = False
                self.blue_led_on = False
            elif key_lower == "q" or key == "\x1b":
                self.quit_requested = True
                self.running = False

    def stop(self) -> None:
        self.running = False


def create_keyboard(backend: str = "pynput") -> PynputKeyboardTeleop | TermiosKeyboardTeleop:
    if backend == "pynput":
        return PynputKeyboardTeleop()
    if backend == "terminal":
        return TermiosKeyboardTeleop()
    raise ValueError(f"Unknown keyboard backend: {backend}")


# Default export
KeyboardTeleop = PynputKeyboardTeleop
