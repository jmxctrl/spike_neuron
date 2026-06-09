"""Non-blocking keyboard teleop state (runs on laptop)."""

from __future__ import annotations

import sys
import threading
import time

from .protocol import RobotCommand

try:
    import termios
    import tty

    TERMIOS_AVAILABLE = True
except ImportError:
    TERMIOS_AVAILABLE = False


class KeyboardTeleop:
    """Reads keyboard input in a background thread and exposes a RobotCommand snapshot."""

    def __init__(self):
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
        if not TERMIOS_AVAILABLE:
            raise RuntimeError("termios is required for keyboard teleop (macOS/Linux)")
        stop_thread = threading.Thread(target=self._stop_on_release, daemon=True)
        stop_thread.start()
        read_thread = threading.Thread(target=self._read_loop, daemon=True)
        read_thread.start()

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
            key = self._read_key()
            self._handle_key(key)

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
            elif key in ["+", "="]:
                self.speed = min(self.max_speed, self.speed + self.speed_step)
            elif key in ["-", "_"]:
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
