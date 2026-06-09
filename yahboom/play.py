#!/usr/bin/env python3
"""
Yahboom Raspbot V1 Keyboard Control

Control your Raspbot from the keyboard.

Controls:
  Movement:
    W / ↑      - Forward
    S / ↓      - Backward
    A / ←      - Turn Left
    D / →      - Turn Right
    
  Speed:
    + / =      - Increase speed
    - / _      - Decrease speed
    1-9        - Set speed level
    
  Servos (Camera Pan/Tilt):
    I          - Tilt up
    K          - Tilt down
    J          - Pan left
    L          - Pan right
    O          - Center servos
    
  Peripherals:
    B          - Beep
    H          - Horn (long beep)
    R          - Toggle red LED
    E          - Toggle blue LED
    X          - LEDs off
    
  Sensors:
    Space      - Read distance
    T          - Read line trackers
    P          - Read IR obstacle sensors
    
  Control:
    Q / Esc    - Quit
"""

import sys
import time
import threading

from raspbot import Raspbot, LineTrackerReading

try:
    import tty
    import termios
    TERMIOS_AVAILABLE = True
except ImportError:
    TERMIOS_AVAILABLE = False


class KeyboardController:
    """Handles keyboard input for robot control."""
    
    def __init__(self, robot: Raspbot):
        self.robot = robot
        self.speed = 100
        self.min_speed = 30
        self.max_speed = 200
        self.speed_step = 20
        
        self.pan_angle = 90
        self.tilt_angle = 90
        self.servo_step = 10
        
        self.running = True
        self.red_led_on = False
        self.blue_led_on = False
        
        self._movement_active = False
        self._last_key_time = 0
        self._key_timeout = 0.6  # Must be longer than OS key repeat delay (~500ms)
    
    def print_status(self):
        """Print current status."""
        print(f"\r[Speed: {self.speed:3d}] [Pan: {self.pan_angle:3d}°] [Tilt: {self.tilt_angle:3d}°]    ", end="", flush=True)
    
    def handle_key(self, key: str):
        """Process a key press."""
        self._last_key_time = time.time()
        key_lower = key.lower()
        
        # Movement keys
        if key in ['w', 'W'] or key == '\x1b[A':  # W or Up arrow
            self.robot.forward(self.speed)
            self._movement_active = True
            
        elif key in ['s', 'S'] or key == '\x1b[B':  # S or Down arrow
            self.robot.backward(self.speed)
            self._movement_active = True
            
        elif key in ['a', 'A'] or key == '\x1b[D':  # A or Left arrow
            self.robot.turn_left(self.speed)
            self._movement_active = True
            
        elif key in ['d', 'D'] or key == '\x1b[C':  # D or Right arrow
            self.robot.turn_right(self.speed)
            self._movement_active = True
            
        # Speed control
        elif key in ['+', '=']:
            self.speed = min(self.max_speed, self.speed + self.speed_step)
            print(f"\nSpeed: {self.speed}")
            self.print_status()
            
        elif key in ['-', '_']:
            self.speed = max(self.min_speed, self.speed - self.speed_step)
            print(f"\nSpeed: {self.speed}")
            self.print_status()
            
        elif key.isdigit() and key != '0':
            self.speed = int(key) * 22  # Maps 1-9 to ~22-198
            print(f"\nSpeed: {self.speed}")
            self.print_status()
            
        # Servo control
        elif key_lower == 'i':  # Tilt up
            self.tilt_angle = max(0, self.tilt_angle - self.servo_step)
            self.robot.set_tilt(self.tilt_angle)
            self.print_status()
            
        elif key_lower == 'k':  # Tilt down
            self.tilt_angle = min(180, self.tilt_angle + self.servo_step)
            self.robot.set_tilt(self.tilt_angle)
            self.print_status()
            
        elif key_lower == 'j':  # Pan left
            self.pan_angle = min(180, self.pan_angle + self.servo_step)
            self.robot.set_pan(self.pan_angle)
            self.print_status()
            
        elif key_lower == 'l':  # Pan right
            self.pan_angle = max(0, self.pan_angle - self.servo_step)
            self.robot.set_pan(self.pan_angle)
            self.print_status()
            
        elif key_lower == 'o':  # Center servos
            self.pan_angle = 90
            self.tilt_angle = 90
            self.robot.center_servos()
            print("\nServos centered")
            self.print_status()
            
        # Buzzer
        elif key_lower == 'b':  # Short beep
            self.robot.beep(0.1, 440)
            
        elif key_lower == 'h':  # Horn (long beep)
            self.robot.beep(0.5, 600)
            
        # LEDs
        elif key_lower == 'r':  # Toggle red LED
            self.red_led_on = not self.red_led_on
            self.robot.led_red(self.red_led_on)
            print(f"\nRed LED: {'ON' if self.red_led_on else 'OFF'}")
            self.print_status()
                
        elif key_lower == 'e':  # Toggle blue LED
            self.blue_led_on = not self.blue_led_on
            self.robot.led_blue(self.blue_led_on)
            print(f"\nBlue LED: {'ON' if self.blue_led_on else 'OFF'}")
            self.print_status()
                
        elif key_lower == 'x':  # LEDs off
            self.robot.leds_off()
            self.red_led_on = False
            self.blue_led_on = False
            print("\nLEDs OFF")
            self.print_status()
            
        # Sensors
        elif key == ' ':  # Space - distance
            distance = self.robot.read_distance()
            if distance > 0:
                print(f"\nDistance: {distance:.1f} cm")
            else:
                print("\nDistance: timeout/error")
            self.print_status()
            
        elif key_lower == 't':  # Line tracker
            reading = self.robot.read_line_tracker()
            print(f"\nLine Tracker: L1={reading.left1} L2={reading.left2} R1={reading.right1} R2={reading.right2}")
            self.print_status()
            
        elif key_lower == 'p':  # IR obstacle
            left, right = self.robot.read_ir_obstacle()
            print(f"\nIR Obstacle: Left={left} Right={right}")
            self.print_status()
            
        # Quit
        elif key_lower == 'q' or key == '\x1b':  # q or Escape
            print("\nQuitting...")
            self.running = False
            
    def stop_movement_on_release(self):
        """Background thread to stop movement when keys are released."""
        while self.running:
            time.sleep(0.05)
            if self._movement_active:
                if time.time() - self._last_key_time > self._key_timeout:
                    self.robot.stop()
                    self._movement_active = False


def get_key_termios() -> str:
    """Get a single keypress using termios (Unix)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # Handle escape sequences (arrow keys)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                ch3 = sys.stdin.read(1)
                return f'\x1b[{ch3}'
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def run_termios(robot: Raspbot):
    """Run keyboard control using termios."""
    controller = KeyboardController(robot)
    
    # Start background thread to handle key release
    stop_thread = threading.Thread(target=controller.stop_movement_on_release, daemon=True)
    stop_thread.start()
    
    print("\nReady! Use WASD or arrow keys to move.")
    controller.print_status()
    
    while controller.running:
        try:
            key = get_key_termios()
            controller.handle_key(key)
        except KeyboardInterrupt:
            break
    
    robot.stop()
    print("\nStopped.")


def main():
    """Main entry point."""
    print("=" * 50)
    print("  Yahboom Raspbot V1 Keyboard Controller")
    print("=" * 50)
    print()
    
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print(__doc__)
        return
    
    # Initialize robot
    print("Initializing robot...")
    robot = Raspbot()
    
    try:
        robot.init()
        print("Robot initialized!")
        
        # Short beep on startup
        robot.beep(0.1, 440)
        time.sleep(0.1)
        
        # Center servos
        robot.center_servos()
        
        print()
        print("Controls:")
        print("  Movement:  WASD or Arrow keys")
        print("  Speed:     +/- or 1-9")
        print("  Camera:    IJKL (pan/tilt), O (center)")
        print("  Sound:     B (beep), H (horn)")
        print("  LEDs:      R (red), E (blue), X (off)")
        print("  Sensors:   Space (distance), T (tracker), P (IR)")
        print("  Quit:      Q or Escape")
        print()
        
        if TERMIOS_AVAILABLE:
            run_termios(robot)
        else:
            print("Error: termios not available. This script requires Unix/Linux.")
            return
        
        # Shutdown
        print("\nShutting down...")
        robot.leds_off()
        robot.beep(0.1, 330)
        
    except KeyboardInterrupt:
        print("\nInterrupted!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        robot.cleanup()
        print("Goodbye!")


if __name__ == "__main__":
    main()
