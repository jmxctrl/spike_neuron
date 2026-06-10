#!/usr/bin/env python3
"""
Yahboom Raspbot remote teleop (runs on your laptop).

Keyboard input and Rerun visualization run locally. Commands are sent to the
robot host over ZMQ, similar to LeRobot's LeKiwi teleop flow.

Start the host on the Raspberry Pi first:
    sudo PYTHONPATH=. python3 -m yahboom.host --camera-backend picamera2

Then on your laptop:
    uv run python -m yahboom.play --remote-ip 192.168.1.100

Controls work globally (even while Rerun is focused) via pynput.
Hold movement keys to drive; release to stop.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .client import RaspbotClient
from .keyboard import PYNPUT_AVAILABLE, TERMIOS_AVAILABLE, create_keyboard
from .protocol import DEFAULT_CMD_PORT, DEFAULT_FPS, DEFAULT_OBS_PORT
from .rerun_viz import init_rerun, log_teleop_data

logger = logging.getLogger(__name__)


def print_controls(keyboard_backend: str) -> None:
    focus = (
        "Keys work globally (Rerun or any window)."
        if keyboard_backend == "pynput"
        else "Focus this terminal for keyboard input."
    )
    print(f"""
Controls ({focus})
  Movement (hold key):
    W / ↑      Forward          A / ←      Turn left
    S / ↓      Backward         D / →      Turn right
  Speed:
    + / =      Increase         - / _      Decrease         1-9  Set level
  Camera pan/tilt:
    I          Tilt up          K          Tilt down
    J          Pan left         L          Pan right        O    Center
  Sensors (print to this terminal):
    Space      Ultrasonic distance
    T          Line trackers
    P          IR obstacle sensors
  Peripherals:
    B          Beep             H          Horn
    R          Toggle red LED   E          Toggle blue LED  X    LEDs off
  Quit:
    Q / Esc

macOS: if keys don't work globally, grant Accessibility (and Input Monitoring)
       to your terminal app in System Settings → Privacy & Security.
""")


def format_sensor_reading(query: str, obs: dict) -> str:
    if query == "distance":
        d = obs.get("distance_cm", -1)
        if d is None or d < 0:
            return "Distance: timeout/error"
        return f"Distance: {d:.1f} cm"
    if query == "line":
        return (
            "Line tracker: "
            f"L1={bool(obs.get('line_left1'))} L2={bool(obs.get('line_left2'))} "
            f"R1={bool(obs.get('line_right1'))} R2={bool(obs.get('line_right2'))}"
        )
    if query == "ir":
        return (
            "IR obstacle: "
            f"Left={bool(obs.get('ir_left'))} Right={bool(obs.get('ir_right'))}"
        )
    return ""


def precise_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote Yahboom teleop with Rerun")
    parser.add_argument("--remote-ip", required=True, help="Robot Raspberry Pi IP address")
    parser.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT)
    parser.add_argument("--obs-port", type=int, default=DEFAULT_OBS_PORT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--poll-timeout-ms", type=int, default=50)
    parser.add_argument("--no-rerun", action="store_true", help="Disable Rerun viewer")
    parser.add_argument(
        "--keyboard-backend",
        choices=["pynput", "terminal"],
        default="pynput",
        help="pynput = global keys (works while Rerun focused); terminal = stdin only",
    )
    args = parser.parse_args()

    if args.keyboard_backend == "pynput" and not PYNPUT_AVAILABLE:
        print("Error: pynput not installed. Run: uv sync")
        sys.exit(1)
    if args.keyboard_backend == "terminal" and not TERMIOS_AVAILABLE:
        print("Error: termios not available on this platform.")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 50)
    print("  Yahboom Raspbot Remote Teleop")
    print("=" * 50)

    client = RaspbotClient(
        remote_ip=args.remote_ip,
        cmd_port=args.cmd_port,
        obs_port=args.obs_port,
        connect_timeout_s=args.connect_timeout_s,
        polling_timeout_ms=args.poll_timeout_ms,
    )
    keyboard = create_keyboard(backend=args.keyboard_backend)

    try:
        client.connect()
        keyboard.start()
        if not args.no_rerun:
            init_rerun(session_name="yahboom_teleop")

        print_controls(args.keyboard_backend)
        if not args.no_rerun:
            print("Rerun viewer should open automatically.\n")

        frame_idx = 0
        while True:
            t0 = time.perf_counter()

            command = keyboard.get_command()
            if command.quit:
                client.send_command(command)
                break

            client.send_command(command)
            client.poll_observation()

            sensor_query = keyboard.pop_sensor_query()
            if sensor_query:
                msg = format_sensor_reading(sensor_query, client.get_observation_dict())
                if msg:
                    print(msg)

            if not args.no_rerun:
                log_teleop_data(
                    observation=client.get_observation_dict(),
                    command=command,
                    frame_idx=frame_idx,
                )
            frame_idx += 1

            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\nInterrupted")
    except ConnectionError as exc:
        print(f"\nConnection error: {exc}")
        sys.exit(1)
    finally:
        keyboard.stop()
        try:
            client.disconnect()
        except Exception:
            pass
        print("Goodbye!")


if __name__ == "__main__":
    main()
