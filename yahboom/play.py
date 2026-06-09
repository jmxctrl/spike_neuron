#!/usr/bin/env python3
"""
Yahboom Raspbot remote teleop (runs on your laptop).

Keyboard input and Rerun visualization run locally. Commands are sent to the
robot host over ZMQ, similar to LeRobot's LeKiwi teleop flow.

Start the host on the Raspberry Pi first:
    sudo python3 -m yahboom.host

Then on your laptop:
    python -m yahboom.play --remote-ip 192.168.1.100

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

  Control:
    Q / Esc    - Quit
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .client import RaspbotClient
from .keyboard import KeyboardTeleop, TERMIOS_AVAILABLE
from .protocol import DEFAULT_CMD_PORT, DEFAULT_FPS, DEFAULT_OBS_PORT
from .rerun_viz import init_rerun, log_teleop_data

logger = logging.getLogger(__name__)


def precise_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def print_controls() -> None:
    print(__doc__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote Yahboom teleop with Rerun")
    parser.add_argument("--remote-ip", required=True, help="Robot Raspberry Pi IP address")
    parser.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT)
    parser.add_argument("--obs-port", type=int, default=DEFAULT_OBS_PORT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--no-rerun", action="store_true", help="Disable Rerun viewer")
    args = parser.parse_args()

    if not TERMIOS_AVAILABLE:
        print("Error: termios is required for keyboard teleop (macOS/Linux).")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 50)
    print("  Yahboom Raspbot Remote Teleop")
    print("=" * 50)
    print_controls()

    client = RaspbotClient(
        remote_ip=args.remote_ip,
        cmd_port=args.cmd_port,
        obs_port=args.obs_port,
        connect_timeout_s=args.connect_timeout_s,
    )
    keyboard = KeyboardTeleop()

    try:
        client.connect()
        keyboard.start()
        if not args.no_rerun:
            init_rerun(session_name="yahboom_teleop")

        print("\nReady! Focus this terminal and use WASD / arrow keys.")
        print("Rerun viewer should open automatically (unless --no-rerun).\n")

        while True:
            t0 = time.perf_counter()

            command = keyboard.get_command()
            if command.quit:
                client.send_command(command)
                break

            client.send_command(command)
            client.poll_observation()

            if not args.no_rerun:
                log_teleop_data(
                    observation=client.get_observation_dict(),
                    command=command,
                )

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
