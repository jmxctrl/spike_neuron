#!/usr/bin/env python3
"""
Remote camera SNN deploy (runs on your laptop).

The Pi only runs yahboom.host. This script receives the camera stream, computes
lane sensors, runs SNN inference, and sends motor commands back over ZMQ.

Pi:
    sudo PYTHONPATH=. python3 -m yahboom.host

Mac:
    uv run python -m camera.play --remote-ip <PI_IP> --weights camera/trained_weights/....npy
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from yahboom.client import RaspbotClient
from yahboom.protocol import DEFAULT_CMD_PORT, DEFAULT_FPS, DEFAULT_OBS_PORT, RobotCommand

from .controller import ACTION_LEFT, ACTION_RIGHT, ACTION_STRAIGHT, CameraSNNController
from .remote_driver import YahboomRemoteDriver
from .rerun_viz import init_rerun, log_camera_snn
from .train import WEIGHTS_DIR, list_weights
from .vision import FrameLaneSensor, annotate_debug_frame, interpret_lane_state

logger = logging.getLogger(__name__)

ACTION_NAMES = {ACTION_LEFT: "LEFT", ACTION_STRAIGHT: "FWD", ACTION_RIGHT: "RIGHT"}


def _default_weights() -> str:
    files = list_weights()
    if not files:
        raise FileNotFoundError(
            f"No weights in {WEIGHTS_DIR}. Train first: uv run python -m camera.run_training"
        )
    return os.path.join(WEIGHTS_DIR, sorted(files)[-1])


def precise_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote camera SNN deploy with Rerun debug")
    from yahboom.robot_config import DEFAULT_ROBOT_IP

    parser.add_argument(
        "--remote-ip",
        default=DEFAULT_ROBOT_IP,
        help=f"Raspberry Pi IP address (default: {DEFAULT_ROBOT_IP})",
    )
    parser.add_argument("--weights", type=str, default=None, help="Path to .npy weights")
    parser.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT)
    parser.add_argument("--obs-port", type=int, default=DEFAULT_OBS_PORT)
    parser.add_argument("--hz", type=float, default=15.0, help="Control loop rate")
    parser.add_argument("--threshold", type=int, default=60, help="Dark-tape threshold (black tape only)")
    parser.add_argument(
        "--tape-color",
        choices=["auto", "purple", "dark"],
        default="auto",
        help="Tape detection: auto=purple+black (default)",
    )
    parser.add_argument("--base-speed", type=int, default=35)
    parser.add_argument("--steer-delta", type=int, default=18)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--no-rerun", action="store_true", help="Disable Rerun viewer")
    parser.add_argument("--no-drive", action="store_true", help="Debug only: do not send motor commands")
    parser.add_argument("--iterations", type=int, default=0, help="0 = until Ctrl+C")
    args = parser.parse_args()

    weights_path = args.weights or _default_weights()
    if not os.path.isfile(weights_path):
        print(f"Weights not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  CAMERA SNN — Remote Deploy")
    print("=" * 60)
    print(f"Robot:   {args.remote_ip}")
    print(f"Weights: {weights_path}")
    print(f"Speed:   base={args.base_speed} steer={args.steer_delta}")
    if args.no_drive:
        print("Mode:    DEBUG (motors disabled)")
    print("Ctrl+C to stop.\n")

    client = RaspbotClient(
        remote_ip=args.remote_ip,
        cmd_port=args.cmd_port,
        obs_port=args.obs_port,
        connect_timeout_s=args.connect_timeout_s,
        polling_timeout_ms=50,
    )

    try:
        client.connect()
    except ConnectionError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if not args.no_rerun:
        init_rerun("camera_snn_deploy")

    sensor = FrameLaneSensor(window_size=5, threshold=args.threshold, tape_color=args.tape_color)
    controller = CameraSNNController(weights_path)
    driver = YahboomRemoteDriver(client, base_speed=args.base_speed, steer_delta=args.steer_delta)

    period = 1.0 / args.hz
    step = 0

    try:
        while True:
            loop_start = time.perf_counter()
            client.poll_observation()
            frame = client.last_frame
            sensors = sensor.read_from_frame(frame)
            action, sensors = controller.run_inference(sensors)
            lane_state = interpret_lane_state(sensors)

            if not args.no_drive:
                driver.execute_action(action)
            else:
                client.send_command(RobotCommand(movement="stop"))

            if step % 3 == 0:
                print(
                    f"[{step:4d}] {lane_state:40s}  "
                    f"L={sensors[0]:.2f} C={sensors[1]:.2f} R={sensors[2]:.2f}  "
                    f"action={ACTION_NAMES.get(action, action)}"
                )

            if not args.no_rerun and frame is not None:
                debug_rgb = annotate_debug_frame(
                    frame,
                    sensors,
                    sensor.last_column_darkness,
                    action=action,
                    threshold=args.threshold,
                    tape_color=args.tape_color,
                )
                log_camera_snn(
                    frame_idx=step,
                    camera_rgb=frame,
                    debug_rgb=debug_rgb,
                    sensors=sensors,
                    column_darkness=sensor.last_column_darkness,
                    lane_state=lane_state,
                    action=action,
                )

            step += 1
            if args.iterations and step >= args.iterations:
                break

            elapsed = time.perf_counter() - loop_start
            precise_sleep(max(period - elapsed, 0.0))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        driver.stop()
        client.disconnect()
        print("Stopped.")


if __name__ == "__main__":
    main()
