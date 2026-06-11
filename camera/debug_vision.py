#!/usr/bin/env python3
"""
Debug camera lane sensors from the robot stream (laptop only, no driving).

Pi:
    sudo PYTHONPATH=. python3 -m yahboom.host

Mac:
    uv run python -m camera.debug_vision --remote-ip <PI_IP>
    uv run python -m camera.debug_vision --remote-ip <PI_IP> --threshold 45
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from yahboom.client import RaspbotClient
from yahboom.protocol import DEFAULT_CMD_PORT, DEFAULT_OBS_PORT, RobotCommand

from .rerun_viz import init_rerun, log_camera_snn
from .sensors import add_lane_sensor_args, build_lane_sensor
from .vision import annotate_debug_frame, interpret_lane_state

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug lane sensors from robot camera (Rerun)")
    from yahboom.robot_config import DEFAULT_ROBOT_IP

    parser.add_argument(
        "--remote-ip",
        default=DEFAULT_ROBOT_IP,
        help=f"Raspberry Pi IP address (default: {DEFAULT_ROBOT_IP})",
    )
    parser.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT)
    parser.add_argument("--obs-port", type=int, default=DEFAULT_OBS_PORT)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--sensor-window", type=int, default=5)
    parser.add_argument("--threshold", type=int, default=60, help="Dark-tape threshold (black tape only)")
    parser.add_argument(
        "--tape-color",
        choices=["auto", "purple", "dark"],
        default="auto",
        help="Tape detection: auto=purple+black (default), purple, dark",
    )
    add_lane_sensor_args(parser)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--no-rerun", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  CAMERA VISION DEBUG")
    print("=" * 60)
    print("Centered in corridor → C high. Drift left → L high. Drift right → R high.")
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

    client.send_command(RobotCommand(movement="stop"))

    if not args.no_rerun:
        init_rerun("camera_vision_debug")

    if args.sensor_backend == "cv" and not args.cv_model:
        print("--sensor-backend cv requires --cv-model path/to/best.pt", file=sys.stderr)
        sys.exit(1)
    sensor = build_lane_sensor(args)
    period = 1.0 / args.hz
    step = 0

    try:
        while True:
            t0 = time.perf_counter()
            client.poll_observation()
            frame = client.last_frame
            sensors = sensor.read_from_frame(frame)
            lane_state = interpret_lane_state(sensors)

            print(
                f"[{step:4d}] {lane_state:40s}  "
                f"L={sensors[0]:.2f} C={sensors[1]:.2f} R={sensors[2]:.2f}  "
                f"dark L={sensor.last_column_darkness[0]:.2f} "
                f"C={sensor.last_column_darkness[1]:.2f} R={sensor.last_column_darkness[2]:.2f}"
            )

            if not args.no_rerun and frame is not None:
                debug_rgb = annotate_debug_frame(
                    frame,
                    sensors,
                    sensor.last_column_darkness,
                    threshold=args.threshold,
                    tape_color=args.tape_color,
                    cv_lane_mask=sensor.lane_mask_for_debug(frame),
                    backend_label=sensor.last_backend_used,
                )
                log_camera_snn(
                    frame_idx=step,
                    camera_rgb=frame,
                    debug_rgb=debug_rgb,
                    sensors=sensors,
                    column_darkness=sensor.last_column_darkness,
                    lane_state=lane_state,
                )

            step += 1
            elapsed = time.perf_counter() - t0
            time.sleep(max(period - elapsed, 0.0))
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        client.send_command(RobotCommand(movement="stop"))
        client.disconnect()


if __name__ == "__main__":
    main()
