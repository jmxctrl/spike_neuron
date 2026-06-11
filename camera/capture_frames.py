#!/usr/bin/env python3
"""
Save camera frames from the robot for Roboflow labeling.

Pi:  sudo PYTHONPATH=. python3 -m yahboom.host

Mac:
  uv run python -m camera.capture_frames --duration 60 --out datasets/roboflow_raw
  # Upload images/ to Roboflow → annotate purple_line + black_bar → Train YOLOv8n-seg
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from yahboom.client import RaspbotClient
from yahboom.protocol import DEFAULT_CMD_PORT, DEFAULT_OBS_PORT, RobotCommand


def main() -> None:
    from yahboom.robot_config import DEFAULT_ROBOT_IP

    parser = argparse.ArgumentParser(description="Capture robot camera frames for CV training")
    parser.add_argument("--remote-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT)
    parser.add_argument("--obs-port", type=int, default=DEFAULT_OBS_PORT)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--hz", type=float, default=2.0, help="Capture rate (keep low; 2 Hz is enough)")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument(
        "--out",
        type=str,
        default="datasets/roboflow_raw",
        help="Output directory (upload contents to Roboflow)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out) / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

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
    period = 1.0 / args.hz
    start = time.perf_counter()
    n = 0
    print(f"Saving to {out_dir}  ({args.hz} Hz for {args.duration:.0f}s)")
    print("Move the robot / push it through different positions and lighting.\n")

    try:
        while time.perf_counter() - start < args.duration:
            t0 = time.perf_counter()
            client.poll_observation()
            frame = client.last_frame
            if frame is not None and frame.size > 0:
                if frame.ndim == 3 and frame.shape[2] == 3:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    bgr = frame
                path = out_dir / f"frame_{n:04d}.jpg"
                cv2.imwrite(str(path), bgr)
                n += 1
                if n % 10 == 0:
                    print(f"  saved {n} frames")
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        client.disconnect()

    print(f"\nDone: {n} frames in {out_dir}")
    print("Next: upload to Roboflow, segment-label purple_line (+ black_bar), train YOLOv8n-seg.")


if __name__ == "__main__":
    main()
