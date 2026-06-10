#!/usr/bin/env python3
"""
Run camera corridor SNN on Yahboom robot (Raspberry Pi).

  sudo PYTHONPATH=. python3 -m camera.deploy --weights camera/trained_weights/camera_....npy
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time

from yahboom.camera import DEFAULT_CAMERA_DEVICE, open_camera

from .controller import CameraSNNController
from .motors import YahboomDriver
from .train import WEIGHTS_DIR, list_weights
from .vision import CameraLaneSensor


def _default_weights() -> str:
    files = list_weights()
    if not files:
        raise FileNotFoundError(
            f"No weights in {WEIGHTS_DIR}. Train first: uv run python -m camera.run_training"
        )
    return os.path.join(WEIGHTS_DIR, sorted(files)[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy camera SNN on Yahboom")
    parser.add_argument("--weights", type=str, default=None, help="Path to .npy weights")
    parser.add_argument("--camera-device", type=str, default=DEFAULT_CAMERA_DEVICE)
    parser.add_argument("--threshold", type=int, default=60, help="Tape darkness threshold")
    parser.add_argument("--hz", type=float, default=15.0, help="Control loop rate")
    parser.add_argument("--base-speed", type=int, default=70)
    parser.add_argument("--steer-delta", type=int, default=35)
    parser.add_argument("--iterations", type=int, default=0, help="0 = run until Ctrl+C")
    args = parser.parse_args()

    weights_path = args.weights or _default_weights()
    if not os.path.isfile(weights_path):
        print(f"Weights not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    cam = open_camera(device=args.camera_device, backend="v4l2")
    if cam is None:
        print("Failed to open camera", file=sys.stderr)
        sys.exit(1)

    sensor = CameraLaneSensor(cam, window_size=5, threshold=args.threshold)
    controller = CameraSNNController(weights_path, sensor)
    driver = YahboomDriver(base_speed=args.base_speed, steer_delta=args.steer_delta)

    running = True

    def _stop(*_args) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print("=" * 60)
    print("CAMERA SNN DEPLOY — Yahboom")
    print("=" * 60)
    print(f"Weights: {weights_path}")
    print(f"Camera:  {args.camera_device}")
    print(f"Speed:   base={args.base_speed} steer={args.steer_delta}")
    print("Ctrl+C to stop.\n")

    driver.start()
    period = 1.0 / args.hz
    step = 0

    try:
        while running:
            t0 = time.perf_counter()
            action, sensors = controller.run_inference()
            driver.execute_action(action)

            if step % 5 == 0:
                names = {-1: "LEFT", 0: "FWD", 1: "RIGHT"}
                print(
                    f"[{step:4d}] action={names.get(action, action):5s}  "
                    f"sensors L={sensors[0]:.2f} C={sensors[1]:.2f} R={sensors[2]:.2f}"
                )

            step += 1
            if args.iterations and step >= args.iterations:
                break

            elapsed = time.perf_counter() - t0
            time.sleep(max(period - elapsed, 0.0))
    finally:
        driver.close()
        sensor.close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
