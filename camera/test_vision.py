#!/usr/bin/env python3
"""
Live preview of camera lane sensors (run on Pi).

  sudo PYTHONPATH=. python3 -m camera.test_vision
  sudo PYTHONPATH=. python3 -m camera.test_vision --threshold 45
"""

from __future__ import annotations

import argparse
import sys
import time

from yahboom.camera import DEFAULT_CAMERA_DEVICE, open_camera

from .vision import CameraLaneSensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Test camera lane sensor extraction")
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--device", type=str, default=DEFAULT_CAMERA_DEVICE)
    args = parser.parse_args()

    cam = open_camera(device=args.device, backend="v4l2")
    if cam is None:
        print("Failed to open camera", file=sys.stderr)
        sys.exit(1)

    sensor = CameraLaneSensor(cam, window_size=5, threshold=args.threshold)
    print("Lane sensors [left, center, right] in [0,1]. Ctrl+C to stop.")
    print("Centered in corridor → center high. Drift left → left high. Drift right → right high.\n")

    try:
        while True:
            s = sensor.read()
            print(f"L={s[0]:.3f}  C={s[1]:.3f}  R={s[2]:.3f}")
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        sensor.close()


if __name__ == "__main__":
    main()
