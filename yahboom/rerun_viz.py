"""Rerun visualization helpers for remote Yahboom teleop."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from .protocol import RobotCommand


def init_rerun(session_name: str = "yahboom_teleop") -> None:
    import rerun as rr

    rr.init(session_name, spawn=True)


def log_teleop_data(
    observation: dict[str, Any] | None = None,
    command: RobotCommand | None = None,
    frame_idx: int = 0,
) -> None:
    import rerun as rr

    rr.set_time("frame", sequence=frame_idx)

    if observation:
        camera_jpeg_b64 = observation.get("camera_jpeg_b64")
        camera = observation.get("camera")

        if camera_jpeg_b64:
            try:
                jpeg_bytes = base64.b64decode(camera_jpeg_b64)
                rr.log(
                    "observation/camera",
                    rr.EncodedImage(contents=jpeg_bytes, media_type="image/jpeg"),
                )
            except (TypeError, ValueError):
                pass
        elif isinstance(camera, np.ndarray) and camera.ndim >= 2:
            rr.log("observation/camera", rr.Image(camera))

        for key, value in observation.items():
            if key in {"camera_jpeg_b64", "camera"} or value is None:
                continue
            path = f"observation/{key}"
            if isinstance(value, (bool, int, float, np.number)):
                rr.log(path, rr.Scalars(float(value)))

    if command:
        cmd = command.to_dict()
        for key, value in cmd.items():
            if value is None or key in {"beep", "center_servos", "servo_dirty"}:
                continue
            if isinstance(value, (bool, int, float, str)):
                if key == "movement":
                    rr.log(f"action/{key}", rr.TextLog(str(value)))
                else:
                    rr.log(
                        f"action/{key}",
                        rr.Scalars(float(value) if isinstance(value, bool) else value),
                    )
