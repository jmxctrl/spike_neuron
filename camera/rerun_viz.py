"""Rerun visualization for remote camera SNN deploy."""

from __future__ import annotations

from typing import Any

import numpy as np


def init_rerun(session_name: str = "camera_snn") -> None:
    import rerun as rr

    rr.init(session_name, spawn=True)


def log_camera_snn(
    *,
    frame_idx: int,
    camera_rgb: np.ndarray | None = None,
    debug_rgb: np.ndarray | None = None,
    sensors: list[float] | None = None,
    column_darkness: tuple[float, float, float] | None = None,
    lane_state: str | None = None,
    action: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    import rerun as rr

    rr.set_time("frame", sequence=frame_idx)

    if camera_rgb is not None and camera_rgb.ndim >= 2:
        rr.log("camera/raw", rr.Image(camera_rgb))

    if debug_rgb is not None and debug_rgb.ndim >= 2:
        rr.log("camera/debug", rr.Image(debug_rgb))

    if sensors is not None:
        names = ("left", "center", "right")
        for name, val in zip(names, sensors):
            rr.log(f"sensors/{name}", rr.Scalars(float(val)))

    if column_darkness is not None:
        for name, val in zip(("left_dark", "center_dark", "right_dark"), column_darkness):
            rr.log(f"columns/{name}", rr.Scalars(float(val)))

    if lane_state:
        rr.log("debug/lane_state", rr.TextLog(lane_state))

    if action is not None:
        action_names = {-1: "LEFT", 0: "FORWARD", 1: "RIGHT"}
        rr.log("action/steer", rr.TextLog(action_names.get(action, str(action))))
        rr.log("action/value", rr.Scalars(float(action)))

    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            if isinstance(value, (bool, int, float, np.number)):
                rr.log(f"extra/{key}", rr.Scalars(float(value)))
