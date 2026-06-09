"""Rerun visualization helpers for remote Yahboom teleop."""

from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import RobotCommand


def init_rerun(session_name: str = "yahboom_teleop") -> None:
    import rerun as rr

    rr.init(session_name, spawn=True)


def log_teleop_data(
    observation: dict[str, Any] | None = None,
    command: RobotCommand | None = None,
) -> None:
    import rerun as rr

    if observation:
        for key, value in observation.items():
            if value is None:
                continue
            path = f"observation/{key}"
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                rr.log(path, rr.Image(value))
            elif isinstance(value, (bool, int, float, np.number)):
                rr.log(path, rr.Scalars(float(value)))

    if command:
        cmd = command.to_dict()
        for key, value in cmd.items():
            if value is None or key == "beep":
                continue
            if isinstance(value, (bool, int, float, str)):
                if key == "movement":
                    rr.log(f"action/{key}", rr.TextLog(str(value)))
                else:
                    rr.log(
                        f"action/{key}",
                        rr.Scalars(float(value) if isinstance(value, bool) else value),
                    )
