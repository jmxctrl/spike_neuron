"""Shared ZMQ message schema for Yahboom host/client teleop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_CMD_PORT = 5555
DEFAULT_OBS_PORT = 5556
DEFAULT_FPS = 30
DEFAULT_WATCHDOG_MS = 500


@dataclass
class RobotCommand:
    """Command sent from laptop -> robot host each control cycle."""

    movement: str = "stop"  # forward | backward | turn_left | turn_right | stop
    speed: int = 100
    pan: int = 90
    tilt: int = 90
    red_led: bool = False
    blue_led: bool = False
    beep: dict[str, float] | None = None  # one-shot: {"duration": 0.1, "frequency": 440}
    quit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RobotObservation:
    """Observation sent from robot host -> laptop each control cycle."""

    distance_cm: float = -1.0
    line_tracker: dict[str, bool] = field(
        default_factory=lambda: {"left1": False, "left2": False, "right1": False, "right2": False}
    )
    ir_obstacle: dict[str, bool] = field(default_factory=lambda: {"left": False, "right": False})
    camera_jpeg_b64: str | None = None
    pan: int = 90
    tilt: int = 90
    speed: int = 100
    red_led: bool = False
    blue_led: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotObservation:
        return cls(
            distance_cm=float(data.get("distance_cm", -1.0)),
            line_tracker=dict(data.get("line_tracker", {})),
            ir_obstacle=dict(data.get("ir_obstacle", {})),
            camera_jpeg_b64=data.get("camera_jpeg_b64"),
            pan=int(data.get("pan", 90)),
            tilt=int(data.get("tilt", 90)),
            speed=int(data.get("speed", 100)),
            red_led=bool(data.get("red_led", False)),
            blue_led=bool(data.get("blue_led", False)),
        )
