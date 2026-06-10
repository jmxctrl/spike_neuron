"""Ping-pong laps: lane keep → end line → 180° spin → slow center → full speed."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .remote_driver import YahboomRemoteDriver
from .vision import (
    EndLineResult,
    centering_steer_action,
    detect_end_line,
    is_recovery_complete,
)


class LapPhase(str, Enum):
    CRUISE = "CRUISE"
    TURNING = "TURNING"
    SLOW_CENTER = "SLOW_CENTER"


@dataclass
class LapStepResult:
    action: int | None
    status: str
    speed_scale: float = 1.0
    slow_center: bool = False


class LapController:
    """
    Simple lap loop:
      1. CRUISE — SNN lane keeping at full speed
      2. TURNING — fixed 180° spin when end bar detected
      3. SLOW_CENTER — creep forward + steer until centered
      4. CRUISE — full speed again
    """

    def __init__(
        self,
        driver: YahboomRemoteDriver,
        *,
        turn_speed: int = 45,
        turn_seconds: float = 3.0,
        end_confirm_frames: int = 6,
        recover_speed_scale: float = 0.35,
        recover_center_frames: int = 12,
        end_lockout_seconds: float = 4.0,
        bottom_fraction: float = 0.20,
        tape_color: str = "auto",
        threshold: int = 60,
        end_tape_color: str = "dark",
        end_threshold: int = 75,
    ):
        self._driver = driver
        self.turn_speed = turn_speed
        self.turn_seconds = turn_seconds
        self.end_confirm_frames = end_confirm_frames
        self.recover_speed_scale = recover_speed_scale
        self.recover_center_frames = recover_center_frames
        self.end_lockout_seconds = end_lockout_seconds
        self.bottom_fraction = bottom_fraction
        self.tape_color = tape_color
        self.threshold = threshold
        self.end_tape_color = end_tape_color
        self.end_threshold = end_threshold

        self.phase = LapPhase.CRUISE
        self.lap_count = 0
        self._end_streak = 0
        self._center_streak = 0
        self._phase_until = 0.0
        self._end_detect_unlock_at = 0.0
        self.last_end_line = EndLineResult(False, 0.0, 0.0, 0.0)

    def _update_end_line(self, frame: np.ndarray | None) -> None:
        self.last_end_line = detect_end_line(
            frame if frame is not None else np.empty((0, 0, 3), dtype=np.uint8),
            bottom_fraction=self.bottom_fraction,
            tape_color=self.end_tape_color,
            threshold=self.end_threshold,
        )

    def step(
        self,
        frame: np.ndarray | None,
        snn_action: int,
        sensors: list[float],
    ) -> LapStepResult:
        now = time.perf_counter()
        self._update_end_line(frame)

        if self.phase == LapPhase.TURNING:
            remaining = self._phase_until - now
            if remaining > 0:
                self._driver.spin_left(self.turn_speed)
                return LapStepResult(None, f"TURNING 180° ({remaining:.1f}s)")

            self._driver.stop()
            self.lap_count += 1
            self.phase = LapPhase.SLOW_CENTER
            self._center_streak = 0
            steer = centering_steer_action(sensors)
            return LapStepResult(
                steer,
                f"SLOW CENTER — lap {self.lap_count} (forward + steer)",
                self.recover_speed_scale,
                slow_center=True,
            )

        if self.phase == LapPhase.SLOW_CENTER:
            steer = centering_steer_action(sensors)
            if is_recovery_complete(sensors):
                self._center_streak += 1
            else:
                self._center_streak = 0

            if self._center_streak >= self.recover_center_frames:
                self.phase = LapPhase.CRUISE
                self._end_streak = 0
                self._end_detect_unlock_at = now + self.end_lockout_seconds
                return LapStepResult(snn_action, "CRUISE — centered, full speed", 1.0)

            return LapStepResult(
                steer,
                f"SLOW CENTER {self._center_streak}/{self.recover_center_frames}",
                self.recover_speed_scale,
                slow_center=True,
            )

        if now < self._end_detect_unlock_at:
            return LapStepResult(
                snn_action,
                f"CRUISE lap={self.lap_count} (end locked {self._end_detect_unlock_at - now:.0f}s)",
                1.0,
            )

        if self.last_end_line.detected:
            self._end_streak += 1
        else:
            self._end_streak = 0

        if self._end_streak >= self.end_confirm_frames:
            self.phase = LapPhase.TURNING
            self._phase_until = now + self.turn_seconds
            self._end_streak = 0
            self._driver.stop()
            return LapStepResult(None, "END LINE — turning 180°")

        end_tag = f" end={self._end_streak}/{self.end_confirm_frames}" if self._end_streak else ""
        lap_tag = f" lap={self.lap_count}" if self.lap_count else ""
        return LapStepResult(snn_action, f"CRUISE{lap_tag}{end_tag}", 1.0)
