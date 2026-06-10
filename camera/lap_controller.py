"""Ping-pong lap logic: detect end bar, turn 180°, cruise back, repeat."""

from __future__ import annotations

import time
from enum import Enum

import numpy as np

from .remote_driver import YahboomRemoteDriver
from .vision import EndLineResult, detect_end_line


class LapPhase(str, Enum):
    CRUISE = "CRUISE"
    TURNING = "TURNING"
    COOLDOWN = "COOLDOWN"


class LapController:
    """
    State machine for endless back-and-forth laps.

    After a 180° turn the robot faces the opposite direction, so the same
  forward-following SNN applies — no retrain required.
    """

    def __init__(
        self,
        driver: YahboomRemoteDriver,
        *,
        turn_speed: int = 45,
        turn_seconds: float = 2.6,
        end_confirm_frames: int = 6,
        cooldown_seconds: float = 1.0,
        bottom_fraction: float = 0.20,
        tape_color: str = "auto",
        threshold: int = 60,
    ):
        self._driver = driver
        self.turn_speed = turn_speed
        self.turn_seconds = turn_seconds
        self.end_confirm_frames = end_confirm_frames
        self.cooldown_seconds = cooldown_seconds
        self.bottom_fraction = bottom_fraction
        self.tape_color = tape_color
        self.threshold = threshold

        self.phase = LapPhase.CRUISE
        self.lap_count = 0
        self._end_streak = 0
        self._phase_until = 0.0
        self.last_end_line = EndLineResult(False, 0.0, 0.0, 0.0)

    def step(self, frame: np.ndarray | None, snn_action: int) -> tuple[int | None, str]:
        """
        Returns (action_for_driver, status).

        action_for_driver is None when the lap controller owns the motors (turning).
        """
        now = time.perf_counter()

        if self.phase == LapPhase.TURNING:
            if now < self._phase_until:
                self._driver.spin_left(self.turn_speed)
                remaining = self._phase_until - now
                return None, f"TURNING 180° ({remaining:.1f}s left)"
            self._driver.stop()
            self.lap_count += 1
            self.phase = LapPhase.COOLDOWN
            self._phase_until = now + self.cooldown_seconds
            self._end_streak = 0
            return None, f"COOLDOWN — lap {self.lap_count} complete"

        if self.phase == LapPhase.COOLDOWN:
            self._driver.stop()
            if now < self._phase_until:
                return None, f"COOLDOWN ({self._phase_until - now:.1f}s)"
            self.phase = LapPhase.CRUISE
            self._end_streak = 0

        self.last_end_line = detect_end_line(
            frame if frame is not None else np.empty((0, 0, 3), dtype=np.uint8),
            bottom_fraction=self.bottom_fraction,
            tape_color=self.tape_color,
            threshold=self.threshold,
        )

        if self.last_end_line.detected:
            self._end_streak += 1
        else:
            self._end_streak = max(0, self._end_streak - 1)

        if self._end_streak >= self.end_confirm_frames:
            self.phase = LapPhase.TURNING
            self._phase_until = now + self.turn_seconds
            self._end_streak = 0
            return None, "END LINE — turning 180°"

        lap_tag = f" lap={self.lap_count}" if self.lap_count else ""
        end_tag = ""
        if self._end_streak > 0:
            end_tag = f" end={self._end_streak}/{self.end_confirm_frames}"
        return snn_action, f"CRUISE{lap_tag}{end_tag}"
