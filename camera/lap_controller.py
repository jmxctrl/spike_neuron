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
    CLEARING = "CLEARING"


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
        turn_speed: int = 68,
        turn_seconds: float = 1.7,
        end_confirm_frames: int = 6,
        clear_seconds: float = 4.0,
        end_lockout_seconds: float = 3.0,
        bottom_fraction: float = 0.20,
        tape_color: str = "auto",
        threshold: int = 60,
    ):
        self._driver = driver
        self.turn_speed = turn_speed
        self.turn_seconds = turn_seconds
        self.end_confirm_frames = end_confirm_frames
        self.clear_seconds = clear_seconds
        self.end_lockout_seconds = end_lockout_seconds
        self.bottom_fraction = bottom_fraction
        self.tape_color = tape_color
        self.threshold = threshold

        self.phase = LapPhase.CRUISE
        self.lap_count = 0
        self._end_streak = 0
        self._phase_until = 0.0
        self._end_detect_unlock_at = 0.0
        self.last_end_line = EndLineResult(False, 0.0, 0.0, 0.0)

    def _update_end_line(self, frame: np.ndarray | None) -> None:
        self.last_end_line = detect_end_line(
            frame if frame is not None else np.empty((0, 0, 3), dtype=np.uint8),
            bottom_fraction=self.bottom_fraction,
            tape_color=self.tape_color,
            threshold=self.threshold,
        )

    def _end_detection_active(self, now: float) -> bool:
        return self.phase == LapPhase.CRUISE and now >= self._end_detect_unlock_at

    def step(self, frame: np.ndarray | None, snn_action: int) -> tuple[int | None, str]:
        """
        Returns (action_for_driver, status).

        action_for_driver is None when the lap controller owns the motors (turning).
        """
        now = time.perf_counter()
        self._update_end_line(frame)

        if self.phase == LapPhase.TURNING:
            if now < self._phase_until:
                self._driver.spin_left(self.turn_speed)
                return None, f"TURNING 180° ({self._phase_until - now:.1f}s)"
            self._driver.stop()
            self.lap_count += 1
            self.phase = LapPhase.CLEARING
            self._phase_until = now + self.clear_seconds
            self._end_streak = 0
            return snn_action, f"CLEARING — lap {self.lap_count} (drive away)"

        if self.phase == LapPhase.CLEARING:
            if now < self._phase_until:
                return snn_action, f"CLEARING ({self._phase_until - now:.1f}s)"
            self.phase = LapPhase.CRUISE
            self._end_streak = 0
            self._end_detect_unlock_at = now + self.end_lockout_seconds

        if not self._end_detection_active(now):
            lock_tag = ""
            if now < self._end_detect_unlock_at:
                lock_tag = f" end locked {self._end_detect_unlock_at - now:.1f}s"
            lap_tag = f" lap={self.lap_count}" if self.lap_count else ""
            return snn_action, f"CRUISE{lap_tag}{lock_tag}"

        if self.last_end_line.detected:
            self._end_streak += 1
        else:
            self._end_streak = 0

        if self._end_streak >= self.end_confirm_frames:
            self.phase = LapPhase.TURNING
            self._phase_until = now + self.turn_seconds
            self._end_streak = 0
            self._driver.stop()
            return None, "END LINE — turning 180°"

        lap_tag = f" lap={self.lap_count}" if self.lap_count else ""
        end_tag = f" end={self._end_streak}/{self.end_confirm_frames}" if self._end_streak else ""
        return snn_action, f"CRUISE{lap_tag}{end_tag}"
