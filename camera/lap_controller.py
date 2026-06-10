"""Ping-pong lap logic: detect end bar, turn 180°, cruise back, repeat."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .remote_driver import YahboomRemoteDriver
from .vision import (
    EndLineResult,
    centering_steer_action,
    corridor_visible,
    detect_end_line,
    is_lane_centered,
)


class LapPhase(str, Enum):
    CRUISE = "CRUISE"
    TURNING = "TURNING"
    RECOVERING = "RECOVERING"


@dataclass
class LapStepResult:
    action: int | None
    status: str
    speed_scale: float = 1.0


class LapController:
    """
    State machine for endless back-and-forth laps.

    Turnaround stops on vision (corridor visible), then a slow sensor-based
    recovery recenters the robot before full-speed SNN resumes.
    """

    def __init__(
        self,
        driver: YahboomRemoteDriver,
        *,
        turn_speed: int = 50,
        min_turn_seconds: float = 1.0,
        max_turn_seconds: float = 4.0,
        corridor_confirm_frames: int = 4,
        end_confirm_frames: int = 6,
        recover_speed_scale: float = 0.35,
        recover_center_frames: int = 18,
        min_recover_seconds: float = 3.0,
        max_recover_seconds: float = 12.0,
        end_lockout_seconds: float = 3.0,
        bottom_fraction: float = 0.20,
        tape_color: str = "auto",
        threshold: int = 60,
    ):
        self._driver = driver
        self.turn_speed = turn_speed
        self.min_turn_seconds = min_turn_seconds
        self.max_turn_seconds = max_turn_seconds
        self.corridor_confirm_frames = corridor_confirm_frames
        self.end_confirm_frames = end_confirm_frames
        self.recover_speed_scale = recover_speed_scale
        self.recover_center_frames = recover_center_frames
        self.min_recover_seconds = min_recover_seconds
        self.max_recover_seconds = max_recover_seconds
        self.end_lockout_seconds = end_lockout_seconds
        self.bottom_fraction = bottom_fraction
        self.tape_color = tape_color
        self.threshold = threshold

        self.phase = LapPhase.CRUISE
        self.lap_count = 0
        self._end_streak = 0
        self._corridor_streak = 0
        self._center_streak = 0
        self._phase_started_at = 0.0
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

    def _begin_turn(self, now: float) -> LapStepResult:
        self.phase = LapPhase.TURNING
        self._phase_started_at = now
        self._corridor_streak = 0
        self._center_streak = 0
        self._end_streak = 0
        self._driver.stop()
        return LapStepResult(None, "END LINE — turning until corridor visible")

    def _begin_recover(self, now: float) -> None:
        self.lap_count += 1
        self.phase = LapPhase.RECOVERING
        self._phase_started_at = now
        self._center_streak = 0
        self._driver.stop()

    def step(
        self,
        frame: np.ndarray | None,
        snn_action: int,
        sensors: list[float],
    ) -> LapStepResult:
        now = time.perf_counter()
        self._update_end_line(frame)

        if self.phase == LapPhase.TURNING:
            elapsed = now - self._phase_started_at
            if elapsed >= self.min_turn_seconds and corridor_visible(sensors):
                self._corridor_streak += 1
            else:
                self._corridor_streak = 0

            stop_spin = (
                elapsed >= self.min_turn_seconds
                and self._corridor_streak >= self.corridor_confirm_frames
            ) or elapsed >= self.max_turn_seconds

            if not stop_spin:
                self._driver.spin_left(self.turn_speed)
                return LapStepResult(None, f"TURNING ({elapsed:.1f}s, corridor={self._corridor_streak})")

            self._begin_recover(now)
            steer = centering_steer_action(sensors)
            return LapStepResult(
                steer,
                f"RECOVERING — lap {self.lap_count} (slow center)",
                self.recover_speed_scale,
            )

        if self.phase == LapPhase.RECOVERING:
            elapsed = now - self._phase_started_at
            steer = centering_steer_action(sensors)
            if is_lane_centered(sensors):
                self._center_streak += 1
            else:
                self._center_streak = 0

            done = (
                elapsed >= self.min_recover_seconds
                and self._center_streak >= self.recover_center_frames
            ) or elapsed >= self.max_recover_seconds

            if done:
                self.phase = LapPhase.CRUISE
                self._end_streak = 0
                self._end_detect_unlock_at = now + self.end_lockout_seconds
                return LapStepResult(snn_action, f"CRUISE lap={self.lap_count} (recovered)", 1.0)

            return LapStepResult(
                steer,
                f"RECOVERING center={self._center_streak}/{self.recover_center_frames} ({elapsed:.1f}s)",
                self.recover_speed_scale,
            )

        if not self._end_detection_active(now):
            lock_tag = ""
            if now < self._end_detect_unlock_at:
                lock_tag = f" end locked {self._end_detect_unlock_at - now:.1f}s"
            lap_tag = f" lap={self.lap_count}" if self.lap_count else ""
            return LapStepResult(snn_action, f"CRUISE{lap_tag}{lock_tag}", 1.0)

        if self.last_end_line.detected:
            self._end_streak += 1
        else:
            self._end_streak = 0

        if self._end_streak >= self.end_confirm_frames:
            return self._begin_turn(now)

        lap_tag = f" lap={self.lap_count}" if self.lap_count else ""
        end_tag = f" end={self._end_streak}/{self.end_confirm_frames}" if self._end_streak else ""
        return LapStepResult(snn_action, f"CRUISE{lap_tag}{end_tag}", 1.0)
