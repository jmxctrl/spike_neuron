"""Ping-pong laps: lane keep → black end bar / tape lost → 180° → drive back."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .remote_driver import YahboomRemoteDriver
from .vision import (
    EndLineResult,
    detect_end_line,
    is_dead_end,
    is_post_turn_ready,
    is_tape_lost,
)


class LapPhase(str, Enum):
    CRUISE = "CRUISE"
    TURNING = "TURNING"
    SETTLING = "SETTLING"


@dataclass
class LapStepResult:
    action: float | None
    status: str
    speed_scale: float = 1.0
    slow_center: bool = False


class LapController:
    """
    Ping-pong loop — turn ONLY when:
      1. CV/heuristic detects horizontal black end bar, OR
      2. All lane tape lost ahead (corridor ended)

    Drifting onto one purple line (L=1, R=0) does NOT trigger a turn.
    """

    def __init__(
        self,
        driver: YahboomRemoteDriver,
        *,
        turn_speed: int = 40,
        turn_180_seconds: float = 3.0,
        turn_max_seconds: float = 5.0,
        turn_settle_seconds: float = 0.5,
        end_confirm_frames: int = 4,
        tape_lost_confirm_frames: int = 10,
        post_turn_grace_seconds: float = 8.0,
        bottom_fraction: float = 0.20,
        end_tape_color: str = "dark",
        end_threshold: int = 75,
        end_line_fn: Callable[[np.ndarray | None], EndLineResult] | None = None,
    ):
        self._driver = driver
        self._end_line_fn = end_line_fn
        self.turn_speed = turn_speed
        self.turn_180_seconds = turn_180_seconds
        self.turn_max_seconds = turn_max_seconds
        self.turn_settle_seconds = turn_settle_seconds
        self.end_confirm_frames = end_confirm_frames
        self.tape_lost_confirm_frames = tape_lost_confirm_frames
        self.post_turn_grace_seconds = post_turn_grace_seconds
        self.bottom_fraction = bottom_fraction
        self.end_tape_color = end_tape_color
        self.end_threshold = end_threshold

        self.phase = LapPhase.CRUISE
        self.lap_count = 0
        self._end_streak = 0
        self._end_trigger = ""
        self._facing_streak = 0
        self._turn_started_at = 0.0
        self._settle_until = 0.0
        self._post_turn_grace_until = 0.0
        self.last_end_line = EndLineResult(False, 0.0, 0.0, 0.0)

    def _update_end_line(self, frame: np.ndarray | None) -> None:
        if self._end_line_fn is not None:
            self.last_end_line = self._end_line_fn(frame)
        else:
            self.last_end_line = detect_end_line(
                frame if frame is not None else np.empty((0, 0, 3), dtype=np.uint8),
                bottom_fraction=self.bottom_fraction,
                tape_color=self.end_tape_color,
                threshold=self.end_threshold,
            )

    def _black_bar_visible(self) -> bool:
        end = self.last_end_line
        return end.detected and end.width_span >= 0.40 and end.row_coverage >= 0.12

    def _turn_trigger(self, sensors: list[float]) -> str:
        if self._black_bar_visible():
            return "black_bar"
        if is_tape_lost(sensors):
            return "tape_lost"
        return ""

    def _confirm_frames_for(self, trigger: str) -> int:
        if trigger == "black_bar":
            end = self.last_end_line
            if end.width_span >= 0.65 and end.row_coverage >= 0.30:
                return max(2, self.end_confirm_frames // 2)
            return self.end_confirm_frames
        if trigger == "tape_lost":
            return self.tape_lost_confirm_frames
        return self.end_confirm_frames

    def _begin_settling(self, now: float, status: str) -> LapStepResult:
        self._driver.stop()
        self.phase = LapPhase.SETTLING
        self._settle_until = now + self.turn_settle_seconds
        self._facing_streak = 0
        return LapStepResult(None, status)

    def _begin_post_turn_cruise(
        self,
        now: float,
        snn_action: float,
        sensors: list[float],
        status: str,
    ) -> LapStepResult:
        self.lap_count += 1
        self.phase = LapPhase.CRUISE
        self._end_streak = 0
        self._end_trigger = ""
        self._post_turn_grace_until = now + self.post_turn_grace_seconds
        action = 0.0 if is_dead_end(sensors) else snn_action
        return LapStepResult(action, status, 1.0)

    def step(
        self,
        frame: np.ndarray | None,
        snn_action: float,
        sensors: list[float],
    ) -> LapStepResult:
        now = time.perf_counter()
        self._update_end_line(frame)

        if self.phase == LapPhase.TURNING:
            elapsed = now - self._turn_started_at
            self._driver.spin_left(self.turn_speed)

            if is_post_turn_ready(sensors):
                self._facing_streak += 1
            else:
                self._facing_streak = 0

            if elapsed >= self.turn_180_seconds or elapsed >= self.turn_max_seconds:
                return self._begin_settling(now, f"TURN DONE — 180° ({elapsed:.1f}s)")

            return LapStepResult(
                None,
                f"TURNING {elapsed:.1f}s / {self.turn_180_seconds:.1f}s",
            )

        if self.phase == LapPhase.SETTLING:
            self._driver.stop()
            remaining = self._settle_until - now
            if remaining > 0:
                return LapStepResult(None, f"SETTLING ({remaining:.1f}s)")

            return self._begin_post_turn_cruise(
                now,
                snn_action,
                sensors,
                f"CRUISE — drive back (lap {self.lap_count + 1})",
            )

        if now < self._post_turn_grace_until:
            remaining = self._post_turn_grace_until - now
            action = 0.0 if is_dead_end(sensors) else snn_action
            return LapStepResult(
                action,
                f"CRUISE drive-back ({remaining:.0f}s)",
                1.0,
            )

        trigger = self._turn_trigger(sensors)
        if trigger:
            if trigger == self._end_trigger:
                self._end_streak += 1
            else:
                self._end_trigger = trigger
                self._end_streak = 1
        else:
            self._end_streak = 0
            self._end_trigger = ""

        confirm_needed = self._confirm_frames_for(self._end_trigger)

        if self._end_streak >= confirm_needed and self._end_trigger:
            self.phase = LapPhase.TURNING
            self._turn_started_at = now
            self._facing_streak = 0
            self._end_streak = 0
            reason = "black bar" if self._end_trigger == "black_bar" else "tape lost"
            self._end_trigger = ""
            self._driver.stop()
            return LapStepResult(None, f"END ({reason}) — turning 180°")

        if self._end_streak > 0:
            tag = "black bar" if self._end_trigger == "black_bar" else "tape lost"
            return LapStepResult(
                0.0,
                f"END APPROACH {tag} {self._end_streak}/{confirm_needed}",
                0.0,
            )

        lap_tag = f" lap={self.lap_count}" if self.lap_count else ""
        return LapStepResult(snn_action, f"CRUISE{lap_tag}", 1.0)
