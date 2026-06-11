"""Ping-pong laps: lane keep → end line → vision-guided 180° → slow center → full speed."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .remote_driver import YahboomRemoteDriver
from .vision import (
    EndLineResult,
    centering_steer_amount,
    detect_end_line,
    is_facing_corridor,
    is_recovery_complete,
)


class LapPhase(str, Enum):
    CRUISE = "CRUISE"
    TURNING = "TURNING"
    SETTLING = "SETTLING"
    SLOW_CENTER = "SLOW_CENTER"


@dataclass
class LapStepResult:
    action: float | None
    status: str
    speed_scale: float = 1.0
    slow_center: bool = False


class LapController:
    """
    Simple lap loop:
      1. CRUISE — SNN lane keeping at full speed
      2. TURNING — slow spin until camera sees both lane lines (not fixed timer)
      3. SETTLING — brief stop so the chassis stops sliding
      4. SLOW_CENTER — creep forward + steer until centered and aligned
      5. CRUISE — full speed again
    """

    def __init__(
        self,
        driver: YahboomRemoteDriver,
        *,
        turn_speed: int = 32,
        turn_min_seconds: float = 1.4,
        turn_max_seconds: float = 5.5,
        facing_confirm_frames: int = 5,
        turn_settle_seconds: float = 0.5,
        end_confirm_frames: int = 6,
        recover_speed_scale: float = 0.35,
        recover_center_frames: int = 15,
        end_lockout_seconds: float = 4.0,
        bottom_fraction: float = 0.20,
        tape_color: str = "auto",
        threshold: int = 60,
        end_tape_color: str = "dark",
        end_threshold: int = 75,
    ):
        self._driver = driver
        self.turn_speed = turn_speed
        self.turn_min_seconds = turn_min_seconds
        self.turn_max_seconds = turn_max_seconds
        self.facing_confirm_frames = facing_confirm_frames
        self.turn_settle_seconds = turn_settle_seconds
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
        self._facing_streak = 0
        self._center_streak = 0
        self._turn_started_at = 0.0
        self._settle_until = 0.0
        self._end_detect_unlock_at = 0.0
        self.last_end_line = EndLineResult(False, 0.0, 0.0, 0.0)

    def _update_end_line(self, frame: np.ndarray | None) -> None:
        self.last_end_line = detect_end_line(
            frame if frame is not None else np.empty((0, 0, 3), dtype=np.uint8),
            bottom_fraction=self.bottom_fraction,
            tape_color=self.end_tape_color,
            threshold=self.end_threshold,
        )

    def _begin_settling(self, now: float, status: str) -> LapStepResult:
        self._driver.stop()
        self.phase = LapPhase.SETTLING
        self._settle_until = now + self.turn_settle_seconds
        self._facing_streak = 0
        return LapStepResult(None, status)

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

            if is_facing_corridor(sensors):
                self._facing_streak += 1
            else:
                self._facing_streak = 0

            vision_done = (
                elapsed >= self.turn_min_seconds
                and self._facing_streak >= self.facing_confirm_frames
            )
            timed_out = elapsed >= self.turn_max_seconds

            if vision_done:
                return self._begin_settling(
                    now,
                    f"TURN DONE — facing corridor ({elapsed:.1f}s, streak={self._facing_streak})",
                )
            if timed_out:
                return self._begin_settling(
                    now,
                    f"TURN TIMEOUT — max {self.turn_max_seconds:.1f}s (facing={self._facing_streak})",
                )

            return LapStepResult(
                None,
                f"TURNING {elapsed:.1f}s face={self._facing_streak}/{self.facing_confirm_frames}",
            )

        if self.phase == LapPhase.SETTLING:
            self._driver.stop()
            remaining = self._settle_until - now
            if remaining > 0:
                return LapStepResult(None, f"SETTLING ({remaining:.1f}s)")

            self.lap_count += 1
            self.phase = LapPhase.SLOW_CENTER
            self._center_streak = 0
            steer = centering_steer_amount(sensors)
            return LapStepResult(
                steer,
                f"SLOW CENTER — lap {self.lap_count}",
                self.recover_speed_scale,
                slow_center=True,
            )

        if self.phase == LapPhase.SLOW_CENTER:
            steer = centering_steer_amount(sensors)
            if is_recovery_complete(sensors):
                self._center_streak += 1
            else:
                self._center_streak = 0

            if self._center_streak >= self.recover_center_frames:
                self.phase = LapPhase.CRUISE
                self._end_streak = 0
                self._end_detect_unlock_at = now + self.end_lockout_seconds
                return LapStepResult(snn_action, "CRUISE — centered, full speed", 1.0)

            facing = "aligned" if is_facing_corridor(sensors) else "re-align"
            return LapStepResult(
                steer,
                f"SLOW CENTER {self._center_streak}/{self.recover_center_frames} ({facing})",
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
            self._turn_started_at = now
            self._facing_streak = 0
            self._end_streak = 0
            self._driver.stop()
            return LapStepResult(None, "END LINE — turning 180°")

        end_tag = f" end={self._end_streak}/{self.end_confirm_frames}" if self._end_streak else ""
        lap_tag = f" lap={self.lap_count}" if self.lap_count else ""
        return LapStepResult(snn_action, f"CRUISE{lap_tag}{end_tag}", 1.0)
