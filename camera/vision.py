"""
Extract 3 lane sensor values from a forward-facing camera between two floor lines.

Semantics (matches training sim):
  - left:   high when too close to the LEFT line / left line dominates view
  - center: high when well centered in the corridor
  - right:  high when too close to the RIGHT line
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from yahboom.camera import CameraSource


def _purple_tape_mask(column_bgr: np.ndarray) -> np.ndarray:
    """Detect purple/violet painters tape on a light wood floor."""
    b, g, r = cv2.split(column_bgr.astype(np.float32))
    purple_metric = (b + r) / 2.0 - g
    mask_color = purple_metric > 18.0

    hsv = cv2.cvtColor(column_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_hsv = (h >= 105) & (h <= 175) & (s >= 20) & (v >= 25)
    return mask_color | mask_hsv


def column_tape_fraction(
    column: np.ndarray,
    threshold: int = 60,
    tape_color: str = "auto",
) -> float:
    """
    Fraction of pixels detected as lane tape in a BGR column (0=none, 1=all tape).

    tape_color: auto (purple + black), purple, or dark (grayscale only).
    """
    if column.size == 0:
        return 0.0

    masks: list[np.ndarray] = []
    if tape_color in {"auto", "purple"}:
        masks.append(_purple_tape_mask(column))
    if tape_color in {"auto", "dark"}:
        gray = cv2.cvtColor(column, cv2.COLOR_BGR2GRAY)
        masks.append(gray < threshold)

    if not masks:
        return 0.0

    combined = masks[0]
    for mask in masks[1:]:
        combined = combined | mask
    return float(combined.mean())


def column_darkness(
    column: np.ndarray,
    threshold: int = 60,
    tape_color: str = "auto",
) -> float:
    """Alias for column_tape_fraction (kept for older call sites)."""
    return column_tape_fraction(column, threshold=threshold, tape_color=tape_color)


def frame_to_column_scores(
    frame: np.ndarray,
    threshold: int = 60,
    roi_fraction: float = 0.45,
    tape_color: str = "auto",
) -> tuple[float, float, float]:
    """
    Split bottom ROI into left / center / right and score tape darkness per column.
    """
    if frame is None or frame.size == 0:
        return 0.0, 0.0, 0.0

    h, w = frame.shape[:2]
    y0 = int(h * (1.0 - roi_fraction))
    roi = frame[y0:, :]

    third = w // 3
    left_col = roi[:, :third]
    center_col = roi[:, third : 2 * third]
    right_col = roi[:, 2 * third :]

    return (
        column_tape_fraction(left_col, threshold, tape_color=tape_color),
        column_tape_fraction(center_col, threshold, tape_color=tape_color),
        column_tape_fraction(right_col, threshold, tape_color=tape_color),
    )


def tape_mask_for_roi(frame_bgr: np.ndarray, roi_fraction: float = 0.45, tape_color: str = "auto", threshold: int = 60) -> np.ndarray:
    """Binary mask of detected tape in the bottom ROI (for debug overlay)."""
    h, w = frame_bgr.shape[:2]
    y0 = int(h * (1.0 - roi_fraction))
    roi = frame_bgr[y0:, :]
    if tape_color == "purple":
        mask = _purple_tape_mask(roi)
    elif tape_color == "dark":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = gray < threshold
    else:
        mask = _purple_tape_mask(roi)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = mask | (gray < threshold)
    full = np.zeros((h, w), dtype=np.uint8)
    full[y0:, :] = (mask.astype(np.uint8) * 255)
    return full


def columns_to_lane_sensors(
    left_tape: float,
    center_tape: float,
    right_tape: float,
) -> list[float]:
    """
    Map per-column tape scores to 3 lane sensors in [0, 1].

    Centered between lines -> outer columns balanced -> high center.
    Drift left -> more tape in left column -> high left sensor.
    """
    outer = left_tape + right_tape
    if outer < 0.02:
        return [0.0, 0.5, 0.0]

    imbalance = (left_tape / outer) - (right_tape / outer)
    left_sensor = float(np.clip(max(0.0, imbalance) * 2.0, 0.0, 1.0))
    right_sensor = float(np.clip(max(0.0, -imbalance) * 2.0, 0.0, 1.0))
    balance = 1.0 - abs(imbalance)
    center_clear = 1.0 - min(1.0, center_tape * 2.0)
    center_sensor = float(np.clip(balance * center_clear, 0.0, 1.0))
    return [left_sensor, center_sensor, right_sensor]


def lane_sensors_from_frame(
    frame: np.ndarray,
    threshold: int = 60,
    tape_color: str = "auto",
) -> list[float]:
    scores = frame_to_column_scores(frame, threshold=threshold, tape_color=tape_color)
    return columns_to_lane_sensors(*scores)


def _tape_mask_strip(strip_bgr: np.ndarray, tape_color: str, threshold: int) -> np.ndarray:
    masks: list[np.ndarray] = []
    if tape_color in {"auto", "purple"}:
        masks.append(_purple_tape_mask(strip_bgr))
    if tape_color in {"auto", "dark"}:
        gray = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)
        masks.append(gray < threshold)
    if not masks:
        return np.zeros(strip_bgr.shape[:2], dtype=bool)
    combined = masks[0]
    for mask in masks[1:]:
        combined = combined | mask
    return combined


@dataclass
class EndLineResult:
    detected: bool
    confidence: float
    row_coverage: float
    width_span: float


def detect_end_line(
    frame: np.ndarray,
    *,
    bottom_fraction: float = 0.20,
    tape_color: str = "dark",
    threshold: int = 75,
    min_row_coverage: float = 0.12,
    min_width_span: float = 0.35,
) -> EndLineResult:
    """
    Detect a horizontal end-of-lane bar in the bottom strip of the frame.

    Default: fat **black** tape (grayscale darkness only, not purple lane lines).
    Triggers when the bar spans most of the image width near the bottom edge.
    """
    if frame is None or frame.size == 0:
        return EndLineResult(False, 0.0, 0.0, 0.0)

    bgr = _to_bgr(frame)
    h, w = bgr.shape[:2]
    if h == 0 or w == 0:
        return EndLineResult(False, 0.0, 0.0, 0.0)

    y0 = int(h * (1.0 - bottom_fraction))
    strip = bgr[y0:, :]
    mask = _tape_mask_strip(strip, tape_color, threshold)

    row_coverage = float(mask.mean(axis=1).max()) if mask.size else 0.0
    width_span = float(mask.any(axis=0).mean()) if mask.size else 0.0
    confidence = float(min(1.0, (row_coverage + width_span) / 2.0))
    detected = row_coverage >= min_row_coverage and width_span >= min_width_span
    return EndLineResult(detected, confidence, row_coverage, width_span)


def interpret_lane_state(sensors: list[float]) -> str:
    """Human-readable lane position from [left, center, right] sensors."""
    left, center, right = sensors
    if center >= 0.55 and left < 0.4 and right < 0.4:
        return "CENTERED"
    if left >= 0.45 and left > right + 0.1:
        return "DRIFT LEFT — too close to left line"
    if right >= 0.45 and right > left + 0.1:
        return "DRIFT RIGHT — too close to right line"
    if left >= 0.35 and right >= 0.35:
        return "BOTH LINES — corridor visible"
    if center >= 0.4:
        return "MOSTLY CENTERED"
    return "UNCLEAR — check threshold / lighting"


def is_lane_centered(
    sensors: list[float],
    *,
    center_min: float = 0.48,
    side_max: float = 0.38,
) -> bool:
    left, center, right = sensors
    return center >= center_min and left < side_max and right < side_max


def corridor_visible(sensors: list[float]) -> bool:
    """True when lane lines are visible enough to stop a turnaround spin."""
    left, center, right = sensors
    if is_lane_centered(sensors):
        return True
    if left >= 0.12 and right >= 0.12:
        return True
    if center >= 0.42:
        return True
    return False


def is_tape_lost(sensors: list[float]) -> bool:
    """True when no lane tape is visible ahead (corridor physically ended)."""
    left, center, right = sensors
    return left < 0.12 and right < 0.12 and center < 0.25


def is_dead_end(sensors: list[float]) -> bool:
    """One-sided tape reading (drift), NOT a turn trigger — used only for post-turn steering."""
    left, center, right = sensors
    one_sided = (left >= 0.70 and right < 0.20) or (right >= 0.70 and left < 0.20)
    no_path = center < 0.30
    return one_sided and no_path


def is_post_turn_ready(sensors: list[float]) -> bool:
    """After a 180° spin, good enough to creep forward / hand back to SNN."""
    if is_dead_end(sensors):
        return False
    if is_facing_corridor(sensors):
        return True
    left, center, right = sensors
    if left >= 0.12 and right >= 0.12:
        return True
    if center >= 0.30:
        return True
    return max(left, right) >= 0.15


def is_facing_corridor(sensors: list[float]) -> bool:
    """True when heading down-corridor (both lines visible, or centered between them)."""
    if is_lane_centered(sensors):
        return True
    left, center, right = sensors
    # CV masks often land in the center column when centered; sides stay low.
    if center >= 0.50 and left < 0.40 and right < 0.40:
        return True
    if left < 0.12 or right < 0.12:
        return False
    if abs(left - right) > 0.32:
        return False
    return center >= 0.22


def is_recovery_complete(sensors: list[float]) -> bool:
    """True when good enough to hand back to full-speed SNN."""
    if is_lane_centered(sensors):
        return True
    left, center, right = sensors
    if center >= 0.50 and left < 0.40 and right < 0.40:
        return True
    if not is_facing_corridor(sensors):
        return False
    if center >= 0.38 and abs(left - right) < 0.30:
        return True
    if left >= 0.10 and right >= 0.10 and abs(left - right) < 0.22:
        return True
    return False


def centering_steer_amount(sensors: list[float]) -> float:
    """Continuous steer in [-1, 1] from lane imbalance."""
    left, _, right = sensors
    imbalance = left - right
    if abs(imbalance) < 0.06:
        return 0.0
    return float(np.clip(imbalance * 2.0, -1.0, 1.0))


def centering_steer_action(sensors: list[float]) -> int:
    """
    Steer while creeping forward to re-center between lane lines.

    Returns -1 (left), 0 (straight), or +1 (right).
    """
    left, center, right = sensors
    imbalance = left - right
    if imbalance > 0.20:
        return 1
    if imbalance < -0.20:
        return -1
    if center >= 0.38 and abs(imbalance) < 0.15:
        return 0
    if imbalance > 0.06:
        return 1
    if imbalance < -0.06:
        return -1
    return 0


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 3:
        # Heuristic: client frames are RGB; OpenCV ops expect BGR.
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def annotate_debug_frame(
    frame: np.ndarray,
    sensors: list[float],
    column_darkness: tuple[float, float, float],
    action: int | None = None,
    *,
    threshold: int = 60,
    roi_fraction: float = 0.45,
    tape_color: str = "auto",
    cv_lane_mask: np.ndarray | None = None,
    backend_label: str | None = None,
    end_line: EndLineResult | None = None,
    end_line_fraction: float = 0.20,
    end_tape_color: str = "dark",
    end_threshold: int = 75,
) -> np.ndarray:
    """Draw ROI columns, sensor bars, tape mask, and lane state (returns RGB)."""
    bgr = _to_bgr(frame).copy()
    h, w = bgr.shape[:2]
    end_y0 = int(h * (1.0 - end_line_fraction))
    cv2.rectangle(bgr, (0, end_y0), (w - 1, h - 1), (0, 220, 255), 1)
    end_mask = _tape_mask_strip(bgr[end_y0:, :], end_tape_color, end_threshold)
    end_overlay = bgr.copy()
    end_overlay[end_y0:, :][end_mask] = (40, 40, 40)
    bgr = cv2.addWeighted(bgr, 0.85, end_overlay, 0.15, 0)
    if end_line is not None:
        end_color = (0, 255, 0) if end_line.detected else (0, 180, 255)
        end_text = "END LINE" if end_line.detected else "end zone"
        cv2.putText(
            bgr,
            f"{end_text} r={end_line.row_coverage:.2f} w={end_line.width_span:.2f}",
            (8, max(end_y0 - 6, 56)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            end_color,
            1,
            cv2.LINE_AA,
        )

    y0 = int(h * (1.0 - roi_fraction))

    if cv_lane_mask is not None and cv_lane_mask.shape[:2] == (h, w):
        overlay = bgr.copy()
        overlay[cv_lane_mask > 0] = (180, 0, 220)
        bgr = cv2.addWeighted(bgr, 0.72, overlay, 0.28, 0)
    else:
        tape_mask = tape_mask_for_roi(bgr, roi_fraction=roi_fraction, tape_color=tape_color, threshold=threshold)
        overlay = bgr.copy()
        overlay[tape_mask > 0] = (180, 0, 220)
        bgr = cv2.addWeighted(bgr, 0.72, overlay, 0.28, 0)

    third = w // 3
    colors = [(80, 80, 255), (80, 255, 80), (255, 80, 80)]
    labels = ("L", "C", "R")
    for i, x0 in enumerate((0, third, 2 * third)):
        x1 = x0 + third if i < 2 else w
        cv2.rectangle(bgr, (x0, y0), (x1, h), colors[i], 2)
        cv2.putText(
            bgr,
            f"{labels[i]} tape {column_darkness[i]:.2f}",
            (x0 + 4, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colors[i],
            1,
            cv2.LINE_AA,
        )

    cv2.line(bgr, (0, y0), (w, y0), (200, 200, 200), 1)

    bar_x = 8
    bar_w = 14
    bar_h = 60
    bar_gap = 6
    base_y = h - 10
    for i, val in enumerate(sensors):
        x = bar_x + i * (bar_w + bar_gap)
        filled = int(bar_h * float(np.clip(val, 0.0, 1.0)))
        cv2.rectangle(bgr, (x, base_y - bar_h), (x + bar_w, base_y), (40, 40, 40), 1)
        cv2.rectangle(bgr, (x, base_y - filled), (x + bar_w, base_y), colors[i], -1)

    state = interpret_lane_state(sensors)
    title = state if not backend_label else f"{state}  [{backend_label}]"
    cv2.putText(bgr, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if action is not None:
        if isinstance(action, (int, float)):
            a = float(action)
            if abs(a) < 0.08:
                action_text = "FORWARD"
            elif a > 0:
                action_text = f"STEER RIGHT {a:+.2f}"
            else:
                action_text = f"STEER LEFT {a:+.2f}"
        else:
            action_text = str(action)
        cv2.putText(
            bgr,
            action_text,
            (8, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    sensor_text = f"L={sensors[0]:.2f} C={sensors[1]:.2f} R={sensors[2]:.2f}"
    cv2.putText(bgr, sensor_text, (8, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class FrameLaneSensor:
    """Lane sensors from an in-memory camera frame (remote / laptop)."""

    def __init__(self, window_size: int = 5, threshold: int = 60, tape_color: str = "auto"):
        self._threshold = threshold
        self._tape_color = tape_color
        self._buffers = [deque(maxlen=window_size) for _ in range(3)]
        self.last_column_darkness: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_raw_sensors: list[float] = [0.0, 0.5, 0.0]

    def read_from_frame(self, frame: np.ndarray | None) -> list[float]:
        if frame is None or frame.size == 0:
            return self._filtered_or_default()

        bgr = _to_bgr(frame)
        self.last_column_darkness = frame_to_column_scores(
            bgr, threshold=self._threshold, tape_color=self._tape_color
        )
        self.last_raw_sensors = columns_to_lane_sensors(*self.last_column_darkness)
        for i, val in enumerate(self.last_raw_sensors):
            self._buffers[i].append(val)
        return [float(statistics.median(buf)) for buf in self._buffers]

    def _filtered_or_default(self) -> list[float]:
        if self._buffers[0]:
            return [float(statistics.median(buf)) for buf in self._buffers]
        return [0.0, 0.5, 0.0]


class CameraLaneSensor:
    """Reads camera frames and returns filtered [left, center, right] lane sensors."""

    def __init__(
        self,
        camera: CameraSource,
        window_size: int = 5,
        threshold: int = 60,
        tape_color: str = "auto",
    ):
        self._camera = camera
        self._frame_sensor = FrameLaneSensor(
            window_size=window_size, threshold=threshold, tape_color=tape_color
        )

    @property
    def last_column_darkness(self) -> tuple[float, float, float]:
        return self._frame_sensor.last_column_darkness

    @property
    def last_raw_sensors(self) -> list[float]:
        return self._frame_sensor.last_raw_sensors

    def read(self) -> list[float]:
        jpeg_b64 = self._camera.capture_jpeg_b64(quality=75)
        if not jpeg_b64:
            return self._frame_sensor._filtered_or_default()

        import base64

        data = base64.b64decode(jpeg_b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return self._frame_sensor._filtered_or_default()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._frame_sensor.read_from_frame(rgb)

    def close(self) -> None:
        self._camera.close()
