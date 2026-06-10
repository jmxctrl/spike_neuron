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
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from yahboom.camera import CameraSource


def column_darkness(column: np.ndarray, threshold: int = 60) -> float:
    """Fraction of pixels darker than threshold in a BGR column (0=no tape, 1=all dark)."""
    gray = cv2.cvtColor(column, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return float(mask.sum() / 255 / mask.size)


def frame_to_column_scores(
    frame: np.ndarray,
    threshold: int = 60,
    roi_fraction: float = 0.45,
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
        column_darkness(left_col, threshold),
        column_darkness(center_col, threshold),
        column_darkness(right_col, threshold),
    )


def columns_to_lane_sensors(
    left_dark: float,
    center_dark: float,
    right_dark: float,
) -> list[float]:
    """
    Map raw column darkness to 3 lane sensors in [0, 1].

    Matches light_task semantics: centered -> [0, 1, 0]; drift left -> high left
    sensor; drift right -> high right sensor.
    """
    diff = left_dark - right_dark
    left_sensor = float(np.clip(max(0.0, diff) * 1.5, 0.0, 1.0))
    right_sensor = float(np.clip(max(0.0, -diff) * 1.5, 0.0, 1.0))
    center_sensor = float(np.clip(1.0 - left_sensor - right_sensor, 0.0, 1.0))
    return [left_sensor, center_sensor, right_sensor]


def lane_sensors_from_frame(frame: np.ndarray, threshold: int = 60) -> list[float]:
    l_dark, c_dark, r_dark = frame_to_column_scores(frame, threshold=threshold)
    return columns_to_lane_sensors(l_dark, c_dark, r_dark)


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
) -> np.ndarray:
    """Draw ROI columns, sensor bars, and lane state on frame (returns RGB)."""
    bgr = _to_bgr(frame).copy()
    h, w = bgr.shape[:2]
    y0 = int(h * (1.0 - roi_fraction))

    third = w // 3
    colors = [(80, 80, 255), (80, 255, 80), (255, 80, 80)]
    labels = ("L", "C", "R")
    for i, x0 in enumerate((0, third, 2 * third)):
        x1 = x0 + third if i < 2 else w
        cv2.rectangle(bgr, (x0, y0), (x1, h), colors[i], 2)
        cv2.putText(
            bgr,
            f"{labels[i]} {column_darkness[i]:.2f}",
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
    cv2.putText(bgr, state, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if action is not None:
        action_names = {-1: "STEER LEFT", 0: "FORWARD", 1: "STEER RIGHT"}
        cv2.putText(
            bgr,
            action_names.get(action, str(action)),
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

    def __init__(self, window_size: int = 5, threshold: int = 60):
        self._threshold = threshold
        self._buffers = [deque(maxlen=window_size) for _ in range(3)]
        self.last_column_darkness: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_raw_sensors: list[float] = [0.0, 0.5, 0.0]

    def read_from_frame(self, frame: np.ndarray | None) -> list[float]:
        if frame is None or frame.size == 0:
            return self._filtered_or_default()

        bgr = _to_bgr(frame)
        self.last_column_darkness = frame_to_column_scores(bgr, threshold=self._threshold)
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
    ):
        self._camera = camera
        self._frame_sensor = FrameLaneSensor(window_size=window_size, threshold=threshold)

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
