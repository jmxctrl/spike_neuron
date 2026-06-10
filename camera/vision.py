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


class CameraLaneSensor:
    """Reads camera frames and returns filtered [left, center, right] lane sensors."""

    def __init__(
        self,
        camera: CameraSource,
        window_size: int = 5,
        threshold: int = 60,
    ):
        self._camera = camera
        self._threshold = threshold
        self._buffers = [deque(maxlen=window_size) for _ in range(3)]

    def read(self) -> list[float]:
        jpeg_b64 = self._camera.capture_jpeg_b64(quality=75)
        if not jpeg_b64:
            return self._filtered_or_default()

        import base64

        data = base64.b64decode(jpeg_b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return self._filtered_or_default()

        sensors = lane_sensors_from_frame(frame, threshold=self._threshold)
        for i, val in enumerate(sensors):
            self._buffers[i].append(val)
        return [float(statistics.median(buf)) for buf in self._buffers]

    def _filtered_or_default(self) -> list[float]:
        if self._buffers[0]:
            return [float(statistics.median(buf)) for buf in self._buffers]
        return [0.0, 0.5, 0.0]

    def close(self) -> None:
        self._camera.close()
