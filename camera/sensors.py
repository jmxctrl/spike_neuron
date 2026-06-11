"""Lane sensor backends: heuristic (vision.py) or CV segmentation (cv_sensors.py)."""

from __future__ import annotations

import argparse
import statistics
from collections import deque
from pathlib import Path

import numpy as np

from .cv_sensors import CvDetectionResult, CvLaneDetector
from .vision import EndLineResult, _to_bgr, columns_to_lane_sensors, detect_end_line, frame_to_column_scores


def add_lane_sensor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sensor-backend",
        choices=["heuristic", "cv"],
        default="heuristic",
        help="Lane sensor source: heuristic rules or YOLOv8-seg (default: heuristic)",
    )
    parser.add_argument(
        "--cv-model",
        type=str,
        default=None,
        help="Path to YOLOv8-seg .pt (required when --sensor-backend cv)",
    )
    parser.add_argument("--cv-conf", type=float, default=0.25, help="CV detection confidence threshold")
    parser.add_argument("--cv-imgsz", type=int, default=320, help="CV inference image size")
    parser.add_argument(
        "--no-cv-fallback",
        action="store_true",
        help="Do not fall back to heuristic when CV finds no lane mask",
    )


def build_lane_sensor(args: argparse.Namespace) -> "FrameLaneSensor":
    return FrameLaneSensor(
        window_size=args.sensor_window,
        threshold=args.threshold,
        tape_color=args.tape_color,
        backend=args.sensor_backend,
        cv_model=args.cv_model,
        cv_conf=args.cv_conf,
        cv_imgsz=args.cv_imgsz,
        cv_fallback_heuristic=not args.no_cv_fallback,
    )


class FrameLaneSensor:
    """
    Unified lane sensor: median-filtered [L, C, R].

    backend='heuristic' — purple/grayscale rules (default, no ML deps)
    backend='cv'        — YOLOv8-seg masks → column scores → L/C/R
    """

    def __init__(
        self,
        window_size: int = 2,
        threshold: int = 60,
        tape_color: str = "auto",
        *,
        backend: str = "heuristic",
        cv_model: str | Path | None = None,
        cv_conf: float = 0.25,
        cv_imgsz: int = 320,
        cv_fallback_heuristic: bool = True,
    ):
        self._backend = backend
        self._window_size = window_size
        self._threshold = threshold
        self._tape_color = tape_color
        self._cv_fallback = cv_fallback_heuristic
        self._buffers = [deque(maxlen=window_size) for _ in range(3)]
        self.last_column_darkness: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.last_raw_sensors: list[float] = [0.0, 0.5, 0.0]
        self.last_cv_confidence: float = 0.0
        self.last_cv_class_hits: dict[str, float] = {}
        self.last_lane_mask_roi: np.ndarray | None = None
        self.last_backend_used: str = backend

        self._cv: CvLaneDetector | None = None
        if backend == "cv":
            if not cv_model:
                raise ValueError("backend='cv' requires --cv-model path to best.pt")
            self._cv = CvLaneDetector(cv_model, conf=cv_conf, imgsz=cv_imgsz)

    def read_from_frame(self, frame: np.ndarray | None) -> list[float]:
        if frame is None or frame.size == 0:
            return self._filtered_or_default()

        if self._backend == "cv" and self._cv is not None:
            sensors, cols = self._read_cv(frame)
            self.last_backend_used = "cv"
            if (
                self._cv_fallback
                and self.last_cv_confidence < 0.15
                and max(cols) < 0.02
            ):
                sensors, cols = self._read_heuristic(frame)
                self.last_backend_used = "heuristic_fallback"
        else:
            sensors, cols = self._read_heuristic(frame)
            self.last_backend_used = "heuristic"

        self.last_column_darkness = cols
        self.last_raw_sensors = sensors
        for i, val in enumerate(sensors):
            self._buffers[i].append(val)
        return [float(statistics.median(buf)) for buf in self._buffers]

    def _read_heuristic(self, frame: np.ndarray) -> tuple[list[float], tuple[float, float, float]]:
        bgr = _to_bgr(frame)
        cols = frame_to_column_scores(bgr, threshold=self._threshold, tape_color=self._tape_color)
        sensors = columns_to_lane_sensors(*cols)
        self.last_lane_mask_roi = None
        self.last_cv_confidence = 0.0
        self.last_cv_class_hits = {}
        return sensors, cols

    def _read_cv(self, frame: np.ndarray) -> tuple[list[float], tuple[float, float, float]]:
        assert self._cv is not None
        det: CvDetectionResult = self._cv.detect(frame)
        self.last_cv_confidence = det.confidence
        self.last_cv_class_hits = det.class_hits
        self.last_lane_mask_roi = det.lane_mask
        return det.sensors, det.column_scores

    def _filtered_or_default(self) -> list[float]:
        if self._buffers[0]:
            return [float(statistics.median(buf)) for buf in self._buffers]
        return [0.0, 0.5, 0.0]

    def lane_mask_for_debug(self, frame: np.ndarray) -> np.ndarray | None:
        if self._cv is not None and self.last_lane_mask_roi is not None:
            return self._cv.lane_mask_full_frame(frame, self.last_lane_mask_roi)
        return None

    def detect_end_line(
        self,
        frame: np.ndarray | None,
        *,
        bottom_fraction: float = 0.20,
        end_tape_color: str = "dark",
        end_threshold: int = 75,
    ) -> EndLineResult:
        if self._backend == "cv" and self._cv is not None and frame is not None and frame.size > 0:
            return self._cv.detect_end_line(frame, bottom_fraction=bottom_fraction)
        return detect_end_line(
            frame,
            bottom_fraction=bottom_fraction,
            tape_color=end_tape_color,
            threshold=end_threshold,
        )
