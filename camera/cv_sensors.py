"""
Lane sensors from a YOLO segmentation model (Roboflow → Ultralytics export).

Train on Roboflow (recommended), not a random pretrained COCO model:
  1. Collect frames:  uv run python -m camera.capture_frames --duration 120
  2. Upload to Roboflow, label with polygon segmentation:
       - purple_line  (both side tapes)
       - black_bar    (end bar, optional but useful)
  3. Train YOLOv8n-seg (Roboflow train or export YOLOv8-seg and fine-tune locally)
  4. Export best.pt → camera/models/lane_seg.pt
  5. Run:  uv run python -m camera.play --sensor-backend cv --cv-model camera/models/lane_seg.pt

Requires: pip install ultralytics  (optional; heuristic backend works without it)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .vision import EndLineResult, _to_bgr, columns_to_lane_sensors

DEFAULT_LANE_CLASS_NAMES = frozenset(
    {"purple_line", "purple_tape", "lane_line", "purple", "side_line", "purple line"}
)
DEFAULT_END_CLASS_NAMES = frozenset(
    {"horizontal_black_line", "black_bar", "black_line", "end_bar", "black", "horizontal black line"}
)


@dataclass
class CvDetectionResult:
    """Raw CV output before median filtering."""

    column_scores: tuple[float, float, float]
    sensors: list[float]
    lane_mask: np.ndarray | None
    confidence: float
    class_hits: dict[str, float] = field(default_factory=dict)


class CvLaneDetector:
    """YOLOv8-seg → per-column tape fractions → same L/C/R as vision.py heuristics."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        conf: float = 0.25,
        imgsz: int = 320,
        roi_fraction: float = 0.45,
        lane_class_names: frozenset[str] | None = None,
        end_class_names: frozenset[str] | None = None,
        device: str | None = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "CV sensor backend needs ultralytics. Install with:\n"
                "  uv pip install ultralytics\n"
                "Or use --sensor-backend heuristic"
            ) from exc

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"CV model not found: {path}")

        self._model = YOLO(str(path))
        self.conf = conf
        self.imgsz = imgsz
        self.roi_fraction = roi_fraction
        self.lane_class_names = lane_class_names or DEFAULT_LANE_CLASS_NAMES
        self.end_class_names = end_class_names or DEFAULT_END_CLASS_NAMES
        self.device = device
        self._names: dict[int, str] = {}
        if hasattr(self._model, "names") and self._model.names:
            self._names = {int(k): str(v).lower().replace(" ", "_") for k, v in self._model.names.items()}

    def detect(self, frame: np.ndarray) -> CvDetectionResult:
        if frame is None or frame.size == 0:
            return CvDetectionResult((0.0, 0.0, 0.0), [0.0, 0.5, 0.0], None, 0.0)

        bgr = _to_bgr(frame)
        h, w = bgr.shape[:2]
        y0 = int(h * (1.0 - self.roi_fraction))
        roi_h = h - y0

        results = self._model.predict(
            bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        lane_mask = np.zeros((roi_h, w), dtype=np.uint8)
        confidences: list[float] = []
        class_hits: dict[str, float] = {}

        if results and results[0].masks is not None and results[0].boxes is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            box_conf = boxes.conf.cpu().numpy()

            for mask, cls_id, c in zip(masks, cls_ids, box_conf):
                name = self._names.get(int(cls_id), str(cls_id)).lower().replace(" ", "_")
                class_hits[name] = max(class_hits.get(name, 0.0), float(c))
                if name not in self.lane_class_names:
                    continue
                confidences.append(float(c))
                resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                roi_slice = (resized[y0:, :] > 0.5).astype(np.uint8)
                lane_mask = np.maximum(lane_mask, roi_slice)

        third = w // 3
        if lane_mask.sum() == 0:
            col_scores = (0.0, 0.0, 0.0)
        else:
            col_scores = (
                float(lane_mask[:, :third].mean()),
                float(lane_mask[:, third : 2 * third].mean()),
                float(lane_mask[:, 2 * third :].mean()),
            )

        sensors = columns_to_lane_sensors(*col_scores)
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        return CvDetectionResult(col_scores, sensors, lane_mask, mean_conf, class_hits)

    def lane_mask_full_frame(self, frame: np.ndarray, lane_mask_roi: np.ndarray | None) -> np.ndarray:
        """Expand ROI lane mask to full-frame uint8 for debug overlay."""
        if frame is None or lane_mask_roi is None:
            return np.zeros((1, 1), dtype=np.uint8)
        h, w = _to_bgr(frame).shape[:2]
        y0 = int(h * (1.0 - self.roi_fraction))
        full = np.zeros((h, w), dtype=np.uint8)
        if lane_mask_roi.shape[0] == h - y0 and lane_mask_roi.shape[1] == w:
            full[y0:, :] = (lane_mask_roi > 0).astype(np.uint8) * 255
        return full

    def detect_end_line(
        self,
        frame: np.ndarray,
        *,
        bottom_fraction: float = 0.20,
        min_row_coverage: float = 0.12,
        min_width_span: float = 0.40,
    ) -> EndLineResult:
        """End bar from horizontal_black_line segmentation mask in the bottom strip."""
        if frame is None or frame.size == 0:
            return EndLineResult(False, 0.0, 0.0, 0.0)

        bgr = _to_bgr(frame)
        h, w = bgr.shape[:2]
        if h == 0 or w == 0:
            return EndLineResult(False, 0.0, 0.0, 0.0)

        y0 = int(h * (1.0 - bottom_fraction))
        end_mask = np.zeros((h - y0, w), dtype=np.uint8)

        results = self._model.predict(
            bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        confidences: list[float] = []

        if results and results[0].masks is not None and results[0].boxes is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            box_conf = boxes.conf.cpu().numpy()

            for mask, cls_id, c in zip(masks, cls_ids, box_conf):
                name = self._names.get(int(cls_id), str(cls_id)).lower().replace(" ", "_")
                if name not in self.end_class_names:
                    continue
                confidences.append(float(c))
                resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                end_mask = np.maximum(end_mask, (resized[y0:, :] > 0.5).astype(np.uint8))

        row_coverage = float(end_mask.mean(axis=1).max()) if end_mask.size else 0.0
        width_span = float(end_mask.any(axis=0).mean()) if end_mask.size else 0.0
        confidence = float(np.mean(confidences)) if confidences else float(min(1.0, (row_coverage + width_span) / 2.0))
        detected = row_coverage >= min_row_coverage and width_span >= min_width_span
        return EndLineResult(detected, confidence, row_coverage, width_span)
