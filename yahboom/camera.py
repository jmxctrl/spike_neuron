"""Camera helpers for the Yahboom host (Raspberry Pi / V4L2)."""

from __future__ import annotations

import logging
import sys

import cv2

logger = logging.getLogger(__name__)


def open_camera(index: int = 0, width: int = 320, height: int = 240) -> cv2.VideoCapture | None:
    """Try common backends/indices used on Raspberry Pi + USB webcams."""
    if sys.platform == "linux":
        candidates: list[tuple[int, int | None]] = [
            (index, cv2.CAP_V4L2),
            (index, None),
            (1, cv2.CAP_V4L2),
            (1, None),
            (2, cv2.CAP_V4L2),
        ]
    else:
        candidates = [(index, None), (1, None)]

    for cam_index, backend in candidates:
        try:
            cap = (
                cv2.VideoCapture(cam_index, backend)
                if backend is not None
                else cv2.VideoCapture(cam_index)
            )
        except Exception as exc:
            logger.debug("Camera open failed for index=%s backend=%s: %s", cam_index, backend, exc)
            continue

        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warm up — first reads are often black/empty on Pi.
        warmed = False
        for _ in range(15):
            ok, _frame = cap.read()
            if ok:
                warmed = True
                break

        if warmed:
            logger.info("Camera opened on index=%s (backend=%s)", cam_index, backend)
            return cap

        cap.release()

    logger.warning("No camera device found (tried indices %s)", [c[0] for c in candidates])
    return None


def capture_jpeg_b64(cap: cv2.VideoCapture | None, quality: int = 75) -> str | None:
    if cap is None:
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    import base64

    return base64.b64encode(buffer).decode("utf-8")
