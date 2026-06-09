"""Camera helpers for the Yahboom host (Raspberry Pi CSI + USB V4L2)."""

from __future__ import annotations

import base64
import fcntl
import glob
import logging
import os
import struct
import sys
from typing import Protocol

import cv2

logger = logging.getLogger(__name__)

V4L2_CAP_VIDEO_CAPTURE = 0x00000001
VIDIOC_QUERYCAP = 0x80685600


class CameraSource(Protocol):
    def capture_jpeg_b64(self, quality: int = 75) -> str | None: ...
    def close(self) -> None: ...


# none | horizontal (mirror) | vertical (upside-down mount) | both (180°)
FlipMode = str

FLIP_MODES = ("none", "horizontal", "vertical", "both")
_FLIP_AXIS = {"horizontal": 1, "vertical": 0, "both": -1}


def _prepare_frame(frame, flip: FlipMode = "vertical"):
    axis = _FLIP_AXIS.get(flip)
    if axis is not None:
        frame = cv2.flip(frame, axis)
    return frame


def _encode_frame_b64(frame, quality: int = 75, flip: FlipMode = "vertical") -> str | None:
    frame = _prepare_frame(frame, flip=flip)
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return base64.b64encode(buffer).decode("utf-8")


class _Picamera2Source:
    def __init__(self, picam2, width: int, height: int, flip: FlipMode = "vertical"):
        self._picam2 = picam2
        self._width = width
        self._height = height
        self._flip = flip

    def capture_jpeg_b64(self, quality: int = 75) -> str | None:
        try:
            frame = self._picam2.capture_array()
        except Exception as exc:
            logger.debug("picamera2 capture failed: %s", exc)
            return None
        if frame is None:
            return None
        return _encode_frame_b64(frame, quality=quality, flip=self._flip)

    def close(self) -> None:
        try:
            self._picam2.stop()
        except Exception:
            pass


class _V4L2Source:
    def __init__(self, cap: cv2.VideoCapture, label: str, flip: FlipMode = "vertical"):
        self._cap = cap
        self._label = label
        self._flip = flip

    def capture_jpeg_b64(self, quality: int = 75) -> str | None:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return _encode_frame_b64(frame, quality=quality, flip=self._flip)

    def close(self) -> None:
        self._cap.release()


def _suppress_opencv_logs() -> None:
    try:
        cv2.setLogLevel(0)  # type: ignore[attr-defined]
    except Exception:
        pass


def _is_v4l_capture_device(path: str) -> bool:
    """Return True only for nodes that advertise VIDEO_CAPTURE."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        # v4l2_capability: capabilities field at byte offset 84
        buf = bytearray(104)
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, buf)
        capabilities = struct.unpack_from("I", buf, 84)[0]
        return bool(capabilities & V4L2_CAP_VIDEO_CAPTURE)
    except OSError:
        return False
    finally:
        os.close(fd)


def _discover_v4l_devices() -> list[str]:
    devices = []
    for path in sorted(glob.glob("/dev/video*"), key=lambda p: int(p.replace("/dev/video", "") or 0)):
        if _is_v4l_capture_device(path):
            devices.append(path)
    return devices


def _try_picamera2(width: int, height: int, flip: FlipMode = "vertical") -> CameraSource | None:
    try:
        from picamera2 import Picamera2  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("picamera2 not installed")
        return None

    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (width, height)}
        )
        picam2.configure(config)
        picam2.start()
        # Warm up
        for _ in range(10):
            frame = picam2.capture_array()
            if frame is not None:
                logger.info("Camera opened via picamera2 (%dx%d)", width, height)
                return _Picamera2Source(picam2, width, height, flip=flip)
        picam2.stop()
    except Exception as exc:
        logger.debug("picamera2 open failed: %s", exc)
    return None


def _try_v4l2_path(path: str, width: int, height: int, flip: FlipMode = "vertical") -> CameraSource | None:
    _suppress_opencv_logs()
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(15):
        ok, frame = cap.read()
        if ok and frame is not None:
            logger.info("Camera opened via V4L2 at %s", path)
            return _V4L2Source(cap, path, flip=flip)

    cap.release()
    return None


def open_camera(
    index: int = 0,
    width: int = 320,
    height: int = 240,
    device: str | None = None,
    backend: str = "auto",
    flip: FlipMode = "vertical",
) -> CameraSource | None:
    """
    Open a camera on Raspberry Pi or USB webcam.

    backend:
      - auto: picamera2 (CSI) first, then V4L2 capture devices
      - picamera2: CSI camera only
      - v4l2: /dev/video* capture nodes only
    """
    if backend in ("auto", "picamera2") and sys.platform == "linux":
        source = _try_picamera2(width, height, flip=flip)
        if source is not None:
            return source

    if backend in ("auto", "v4l2"):
        if device:
            candidates = [device]
        else:
            candidates = _discover_v4l_devices()
            if not candidates and sys.platform == "linux":
                candidates = [f"/dev/video{index}"]

        for path in candidates:
            source = _try_v4l2_path(path, width, height, flip=flip)
            if source is not None:
                return source

    logger.warning(
        "No camera device found. On Pi CSI cameras install picamera2: "
        "sudo apt install -y python3-picamera2. "
        "For USB cams check `v4l2-ctl --list-devices` and pass --camera-device."
    )
    return None


def capture_jpeg_b64(source: CameraSource | None, quality: int = 75) -> str | None:
    if source is None:
        return None
    return source.capture_jpeg_b64(quality=quality)
