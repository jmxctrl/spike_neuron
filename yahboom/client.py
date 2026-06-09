"""ZMQ client for remote Yahboom teleop (runs on laptop)."""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

import cv2
import numpy as np
import zmq

from .protocol import (
    DEFAULT_CMD_PORT,
    DEFAULT_OBS_PORT,
    RobotCommand,
    RobotObservation,
)

logger = logging.getLogger(__name__)


class RaspbotClient:
    """Connects to a RaspbotHost over ZMQ and exposes send/receive helpers."""

    def __init__(
        self,
        remote_ip: str,
        cmd_port: int = DEFAULT_CMD_PORT,
        obs_port: int = DEFAULT_OBS_PORT,
        connect_timeout_s: float = 5.0,
        polling_timeout_ms: int = 15,
    ):
        self.remote_ip = remote_ip
        self.cmd_port = cmd_port
        self.obs_port = obs_port
        self.connect_timeout_s = connect_timeout_s
        self.polling_timeout_ms = polling_timeout_ms

        self._context: zmq.Context | None = None
        self._cmd_socket: zmq.Socket | None = None
        self._obs_socket: zmq.Socket | None = None
        self._connected = False

        self.last_observation = RobotObservation()
        self.last_frame: np.ndarray | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._context = zmq.Context()
        self._cmd_socket = self._context.socket(zmq.PUSH)
        self._cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self._cmd_socket.connect(f"tcp://{self.remote_ip}:{self.cmd_port}")

        self._obs_socket = self._context.socket(zmq.PULL)
        self._obs_socket.setsockopt(zmq.CONFLATE, 1)
        self._obs_socket.connect(f"tcp://{self.remote_ip}:{self.obs_port}")

        poller = zmq.Poller()
        poller.register(self._obs_socket, zmq.POLLIN)
        socks = dict(poller.poll(int(self.connect_timeout_s * 1000)))
        if self._obs_socket not in socks:
            self.disconnect()
            raise ConnectionError(
                f"Timeout waiting for host at tcp://{self.remote_ip}:{self.obs_port}. "
                "Is `python -m yahboom.host` running on the robot?"
            )

        self._connected = True
        logger.info(
            "Connected to robot at tcp://%s:%s (observations tcp://%s:%s)",
            self.remote_ip,
            self.cmd_port,
            self.remote_ip,
            self.obs_port,
        )

    def disconnect(self) -> None:
        if self._obs_socket is not None:
            self._obs_socket.close()
        if self._cmd_socket is not None:
            self._cmd_socket.close()
        if self._context is not None:
            self._context.term()
        self._obs_socket = None
        self._cmd_socket = None
        self._context = None
        self._connected = False

    def send_command(self, command: RobotCommand) -> None:
        if not self._connected or self._cmd_socket is None:
            raise RuntimeError("Client is not connected")
        self._cmd_socket.send_string(json.dumps(command.to_dict()))

    def poll_observation(self) -> RobotObservation:
        if not self._connected or self._obs_socket is None:
            raise RuntimeError("Client is not connected")

        poller = zmq.Poller()
        poller.register(self._obs_socket, zmq.POLLIN)
        try:
            socks = dict(poller.poll(self.polling_timeout_ms))
        except zmq.ZMQError as exc:
            logger.error("ZMQ polling error: %s", exc)
            return self.last_observation

        if self._obs_socket not in socks:
            return self.last_observation

        latest_msg: str | None = None
        while True:
            try:
                latest_msg = self._obs_socket.recv_string(zmq.NOBLOCK)
            except zmq.Again:
                break

        if latest_msg is None:
            return self.last_observation

        try:
            data = json.loads(latest_msg)
        except json.JSONDecodeError as exc:
            logger.error("Invalid observation JSON: %s", exc)
            return self.last_observation

        self.last_observation = RobotObservation.from_dict(data)
        self.last_frame = self._decode_camera(data.get("camera_jpeg_b64"))
        return self.last_observation

    @staticmethod
    def _decode_camera(image_b64: str | None) -> np.ndarray | None:
        if not image_b64:
            return None
        try:
            jpg_data = base64.b64decode(image_b64)
            arr = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        except (TypeError, ValueError) as exc:
            logger.error("Failed to decode camera frame: %s", exc)
            return None

    def get_observation_dict(self) -> dict[str, Any]:
        obs = self.last_observation
        out: dict[str, Any] = {
            "distance_cm": obs.distance_cm,
            "pan": obs.pan,
            "tilt": obs.tilt,
            "speed": obs.speed,
            "red_led": float(obs.red_led),
            "blue_led": float(obs.blue_led),
            "line_left1": float(obs.line_tracker.get("left1", False)),
            "line_left2": float(obs.line_tracker.get("left2", False)),
            "line_right1": float(obs.line_tracker.get("right1", False)),
            "line_right2": float(obs.line_tracker.get("right2", False)),
            "ir_left": float(obs.ir_obstacle.get("left", False)),
            "ir_right": float(obs.ir_obstacle.get("right", False)),
        }
        if self.last_frame is not None:
            out["camera"] = self.last_frame
        return out
