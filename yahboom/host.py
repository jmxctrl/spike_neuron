#!/usr/bin/env python3
"""
Yahboom Raspbot host (runs on the Raspberry Pi).

Listens for commands over ZMQ and streams observations (sensors + camera) back
to the laptop, similar to LeRobot's LeKiwi host.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time

import cv2
import zmq

from .protocol import (
    DEFAULT_CMD_PORT,
    DEFAULT_FPS,
    DEFAULT_OBS_PORT,
    DEFAULT_WATCHDOG_MS,
    RobotCommand,
    RobotObservation,
)
from .raspbot import Raspbot

logger = logging.getLogger(__name__)


class RaspbotHost:
    def __init__(
        self,
        cmd_port: int = DEFAULT_CMD_PORT,
        obs_port: int = DEFAULT_OBS_PORT,
        watchdog_ms: int = DEFAULT_WATCHDOG_MS,
        max_loop_hz: int = DEFAULT_FPS,
        camera_index: int = 0,
        enable_camera: bool = True,
    ):
        self.watchdog_ms = watchdog_ms
        self.max_loop_hz = max_loop_hz
        self.enable_camera = enable_camera
        self.camera_index = camera_index

        self._context = zmq.Context()
        self._cmd_socket = self._context.socket(zmq.PULL)
        self._cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self._cmd_socket.bind(f"tcp://*:{cmd_port}")

        self._obs_socket = self._context.socket(zmq.PUSH)
        self._obs_socket.setsockopt(zmq.CONFLATE, 1)
        self._obs_socket.bind(f"tcp://*:{obs_port}")

        self._last_cmd_time = time.time()
        self._watchdog_active = False
        self._speed = 100
        self._pan = 90
        self._tilt = 90
        self._red_led = False
        self._blue_led = False

        self._camera = None
        if self.enable_camera:
            self._camera = cv2.VideoCapture(self.camera_index)
            if self._camera.isOpened():
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            else:
                logger.warning("Camera unavailable; continuing without video stream")
                self._camera = None

    def close(self) -> None:
        if self._camera is not None:
            self._camera.release()
        self._obs_socket.close()
        self._cmd_socket.close()
        self._context.term()

    def _read_command(self) -> RobotCommand | None:
        try:
            msg = self._cmd_socket.recv_string(zmq.NOBLOCK)
        except zmq.Again:
            return None
        try:
            data = json.loads(msg)
            return RobotCommand(**data)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("Invalid command JSON: %s", exc)
            return None

    def _apply_command(self, robot: Raspbot, command: RobotCommand) -> None:
        if command.quit:
            return

        self._speed = command.speed
        self._pan = command.pan
        self._tilt = command.tilt
        self._red_led = command.red_led
        self._blue_led = command.blue_led

        movement = command.movement
        speed = command.speed
        if movement == "forward":
            robot.forward(speed)
        elif movement == "backward":
            robot.backward(speed)
        elif movement == "turn_left":
            robot.turn_left(speed)
        elif movement == "turn_right":
            robot.turn_right(speed)
        else:
            robot.stop()

        robot.set_pan(self._pan)
        robot.set_tilt(self._tilt)
        robot.led_red(self._red_led)
        robot.led_blue(self._blue_led)

        if command.beep:
            robot.beep(
                duration=float(command.beep.get("duration", 0.1)),
                frequency=int(command.beep.get("frequency", 440)),
            )

    def _capture_camera_b64(self) -> str | None:
        if self._camera is None:
            return None
        ok, frame = self._camera.read()
        if not ok:
            return None
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        return base64.b64encode(buffer).decode("utf-8")

    def _build_observation(self, robot: Raspbot) -> RobotObservation:
        line = robot.read_line_tracker()
        ir_left, ir_right = robot.read_ir_obstacle()
        return RobotObservation(
            distance_cm=robot.read_distance(),
            line_tracker={
                "left1": line.left1,
                "left2": line.left2,
                "right1": line.right1,
                "right2": line.right2,
            },
            ir_obstacle={"left": ir_left, "right": ir_right},
            camera_jpeg_b64=self._capture_camera_b64(),
            pan=self._pan,
            tilt=self._tilt,
            speed=self._speed,
            red_led=self._red_led,
            blue_led=self._blue_led,
        )

    def run(self, robot: Raspbot, duration_s: float | None = None) -> None:
        logger.info("Waiting for commands on ZMQ...")
        start = time.perf_counter()
        should_run = True

        while should_run:
            loop_start = time.time()
            command = self._read_command()

            if command is not None:
                self._last_cmd_time = time.time()
                self._watchdog_active = False
                if command.quit:
                    logger.info("Quit command received")
                    robot.stop()
                    break
                self._apply_command(robot, command)
            else:
                elapsed_ms = (time.time() - self._last_cmd_time) * 1000
                if elapsed_ms > self.watchdog_ms and not self._watchdog_active:
                    logger.warning("Watchdog timeout; stopping robot")
                    self._watchdog_active = True
                    robot.stop()

            observation = self._build_observation(robot)
            try:
                self._obs_socket.send_string(json.dumps(observation.to_dict()), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            if duration_s is not None and time.perf_counter() - start >= duration_s:
                break

            elapsed = time.time() - loop_start
            time.sleep(max(1.0 / self.max_loop_hz - elapsed, 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Yahboom Raspbot ZMQ host")
    parser.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT)
    parser.add_argument("--obs-port", type=int, default=DEFAULT_OBS_PORT)
    parser.add_argument("--watchdog-ms", type=int, default=DEFAULT_WATCHDOG_MS)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--duration-s", type=float, default=None, help="Stop after N seconds")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    robot = Raspbot()
    host = RaspbotHost(
        cmd_port=args.cmd_port,
        obs_port=args.obs_port,
        watchdog_ms=args.watchdog_ms,
        max_loop_hz=args.fps,
        camera_index=args.camera_index,
        enable_camera=not args.no_camera,
    )

    try:
        robot.init()
        robot.beep(0.1, 440)
        robot.center_servos()
        host.run(robot, duration_s=args.duration_s)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        robot.stop()
        robot.cleanup()
        host.close()
        logger.info("Host stopped")


if __name__ == "__main__":
    main()
