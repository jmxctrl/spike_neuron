#!/usr/bin/env python3
"""
Yahboom Raspbot host (runs on the Raspberry Pi).

Listens for commands over ZMQ and streams observations (sensors + camera) back
to the laptop, similar to LeRobot's LeKiwi host.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

try:
    import zmq
except ModuleNotFoundError:
    raise SystemExit(
        "Missing pyzmq. On the Raspberry Pi host, install system packages:\n"
        "  sudo apt install -y python3-zmq python3-picamera2 python3-opencv\n"
        "Or run: bash yahboom/setup_robot.sh\n"
        "Then start with: sudo PYTHONPATH=. python3 -m yahboom.host"
    ) from None

from .camera import capture_jpeg_b64, open_camera
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


def _configure_socket(socket: zmq.Socket) -> None:
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.SNDHWM, 3)
    socket.setsockopt(zmq.RCVHWM, 3)
    socket.setsockopt(zmq.SNDBUF, 2 * 1024 * 1024)
    socket.setsockopt(zmq.RCVBUF, 2 * 1024 * 1024)


class RaspbotHost:
    def __init__(
        self,
        cmd_port: int = DEFAULT_CMD_PORT,
        obs_port: int = DEFAULT_OBS_PORT,
        watchdog_ms: int = DEFAULT_WATCHDOG_MS,
        max_loop_hz: int = DEFAULT_FPS,
        camera_index: int = 0,
        camera_device: str | None = None,
        camera_backend: str = "auto",
        enable_camera: bool = True,
    ):
        self.watchdog_ms = watchdog_ms
        self.max_loop_hz = max_loop_hz
        self.enable_camera = enable_camera
        self.camera_index = camera_index
        self.camera_device = camera_device
        self.camera_backend = camera_backend

        self._context = zmq.Context()
        self._cmd_socket = self._context.socket(zmq.PULL)
        _configure_socket(self._cmd_socket)
        self._cmd_socket.bind(f"tcp://*:{cmd_port}")

        self._obs_socket = self._context.socket(zmq.PUSH)
        _configure_socket(self._obs_socket)
        self._obs_socket.bind(f"tcp://*:{obs_port}")

        self._last_cmd_time = time.time()
        self._watchdog_active = False
        self._speed = 100
        self._pan = 90
        self._tilt = 90
        self._red_led = False
        self._blue_led = False
        self._applied_pan = 90
        self._applied_tilt = 90
        self._last_distance_cm = -1.0
        self._distance_counter = 0

        self._camera = (
            open_camera(
                index=camera_index,
                device=camera_device,
                backend=camera_backend,
            )
            if enable_camera
            else None
        )
        if enable_camera and self._camera is None:
            logger.warning("Camera unavailable; continuing without video stream")

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()
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
            return RobotCommand.from_dict(data)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.error("Invalid command JSON: %s", exc)
            return None

    def _apply_movement(self, robot: Raspbot, command: RobotCommand) -> None:
        movement = command.movement
        speed = int(command.speed)
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

    def _apply_servos(self, robot: Raspbot, command: RobotCommand) -> None:
        pan = int(command.pan)
        tilt = int(command.tilt)

        if command.center_servos:
            robot.center_servos()
            self._applied_pan = 90
            self._applied_tilt = 90
            self._pan = 90
            self._tilt = 90
            return

        if not command.servo_dirty and pan == self._applied_pan and tilt == self._applied_tilt:
            return

        if pan != self._applied_pan:
            robot.set_pan(pan)
            self._applied_pan = pan
        if tilt != self._applied_tilt:
            robot.set_tilt(tilt)
            self._applied_tilt = tilt

        self._pan = pan
        self._tilt = tilt

    def _apply_command(self, robot: Raspbot, command: RobotCommand) -> None:
        if command.quit:
            return

        self._speed = int(command.speed)
        self._red_led = command.red_led
        self._blue_led = command.blue_led

        self._apply_servos(robot, command)
        self._apply_movement(robot, command)

        robot.led_red(self._red_led)
        robot.led_blue(self._blue_led)

        if command.beep:
            robot.beep(
                duration=float(command.beep.get("duration", 0.1)),
                frequency=int(command.beep.get("frequency", 440)),
            )

    def _read_distance_throttled(self, robot: Raspbot, every_n: int = 10) -> float:
        self._distance_counter += 1
        if self._distance_counter % every_n == 0:
            self._last_distance_cm = robot.read_distance()
        return self._last_distance_cm

    def _build_observation(self, robot: Raspbot) -> RobotObservation:
        line = robot.read_line_tracker()
        ir_left, ir_right = robot.read_ir_obstacle()
        return RobotObservation(
            distance_cm=self._read_distance_throttled(robot),
            line_tracker={
                "left1": line.left1,
                "left2": line.left2,
                "right1": line.right1,
                "right2": line.right2,
            },
            ir_obstacle={"left": ir_left, "right": ir_right},
            camera_jpeg_b64=capture_jpeg_b64(self._camera),
            pan=self._pan,
            tilt=self._tilt,
            speed=self._speed,
            red_led=self._red_led,
            blue_led=self._blue_led,
        )

    def run(self, robot: Raspbot, duration_s: float | None = None) -> None:
        logger.info("Waiting for commands on ZMQ...")
        start = time.perf_counter()

        while True:
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
                logger.debug("Observation dropped (client not keeping up)")

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
    parser.add_argument(
        "--camera-device",
        type=str,
        default=None,
        help="V4L2 device path, e.g. /dev/video0 (from v4l2-ctl --list-devices)",
    )
    parser.add_argument(
        "--camera-backend",
        choices=["auto", "picamera2", "v4l2"],
        default="auto",
        help="auto tries picamera2 (CSI) then V4L2 capture devices",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    robot = Raspbot()
    host = RaspbotHost(
        cmd_port=args.cmd_port,
        obs_port=args.obs_port,
        watchdog_ms=args.watchdog_ms,
        max_loop_hz=args.fps,
        camera_index=args.camera_index,
        camera_device=args.camera_device,
        camera_backend=args.camera_backend,
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
