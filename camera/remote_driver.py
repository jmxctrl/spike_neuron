"""Send arc-steering motor commands to Yahboom host over ZMQ."""

from __future__ import annotations

from yahboom.client import RaspbotClient
from yahboom.protocol import RobotCommand

from .controller import ACTION_LEFT, ACTION_RIGHT, ACTION_STRAIGHT


class YahboomRemoteDriver:
    def __init__(self, client: RaspbotClient, base_speed: int = 52, steer_delta: int = 27):
        self._client = client
        self.base_speed = base_speed
        self.steer_delta = steer_delta

    def execute_action(self, action: int, speed_scale: float = 1.0) -> None:
        scale = max(0.1, min(1.0, speed_scale))
        base = max(1, int(self.base_speed * scale))
        d = max(1, int(self.steer_delta * scale))
        self._send_arc(action, base, d)

    def execute_recovery(self, action: int, speed_scale: float = 0.35) -> None:
        """Slow forward crawl with stronger steering — always rolls both wheels."""
        scale = max(0.15, min(1.0, speed_scale))
        base = max(22, int(self.base_speed * scale))
        d = max(18, int(self.steer_delta * scale * 2.0))
        min_wheel = max(12, int(base * 0.5))
        left, right = self._arc_speeds(action, base, d)
        left = max(min_wheel, left)
        right = max(min_wheel, right)
        self._client.send_command(RobotCommand(left_speed=left, right_speed=right))

    def _arc_speeds(self, action: int, base: int, d: int) -> tuple[int, int]:
        if action == ACTION_STRAIGHT:
            return base, base
        if action == ACTION_LEFT:
            return base - d, base + d
        if action == ACTION_RIGHT:
            return base + d, base - d
        return 0, 0

    def _send_arc(self, action: int, base: int, d: int) -> None:
        if action == ACTION_STRAIGHT:
            self._client.send_command(RobotCommand(left_speed=base, right_speed=base))
        elif action == ACTION_LEFT:
            self._client.send_command(RobotCommand(left_speed=base - d, right_speed=base + d))
        elif action == ACTION_RIGHT:
            self._client.send_command(RobotCommand(left_speed=base + d, right_speed=base - d))
        else:
            self._client.send_command(RobotCommand(movement="stop"))

    def spin_left(self, speed: int = 68) -> None:
        self._client.send_command(RobotCommand(movement="turn_left", speed=speed))

    def spin_right(self, speed: int = 68) -> None:
        self._client.send_command(RobotCommand(movement="turn_right", speed=speed))

    def stop(self) -> None:
        self._client.send_command(RobotCommand(movement="stop"))
