"""Send arc-steering motor commands to Yahboom host over ZMQ."""

from __future__ import annotations

from yahboom.client import RaspbotClient
from yahboom.protocol import RobotCommand

from .controller import ACTION_LEFT, ACTION_RIGHT, ACTION_STRAIGHT


class YahboomRemoteDriver:
    def __init__(self, client: RaspbotClient, base_speed: int = 35, steer_delta: int = 18):
        self._client = client
        self.base_speed = base_speed
        self.steer_delta = steer_delta

    def execute_action(self, action: int) -> None:
        base = self.base_speed
        d = self.steer_delta
        if action == ACTION_STRAIGHT:
            cmd = RobotCommand(left_speed=base, right_speed=base)
        elif action == ACTION_LEFT:
            cmd = RobotCommand(left_speed=base - d, right_speed=base + d)
        elif action == ACTION_RIGHT:
            cmd = RobotCommand(left_speed=base + d, right_speed=base - d)
        else:
            cmd = RobotCommand(movement="stop")
        self._client.send_command(cmd)

    def stop(self) -> None:
        self._client.send_command(RobotCommand(movement="stop"))
