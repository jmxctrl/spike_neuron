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

    def wheel_speeds(self, steer: float, speed_scale: float = 1.0, *, min_wheel: int = 0) -> tuple[int, int]:
        """
        Map steer ∈ [-1, 1] to signed wheel speeds for robot.control_car().

        Matches Yahboom arc convention: steer>0 → left wheel faster → turn right.
        """
        steer = max(-1.0, min(1.0, float(steer)))
        scale = max(0.1, min(1.0, speed_scale))
        base = max(1, int(self.base_speed * scale))
        d = max(1, int(self.steer_delta * scale))
        left = int(round(base + steer * d))
        right = int(round(base - steer * d))
        if min_wheel > 0:
            left = max(min_wheel, left)
            right = max(min_wheel, right)
        left = max(0, min(255, left))
        right = max(0, min(255, right))
        return left, right

    def execute_steer(self, steer: float, speed_scale: float = 1.0, *, min_wheel: int = 0) -> None:
        """Send differential speeds via ZMQ → host → Raspbot.control_car(left, right)."""
        left, right = self.wheel_speeds(steer, speed_scale=speed_scale, min_wheel=min_wheel)
        self._client.send_command(RobotCommand(left_speed=left, right_speed=right))

    def execute_action(self, action: int | float, speed_scale: float = 1.0) -> None:
        """Discrete {-1,0,1} or continuous steer ∈ [-1, 1]."""
        self.execute_steer(float(action), speed_scale=speed_scale)

    def execute_recovery(self, steer: float, speed_scale: float = 0.35) -> None:
        """Slow forward crawl with stronger steering — always rolls both wheels."""
        min_wheel = max(12, int(self.base_speed * max(0.15, min(1.0, speed_scale)) * 0.5))
        self.execute_steer(steer, speed_scale=speed_scale, min_wheel=min_wheel)

    def spin_left(self, speed: int = 68) -> None:
        self._client.send_command(RobotCommand(movement="turn_left", speed=speed))

    def spin_right(self, speed: int = 68) -> None:
        self._client.send_command(RobotCommand(movement="turn_right", speed=speed))

    def stop(self) -> None:
        self._client.send_command(RobotCommand(movement="stop"))
