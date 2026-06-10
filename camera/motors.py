"""Yahboom motor adapter for lane-following (forward + arc steering)."""

from __future__ import annotations

from yahboom.raspbot import Raspbot

from .controller import ACTION_LEFT, ACTION_RIGHT, ACTION_STRAIGHT


class YahboomDriver:
    def __init__(self, base_speed: int = 70, steer_delta: int = 35):
        self.robot = Raspbot()
        self.base_speed = base_speed
        self.steer_delta = steer_delta
        self._initialized = False

    def start(self) -> None:
        self.robot.init()
        self._initialized = True

    def execute_action(self, action: int) -> None:
        base = self.base_speed
        d = self.steer_delta
        if action == ACTION_STRAIGHT:
            self.robot.forward(base)
        elif action == ACTION_LEFT:
            self.robot.control_car(base - d, base + d)
        elif action == ACTION_RIGHT:
            self.robot.control_car(base + d, base - d)
        else:
            self.robot.stop()

    def stop(self) -> None:
        if self._initialized:
            self.robot.stop()

    def close(self) -> None:
        if self._initialized:
            self.robot.stop()
            self.robot.cleanup()
            self._initialized = False
