"""
Yahboom Raspbot V1 Control Package

A Python library for controlling the Yahboom Raspbot V1 robot.
"""

from .raspbot import Raspbot, LineTrackerReading, create_robot

__all__ = [
    "Raspbot",
    "LineTrackerReading", 
    "create_robot",
]

__version__ = "1.0.0"
