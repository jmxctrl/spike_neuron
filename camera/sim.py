"""
Camera sensor simulation for training (two parallel lines, no center line).

Produces the same 3-value format as vision.py so sim matches deployment.
"""

from __future__ import annotations

import numpy as np

from .vision import columns_to_lane_sensors


def simulate_column_darkness(position: float, noise_std: float = 0.04) -> tuple[float, float, float]:
    """
    Simulate per-column tape visibility from lateral position in [-1, 1].

    Lines are at the corridor edges (±1). No center line — middle column is floor.
    """
    pos = float(np.clip(position, -1.2, 1.2))

    # Lines at corridor edges (±1), matching light_task CarState bounds
    left_dark = 0.08 + 0.92 * max(0.0, 1.0 - abs(pos - (-1.0)))
    right_dark = 0.08 + 0.92 * max(0.0, 1.0 - abs(pos - 1.0))
    center_dark = 0.04 + 0.18 * abs(pos)

    if noise_std > 0:
        left_dark += np.random.normal(0, noise_std)
        center_dark += np.random.normal(0, noise_std)
        right_dark += np.random.normal(0, noise_std)

    return (
        float(np.clip(left_dark, 0.0, 1.0)),
        float(np.clip(center_dark, 0.0, 1.0)),
        float(np.clip(right_dark, 0.0, 1.0)),
    )


def get_camera_sensor_values(car_state, noise_std: float = 0.06) -> list[float]:
    """3 lane sensors from simulated camera for a CarState position."""
    position = car_state.position if hasattr(car_state, "position") else float(car_state)
    l_dark, c_dark, r_dark = simulate_column_darkness(position, noise_std=noise_std)
    return columns_to_lane_sensors(l_dark, c_dark, r_dark)
