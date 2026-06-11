"""
Camera sensor simulation for training (two parallel lines, no center line).

Produces the same 3-value format as vision.py so sim matches deployment.
Calibrated against recordings/ LCR logs.
"""

from __future__ import annotations

import numpy as np

from .lcr_data import RecordingBank, infer_position_from_sensors
from .vision import columns_to_lane_sensors


def simulate_column_darkness(position: float, noise_std: float = 0.10) -> tuple[float, float, float]:
    """
    Simulate per-column tape visibility from lateral position in [-1, 1].

    Lines are at the corridor edges (±1). No center line — middle column is floor.
    """
    pos = float(np.clip(position, -1.2, 1.2))

    left_dark = 0.08 + 0.92 * max(0.0, 1.0 - abs(pos - (-1.0)))
    right_dark = 0.08 + 0.92 * max(0.0, 1.0 - abs(pos - 1.0))
    # Real camera logs show higher center readings when roughly in-lane.
    center_dark = 0.10 + 0.22 * max(0.0, 1.0 - abs(pos))

    if noise_std > 0:
        left_dark += np.random.normal(0, noise_std)
        center_dark += np.random.normal(0, noise_std)
        right_dark += np.random.normal(0, noise_std)

    return (
        float(np.clip(left_dark, 0.0, 1.0)),
        float(np.clip(center_dark, 0.0, 1.0)),
        float(np.clip(right_dark, 0.0, 1.0)),
    )


def _saturate_sensors(sensors: list[float], snap_threshold: float = 0.72) -> list[float]:
    """Match real camera: L/R often peg at 0 or 1."""
    left, center, right = sensors
    if left > snap_threshold:
        left = 1.0
    elif left < (1.0 - snap_threshold):
        left = 0.0
    if right > snap_threshold:
        right = 1.0
    elif right < (1.0 - snap_threshold):
        right = 0.0
    return [float(left), float(center), float(right)]


def get_camera_sensor_values(
    car_state,
    noise_std: float = 0.10,
    center_boost: float = 0.12,
) -> list[float]:
    """3 lane sensors from simulated camera for a CarState position."""
    position = car_state.position if hasattr(car_state, "position") else float(car_state)
    l_dark, c_dark, r_dark = simulate_column_darkness(position, noise_std=noise_std)
    sensors = columns_to_lane_sensors(l_dark, c_dark, r_dark)
    sensors[1] = float(np.clip(sensors[1] + center_boost * max(0.0, 1.0 - abs(position)), 0.0, 1.0))
    return _saturate_sensors(sensors)


def get_training_sensors(
    car_state,
    recording_bank: RecordingBank | None = None,
    *,
    real_mix: float = 0.40,
    noise_std: float = 0.10,
    center_boost: float = 0.12,
) -> list[float]:
    """
    Sim sensors, sometimes replaced with a recorded tuple near the car position.

    real_mix: fraction of steps that use logged robot sensors (when bank is non-empty).
    """
    if recording_bank is not None and len(recording_bank) > 0 and np.random.random() < real_mix:
        position = car_state.position if hasattr(car_state, "position") else float(car_state)
        sampled = recording_bank.sample_near_position(position)
        if sampled is not None:
            # Keep physics and sensors aligned when injecting real logs.
            if hasattr(car_state, "position"):
                car_state.position = infer_position_from_sensors(sampled)
            return sampled
    return get_camera_sensor_values(car_state, noise_std=noise_std, center_boost=center_boost)
