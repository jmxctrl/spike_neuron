"""Load recorded L/C/R CSVs and sample realistic sensor tuples for training."""

from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass

import numpy as np


def infer_position_from_sensors(sensors: list[float]) -> float:
    """Rough lateral position in [-1, 1] from lane sensors (R high -> +1)."""
    left, _, right = sensors
    return float(np.clip(right - left, -1.0, 1.0))


@dataclass
class RecordingBank:
    """Recorded (position estimate, sensors) samples from robot logs."""

    samples: list[tuple[float, list[float]]]

    @classmethod
    def from_glob(cls, pattern: str) -> RecordingBank:
        paths = sorted(glob.glob(pattern))
        samples: list[tuple[float, list[float]]] = []
        for path in paths:
            samples.extend(_load_csv(path))
        return cls(samples)

    @classmethod
    def from_dir(cls, directory: str) -> RecordingBank:
        return cls.from_glob(os.path.join(directory, "lcr_*.csv"))

    def __len__(self) -> int:
        return len(self.samples)

    def sample_near_position(self, position: float, tolerance: float = 0.45) -> list[float] | None:
        if not self.samples:
            return None
        matches = [s for p, s in self.samples if abs(p - position) <= tolerance]
        pool = matches if matches else [s for _, s in self.samples]
        return pool[int(np.random.randint(0, len(pool)))]


def _load_csv(path: str) -> list[tuple[float, list[float]]]:
    rows: list[tuple[float, list[float]]] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(line for line in fh if not line.startswith("#"))
        for row in reader:
            sensors = [float(row["L"]), float(row["C"]), float(row["R"])]
            rows.append((infer_position_from_sensors(sensors), sensors))
    return rows
