"""SNN inference controller for camera lane following."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from light_task import encode_sensors_to_spikes
from neuromodulation import classify_actions
from spike_vectorized import run_vectorized_lif

from .train import NUM_NEURONS, NUM_STEPS, load_weights

ACTION_STRAIGHT = 0
ACTION_LEFT = -1
ACTION_RIGHT = 1


@dataclass
class InferenceConfig:
    """Runtime SNN inference knobs (defaults match original deploy)."""

    num_steps: int = NUM_STEPS
    p_max: float = 0.2
    action_window: int = 10
    pool_size: int = 33
    lif_tau: float = 20.0
    lif_threshold: float = 1.0


class CameraSNNController:
    def __init__(self, weights_path: str, config: InferenceConfig | None = None):
        self.weights = load_weights(weights_path)
        self.config = config or InferenceConfig()
        self.last_sensors: list[float] = [0.0, 0.5, 0.0]
        self.last_action: int = ACTION_STRAIGHT

    def run_inference(self, sensors: list[float] | None = None) -> tuple[int, list[float]]:
        if sensors is not None:
            self.last_sensors = sensors
        else:
            raise ValueError("sensors required (use FrameLaneSensor.read_from_frame on laptop)")

        cfg = self.config
        input_spikes = encode_sensors_to_spikes(self.last_sensors, T=cfg.num_steps, p_max=cfg.p_max)

        _, _, pathway_history = run_vectorized_lif(
            W=self.weights,
            external_spikes=input_spikes,
            num_steps=cfg.num_steps,
            num_neurons=NUM_NEURONS,
            tau=cfg.lif_tau,
            threshold=cfg.lif_threshold,
            plot=False,
        )

        self.last_action = int(
            classify_actions(pathway_history, pool_size=cfg.pool_size, window=cfg.action_window)
        )
        return self.last_action, self.last_sensors
