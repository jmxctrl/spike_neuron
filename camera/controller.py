"""SNN inference controller for camera lane following."""

from __future__ import annotations

import numpy as np

from light_task import encode_sensors_to_spikes
from neuromodulation import classify_actions
from spike_vectorized import run_vectorized_lif

from .train import NUM_NEURONS, NUM_STEPS, load_weights

ACTION_STRAIGHT = 0
ACTION_LEFT = -1
ACTION_RIGHT = 1


class CameraSNNController:
    def __init__(self, weights_path: str):
        self.weights = load_weights(weights_path)
        self.last_sensors: list[float] = [0.0, 0.5, 0.0]
        self.last_action: int = ACTION_STRAIGHT

    def run_inference(self, sensors: list[float] | None = None) -> tuple[int, list[float]]:
        if sensors is not None:
            self.last_sensors = sensors
        else:
            raise ValueError("sensors required (use FrameLaneSensor.read_from_frame on laptop)")
        input_spikes = encode_sensors_to_spikes(self.last_sensors, T=NUM_STEPS, p_max=0.2)

        _, _, pathway_history = run_vectorized_lif(
            W=self.weights,
            external_spikes=input_spikes,
            num_steps=NUM_STEPS,
            num_neurons=NUM_NEURONS,
            plot=False,
        )

        self.last_action = int(classify_actions(pathway_history))
        return self.last_action, self.last_sensors
