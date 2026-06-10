"""Train SNN for camera-based corridor following (3 inputs)."""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np

from light_task import CarState, encode_sensors_to_spikes
from neuromodulation import apply_dopamine, classify_actions, compute_stdp_eligibility, reward_calculator
from spike_vectorized import create_weight_matrix, run_vectorized_lif

from .sim import get_camera_sensor_values

NUM_NEURONS = 100
NUM_STEPS = 10
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "trained_weights")


def save_weights(weights: np.ndarray, reward: float | None = None) -> str:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"reward{reward:.1f}_" if reward is not None else ""
    path = os.path.join(WEIGHTS_DIR, f"camera_{tag}{timestamp}.npy")
    np.save(path, weights)
    return path


def load_weights(path: str) -> np.ndarray:
    return np.load(path)


def list_weights() -> list[str]:
    if not os.path.isdir(WEIGHTS_DIR):
        return []
    return sorted(f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".npy"))


def lr_scheduler(episode: int, total: int, warmup: int = 50, max_lr: float = 0.01, min_lr: float = 1e-4) -> float:
    if episode < warmup:
        return max_lr * (episode / warmup)
    progress = (episode - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))


def _start_position_for_episode(episode: int, total: int) -> float:
    """Curriculum: easy centered starts early, wider range later."""
    progress = min(1.0, episode / max(total * 0.6, 1))
    half_range = 0.15 + 0.35 * progress
    return float(np.random.uniform(-half_range, half_range))


def _explore_prob_for_episode(episode: int, total: int, max_prob: float = 0.08) -> float:
    return max(0.02, max_prob * (1.0 - episode / max(total, 1)))


def run_episode(
    weights: np.ndarray,
    max_steps: int = 100,
    start_position: float | None = None,
    explore_prob: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    if start_position is None:
        start_position = float(np.random.uniform(-0.5, 0.5))

    car = CarState(position=start_position)
    prev_action = 0
    total_reward = 0.0
    spike_history = []
    reward_history = []
    eligibility_history = []

    for _ in range(max_steps):
        sensors = get_camera_sensor_values(car)
        input_spikes = encode_sensors_to_spikes(sensors, T=NUM_STEPS, p_max=0.2)

        spikes, _, pathway_history = run_vectorized_lif(
            W=weights,
            external_spikes=input_spikes,
            num_steps=NUM_STEPS,
            num_neurons=NUM_NEURONS,
            plot=False,
        )

        eligibility_history.append(compute_stdp_eligibility(spikes))
        action = classify_actions(pathway_history)

        if np.random.random() < explore_prob:
            action = int(np.random.choice([-1, 0, 1]))

        car.update(action)
        reward = reward_calculator(car, action, prev_action)

        spike_history.append(spikes[-1, :])
        reward_history.append(reward)
        total_reward += reward
        prev_action = action

        if car.is_crashed():
            break

    return (
        np.array(spike_history),
        np.array(reward_history),
        total_reward,
        np.array(eligibility_history),
    )


def train_snn(
    num_episodes: int = 500,
    learning_rate: float = 0.01,
    baseline_lr: float = 0.1,
    patience: int = 150,
    min_episodes: int = 200,
) -> tuple[np.ndarray, list[float], float]:
    W = create_weight_matrix(NUM_NEURONS)
    baseline = 0.0
    best_reward = -np.inf
    best_weights = W.copy()
    episode_rewards: list[float] = []
    stale = 0

    for episode in range(num_episodes):
        start_pos = _start_position_for_episode(episode, num_episodes)
        explore = _explore_prob_for_episode(episode, num_episodes)
        _, reward_history, total_reward, eligibility_history = run_episode(
            W, start_position=start_pos, explore_prob=explore
        )
        current_lr = lr_scheduler(episode, num_episodes, max_lr=learning_rate)

        W, baseline = apply_dopamine(
            W,
            eligibility_history,
            reward_history,
            baseline,
            baseline_lr=baseline_lr,
            learning_rate=current_lr,
        )

        episode_rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            best_weights = W.copy()
            stale = 0
            print(f"  New best episode {episode}: reward={total_reward:.2f}")
        else:
            stale += 1

        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}: reward={total_reward:.2f}, baseline={baseline:.2f}")

        if episode >= min_episodes and stale >= patience:
            print(f"Early stop at episode {episode} (no improvement for {patience} episodes)")
            break

    print(f"Best reward: {best_reward:.2f}")
    return best_weights, episode_rewards, best_reward
