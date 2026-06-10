#!/usr/bin/env python3
"""Train camera-corridor SNN. Run from repo root: uv run python -m camera.run_training"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from .train import save_weights, train_snn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SNN for camera corridor following")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    print("=" * 60)
    print("TRAINING CAMERA CORRIDOR SNN")
    print("=" * 60)

    weights, rewards, best_reward = train_snn(num_episodes=args.episodes, learning_rate=args.lr)
    path = save_weights(weights, reward=best_reward)
    print(f"\nSaved weights: {path}")
    print(f"Best reward: {best_reward:.2f}, mean last 20: {np.mean(rewards[-20:]):.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, marker="o", markersize=2, linewidth=1)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Camera corridor SNN training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "trained_weights")
    os.makedirs(out_dir, exist_ok=True)
    curve_path = os.path.join(out_dir, "training_curve.png")
    plt.savefig(curve_path, dpi=120)
    print(f"Saved plot: {curve_path}")


if __name__ == "__main__":
    main()
