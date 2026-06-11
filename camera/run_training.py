#!/usr/bin/env python3
"""Train camera-corridor SNN. Run from repo root: uv run python -m camera.run_training"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from .lcr_data import RecordingBank
from .train import DEFAULT_RECORDINGS_DIR, save_weights, train_snn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SNN for camera corridor following")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--recordings-dir",
        type=str,
        default=DEFAULT_RECORDINGS_DIR,
        help="Directory with lcr_*.csv logs for sim-real calibration",
    )
    parser.add_argument(
        "--real-mix",
        type=float,
        default=0.25,
        help="Fraction of training steps using recorded sensors",
    )
    parser.add_argument("--no-recordings", action="store_true", help="Sim only, ignore CSV logs")
    parser.add_argument(
        "--init-weights",
        type=str,
        default=None,
        help="Fine-tune from an existing .npy instead of random init",
    )
    args = parser.parse_args()

    bank: RecordingBank | None = None
    if not args.no_recordings and os.path.isdir(args.recordings_dir):
        bank = RecordingBank.from_dir(args.recordings_dir)
        print(f"Loaded {len(bank)} recorded sensor samples from {args.recordings_dir}")
    else:
        print("Training with sim-only sensors (no recordings)")

    print("=" * 60)
    print("TRAINING CAMERA CORRIDOR SNN")
    print("=" * 60)

    init_w = np.load(args.init_weights) if args.init_weights else None
    if init_w is not None:
        print(f"Fine-tuning from {args.init_weights}")

    weights, rewards, best_reward = train_snn(
        num_episodes=args.episodes,
        learning_rate=args.lr,
        recording_bank=bank,
        real_mix=args.real_mix,
        init_weights=init_w,
    )
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
