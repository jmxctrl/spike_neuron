#!/usr/bin/env python3
"""
Visualize the camera training simulator.

Shows top-down corridor, per-column tape scores, and L/C/R sensors vs position.

  uv run python -m camera.view_sim
  uv run python -m camera.view_sim --drive --weights camera/trained_weights/camera_reward295.7_....npy
  uv run python -m camera.view_sim --real-mix 0.25   # same sensor mix as training
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation

from light_task import CarState, encode_sensors_to_spikes
from neuromodulation import compute_steer
from spike_vectorized import run_vectorized_lif

from .controller import InferenceConfig
from .lcr_data import RecordingBank
from .sim import get_camera_sensor_values, get_training_sensors, simulate_column_darkness
from .train import DEFAULT_RECORDINGS_DIR, NUM_NEURONS, NUM_STEPS, P_MAX, list_weights, load_weights

ACTION_NAMES = {-1: "LEFT", 0: "FWD", 1: "RIGHT"}


def _default_weights() -> str | None:
    files = list_weights()
    if not files:
        return None
    return os.path.join(os.path.dirname(__file__), "trained_weights", sorted(files)[-1])


def _infer_steer(weights: np.ndarray, sensors: list[float], cfg: InferenceConfig) -> float:
    spikes = encode_sensors_to_spikes(sensors, T=cfg.num_steps, p_max=cfg.p_max)
    _, _, pathway = run_vectorized_lif(
        W=weights,
        external_spikes=spikes,
        num_steps=cfg.num_steps,
        num_neurons=NUM_NEURONS,
        tau=cfg.lif_tau,
        threshold=cfg.lif_threshold,
        plot=False,
    )
    return compute_steer(
        pathway,
        window=cfg.action_window,
        gain=cfg.steer_gain,
        deadzone=cfg.steer_deadzone,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize camera corridor training sim")
    parser.add_argument("--drive", action="store_true", help="Autopilot with SNN weights (else sweep position)")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--start", type=float, default=0.0, help="Start position in [-1, 1]")
    parser.add_argument("--real-mix", type=float, default=0.0, help="Training-style recording injection (0..1)")
    parser.add_argument("--recordings-dir", type=str, default=DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--interval", type=int, default=80, help="Animation ms per frame")
    args = parser.parse_args()

    bank: RecordingBank | None = None
    if args.real_mix > 0 and os.path.isdir(args.recordings_dir):
        bank = RecordingBank.from_dir(args.recordings_dir)
        print(f"Recording mix {args.real_mix:.0%} ({len(bank)} samples)")

    weights: np.ndarray | None = None
    cfg = InferenceConfig()
    if args.drive:
        path = args.weights or _default_weights()
        if not path or not os.path.isfile(path):
            raise SystemExit("Need --weights or trained weights in camera/trained_weights/")
        weights = load_weights(path)
        print(f"Driving with {path}")

    car = CarState(position=args.start)
    mode = "drive" if args.drive else "sweep"
    sweep_positions = np.linspace(-0.95, 0.95, args.steps)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#1a1d27")
    for ax in axes:
        ax.set_facecolor("#252836")

    ax_corridor, ax_camera, ax_bars = axes
    title = fig.suptitle("", color="white", fontsize=12)

    def draw(frame: int):
        for ax in axes:
            ax.clear()
            ax.set_facecolor("#252836")
            ax.tick_params(colors="#aaa")

        if mode == "sweep":
            car.position = float(sweep_positions[frame % len(sweep_positions)])
            sensors = get_camera_sensor_values(car)
            steer = 0.0
            source = "sim"
        else:
            sensors = get_training_sensors(car, bank, real_mix=args.real_mix)
            source = "recording" if bank and args.real_mix > 0 else "sim"
            assert weights is not None
            steer = _infer_steer(weights, sensors, cfg)
            car.update(steer)

        cols = simulate_column_darkness(car.position, noise_std=0.0)
        l, c, r = sensors

        # --- corridor ---
        ax_corridor.set_xlim(-1.4, 1.4)
        ax_corridor.set_ylim(-0.3, 1.2)
        ax_corridor.axvline(-1.0, color="#b060ff", linewidth=3, label="left line")
        ax_corridor.axvline(1.0, color="#b060ff", linewidth=3, label="right line")
        ax_corridor.fill_between([-1.0, 1.0], 0, 1, color="#2a3d2a", alpha=0.5)
        ax_corridor.plot(car.position, 0.5, "o", color="#ffd93d", markersize=14, zorder=5)
        ax_corridor.annotate(
            "",
            xy=(car.position + steer * 0.15, 0.65),
            xytext=(car.position, 0.5),
            arrowprops=dict(arrowstyle="->", color="#4ecdc4", lw=2),
        )
        ax_corridor.set_title("Top-down corridor", color="white")
        ax_corridor.set_xlabel("position (−1=left edge, +1=right)", color="#aaa")
        ax_corridor.set_yticks([])

        # --- fake camera columns ---
        ax_camera.set_xlim(0, 3)
        ax_camera.set_ylim(0, 1)
        colors = ["#ff6b6b", "#6bff95", "#6b9fff"]
        labels = ("L col", "C col", "R col")
        for i, (val, col, lab) in enumerate(zip(cols, colors, labels)):
            ax_camera.bar(i + 0.15, val, width=0.7, color=col, alpha=0.35 + 0.65 * val)
            ax_camera.text(i + 0.5, 0.02, f"{lab}\n{val:.2f}", ha="center", color="white", fontsize=9)
        ax_camera.set_xticks([])
        ax_camera.set_title("Column tape (noise-free)", color="white")
        ax_camera.set_ylabel("tape fraction", color="#aaa")

        # --- LCR bars ---
        ax_bars.set_xlim(0, 3)
        ax_bars.set_ylim(0, 1.05)
        names = ("L", "C", "R")
        for i, (val, col, name) in enumerate(zip([l, c, r], colors, names)):
            ax_bars.bar(i + 0.15, val, width=0.7, color=col, alpha=0.85)
            ax_bars.text(i + 0.5, val + 0.02, f"{name}={val:.2f}", ha="center", color="white", fontsize=9)
        ax_bars.set_xticks([])
        ax_bars.set_title("Lane sensors → SNN input", color="white")

        title.set_text(
            f"mode={mode}  pos={car.position:+.2f}  source={source}  "
            f"steer={steer:+.2f}  real_mix={args.real_mix:.0%}"
        )

        if car.is_crashed():
            title.set_text(title.get_text() + "  CRASHED")

        return []

    frames = args.steps if mode == "drive" else len(sweep_positions)
    plt.tight_layout()
    draw(0)  # first frame immediately (don't wait for timer)
    # Must keep a reference — otherwise macOS/backend garbage-collects before anything draws.
    _anim = FuncAnimation(
        fig,
        draw,
        frames=frames,
        interval=args.interval,
        repeat=mode == "sweep",
        blit=False,
    )
    print("Close the window to exit.")
    print("  Left: car between purple lines | Middle: raw column tape | Right: L/C/R after vision.py mapping")
    plt.show()


if __name__ == "__main__":
    main()
