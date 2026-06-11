"""
Minimal Spiking Neural Network — learning tutorial
==================================================

Network: 2 input neurons + 1 output neuron (3 total)
Task:    tell "warm" from "cold" using spike patterns
Learn:   STDP (Spike-Timing Dependent Plasticity)

Run:
    python example.py          # train, then open live visualization
    python example.py --train  # train only, save weights to example_weights.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch

# ---------------------------------------------------------------------------
# Network layout
# ---------------------------------------------------------------------------
N_INPUT = 2
N_OUTPUT = 1
N = N_INPUT + N_OUTPUT  # neurons 0,1 = inputs; neuron 2 = output

INPUT_NAMES = ["warm sensor", "cold sensor"]
OUTPUT_NAME = "decision"

# LIF neuron parameters (discrete time step = 1 ms)
DT = 1.0
TAU = 20.0          # membrane leak — higher = slower decay
V_REST = 0.0
V_THRESHOLD = 0.55  # fire when membrane potential crosses this
V_RESET = 0.0

# Rate encoding: value in [0,1] → spike probability per timestep
T_SIM = 80
P_MAX = 0.4

# Training patterns (what the two sensors "see")
PATTERN_WARM = np.array([0.9, 0.1])   # mostly warm → output should fire
PATTERN_COLD = np.array([0.1, 0.9])   # mostly cold → output should stay quiet

# STDP learning rates
A_PLUS = 0.12    # strengthen when pre fires before post (causal)
A_MINUS = 0.10   # weaken when post fires before pre
TAU_PLUS = 15.0
TAU_MINUS = 15.0
COLD_LTD = 0.08  # extra weakening on cold trials (no teacher spike)

WEIGHTS_PATH = Path(__file__).with_name("example_weights.npy")


# ---------------------------------------------------------------------------
# 1. Encode real-world values as spike trains (rate coding)
# ---------------------------------------------------------------------------
def rate_encode(values: np.ndarray, T: int, p_max: float, seed: int | None = None) -> np.ndarray:
    """
    Higher sensor value → more spikes over time.

    Returns shape (N_INPUT, T), values 0 or 1.
    """
    rng = np.random.default_rng(seed)
    values = np.clip(values, 0.0, 1.0)
    prob = values * p_max
    return (rng.random((len(values), T)) < prob[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. LIF neuron simulation
# ---------------------------------------------------------------------------
def simulate(
    W: np.ndarray,
    input_spikes: np.ndarray,
    force_output_spike_at: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run one trial.

    W:              (N, N) synaptic weights, W[i,j] = j → i
    input_spikes:   (N_INPUT, T) external spikes for input neurons
    force_output_spike_at: if set, teacher-forces output spike at this timestep

    Returns:
        spikes: (T, N)  — 1 where a neuron fired
        V:      (T, N)  — membrane potential trace
    """
    T = input_spikes.shape[1]
    spikes = np.zeros((T, N), dtype=np.float32)
    V = np.zeros((T, N), dtype=np.float32)
    v = np.full(N, V_REST, dtype=np.float32)

    for t in range(T):
        # Input neurons are driven externally (they don't integrate like outputs)
        spikes[t, :N_INPUT] = input_spikes[:, t]

        # Synaptic current into each neuron from previous timestep's spikes
        if t == 0:
            I_syn = np.zeros(N, dtype=np.float32)
        else:
            I_syn = 1.8 * (W @ spikes[t - 1])

        # LIF update for all neurons (inputs get overwritten above each step)
        v = v + (DT / TAU) * (-(v - V_REST) + I_syn)

        # Output neuron(s) can spike from membrane dynamics
        out_idx = N_INPUT
        if v[out_idx] >= V_THRESHOLD:
            spikes[t, out_idx] = 1.0
            v[out_idx] = V_RESET

        # Teacher signal during training
        if force_output_spike_at is not None and t == force_output_spike_at:
            spikes[t, out_idx] = 1.0
            v[out_idx] = V_RESET

        V[t] = v

    return spikes, V


# ---------------------------------------------------------------------------
# 3. STDP weight update
# ---------------------------------------------------------------------------
def stdp_update(spikes: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    For every pair (pre=j, post=i), if pre spiked before post → strengthen;
    if post spiked before pre → weaken.  Matches the rule in README.md.
    """
    T = spikes.shape[0]
    delta_W = np.zeros_like(W)

    for post in range(N):
        for pre in range(N):
            if pre == post:
                continue
            t_pre = np.where(spikes[:, pre] > 0)[0]
            t_post = np.where(spikes[:, post] > 0)[0]
            if len(t_pre) == 0 or len(t_post) == 0:
                continue

            dw = 0.0
            for tp in t_post:
                for tq in t_pre:
                    dt = tp - tq  # post minus pre
                    if dt > 0:
                        dw += A_PLUS * np.exp(-dt / TAU_PLUS)   # LTP: causal
                    elif dt < 0:
                        dw -= A_MINUS * np.exp(dt / TAU_MINUS)  # LTD: anti-causal
            delta_W[post, pre] = dw

    return delta_W


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------
def init_weights() -> np.ndarray:
    W = np.zeros((N, N), dtype=np.float32)
    W[N_INPUT:, :N_INPUT] = np.random.uniform(0.05, 0.15, (N_OUTPUT, N_INPUT))
    return W


def train(num_epochs: int = 80, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = init_weights()
    teacher_time = T_SIM - 8  # force correct output spike near end of warm trials

    print("Training 2→1 spiking network with STDP")
    print(f"  warm pattern: {PATTERN_WARM}  → output should FIRE")
    print(f"  cold pattern: {PATTERN_COLD}  → output should stay QUIET\n")

    for epoch in range(num_epochs):
        # Alternate warm / cold; shuffle order each epoch
        trials = [(PATTERN_WARM, True), (PATTERN_COLD, False)]
        rng.shuffle(trials)

        for pattern, should_fire in trials:
            inp = rate_encode(pattern, T_SIM, P_MAX, seed=int(rng.integers(1_000_000)))
            force_at = teacher_time if should_fire else None
            spikes, _ = simulate(W, inp, force_output_spike_at=force_at)

            if should_fire:
                # STDP: input spikes before teacher output → strengthen warm→out
                delta = stdp_update(spikes, W)
                delta[:N_INPUT, :] = 0.0
                delta[:, N_INPUT:] = 0.0
                # Warm trial: only strengthen warm→out, weaken cold→out
                delta[2, 0] = max(delta[2, 0], 0.0)
                delta[2, 1] = min(delta[2, 1], 0.0)
                W += delta
            else:
                # Cold trial: weaken cold→out path
                cold_spike_count = inp[1].sum()
                W[2, 1] -= COLD_LTD * cold_spike_count / T_SIM
                W[2, 0] -= 0.01 * cold_spike_count / T_SIM  # slight push apart

            W = np.clip(W, 0.0, 1.5)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            warm_out = test_pattern(W, PATTERN_WARM, seed=0)
            cold_out = test_pattern(W, PATTERN_COLD, seed=1)
            print(
                f"epoch {epoch:3d}  "
                f"W=[{W[2,0]:.2f}, {W[2,1]:.2f}]  "
                f"warm spikes={warm_out:.0f}  cold spikes={cold_out:.0f}"
            )

    return W


def test_pattern(W: np.ndarray, pattern: np.ndarray, seed: int = 0) -> float:
    """Return number of output spikes for one pattern (no teacher forcing)."""
    inp = rate_encode(pattern, T_SIM, P_MAX, seed=seed)
    spikes, _ = simulate(W, inp)
    return spikes[:, N_INPUT:].sum()


# ---------------------------------------------------------------------------
# 5. Live visualization (trained model)
# ---------------------------------------------------------------------------
def run_visualization(W: np.ndarray) -> None:
    """
    Animate the trained network cycling warm → cold → warm …

    Layout:
      top-left  — network diagram (circles flash on spike)
      top-right — weight bars (thickness ∝ learned strength)
      bottom    — membrane potentials + spike raster
    """
    patterns = [
        ("WARM — output should fire", PATTERN_WARM, 0),
        ("COLD — output should stay quiet", PATTERN_COLD, 1),
    ]
    pattern_idx = 0
    trial_spikes: np.ndarray | None = None
    trial_V: np.ndarray | None = None
    trial_inp: np.ndarray | None = None
    t_cursor = 0
    pause_between = 20
    pause_counter = 0

    fig = plt.figure(figsize=(11, 7))
    fig.patch.set_facecolor("#0f1117")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.3)

    ax_net = fig.add_subplot(gs[0, 0])
    ax_w = fig.add_subplot(gs[0, 1])
    ax_trace = fig.add_subplot(gs[1, :])

    for ax in (ax_net, ax_w, ax_trace):
        ax.set_facecolor("#1a1d27")

    neuron_pos = {
        0: (-0.8, 0.5),
        1: (-0.8, -0.5),
        2: (0.9, 0.0),
    }
    colors = ["#ff6b6b", "#4ecdc4", "#ffd93d"]

    title = fig.suptitle("", color="white", fontsize=14, y=0.98)

    def start_trial():
        nonlocal trial_spikes, trial_V, trial_inp, t_cursor, pause_counter
        label, pattern, pattern_idx_local = patterns[pattern_idx]
        title.set_text(f"Live inference — {label}")
        inp = rate_encode(pattern, T_SIM, P_MAX, seed=42 + pattern_idx_local)
        spikes, V = simulate(W, inp)
        trial_spikes, trial_V, trial_inp = spikes, V, inp
        t_cursor = 0
        pause_counter = 0

    def draw_network_frame(t: int):
        ax_net.clear()
        ax_net.set_facecolor("#1a1d27")
        ax_net.set_xlim(-1.3, 1.3)
        ax_net.set_ylim(-1.0, 1.0)
        ax_net.axis("off")
        ax_net.set_title("Network", color="white", fontsize=11)

        assert trial_spikes is not None
        for pre in range(N_INPUT):
            for post in range(N_INPUT, N):
                w = W[post, pre]
                x0, y0 = neuron_pos[pre]
                x1, y1 = neuron_pos[post]
                ax_net.add_patch(
                    FancyArrowPatch(
                        (x0 + 0.15, y0 * 0.3),
                        (x1 - 0.15, y1 * 0.3),
                        arrowstyle="-|>",
                        color="#888",
                        linewidth=1 + 4 * w,
                        alpha=0.4 + 0.4 * min(w, 1.0),
                        mutation_scale=12,
                    )
                )

        labels = INPUT_NAMES + [OUTPUT_NAME]
        for i, (x, y) in neuron_pos.items():
            fired = trial_spikes[t, i] > 0
            glow = 0.95 if fired else 0.35
            radius = 0.18 if fired else 0.14
            ax_net.add_patch(
                Circle(
                    (x, y),
                    radius,
                    facecolor=colors[i],
                    edgecolor="white" if fired else "#555",
                    linewidth=2 if fired else 1,
                    alpha=glow,
                )
            )
            ax_net.text(x, y - 0.32, labels[i], ha="center", color="white", fontsize=9)

    def draw_weights():
        ax_w.clear()
        ax_w.set_facecolor("#1a1d27")
        ax_w.set_title("Learned synaptic weights", color="white", fontsize=11)
        labels = ["warm→out", "cold→out"]
        vals = [W[2, 0], W[2, 1]]
        bars = ax_w.barh(labels, vals, color=["#ff6b6b", "#4ecdc4"], alpha=0.85)
        ax_w.set_xlim(0, max(1.2, max(vals) * 1.2))
        ax_w.tick_params(colors="white")
        for spine in ax_w.spines.values():
            spine.set_color("#444")
        ax_w.set_xlabel("weight strength", color="#aaa")
        for bar, v in zip(bars, vals):
            ax_w.text(v + 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", color="white")

    def draw_traces(t: int):
        ax_trace.clear()
        ax_trace.set_facecolor("#1a1d27")
        ax_trace.set_title("Membrane potential (output neuron) + spike raster", color="white", fontsize=11)

        assert trial_spikes is not None and trial_V is not None and trial_inp is not None
        T = trial_spikes.shape[0]
        times = np.arange(T)

        # Output voltage trace up to current time
        ax_trace.plot(times[: t + 1], trial_V[: t + 1, 2], color="#ffd93d", linewidth=2, label="output V")
        ax_trace.axhline(V_THRESHOLD, color="#666", linestyle="--", linewidth=1, label="threshold")

        # Spike raster for all neurons
        for i in range(N):
            spike_times = np.where(trial_spikes[: t + 1, i] > 0)[0]
            ax_trace.scatter(
                spike_times,
                np.full_like(spike_times, 0.15 + i * 0.12),
                marker="|",
                s=200,
                color=colors[i],
                linewidths=2,
            )

        ax_trace.set_xlim(0, T)
        ax_trace.set_ylim(-0.1, 1.25)
        ax_trace.set_xlabel("time step (ms)", color="#aaa")
        ax_trace.set_ylabel("potential / spikes", color="#aaa")
        ax_trace.tick_params(colors="white")
        for spine in ax_trace.spines.values():
            spine.set_color("#444")
        ax_trace.legend(loc="upper right", fontsize=8, facecolor="#1a1d27", labelcolor="white")

        out_count = trial_spikes[: t + 1, 2].sum()
        ax_trace.text(
            0.02,
            0.95,
            f"output spikes so far: {int(out_count)}",
            transform=ax_trace.transAxes,
            color="white",
            fontsize=10,
            va="top",
        )

    def update(_frame):
        nonlocal t_cursor, pattern_idx, pause_counter

        if trial_spikes is None:
            start_trial()
            draw_network_frame(0)
            draw_weights()
            draw_traces(0)
            return []

        if t_cursor < T_SIM - 1:
            t_cursor += 1
        else:
            pause_counter += 1
            if pause_counter >= pause_between:
                pattern_idx = (pattern_idx + 1) % len(patterns)
                start_trial()
            t_cursor = min(t_cursor, T_SIM - 1)

        draw_network_frame(t_cursor)
        draw_weights()
        draw_traces(t_cursor)
        return []

    start_trial()
    anim = FuncAnimation(fig, update, frames=T_SIM * 8, interval=80, blit=False, repeat=True)
    plt.tight_layout()
    print("\nVisualization running — close the window to exit.")
    print("  Top-left:  neurons flash yellow/red/teal when they spike")
    print("  Top-right: learned weights (warm→output should be larger)")
    print("  Bottom:    output voltage climbing toward threshold + spike ticks\n")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SNN tutorial")
    parser.add_argument("--train", action="store_true", help="Train and save weights only")
    parser.add_argument("--viz", action="store_true", help="Skip training, load saved weights")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    if args.viz and WEIGHTS_PATH.exists():
        W = np.load(WEIGHTS_PATH)
        print(f"Loaded weights from {WEIGHTS_PATH}")
    else:
        W = train(num_epochs=args.epochs)
        np.save(WEIGHTS_PATH, W)
        print(f"\nSaved weights to {WEIGHTS_PATH}")

    if args.train:
        print("\nDone (--train). Run without flags to visualize.")
        return

    warm = test_pattern(W, PATTERN_WARM)
    cold = test_pattern(W, PATTERN_COLD)
    print(f"\nFinal test (no teacher): warm output spikes={warm:.0f}, cold output spikes={cold:.0f}")
    if warm > cold:
        print("Network learned the task.")
    else:
        print("Try more epochs: python example.py --epochs 120")

    run_visualization(W)


if __name__ == "__main__":
    main()
