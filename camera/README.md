# Camera corridor SNN

Drive the Yahboom robot **between two parallel floor lines** using a spiking neural network and a **forward-facing USB camera**.

No center line is required — the camera sees the left and right tape in the lower image; the SNN steers to stay centered.

---

## Architecture (remote deploy)

The Pi **only runs the host** (motors + camera). Your Mac runs vision, SNN inference, debugging, and Rerun.

```
┌─────────────────────────────┐         ZMQ          ┌──────────────────────────┐
│  Mac (camera.play)          │  ── motor cmds :5555 ►│  Pi (yahboom.host)       │
│  • lane sensors from JPEG   │  ◄─ camera JPEG :5556 │  • USB cam /dev/video8   │
│  • SNN inference            │                       │  • differential motors   │
│  • Rerun debug overlay      │                       │                          │
└─────────────────────────────┘                       └──────────────────────────┘
```

| Command | Where | Purpose |
|---------|-------|---------|
| `yahboom.host` | **Pi** | Camera stream + motor execution |
| `camera.debug_vision` | **Mac** | See what the car sees (no driving) |
| `camera.play` | **Mac** | Autonomous SNN + live debug |
| `camera.run_training` | **Mac** | Train weights (simulation) |

---

## How it works

```
        camera view (forward, bottom of frame)
   ┌─────────┬─────────┬─────────┐
   │  LEFT   │ CENTER  │  RIGHT  │  ← 3 columns
   │  tape?  │  floor  │  tape?  │
   └─────────┴─────────┴─────────┘
              ↓
   3 values [left, center, right] in [0, 1]
              ↓
   rate-encoded spikes → 100-neuron SNN → steer left / forward / right
```

| Sensor | Meaning |
|--------|---------|
| **left** | High → too close to left line (or left line dominates) |
| **center** | High → well centered between both lines |
| **right** | High → too close to right line |

Debug output also shows human-readable state, e.g. `DRIFT LEFT — too close to left line`.

---

## Ping-pong laps (end bar → turn 180° → repeat)

**No retrain needed.** After a 180° turn the robot faces back down the corridor — same forward-following SNN.

Add a **horizontal purple tape bar** across the lane at each end:

```
══════════════════════════════════  ← end bar (perpendicular)
        ║                  ║
        ║    corridor      ║
        ║                  ║
══════════════════════════════════  ← start / other end
```

When the end bar fills the **bottom 20%** of the camera view for a few frames, the robot spins 180°, **drives away** from the bar (SNN steering), then ignores the end zone for a few seconds before re-arming. This repeats forever (default in `camera.play`).

```bash
uv run python -m camera.play \
  --weights camera/trained_weights/<file>.npy
```

Tune turn if it under/over-rotates:

```bash
uv run python -m camera.play \
  --weights camera/trained_weights/<file>.npy \
  --turn-seconds 3.0 \
  --turn-speed 50
```

Single-direction only (no turns): `--no-ping-pong`

Debug overlay shows a **cyan box** at the bottom = end detection zone. Text `END LINE` appears when triggered.

---

## Floor markings (30 cm apart)

**Two solid parallel lines are enough.** You do **not** need a center line.

| Item | Recommendation |
|------|----------------|
| **Lines** | 2 parallel black tape lines on white/light surface |
| **Spacing** | **30 cm** between inner edges |
| **Width** | **2–3 cm** tape |
| **Center** | Empty floor between lines — no third line |

---

## Quick start

### 1. Pi — start host (always first)

```bash
cd ~/spike_neuron
git pull
bash yahboom/setup_robot.sh   # first time only

sudo PYTHONPATH=. python3 -m yahboom.host
```

USB webcam default: `/dev/video8`. CSI module: add `--camera-backend picamera2`.

### 2. Mac — train weights (once)

```bash
cd ~/dev/spike_neuron
uv sync
uv run python -m camera.run_training
```

Weights saved to `camera/trained_weights/camera_rewardXXX_*.npy`.

### 3. Mac — debug vision (no driving)

Place the robot centered on the track. Rerun opens with **raw camera** and **debug overlay** (ROI columns + sensor bars).

```bash
uv run python -m camera.debug_vision --remote-ip <PI_IP>
uv run python -m camera.debug_vision --remote-ip <PI_IP> --threshold 45
```

**Terminal output example:**

```
[   0] CENTERED                                  L=0.05 C=0.91 R=0.04  dark L=0.12 C=0.04 R=0.11
[   3] DRIFT LEFT — too close to left line       L=0.62 C=0.38 R=0.00  dark L=0.45 C=0.08 R=0.10
```

**Rerun panels:**

| Path | Content |
|------|---------|
| `camera/raw` | Live JPEG from robot |
| `camera/debug` | ROI overlay + sensor bars + lane state |
| `sensors/left`, `center`, `right` | Filtered lane values |
| `columns/*` | Raw column darkness |
| `debug/lane_state` | Text interpretation |

Purple tape is detected by **color** (default `--tape-color auto`). Black tape also works via `--threshold`.

Tune until centered → **C high**, push left → **L rises**, push right → **R rises**. Debug overlay highlights detected tape in **magenta**.

### 4. Mac — autonomous deploy

```bash
uv run python -m camera.play \
  --remote-ip <PI_IP> \
  --weights camera/trained_weights/camera_reward254.5_20260610_104045.npy \
  --threshold 60
```

**Dry run** (vision + SNN, motors off):

```bash
uv run python -m camera.play --remote-ip <PI_IP> --weights ... --no-drive
```

**Ctrl+C** stops motors. The Pi host keeps running — you can reconnect immediately.

---

## Commands reference

### Pi

```bash
sudo PYTHONPATH=. python3 -m yahboom.host
sudo PYTHONPATH=. python3 -m yahboom.host --camera-device /dev/video8
```

### Mac

```bash
# Train
uv run python -m camera.run_training
uv run python -m camera.run_training --episodes 200

# Debug vision (Rerun, no motors)
uv run python -m camera.debug_vision --remote-ip 192.168.1.170

# Autonomous SNN
uv run python -m camera.play --remote-ip 192.168.1.170 --weights camera/trained_weights/<file>.npy

# Teleop (manual driving while building track)
uv run python -m yahboom.play --remote-ip 192.168.1.170
```

### `camera.play` flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--remote-ip` | *(required)* | Pi IP address |
| `--weights` | latest in `trained_weights/` | SNN weights |
| `--threshold` | `60` | Black-tape darkness threshold |
| `--tape-color` | `auto` | `auto` (purple+black), `purple`, or `dark` |
| `--base-speed` | `35` | Forward speed |
| `--steer-delta` | `18` | Turn strength |
| `--hz` | `15` | Control loop rate |
| `--no-drive` | off | Debug only — don't move motors |
| `--no-rerun` | off | Terminal output only |
| `--no-ping-pong` | off | Disable end-bar 180° turns |
| `--turn-seconds` | `1.7` | Spin duration for ~180° |
| `--turn-speed` | `68` | In-place turn speed |
| `--clear-seconds` | `4.0` | Drive away after turn (end detect off) |
| `--end-lockout-seconds` | `3.0` | Extra ignore time after clearing |
| `--end-zone-fraction` | `0.20` | Bottom of frame for end bar |
| `--base-speed` | `52` | Forward speed |
| `--steer-delta` | `27` | Turn strength |
| `--iterations` | `0` | `0` = until Ctrl+C |

---

## Project layout

```
camera/
├── vision.py           # Frame → lane sensors + debug overlay
├── sim.py              # Training simulator
├── train.py            # SNN training loop
├── run_training.py     # Train on Mac
├── controller.py       # SNN inference
├── remote_driver.py    # ZMQ motor commands (arc steering)
├── play.py             # Remote autonomous deploy (Mac)
├── debug_vision.py     # Remote vision debug (Mac)
├── rerun_viz.py        # Rerun logging
├── motors.py           # (legacy local Pi driver)
├── deploy.py           # Deprecated — prints instructions
├── trained_weights/
└── README.md
```

---

## Tuning on the real robot

1. Run `debug_vision` first — fix `--threshold` until sensors respond correctly
2. Start autonomous with **low speed**: `--base-speed 25 --steer-delta 12`
3. If it zigzags, lower `--steer-delta`
4. If it doesn't turn enough, raise `--steer-delta` slightly
5. If it drove off left but sensors showed centered, threshold/lighting is wrong — fix vision before retraining

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Connection timeout | Is `yahboom.host` running on Pi? Same Wi‑Fi? |
| Rerun doesn't open | `uv sync` on Mac; try `--no-rerun` for terminal-only |
| All sensors ~0 | Use `--tape-color purple`; check magenta overlay hits tape |
| Purple tape, wrong sensors | `git pull` — purple color detection is default now |
| Center always low | Robot not centered; check debug overlay ROI |
| Went off left, sensors said centered | Purple wasn't detected (old darkness-only mode); verify magenta mask on tape |
| Robot spins | Wrong weights; verify sensors change when you push robot |
| `camera.deploy` error | Expected — use `camera.play` from Mac instead |

---

## Steering semantics

| SNN action | Motors |
|------------|--------|
| `0` | Forward (both wheels) |
| `-1` | Arc left (slow left, fast right) |
| `+1` | Arc right |

---

## Optional: teleop while building track

```bash
# Pi
sudo PYTHONPATH=. python3 -m yahboom.host

# Mac
uv run python -m yahboom.play --remote-ip <PI_IP>
```

Then switch to `camera.debug_vision` and `camera.play` for autonomous runs (host stays up).
