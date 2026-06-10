# Camera corridor SNN

Drive the Yahboom robot **between two parallel floor lines** using a spiking neural network and a **forward-facing USB camera**.

- **Side lines:** purple painters tape (~30 cm apart, no center line)
- **End bars:** fat **black** tape, perpendicular across the lane (ping-pong turns)
- **Pi** runs the host only; **Mac** runs vision, SNN, and Rerun

Default robot IP: **`192.168.1.170`** (berry) — configured in `yahboom/robot_config.py`.

---

## Architecture

```
┌─────────────────────────────┐         ZMQ          ┌──────────────────────────┐
│  Mac                        │  ── motor cmds :5555 ►│  Pi (yahboom.host)       │
│  camera.debug_vision        │  ◄─ camera JPEG :5556 │  USB cam /dev/video8     │
│  camera.play                │                       │  motors + GPIO           │
└─────────────────────────────┘                       └──────────────────────────┘
```

| Port | Direction | Purpose |
|------|-----------|---------|
| **5555** | Mac → Pi | Motor / turn commands |
| **5556** | Pi → Mac | Camera JPEG + sensor state |

| Command | Machine | Purpose |
|---------|---------|---------|
| `yahboom.host` | Pi | Camera + motors (always start first) |
| `camera.run_training` | Mac | Train SNN weights (simulation) |
| `camera.debug_vision` | Mac | Live vision debug, no driving |
| `camera.play` | Mac | Autonomous driving + ping-pong laps |
| `yahboom.play` | Mac | Manual teleop while building track |

---

## How it works

```
        camera view (bottom of frame)
   ┌─────────┬─────────┬─────────┐
   │  LEFT   │ CENTER  │  RIGHT  │  ← 3 columns
   │  tape?  │  floor  │  tape?  │
   └─────────┴─────────┴─────────┘
              ↓
   [left, center, right] in [0, 1]
              ↓
   spikes → 100-neuron SNN → steer left / forward / right
```

| Sensor | Meaning |
|--------|---------|
| **left** | High → too close to left line |
| **center** | High → centered in corridor |
| **right** | High → too close to right line |

Side lines use **purple color detection** (`--tape-color auto`). Debug overlay highlights tape in **magenta**.

---

## Floor layout

### Side lines (lane keeping)

| Item | Recommendation |
|------|----------------|
| **Tape** | Purple painters tape on light wood / matte white |
| **Count** | 2 parallel lines only — **no center line** |
| **Spacing** | ~**30 cm** between inner edges |
| **Width** | 2–3 cm |

### End bars (ping-pong turns)

| Item | Recommendation |
|------|----------------|
| **Tape** | Fat **black** tape, perpendicular across full lane width |
| **Detection** | Bottom **20%** of camera frame (`--end-zone-fraction 0.20`) |
| **Separate from side lines** | End detection uses **black only** — won't trigger on purple |

```
██████████████████████████████████  ← fat black end bar
        ║                      ║
        ║   purple side lines  ║
        ║                      ║
██████████████████████████████████  ← other end
```

---

## Ping-pong lap behavior (default in `camera.play`)

**No retrain needed** for laps. After a 180° turn the robot faces back down the corridor — same SNN.

| Phase | What happens |
|-------|----------------|
| **CRUISE** | SNN lane keeping at full speed |
| **TURNING** | Fixed ~180° spin when black end bar detected |
| **SLOW CENTER** | Slow forward crawl + sensor steering until centered |
| **CRUISE** | Full speed again (repeat forever) |

Disable laps: `--no-ping-pong`

---

## Setup

### Pi (first time)

```bash
ssh berry@192.168.1.170

cd ~/spike_neuron
git pull
bash yahboom/setup_robot.sh   # apt: zmq, opencv, picamera2, GPIO
```

Confirm USB camera:

```bash
v4l2-ctl --list-devices
# GENERAL WEBCAM → /dev/video8
```

**Use system `python3` on the Pi** (not `uv run`) — picamera2/GPIO need apt packages.

### Mac (first time)

```bash
cd ~/dev/spike_neuron
git pull
uv sync
```

---

## Step-by-step workflow

### 1. Start host on Pi (leave running)

```bash
cd ~/spike_neuron
sudo PYTHONPATH=. python3 -m yahboom.host
```

CSI camera instead of USB:

```bash
sudo PYTHONPATH=. python3 -m yahboom.host --camera-backend picamera2
```

Expected: `Waiting for commands on ZMQ...`

### 2. Train weights on Mac (once, no robot needed)

```bash
cd ~/dev/spike_neuron
uv run python -m camera.run_training
```

Weights save to:

```
camera/trained_weights/camera_rewardXXX_YYYYMMDD_HHMMSS.npy
```

Good training: best reward **> 150**. Plot saved to `camera/trained_weights/training_curve.png`.

Faster test:

```bash
uv run python -m camera.run_training --episodes 200
```

### 3. Debug vision on Mac (no driving)

Place robot centered between purple lines. Rerun opens automatically.

```bash
uv run python -m camera.debug_vision
```

Default IP is `192.168.1.170` — no `--remote-ip` needed unless it changed.

**What to verify:**

- Centered → `CENTERED`, **C high**, L and R low
- Push toward left line → **L rises**
- Push toward right line → **R rises**
- Magenta overlay sits on purple tape

Tune side-line sensitivity:

```bash
uv run python -m camera.debug_vision --threshold 45
uv run python -m camera.debug_vision --tape-color purple
```

**Rerun panels:**

| Path | Content |
|------|---------|
| `camera/raw` | Live camera |
| `camera/debug` | ROI columns, sensor bars, lane state |
| `sensors/*` | Filtered L / C / R values |
| `columns/*` | Raw column tape scores |

### 4. Dry-run autonomous on Mac (motors off)

```bash
uv run python -m camera.play \
  --weights camera/trained_weights/camera_reward254.5_20260610_104045.npy \
  --no-drive
```

Check actions and sensors look sensible before enabling motors.

### 5. Autonomous driving with ping-pong laps

```bash
uv run python -m camera.play \
  --weights camera/trained_weights/camera_reward254.5_20260610_104045.npy
```

**Ctrl+C** stops motors on Mac. Pi host keeps running — reconnect anytime.

---

## All commands (copy-paste)

### Pi

```bash
# Host (always first)
sudo PYTHONPATH=. python3 -m yahboom.host

# USB webcam explicitly
sudo PYTHONPATH=. python3 -m yahboom.host --camera-device /dev/video8
```

### Mac

```bash
# Train
uv run python -m camera.run_training

# Vision debug (no motors)
uv run python -m camera.debug_vision

# Autonomous + ping-pong laps
uv run python -m camera.play \
  --weights camera/trained_weights/camera_reward254.5_20260610_104045.npy

# Dry run (no motors)
uv run python -m camera.play \
  --weights camera/trained_weights/camera_reward254.5_20260610_104045.npy \
  --no-drive

# Manual teleop (build / test track)
uv run python -m yahboom.play
```

### Copy weights Mac → Pi (if needed)

```bash
scp camera/trained_weights/camera_reward*.npy \
  berry@192.168.1.170:~/spike_neuron/camera/trained_weights/
```

---

## Tuning guide

### Side lines (purple)

Run `debug_vision` first. Fix vision before driving.

| Symptom | Try |
|---------|-----|
| All sensors ~0 | `--tape-color purple` or lower `--threshold` |
| Magenta misses purple tape | `--tape-color purple`; improve lighting |
| Centered but C low | Robot not centered; check ROI in debug overlay |

### Driving speed

Defaults: `--base-speed 52 --steer-delta 27` (50% slower than original).

| Symptom | Try |
|---------|-----|
| Too fast / falls off | `--base-speed 35 --steer-delta 18` |
| Zigzag | Lower `--steer-delta` |
| Won't turn enough | Raise `--steer-delta` slightly |

### 180° turn

Default `--turn-seconds 3.0` (~180° on Yahboom). **2.0s ≈ 120° only.**

| Symptom | Try |
|---------|-----|
| Turn too short | `--turn-seconds 3.2` or `3.5` |
| Turn too long | `--turn-seconds 2.7` |
| Spin too weak | `--turn-speed 55` |

Rule of thumb: `turn_seconds ≈ 3.0 × (degrees / 180)`.

### After-turn centering (SLOW CENTER phase)

Terminal shows `SLOW CENTER 5/12` — counts centered frames before full speed.

| Symptom | Try |
|---------|-----|
| Stuck drifted left (L=1, action=RIGHT) | `--recover-speed-scale 0.45` |
| Exits before truly centered | `--recover-center-frames 20` |
| Takes too long | `--recover-center-frames 8` |

### End bar (black)

Debug overlay: **cyan box** at bottom = end zone. **`END LINE`** when triggered.

| Symptom | Try |
|---------|-----|
| End bar not detected | `--end-threshold 65` (more sensitive) |
| Shadows trigger early | `--end-threshold 85` |
| Purple side line triggers turn | Keep `--end-tape-color dark` (default) |

### End bar lockout after turn

Default `--end-lockout-seconds 4.0` ignores end bar briefly after SLOW CENTER completes (prevents double-turn).

---

## `camera.play` flags (defaults)

| Flag | Default | Purpose |
|------|---------|---------|
| `--remote-ip` | `192.168.1.170` | Pi address |
| `--weights` | latest in `trained_weights/` | SNN `.npy` file |
| `--tape-color` | `auto` | Side lines: purple + black |
| `--threshold` | `60` | Side line threshold |
| `--end-tape-color` | `dark` | End bar: black only |
| `--end-threshold` | `75` | End bar darkness |
| `--end-zone-fraction` | `0.20` | Bottom of frame for end detect |
| `--base-speed` | `52` | Cruise forward speed |
| `--steer-delta` | `27` | Cruise turn strength |
| `--hz` | `15` | Control loop rate |
| `--turn-seconds` | `3.0` | Spin duration (~180°) |
| `--turn-speed` | `45` | Spin speed |
| `--recover-speed-scale` | `0.35` | SLOW CENTER speed fraction |
| `--recover-center-frames` | `12` | Centered frames before full speed |
| `--end-lockout-seconds` | `4.0` | Ignore end bar after lap |
| `--end-confirm-frames` | `6` | Frames end bar must be visible |
| `--no-ping-pong` | off | Single direction, no turns |
| `--no-drive` | off | Debug only, no motors |
| `--no-rerun` | off | Terminal output only |
| `--iterations` | `0` | `0` = until Ctrl+C |

---

## Steering semantics

| SNN / sensor action | Motors |
|---------------------|--------|
| `0` (straight) | Forward, both wheels equal |
| `-1` (left) | Forward + arc left |
| `+1` (right) | Forward + arc right |

During **SLOW CENTER**, sensor steering replaces the SNN until the robot is centered.

---

## Project layout

```
camera/
├── vision.py           # Purple/black tape → lane sensors + end bar detect
├── sim.py              # Training simulator
├── train.py            # SNN training loop
├── run_training.py     # CLI: train on Mac
├── controller.py       # SNN inference
├── lap_controller.py   # Ping-pong: turn → slow center → cruise
├── remote_driver.py    # ZMQ motor commands
├── play.py             # Autonomous deploy (Mac)
├── debug_vision.py     # Vision debug (Mac)
├── rerun_viz.py        # Rerun logging
├── trained_weights/    # Saved .npy weights
└── README.md
```

Legacy (deprecated — print instructions only):

- `deploy.py` — use `camera.play` from Mac instead
- `test_vision.py` — use `camera.debug_vision` from Mac instead

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection timeout` | Is `yahboom.host` running on Pi? Same Wi‑Fi? IP `192.168.1.170`? |
| `ModuleNotFoundError: matplotlib` on Pi | Don't run SNN on Pi — use `camera.play` on Mac |
| `No weights in trained_weights` | Run `camera.run_training` on Mac |
| Rerun doesn't open | `uv sync` on Mac; or `--no-rerun` |
| Sensors wrong after turn | Normal briefly — wait for `SLOW CENTER` to finish |
| Double turn at end bar | `git pull`; lockout + SLOW CENTER should prevent this |
| Turn only ~120° | Use `--turn-seconds 3.0` (not 2.0) |
| Stuck in SLOW CENTER forever | `--recover-speed-scale 0.45`; check purple lines visible |
| End bar not detected | `--end-threshold 65` |
| End bar false triggers | `--end-threshold 85`; use fat black tape only |
| `camera.deploy` on Pi | Deprecated — run `camera.play` on Mac |

---

## Optional: teleop while building track

```bash
# Pi
sudo PYTHONPATH=. python3 -m yahboom.host

# Mac — WASD drive, Rerun camera view
uv run python -m yahboom.play
```

Host stays up when you switch to `debug_vision` or `play`.

---

## Retraining

Retrain if you change line spacing, tape color, or lighting significantly:

```bash
uv run python -m camera.run_training
```

The sim uses normalized position between lines — physical spacing (~30 cm) maps to training semantics, not exact metric distance.

---

## Related docs

- Yahboom teleop + host details: [`yahboom/README.md`](../yahboom/README.md)
- Pi apt setup script: [`yahboom/setup_robot.sh`](../yahboom/setup_robot.sh)
