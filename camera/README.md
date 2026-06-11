# Camera corridor SNN

Drive the Yahboom robot **between two parallel floor lines** using a spiking neural network and a **forward-facing USB camera**.

- **Side lines:** purple painters tape (~30 cm apart, no center line)
- **End bars:** fat **black** tape, perpendicular across the lane (optional ping-pong laps)
- **Pi** runs the host only; **Mac** runs vision, SNN, and Rerun
- **Steering:** continuous `steer ∈ [-1, 1]` → differential wheel speeds via ZMQ

Default robot IP: **`192.168.1.170`** (berry) — configured in `yahboom/robot_config.py`.

**Latest weights (continuous steer):**

```bash
W=camera/trained_weights/camera_reward277.2_20260610_201154.npy
```

---

## Architecture

```
┌─────────────────────────────┐         ZMQ          ┌──────────────────────────┐
│  Mac                        │  ── motor cmds :5555 ►│  Pi (yahboom.host)       │
│  camera.debug_vision        │  ◄─ camera JPEG :5556 │  USB cam /dev/video8     │
│  camera.play                │                       │  Raspbot.control_car()   │
│  camera.record_lcr          │                       │  motors + GPIO           │
└─────────────────────────────┘                       └──────────────────────────┘
```

| Port | Direction | Purpose |
|------|-----------|---------|
| **5555** | Mac → Pi | `RobotCommand(left_speed, right_speed)` or movement |
| **5556** | Pi → Mac | Camera JPEG + sensor state |

| Command | Machine | Purpose |
|---------|---------|---------|
| `yahboom.host` | Pi | Camera + motors (always start first) |
| `camera.run_training` | Mac | Train / fine-tune SNN (sim + optional recordings) |
| `camera.view_sim` | Mac | Visualize training sim (top-down + L/C/R) |
| `camera.record_lcr` | Mac | Log lane sensors while driving (sim-real calibration) |
| `camera.debug_vision` | Mac | Live vision debug, no driving |
| `camera.play` | Mac | Autonomous driving (lane follow; laps optional) |
| `yahboom.play` | Mac | Manual teleop while building track |

### Motor path (continuous steer)

```
SNN → compute_steer() → float in [-1, 1]
  → remote_driver.execute_steer(steer)
  → RobotCommand(left_speed, right_speed)
  → host._apply_movement() → Raspbot.control_car(left, right)
```

`steer > 0` → left wheel faster → arc **right**. Same convention as discrete `{-1, 0, +1}`.

Example at defaults (`base_speed=52`, `steer_delta=27`):

| steer | left wheel | right wheel |
|-------|------------|-------------|
| `0.0` | 52 | 52 |
| `+0.5` | 65 | 39 |
| `+1.0` | 79 | 25 |

---

## How it works

```
        camera view (bottom of frame)
   ┌─────────┬─────────┬─────────┐
   │  LEFT   │ CENTER  │  RIGHT  │  ← 3 columns
   │  tape?  │  floor  │  tape?  │
   └─────────┴─────────┴─────────┘
              ↓
   [left, center, right] in [0, 1]   (median filter, default window=2)
              ↓
   spikes → 100-neuron SNN → compute_steer() → [-1, 1]
              ↓
   differential wheel speeds
```

| Sensor | Meaning |
|--------|---------|
| **left** | High → too close to left line → steer **right** (+) |
| **center** | High → centered in corridor |
| **right** | High → too close to right line → steer **left** (−) |

**SNN readout:** motor pools exclude input neurons 0–2. Lateral pool difference maps to steer via `tanh(gain × (pool_left − pool_right))`.

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

### End bars (optional ping-pong)

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

## Ping-pong laps (opt-in)

**Off by default.** Enable with `--ping-pong`. No separate weights needed.

| Phase | What happens |
|-------|----------------|
| **CRUISE** | SNN continuous lane keeping |
| **TURNING** | Slow spin until **both lane lines visible** (vision-guided, not blind timer) |
| **SETTLING** | Brief full stop so the chassis doesn't slide |
| **SLOW CENTER** | Slow crawl + proportional sensor steering until aligned |
| **CRUISE** | Full speed again |

```bash
uv run python -m camera.play --weights $W --ping-pong
```

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

### 2. Train weights on Mac (no robot needed)

**From scratch:**

```bash
uv run python -m camera.run_training
```

**Fine-tune** from existing weights with recorded sensor logs:

```bash
W=camera/trained_weights/camera_reward277.2_20260610_201154.npy

uv run python -m camera.run_training \
  --init-weights $W \
  --episodes 300 \
  --lr 0.005 \
  --real-mix 0.25
```

| Flag | Purpose |
|------|---------|
| `--init-weights` | Fine-tune from `.npy` instead of random init |
| `--real-mix 0.25` | 25% of sim steps use rows from `recordings/lcr_*.csv` |
| `--recordings-dir recordings` | Where CSV logs live (default) |
| `--no-recordings` | Sim only, ignore CSVs |
| `--episodes` | Training episodes (default 500) |
| `--lr` | Learning rate (default 0.01; use 0.005 for fine-tune) |

Weights save to:

```
camera/trained_weights/camera_rewardXXX_YYYYMMDD_HHMMSS.npy
```

Plot: `camera/trained_weights/training_curve.png`. Good training: best reward **> 250**.

### 3. Visualize the training sim (optional)

```bash
# Sweep position left→right, watch L/C/R bars
uv run python -m camera.view_sim

# SNN driving in sim
uv run python -m camera.view_sim --drive --weights $W

# Same 25% recording mix as training
uv run python -m camera.view_sim --drive --weights $W --real-mix 0.25
```

### 4. Record real sensor data (robot needed)

Autonomous drive + log L/C/R to CSV for sim calibration:

```bash
uv run python -m camera.record_lcr \
  --remote-ip 192.168.1.170 \
  --weights $W \
  --note "autonomous_lap" \
  --duration 120
```

Output: `recordings/lcr_<timestamp>.csv` (filtered L/C/R, raw values, column tape scores, steer).

Log only (motors stopped, push car by hand):

```bash
uv run python -m camera.record_lcr --remote-ip 192.168.1.170 --note "hand_push"
```

Then fine-tune with `--real-mix` (step 2). Training is **offline** — it does not learn while you drive.

### 5. Debug vision on Mac (no driving)

```bash
uv run python -m camera.debug_vision
```

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

### 6. Dry-run autonomous on Mac (motors off)

```bash
uv run python -m camera.play --weights $W --no-drive
```

Check `steer=RIGHT +0.62` etc. before enabling motors.

### 7. Autonomous driving

```bash
uv run python -m camera.play --remote-ip 192.168.1.170 --weights $W
```

**Ctrl+C** stops motors on Mac. Pi host keeps running.

---

## All commands (copy-paste)

### Pi

```bash
sudo PYTHONPATH=. python3 -m yahboom.host
```

### Mac

```bash
W=camera/trained_weights/camera_reward277.2_20260610_201154.npy

# Train / fine-tune
uv run python -m camera.run_training
uv run python -m camera.run_training --init-weights $W --episodes 300 --lr 0.005 --real-mix 0.25

# Sim visualization
uv run python -m camera.view_sim
uv run python -m camera.view_sim --drive --weights $W

# Record L/C/R while driving
uv run python -m camera.record_lcr --weights $W --duration 120

# Vision debug (no motors)
uv run python -m camera.debug_vision

# Autonomous lane follow (default)
uv run python -m camera.play --weights $W

# With ping-pong laps
uv run python -m camera.play --weights $W --ping-pong

# Dry run
uv run python -m camera.play --weights $W --no-drive

# Manual teleop
uv run python -m yahboom.play
```

---

## Tuning guide

### SNN / reaction (defaults tuned for real robot)

| Flag | Default | Effect |
|------|---------|--------|
| `--sensor-window` | `2` | Median filter on L/C/R (lower = faster, noisier) |
| `--hz` | `25` | Control loop rate |
| `--action-window` | `3` | SNN pathway timesteps averaged for steer |
| `--p-max` | `0.35` | Spike encoding strength |
| `--lif-tau` | `10` | Membrane time constant (lower = snappier) |
| `--lif-threshold` | `0.8` | Spike threshold |
| `--steer-gain` | `12000` | Lateral pool → steer scaling |
| `--steer-deadzone` | `0.08` | \|steer\| below this → straight |

Example faster reactions:

```bash
uv run python -m camera.play --weights $W \
  --sensor-window 2 --hz 25 --action-window 3 --steer-gain 15000
```

### Driving speed

Defaults: `--base-speed 52 --steer-delta 27`.

| Symptom | Try |
|---------|-----|
| Too fast / falls off | `--base-speed 35 --steer-delta 18` |
| Zigzag | Lower `--steer-delta` or raise `--steer-deadzone` |
| Won't turn enough | Raise `--steer-delta` or `--steer-gain` |
| Too twitchy | `--steer-deadzone 0.12` |

### 180° turn (`--ping-pong` only)

Vision-guided spin (not a fixed timer). Defaults: slow spin, stop when both lane lines visible.

| Flag | Default | Purpose |
|------|---------|---------|
| `--turn-speed` | `32` | In-place spin speed (lower = less tipping) |
| `--turn-min-seconds` | `1.4` | Minimum spin before vision can stop turn |
| `--turn-max-seconds` | `5.5` | Safety cap if lines never appear |
| `--facing-confirm-frames` | `5` | Consecutive “facing corridor” frames to stop |
| `--turn-settle-seconds` | `0.5` | Pause after spin before creep forward |
| `--recover-center-frames` | `15` | Good frames before full-speed CRUISE |

| Symptom | Try |
|---------|-----|
| Robot tips during spin | `--turn-speed 28` |
| Stops too early | `--turn-min-seconds 1.8 --facing-confirm-frames 7` |
| Spins forever | `--turn-max-seconds 6`; check tape visible after turn |
| Leaves at angle after turn | `--facing-confirm-frames 8 --recover-center-frames 20` |

### Side lines (purple)

Run `debug_vision` first.

| Symptom | Try |
|---------|-----|
| All sensors ~0 | `--tape-color purple` or lower `--threshold` |
| Long stretches `C=0.50 L=0 R=0` | Lost line of sight — lighting, speed, or threshold |
| Centered but C low | Robot not centered; check ROI in debug overlay |

### End bar (black)

| Symptom | Try |
|---------|-----|
| End bar not detected | `--end-threshold 65` |
| Shadows trigger early | `--end-threshold 85` |

---

## `camera.play` flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--remote-ip` | `192.168.1.170` | Pi address |
| `--weights` | latest in `trained_weights/` | SNN `.npy` |
| `--hz` | `25` | Control loop rate |
| `--sensor-window` | `2` | L/C/R median filter |
| `--action-window` | `3` | SNN pathway average window |
| `--p-max` | `0.35` | Spike encoding |
| `--lif-tau` | `10` | LIF time constant |
| `--lif-threshold` | `0.8` | LIF threshold |
| `--steer-gain` | `12000` | Continuous steer gain |
| `--steer-deadzone` | `0.08` | Straight deadband |
| `--tape-color` | `auto` | Side lines: purple + black |
| `--threshold` | `60` | Side line threshold |
| `--base-speed` | `52` | Cruise forward speed |
| `--steer-delta` | `27` | Max wheel differential at steer=±1 |
| `--ping-pong` | off | Enable end-line lap mode |
| `--turn-speed` | `32` | Ping-pong spin speed |
| `--turn-min-seconds` | `1.4` | Min spin time |
| `--turn-max-seconds` | `5.5` | Max spin time |
| `--facing-confirm-frames` | `5` | Frames to confirm facing corridor |
| `--turn-settle-seconds` | `0.5` | Pause after spin |
| `--recover-speed-scale` | `0.35` | SLOW CENTER speed |
| `--recover-center-frames` | `15` | Frames before full speed |
| `--end-lockout-seconds` | `4.0` | Ignore end bar after lap |
| `--end-confirm-frames` | `6` | End bar confirm frames |
| `--end-tape-color` | `dark` | End bar: black only |
| `--end-threshold` | `75` | End bar darkness |
| `--end-zone-fraction` | `0.20` | Bottom ROI for end bar |
| `--no-drive` | off | Debug only |
| `--no-rerun` | off | Terminal only |
| `--iterations` | `0` | `0` = until Ctrl+C |

---

## Steering semantics

| steer | Meaning | Wheels (defaults) |
|-------|---------|-------------------|
| `0.0` | Straight | `(52, 52)` |
| `+0.5` | Moderate right | `(65, 39)` |
| `+1.0` | Full right | `(79, 25)` |
| `−0.5` | Moderate left | `(39, 65)` |
| `−1.0` | Full left | `(25, 79)` |

During **SLOW CENTER** (ping-pong), proportional sensor steering replaces the SNN until aligned.

---

## Sim ↔ real calibration

The training sim (`camera/sim.py`) models the same L/C/R pipeline as `vision.py`, with:

- Higher center readings when in-lane
- L/R saturation (values peg near 0 or 1 like the real camera)
- Optional injection of `recordings/lcr_*.csv` rows during training (`--real-mix`)

Workflow:

1. `camera.record_lcr` while driving → CSVs in `recordings/`
2. `camera.view_sim --real-mix 0.25` → compare sim vs logs
3. `camera.run_training --init-weights $W --real-mix 0.25` → fine-tune offline

---

## Project layout

```
camera/
├── vision.py           # Tape → lane sensors, end bar, centering helpers
├── sim.py              # Training simulator (calibrated to recordings)
├── lcr_data.py         # Load recordings/lcr_*.csv for training mix
├── train.py            # SNN training loop (continuous steer)
├── run_training.py     # CLI: train / fine-tune on Mac
├── controller.py       # SNN inference → compute_steer()
├── lap_controller.py   # Optional ping-pong laps
├── remote_driver.py    # ZMQ → differential wheel speeds
├── play.py             # Autonomous deploy (Mac)
├── record_lcr.py       # Log L/C/R while driving
├── view_sim.py         # Visualize training sim
├── debug_vision.py     # Vision debug (Mac)
├── rerun_viz.py        # Rerun logging
├── trained_weights/    # Saved .npy weights
└── README.md

recordings/             # lcr_*.csv logs from record_lcr (gitignored optional)
```

Legacy (deprecated):

- `deploy.py` — use `camera.play` from Mac
- `test_vision.py` — use `camera.debug_vision`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection timeout` | Is `yahboom.host` running? Same Wi‑Fi? IP `192.168.1.170`? |
| `ModuleNotFoundError` on Pi | Don't run SNN on Pi — use `camera.play` on Mac |
| `No weights in trained_weights` | Run `camera.run_training` |
| Drift left but no right steer | Retrain / fine-tune; check `steer` in `--no-drive` log |
| `steer=0` while L=1 | Raise `--steer-gain`; fine-tune with `record_lcr` data |
| Long `C=0.50 L=0 R=0` | Default when no tape visible — slow down or fix lighting |
| Robot tips on ping-pong spin | `--turn-speed 28`; ensure `--ping-pong` only when needed |
| Turn leaves at wrong angle | `--facing-confirm-frames 8`; more `recover-center-frames` |
| `view_sim` blank window | Update repo; first frame draws immediately |
| Training doesn't help | Use `--init-weights $W` + `--real-mix 0.25`, not from scratch only |

---

## Optional: teleop while building track

```bash
# Pi
sudo PYTHONPATH=. python3 -m yahboom.host

# Mac
uv run python -m yahboom.play
```

---

## Related docs

- Yahboom teleop + host: [`yahboom/README.md`](../yahboom/README.md)
- Pi setup: [`yahboom/setup_robot.sh`](../yahboom/setup_robot.sh)
