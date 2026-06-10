# Camera corridor SNN

Drive the Yahboom robot **between two parallel floor lines** using a spiking neural network and a **forward-facing USB camera** (`/dev/video8`).

No center line is required — the camera sees the left and right tape in the lower image; the SNN steers to stay centered.

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

Training uses a **camera simulator** (`sim.py`) that matches this logic. Deployment uses the real USB camera (`vision.py`).

---

## Floor markings (30 cm apart)

**Two solid parallel lines are enough.** You do **not** need a center line.

### Recommended layout

```
══════════════════════════════════════════  ← start / optional end bar
        ║                          ║
        ║    30 cm corridor        ║
        ║         ↑ robot          ║
        ║                          ║
══════════════════════════════════════════
```

| Item | Recommendation |
|------|----------------|
| **Lines** | 2 parallel lines only (left boundary + right boundary) |
| **Spacing** | **30 cm** between inner edges is fine for Yahboom (~12–15 cm wide → ~7–8 cm clearance per side) |
| **Material** | Black electrical tape on **white/light** surface (poster board, white mat, light floor) |
| **Width** | **2–3 cm** tape width — visible in camera, not so wide the corridor feels narrow |
| **Length** | At least **2–3 m** for testing; longer for stable driving |
| **Center** | Leave **empty floor** between the lines — no third line |

### Tips

- Matte white base works better than glossy or dark gray floors.
- Avoid strong shadows across the track; diffuse room lighting helps.
- Tape should be flat (no wrinkles) at least in the camera’s view ahead of the robot.
- Optional: short perpendicular tape at the **start** so you can place the robot centered by hand.

### What does *not* work well

- Single center line only (SNN expects left + right boundaries).
- Same-color tape and floor (low contrast).
- Lines much narrower than ~1 cm (hard to see at 320×240).

---

## Project layout

```
camera/
├── vision.py          # Camera → 3 lane sensors
├── sim.py             # Training-time camera simulator
├── train.py           # SNN training loop
├── run_training.py    # Train on laptop
├── controller.py      # SNN inference
├── motors.py          # Yahboom forward + arc steering
├── deploy.py          # Run on Pi
├── test_vision.py     # Debug sensor values on Pi
├── trained_weights/   # Saved .npy weights
└── README.md
```

---

## Part 1 — Train (laptop)

Training is **simulation only** (no robot needed). Takes a few minutes.

```bash
cd ~/spike_neuron
uv sync

# Train (default 500 episodes)
uv run python -m camera.run_training

# Faster test run
uv run python -m camera.run_training --episodes 200
```

Weights are saved to:

```
camera/trained_weights/camera_rewardXXX_YYYYMMDD_HHMMSS.npy
```

Note the path of the best file — you’ll copy it to the Pi.

**What good training looks like:** reward climbing over episodes; best reward often **> 150** (depends on episode length). A learning curve plot opens at the end.

---

## Part 2 — Prepare the Pi

Same setup as Yahboom teleop:

```bash
cd ~/spike_neuron
git pull
bash yahboom/setup_robot.sh   # once: python3-zmq, opencv, GPIO, etc.
```

Confirm USB camera:

```bash
v4l2-ctl --list-devices
# GENERAL WEBCAM → /dev/video8
```

---

## Part 3 — Calibrate vision (Pi)

Lay your tape corridor. Place the robot **centered** between the lines, camera facing along the track.

```bash
sudo PYTHONPATH=. python3 -m camera.test_vision
```

**Expected readings when centered:**

```
L ≈ 0.2–0.5   C ≈ 0.6–1.0   R ≈ 0.2–0.5
```

Push robot toward **left line** → **L** should rise.  
Push toward **right line** → **R** should rise.

If values are wrong, tune threshold:

```bash
sudo PYTHONPATH=. python3 -m camera.test_vision --threshold 45
# or --threshold 75
```

Use the same `--threshold` in `deploy.py`.

---

## Part 4 — Deploy SNN (Pi)

Copy trained weights to the Pi (if trained on laptop):

```bash
# from laptop
scp camera/trained_weights/camera_reward*.npy berry@192.168.1.170:~/spike_neuron/camera/trained_weights/
```

Run autonomous driving:

```bash
cd ~/spike_neuron
sudo PYTHONPATH=. python3 -m camera.deploy \
  --weights camera/trained_weights/camera_rewardXXX.npy \
  --threshold 60 \
  --base-speed 60 \
  --steer-delta 30
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--weights` | latest in `trained_weights/` | Trained SNN weights |
| `--camera-device` | `/dev/video8` | USB webcam |
| `--threshold` | `60` | Tape detection (match `test_vision`) |
| `--base-speed` | `35` | Forward speed (lower = easier) |
| `--steer-delta` | `18` | Turn strength |
| `--hz` | `15` | Control loop rate |
| `--iterations` | `0` | `0` = until Ctrl+C |

**Ctrl+C** stops motors.

### Tuning on the real robot

1. Start with **low speed**: `--base-speed 25 --steer-delta 12`
2. If it oscillates (zigzag), lower `--steer-delta` or train longer
3. If it doesn’t turn enough, raise `--steer-delta` slightly
4. If sensors look noisy, raise threshold or improve lighting/tape contrast
5. If behavior is poor after good `test_vision`, **retrain** — sim may need more episodes

---

## Part 5 — Full workflow checklist

- [ ] Tape two parallel black lines ~**30 cm** apart on a light surface
- [ ] `uv run python -m camera.run_training` on laptop
- [ ] `scp` weights to Pi
- [ ] `sudo PYTHONPATH=. python3 -m camera.test_vision` — sensors respond correctly
- [ ] `sudo PYTHONPATH=. python3 -m camera.deploy --weights ...` — robot follows corridor
- [ ] Tune `--threshold`, `--base-speed`, `--steer-delta` as needed

---

## Steering semantics

| SNN action | Motors |
|------------|--------|
| `0` | Forward |
| `-1` | Forward + arc left |
| `+1` | Forward + arc right |

The network learns from reward to **increase center sensor** and avoid crashing past the corridor edges (simulated as `|position| > 1`).

---

## Retraining after tape changes

If you change line spacing, color, or lighting significantly, **retrain** (`run_training`) and optionally adjust `simulate_column_darkness()` in `sim.py` if your corridor width differs a lot from the default sim.

For a **different physical spacing** (e.g. 20 cm vs 40 cm), the sim still uses normalized position `[-1, 1]` between lines — 30 cm is a good real-world scale; the network learns relative centering, not metric distance.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No weights in trained_weights` | Run `camera.run_training` first |
| Camera won't open | Check `v4l2-ctl --list-devices`, use `--camera-device /dev/video8` |
| All sensors ~0 | Lower threshold; improve tape contrast |
| Center always low | Robot not centered; check tape visible in camera view |
| Robot spins in place | Wrong weights or use `motors.py` arc mode (already default in `deploy`) |
| Ignores lines | Retrain; verify `test_vision` shows L/R changing when pushed |

---

## Optional: teleop while debugging

Use Yahboom teleop to verify motors and camera while building the track:

```bash
# Pi
sudo PYTHONPATH=. python3 -m yahboom.host

# Laptop
uv run python -m yahboom.play --remote-ip <PI_IP>
```

Then switch to `camera.test_vision` and `camera.deploy` for autonomous runs.
