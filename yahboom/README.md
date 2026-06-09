# Yahboom Raspbot V1 — Remote Teleop

Python library and **LeRobot-style remote teleop** for the Yahboom Raspbot V1.

- **Robot (Pi)** runs `yahboom.host` — GPIO, motors, sensors, camera
- **Laptop** runs `yahboom.play` — global keyboard control + Rerun visualization
- Commands and observations travel over **ZMQ** (same pattern as [LeKiwi](https://github.com/huggingface/lerobot/blob/main/docs/source/lekiwi.mdx))

---

## Architecture

```
┌─────────────────────────┐         ZMQ          ┌──────────────────────────┐
│  Laptop (play.py)       │  ── commands :5555 ─► │  Raspberry Pi (host.py)  │
│  • pynput keyboard      │  ◄─ observations :5556│  • Raspbot GPIO/I2C      │
│  • Rerun viewer         │                        │  • picamera2 / V4L2      │
└─────────────────────────┘                        └──────────────────────────┘
```

| Port | Direction | Purpose |
|------|-----------|---------|
| **5555** | Laptop → Pi | JSON commands (movement, servos, LEDs, beep) |
| **5556** | Pi → Laptop | JSON observations (sensors, camera JPEG, state) |

Default control rate: **30 Hz**. The host **watchdog** stops motors if no command arrives within **500 ms**.

---

## Requirements

### Robot (Raspberry Pi)

- Yahboom Raspbot V1
- Raspberry Pi OS with I2C + camera enabled:
  - `sudo raspi-config` → Interface Options → **I2C** → Enable
  - `sudo raspi-config` → Interface Options → **Camera** → Enable (for video)

**Use system `python3` + apt packages on the Pi** (not `uv run`):

| Package | Why |
|---------|-----|
| `python3-zmq` | ZMQ networking |
| `python3-picamera2` | CSI camera module |
| `python3-opencv` | JPEG encode + flip |
| `python3-lgpio` / `python3-rpi-lgpio` | GPIO on Pi 5 |
| `python3-smbus` | I2C motors/servos |
| `v4l-utils` | USB camera debugging |

One-shot setup:

```bash
cd ~/spike_neuron
bash yahboom/setup_robot.sh
```

### Laptop (macOS / Linux)

- Python 3.11+
- Network access to Pi on ports **5555–5556**

```bash
cd ~/spike_neuron
uv sync
```

Dependencies: `pyzmq`, `rerun-sdk`, `pynput`, `opencv-python`, `numpy`

**macOS:** grant your terminal app **Input Monitoring** permission so global keyboard works while Rerun is focused:

> System Settings → Privacy & Security → Input Monitoring → enable Terminal (or iTerm, etc.)

---

## Quick start

### 1. Robot — start the host

```bash
cd ~/spike_neuron
git pull
bash yahboom/setup_robot.sh          # first time only

sudo PYTHONPATH=. python3 -m yahboom.host --camera-backend picamera2
```

Expected log line:

```
INFO: Camera opened via picamera2 (320x240)
INFO: Waiting for commands on ZMQ...
```

### 2. Laptop — teleop + Rerun

```bash
cd ~/spike_neuron
uv sync                               # installs pynput if needed
uv run python -m yahboom.play --remote-ip <PI_IP>
```

Replace `<PI_IP>` with the Pi address (e.g. `192.168.1.100`).

- Rerun opens automatically with live camera + sensor plots
- **Keys work globally** — you can drive while focused on Rerun (hold WASD to move)

---

## Keyboard controls

**Hold** movement keys to drive; **release** to stop (LeKiwi-style).

| Key | Action |
|-----|--------|
| **Movement** | |
| W / ↑ | Forward (hold) |
| S / ↓ | Backward (hold) |
| A / ← | Turn left (hold) |
| D / → | Turn right (hold) |
| **Speed** | |
| + / = | Increase speed |
| - / _ | Decrease speed |
| 1–9 | Set speed level |
| **Camera pan/tilt** | |
| I | Tilt up |
| K | Tilt down |
| J | Pan left |
| L | Pan right |
| O | Center servos |
| **Sound** | |
| B | Short beep |
| H | Horn (long beep) |
| **LEDs** | |
| R | Toggle red LED |
| E | Toggle blue LED |
| X | Turn off both LEDs |
| **Control** | |
| Q / Esc | Quit |

### Keyboard backends

| Backend | Flag | Behavior |
|---------|------|----------|
| **pynput** (default) | `--keyboard-backend pynput` | Global keys — works while Rerun focused |
| terminal | `--keyboard-backend terminal` | Terminal must be focused (old behavior) |

---

## Rerun visualization

Logged every frame under:

| Path | Content |
|------|---------|
| `observation/camera` | Live JPEG from robot (horizontally flipped) |
| `observation/distance_cm` | Ultrasonic distance |
| `observation/line_*` | Line tracker booleans |
| `observation/ir_*` | IR obstacle sensors |
| `observation/pan`, `tilt`, `speed` | Robot state |
| `action/movement` | Current movement command |
| `action/pan`, `tilt`, … | Sent command values |

Disable Rerun: `--no-rerun`

---

## Host options

```bash
sudo PYTHONPATH=. python3 -m yahboom.host --help
```

Common flags:

```bash
# Pi CSI camera (recommended)
sudo PYTHONPATH=. python3 -m yahboom.host --camera-backend picamera2

# USB webcam
v4l2-ctl --list-devices
sudo PYTHONPATH=. python3 -m yahboom.host --camera-device /dev/video0 --camera-backend v4l2

# Tuning
sudo PYTHONPATH=. python3 -m yahboom.host --fps 30 --watchdog-ms 2000
sudo PYTHONPATH=. python3 -m yahboom.host --no-camera
sudo PYTHONPATH=. python3 -m yahboom.host --no-flip-camera   # disable mirror
```

Camera feed is **horizontally flipped by default** (mirror view). Pass `--no-flip-camera` to disable.

---

## Client (play) options

```bash
uv run python -m yahboom.play --remote-ip 192.168.1.100
uv run python -m yahboom.play --remote-ip 192.168.1.100 --fps 30
uv run python -m yahboom.play --remote-ip 192.168.1.100 --no-rerun
uv run python -m yahboom.play --remote-ip 192.168.1.100 --keyboard-backend terminal
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'zmq'` on Pi

You ran with system Python but didn't install apt packages:

```bash
bash yahboom/setup_robot.sh
```

### Camera not found / OpenCV V4L2 warnings

Pi CSI cameras need **picamera2**, not `cv2.VideoCapture(0)`:

```bash
libcamera-hello --list-cameras    # verify camera hardware
sudo apt install -y python3-picamera2
sudo PYTHONPATH=. python3 -m yahboom.host --camera-backend picamera2
```

### `Watchdog timeout; stopping robot`

Normal if the laptop client isn't connected yet. Start `play.py`, or increase timeout:

```bash
sudo PYTHONPATH=. python3 -m yahboom.host --watchdog-ms 5000
```

### Keys don't work while Rerun is focused

1. Use default pynput backend (not `terminal`)
2. On macOS, grant **Input Monitoring** to your terminal app
3. Run `uv sync` to install `pynput`

### Pan/tilt (I/J/K/L) not moving servos

Servo commands are sent on key press. Make sure the host is receiving commands (watchdog shouldn't fire constantly). Press **O** to re-center.

### `uv run` on the Pi

Don't use `uv run` for the **host** — the venv lacks `picamera2`. Use system Python as shown above. `uv run` is fine on the **laptop** for `play.py`.

---

## Sensors on the robot

| Sensor | Count | API | Detects |
|--------|-------|-----|---------|
| Ultrasonic | 1 | `read_distance()` | Distance ahead (cm) |
| Line trackers | 4 | `read_line_tracker()` | Dark line under chassis |
| IR obstacle | 2 | `read_ir_obstacle()` | Close object front-left/right |

Sensor values stream continuously to Rerun under `observation/*`.

---

## Hardware connections

### I2C (address `0x16`)

| Register | Function |
|----------|----------|
| `0x01` | Motors |
| `0x03` | Pan/tilt servos |

### GPIO (BOARD pin numbers)

| Component | Pin |
|-----------|-----|
| Buzzer | 32 |
| LED Red | 40 |
| LED Blue | 38 |
| Ultrasonic Trig | 16 |
| Ultrasonic Echo | 18 |
| Line Tracker L1 | 13 |
| Line Tracker L2 | 15 |
| Line Tracker R1 | 11 |
| Line Tracker R2 | 7 |
| IR Avoid Left | 21 |
| IR Avoid Right | 19 |
| IR Avoid Enable | 22 |

---

## Python API (direct control)

For scripts running **on the Pi** without ZMQ:

```python
from yahboom.raspbot import Raspbot

robot = Raspbot()
robot.init()

robot.forward(100)
robot.turn_left(100)
robot.stop()

robot.set_pan(90)
robot.set_tilt(90)

distance = robot.read_distance()
line = robot.read_line_tracker()
ir_left, ir_right = robot.read_ir_obstacle()

robot.cleanup()
```

---

## Project layout

```
yahboom/
├── host.py           # Robot-side ZMQ server (run on Pi)
├── play.py           # Laptop teleop + Rerun client
├── client.py         # ZMQ client wrapper
├── keyboard.py       # pynput global keyboard (default)
├── camera.py         # picamera2 + V4L2 capture
├── protocol.py       # Command/observation JSON schema
├── rerun_viz.py      # Rerun logging helpers
├── raspbot.py        # Low-level GPIO/I2C driver
├── setup_robot.sh    # Pi apt dependency installer
└── README.md         # This file
```

---

## Why system Python on Pi but uv on laptop?

| | Pi (`host`) | Laptop (`play`) |
|--|-------------|-----------------|
| `picamera2` | apt / system | not needed |
| `pyzmq` | apt (`python3-zmq`) | uv venv |
| `pynput` | not needed | uv venv |
| `gpiozero` / `lgpio` | apt / system | not needed |

This split matches how LeKiwi runs hardware on the robot and teleop on the laptop.
