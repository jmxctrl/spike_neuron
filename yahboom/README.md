# Yahboom Raspbot V1 Control

Python library and remote teleop for the Yahboom Raspbot V1 robot, with a
LeRobot-style host/client split over ZMQ.

## Architecture

| Machine | Script | Role |
|---------|--------|------|
| Raspberry Pi (robot) | `python -m yahboom.host` | GPIO, motors, sensors, camera → streams observations |
| Laptop (main computer) | `python -m yahboom.play --remote-ip <PI_IP>` | Keyboard control + Rerun visualization |

ZMQ ports (defaults):

- **5555** — commands (laptop PUSH → robot PULL)
- **5556** — observations (robot PUSH → laptop PULL)

## Requirements

### Robot (Raspberry Pi)

- Raspberry Pi with Yahboom Raspbot V1
- I2C enabled (`sudo raspi-config` → Interface Options → I2C)

```bash
sudo apt install python3-lgpio python3-rpi-lgpio python3-smbus
```

### Laptop

- Python 3.11+
- Network access to the Pi on ports 5555–5556

```bash
uv sync
# or: pip install pyzmq rerun-sdk opencv-python numpy
```

On macOS, grant Terminal **Input Monitoring** permission for keyboard teleop
(System Settings → Privacy & Security → Input Monitoring).

## Quick start

**1. On the robot (SSH into the Pi):**

```bash
cd ~/spike_neuron
sudo uv run python -m yahboom.host
```

> `sudo` is required for GPIO access on the Pi.

**2. On your laptop:**

```bash
cd ~/spike_neuron
uv run python -m yahboom.play --remote-ip 192.168.1.100
```

Replace `192.168.1.100` with your Pi's IP address. A Rerun viewer window opens
with the camera feed, sensor readings, and current action.

## Controls (laptop keyboard)

| Key | Action |
|-----|--------|
| **Movement** | |
| W / ↑ | Forward |
| S / ↓ | Backward |
| A / ← | Turn left |
| D / → | Turn right |
| **Speed** | |
| + / = | Increase speed |
| - / _ | Decrease speed |
| 1-9 | Set speed level |
| **Camera (Pan/Tilt)** | |
| I | Tilt up |
| K | Tilt down |
| J | Pan left |
| L | Pan right |
| O | Center servos |
| **Sound** | |
| B | Beep |
| H | Horn |
| **LEDs** | |
| R | Toggle red LED |
| E | Toggle blue LED |
| X | Turn off LEDs |
| **Control** | |
| Q / Esc | Quit |

Sensor readings (distance, line trackers, IR) stream continuously to Rerun
under `observation/*`.

## Host options

```bash
sudo python -m yahboom.host --help

# Common flags:
sudo python -m yahboom.host --fps 30
sudo python -m yahboom.host --watchdog-ms 500
sudo python -m yahboom.host --no-camera
sudo python -m yahboom.host --duration-s 3600

# Pi CSI camera (Camera Module):
sudo apt install -y python3-picamera2
sudo python -m yahboom.host --camera-backend picamera2

# USB camera — find device with v4l2-ctl --list-devices:
sudo python -m yahboom.host --camera-device /dev/video0 --camera-backend v4l2
```

The host stops motors automatically if no command arrives within the watchdog
timeout (default 500 ms).

## Client options

```bash
python -m yahboom.play --remote-ip 192.168.1.100 --fps 30
python -m yahboom.play --remote-ip 192.168.1.100 --no-rerun
```

## Using the library directly

```python
from yahboom.raspbot import Raspbot

robot = Raspbot()
robot.init()

robot.forward(100)
robot.stop()
robot.cleanup()
```

## Hardware connections

### I2C (address 0x16)

- Motors (register 0x01)
- Servos (register 0x03)

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
