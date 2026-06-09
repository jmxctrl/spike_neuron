# Yahboom Raspbot V1 Control

Python library and keyboard controller for the Yahboom Raspbot V1 robot.

## Requirements

- Raspberry Pi (tested on Pi 5)
- Yahboom Raspbot V1
- I2C enabled (`sudo raspi-config` → Interface Options → I2C)

### System packages (for GPIO):
```bash
sudo apt install python3-lgpio python3-rpi-lgpio python3-smbus
```

## Usage

### Run the keyboard controller:
```bash
cd ~/berrybot
sudo python3 yahboom/play.py
```

> **Note:** `sudo` is required for GPIO access. Using `uv run` won't work because the virtual environment can't access system GPIO libraries.

### Controls

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
| **Sensors** | |
| Space | Read ultrasonic distance |
| T | Read line trackers |
| P | Read IR obstacle sensors |
| **Control** | |
| Q / Esc | Quit |

## Using the Library

```python
from yahboom.raspbot import Raspbot

# Create and initialize robot
robot = Raspbot()
robot.init()

# Movement
robot.forward(100)   # Speed 0-255
robot.backward(100)
robot.turn_left(100)
robot.turn_right(100)
robot.stop()

# Servos
robot.set_pan(90)    # 0-180 degrees
robot.set_tilt(90)
robot.center_servos()

# LEDs
robot.led_red(True)
robot.led_blue(True)
robot.leds_off()

# Buzzer
robot.beep(0.1, 440)  # duration, frequency

# Sensors
distance = robot.read_distance()           # cm
line = robot.read_line_tracker()           # LineTrackerReading
left, right = robot.read_ir_obstacle()     # (bool, bool)

# Cleanup
robot.cleanup()
```

## Hardware Connections

### I2C (address 0x16):
- Motors (register 0x01)
- Servos (register 0x03)

### GPIO (BOARD pin numbers):
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
