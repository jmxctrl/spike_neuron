#!/usr/bin/env bash
# Install system packages needed to run yahboom.host on the Raspberry Pi.
# Uses apt (not uv) so picamera2 + GPIO libs are available to system python3.
set -euo pipefail

echo "Installing Yahboom robot host dependencies (apt)..."
sudo apt update
sudo apt install -y \
  python3-zmq \
  python3-opencv \
  python3-picamera2 \
  python3-lgpio \
  python3-rpi-lgpio \
  python3-smbus \
  v4l-utils

echo ""
echo "Done. Run the host with:"
echo "  cd ~/spike_neuron"
echo "  sudo PYTHONPATH=. python3 -m yahboom.host"
echo "  # CSI cam: add --camera-backend picamera2"
echo ""
echo "Camera SNN (from Mac, after host is running):"
echo "  uv run python -m camera.debug_vision --remote-ip <PI_IP>"
echo "  uv run python -m camera.play --remote-ip <PI_IP> --weights camera/trained_weights/<file>.npy"
