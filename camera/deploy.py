#!/usr/bin/env python3
"""
Deprecated: autonomous deploy now runs from the laptop via ZMQ.

Use instead:
  Pi:  sudo PYTHONPATH=. python3 -m yahboom.host
  Mac: uv run python -m camera.play --remote-ip <PI_IP> --weights ...
"""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "camera.deploy is deprecated.\n\n"
        "The Pi should only run the host. Deploy and debug from your Mac:\n\n"
        "  # On Pi\n"
        "  sudo PYTHONPATH=. python3 -m yahboom.host\n\n"
        "  # Debug vision (Rerun + lane sensors, no driving)\n"
        "  uv run python -m camera.debug_vision --remote-ip <PI_IP>\n\n"
        "  # Autonomous SNN\n"
        "  uv run python -m camera.play --remote-ip <PI_IP> "
        "--weights camera/trained_weights/<file>.npy\n",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
