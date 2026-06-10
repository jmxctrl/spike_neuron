#!/usr/bin/env python3
"""
Deprecated on Pi — use remote vision debug from your laptop instead.

  Pi:  sudo PYTHONPATH=. python3 -m yahboom.host
  Mac: uv run python -m camera.debug_vision --remote-ip <PI_IP>
"""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "camera.test_vision on the Pi is deprecated.\n\n"
        "Start the host on the Pi, then debug from your Mac:\n\n"
        "  sudo PYTHONPATH=. python3 -m yahboom.host\n"
        "  uv run python -m camera.debug_vision --remote-ip <PI_IP>\n",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
