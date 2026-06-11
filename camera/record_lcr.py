#!/usr/bin/env python3
"""
Record left / center / right lane sensors from the live camera stream.

Use the CSV to compare real robot sensors against camera/sim.py and tune noise_std.

Pi:
    sudo PYTHONPATH=. python3 -m yahboom.host

Mac (no driving, log only):
    uv run python -m camera.record_lcr --remote-ip 192.168.1.170

Mac (drive with SNN while logging):
    uv run python -m camera.record_lcr --remote-ip 192.168.1.170 --weights camera/trained_weights/....npy
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

from yahboom.client import RaspbotClient
from yahboom.protocol import DEFAULT_CMD_PORT, DEFAULT_OBS_PORT, RobotCommand

from .controller import CameraSNNController, InferenceConfig
from .remote_driver import YahboomRemoteDriver
from .sensors import add_lane_sensor_args, build_lane_sensor
from .vision import interpret_lane_state


def main() -> None:
    from yahboom.robot_config import DEFAULT_ROBOT_IP

    parser = argparse.ArgumentParser(description="Record L/C/R lane sensors from live camera")
    parser.add_argument("--remote-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT)
    parser.add_argument("--obs-port", type=int, default=DEFAULT_OBS_PORT)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--hz", type=float, default=15.0)
    parser.add_argument("--sensor-window", type=int, default=5)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--tape-color", choices=["auto", "purple", "dark"], default="auto")
    add_lane_sensor_args(parser)
    parser.add_argument("--weights", type=str, default=None, help="Optional: run SNN and log actions")
    parser.add_argument("--action-window", type=int, default=3)
    parser.add_argument("--p-max", type=float, default=0.35)
    parser.add_argument("--lif-tau", type=float, default=10.0)
    parser.add_argument("--lif-threshold", type=float, default=0.8)
    parser.add_argument("--steer-gain", type=float, default=12000.0)
    parser.add_argument("--steer-deadzone", type=float, default=0.08)
    parser.add_argument("--base-speed", type=int, default=52)
    parser.add_argument("--steer-delta", type=int, default=27)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: recordings/lcr_<timestamp>.csv)",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to record (0 = until Ctrl+C)")
    parser.add_argument("--note", type=str, default="", help="Free-text tag stored in CSV header")
    args = parser.parse_args()

    os.makedirs("recordings", exist_ok=True)
    if args.out:
        out_path = args.out
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("recordings", f"lcr_{stamp}.csv")

    client = RaspbotClient(
        remote_ip=args.remote_ip,
        cmd_port=args.cmd_port,
        obs_port=args.obs_port,
        connect_timeout_s=args.connect_timeout_s,
        polling_timeout_ms=50,
    )
    try:
        client.connect()
    except ConnectionError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if args.sensor_backend == "cv" and not args.cv_model:
        print("--sensor-backend cv requires --cv-model path/to/best.pt", file=sys.stderr)
        sys.exit(1)
    sensor = build_lane_sensor(args)
    controller: CameraSNNController | None = None
    driver: YahboomRemoteDriver | None = None

    if args.weights:
        if not os.path.isfile(args.weights):
            print(f"Weights not found: {args.weights}", file=sys.stderr)
            sys.exit(1)
        controller = CameraSNNController(
            args.weights,
            config=InferenceConfig(
                p_max=args.p_max,
                action_window=args.action_window,
                lif_tau=args.lif_tau,
                lif_threshold=args.lif_threshold,
                steer_gain=args.steer_gain,
                steer_deadzone=args.steer_deadzone,
            ),
        )
        driver = YahboomRemoteDriver(client, base_speed=args.base_speed, steer_delta=args.steer_delta)

    period = 1.0 / args.hz
    drive = args.weights is not None
    print(f"Recording to {out_path}")
    print(f"Robot: {args.remote_ip}  hz={args.hz}  sensor_window={args.sensor_window}")
    if drive:
        print(f"Driving with weights: {args.weights}")
    else:
        print("Motors disabled — log only (pass --weights to drive while recording)")
    print("Ctrl+C to stop.\n")

    fieldnames = [
        "t_ms",
        "step",
        "L",
        "C",
        "R",
        "raw_L",
        "raw_C",
        "raw_R",
        "col_L",
        "col_C",
        "col_R",
        "lane_state",
        "action",
    ]

    start = time.perf_counter()
    step = 0

    try:
        with open(out_path, "w", newline="") as f:
            f.write(f"# note={args.note}\n")
            f.write(f"# remote_ip={args.remote_ip}\n")
            f.write(f"# hz={args.hz} sensor_window={args.sensor_window} threshold={args.threshold}\n")
            f.write(f"# tape_color={args.tape_color}\n")
            if args.weights:
                f.write(f"# weights={args.weights}\n")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                loop_start = time.perf_counter()
                client.poll_observation()
                frame = client.last_frame
                sensors = sensor.read_from_frame(frame)
                raw = sensor.last_raw_sensors
                cols = sensor.last_column_darkness
                lane_state = interpret_lane_state(sensors)
                action = ""

                if controller is not None and driver is not None:
                    steer, sensors = controller.run_inference(sensors)
                    driver.execute_steer(steer)
                    action = f"{steer:.3f}"
                else:
                    client.send_command(RobotCommand(movement="stop"))

                elapsed_ms = (time.perf_counter() - start) * 1000.0
                writer.writerow(
                    {
                        "t_ms": f"{elapsed_ms:.1f}",
                        "step": step,
                        "L": f"{sensors[0]:.4f}",
                        "C": f"{sensors[1]:.4f}",
                        "R": f"{sensors[2]:.4f}",
                        "raw_L": f"{raw[0]:.4f}",
                        "raw_C": f"{raw[1]:.4f}",
                        "raw_R": f"{raw[2]:.4f}",
                        "col_L": f"{cols[0]:.4f}",
                        "col_C": f"{cols[1]:.4f}",
                        "col_R": f"{cols[2]:.4f}",
                        "lane_state": lane_state,
                        "action": action,
                    }
                )
                f.flush()

                if step % 15 == 0:
                    print(
                        f"[{step:4d}] L={sensors[0]:.2f} C={sensors[1]:.2f} R={sensors[2]:.2f}  "
                        f"raw L={raw[0]:.2f} R={raw[2]:.2f}  {lane_state}"
                    )

                step += 1
                if args.duration > 0 and (time.perf_counter() - start) >= args.duration:
                    break

                sleep_s = period - (time.perf_counter() - loop_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if driver is not None:
            driver.stop()
        else:
            client.send_command(RobotCommand(movement="stop"))
        client.disconnect()
        print(f"Saved {step} rows to {out_path}")


if __name__ == "__main__":
    main()
