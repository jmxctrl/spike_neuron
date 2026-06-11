#!/usr/bin/env python3
"""
Remote camera SNN deploy (runs on your laptop).

The Pi only runs yahboom.host. This script receives the camera stream, computes
lane sensors, runs SNN inference, and sends motor commands back over ZMQ.

Pi:
    sudo PYTHONPATH=. python3 -m yahboom.host

Mac:
    uv run python -m camera.play --remote-ip <PI_IP> --weights camera/trained_weights/....npy
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from yahboom.client import RaspbotClient
from yahboom.protocol import DEFAULT_CMD_PORT, DEFAULT_FPS, DEFAULT_OBS_PORT, RobotCommand

from .controller import ACTION_LEFT, ACTION_RIGHT, ACTION_STRAIGHT, CameraSNNController, InferenceConfig
from .lap_controller import LapController
from .remote_driver import YahboomRemoteDriver
from .rerun_viz import init_rerun, log_camera_snn
from .train import WEIGHTS_DIR, list_weights
from .vision import FrameLaneSensor, annotate_debug_frame, detect_end_line, interpret_lane_state

logger = logging.getLogger(__name__)

ACTION_NAMES = {ACTION_LEFT: "LEFT", ACTION_STRAIGHT: "FWD", ACTION_RIGHT: "RIGHT"}


def _default_weights() -> str:
    files = list_weights()
    if not files:
        raise FileNotFoundError(
            f"No weights in {WEIGHTS_DIR}. Train first: uv run python -m camera.run_training"
        )
    return os.path.join(WEIGHTS_DIR, sorted(files)[-1])


def precise_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote camera SNN deploy with Rerun debug")
    from yahboom.robot_config import DEFAULT_ROBOT_IP

    parser.add_argument(
        "--remote-ip",
        default=DEFAULT_ROBOT_IP,
        help=f"Raspberry Pi IP address (default: {DEFAULT_ROBOT_IP})",
    )
    parser.add_argument("--weights", type=str, default=None, help="Path to .npy weights")
    parser.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT)
    parser.add_argument("--obs-port", type=int, default=DEFAULT_OBS_PORT)
    parser.add_argument("--hz", type=float, default=15.0, help="Control loop rate")
    parser.add_argument(
        "--sensor-window",
        type=int,
        default=5,
        help="Median filter window for lane sensors (lower = faster reaction)",
    )
    parser.add_argument(
        "--action-window",
        type=int,
        default=10,
        help="SNN pathway timesteps averaged for action (lower = faster reaction)",
    )
    parser.add_argument(
        "--p-max",
        type=float,
        default=0.2,
        help="Max spike probability per sensor timestep (higher = stronger input)",
    )
    parser.add_argument(
        "--lif-tau",
        type=float,
        default=20.0,
        help="LIF membrane time constant (lower = faster neuron response)",
    )
    parser.add_argument(
        "--lif-threshold",
        type=float,
        default=1.0,
        help="LIF spike threshold (lower = easier firing)",
    )
    parser.add_argument("--threshold", type=int, default=60, help="Lane tape threshold (purple side lines)")
    parser.add_argument(
        "--tape-color",
        choices=["auto", "purple", "dark"],
        default="auto",
        help="Side lane lines: auto=purple+black (default)",
    )
    parser.add_argument(
        "--end-threshold",
        type=int,
        default=75,
        help="End bar threshold — fat black line (grayscale, default 75)",
    )
    parser.add_argument(
        "--end-tape-color",
        choices=["dark", "auto", "purple"],
        default="dark",
        help="End bar detection: dark=black only (default)",
    )
    parser.add_argument("--base-speed", type=int, default=52)
    parser.add_argument("--steer-delta", type=int, default=27)
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--no-rerun", action="store_true", help="Disable Rerun viewer")
    parser.add_argument("--no-drive", action="store_true", help="Debug only: do not send motor commands")
    parser.add_argument("--no-ping-pong", action="store_true", help="Disable end-line 180° turn laps")
    parser.add_argument("--turn-speed", type=int, default=45, help="In-place turn speed")
    parser.add_argument(
        "--turn-seconds",
        type=float,
        default=3.0,
        help="Spin duration for ~180° (2.0s ≈ 120° on Yahboom; use 3.0 for full 180°)",
    )
    parser.add_argument("--end-confirm-frames", type=int, default=6, help="End-line frames before turn")
    parser.add_argument("--recover-speed-scale", type=float, default=0.35, help="Slow forward speed after turn")
    parser.add_argument("--recover-center-frames", type=int, default=12, help="Good-enough frames before full speed")
    parser.add_argument("--end-lockout-seconds", type=float, default=4.0, help="Ignore end bar after recovery")
    parser.add_argument("--end-zone-fraction", type=float, default=0.20, help="Bottom fraction for end bar")
    parser.add_argument("--iterations", type=int, default=0, help="0 = run until Ctrl+C")
    args = parser.parse_args()

    weights_path = args.weights or _default_weights()
    if not os.path.isfile(weights_path):
        print(f"Weights not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  CAMERA SNN — Remote Deploy")
    print("=" * 60)
    print(f"Robot:   {args.remote_ip}")
    print(f"Weights: {weights_path}")
    print(f"Speed:   base={args.base_speed} steer={args.steer_delta}")
    print(
        f"SNN:     sensor_window={args.sensor_window} hz={args.hz} "
        f"action_window={args.action_window} p_max={args.p_max} "
        f"lif_tau={args.lif_tau} lif_threshold={args.lif_threshold}"
    )
    if args.no_drive:
        print("Mode:    DEBUG (motors disabled)")
    elif not args.no_ping_pong:
        print("Mode:    PING-PONG (turn 180° at end bar, repeat forever)")
    print("Ctrl+C to stop.\n")

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

    if not args.no_rerun:
        init_rerun("camera_snn_deploy")

    sensor = FrameLaneSensor(
        window_size=args.sensor_window, threshold=args.threshold, tape_color=args.tape_color
    )
    inference_config = InferenceConfig(
        p_max=args.p_max,
        action_window=args.action_window,
        lif_tau=args.lif_tau,
        lif_threshold=args.lif_threshold,
    )
    controller = CameraSNNController(weights_path, config=inference_config)
    driver = YahboomRemoteDriver(client, base_speed=args.base_speed, steer_delta=args.steer_delta)
    lap: LapController | None = None
    if not args.no_ping_pong and not args.no_drive:
        lap = LapController(
            driver,
            turn_speed=args.turn_speed,
            turn_seconds=args.turn_seconds,
            end_confirm_frames=args.end_confirm_frames,
            recover_speed_scale=args.recover_speed_scale,
            recover_center_frames=args.recover_center_frames,
            end_lockout_seconds=args.end_lockout_seconds,
            bottom_fraction=args.end_zone_fraction,
            tape_color=args.tape_color,
            threshold=args.threshold,
            end_tape_color=args.end_tape_color,
            end_threshold=args.end_threshold,
        )

    period = 1.0 / args.hz
    step = 0
    status = "CRUISE"

    try:
        while True:
            loop_start = time.perf_counter()
            client.poll_observation()
            frame = client.last_frame
            sensors = sensor.read_from_frame(frame)
            action, sensors = controller.run_inference(sensors)
            lane_state = interpret_lane_state(sensors)
            speed_scale = 1.0

            if lap is not None:
                lap_result = lap.step(frame, action, sensors)
                action = lap_result.action
                status = lap_result.status
                speed_scale = lap_result.speed_scale
                end_line = lap.last_end_line
            elif not args.no_ping_pong:
                end_line = detect_end_line(
                    frame if frame is not None else None,
                    bottom_fraction=args.end_zone_fraction,
                    tape_color=args.end_tape_color,
                    threshold=args.end_threshold,
                )
                speed_scale = 1.0
            else:
                end_line = None
                speed_scale = 1.0

            if not args.no_drive:
                if action is not None:
                    if lap is not None and lap_result.slow_center:
                        driver.execute_recovery(action, speed_scale=speed_scale)
                    else:
                        driver.execute_action(action, speed_scale=speed_scale)
            else:
                client.send_command(RobotCommand(movement="stop"))

            if step % 3 == 0:
                action_name = "TURN" if action is None else ACTION_NAMES.get(action, action)
                print(
                    f"[{step:4d}] {status:22s} {lane_state:32s}  "
                    f"L={sensors[0]:.2f} C={sensors[1]:.2f} R={sensors[2]:.2f}  "
                    f"action={action_name}"
                )

            if not args.no_rerun and frame is not None:
                debug_rgb = annotate_debug_frame(
                    frame,
                    sensors,
                    sensor.last_column_darkness,
                    action=action if action is not None else ACTION_STRAIGHT,
                    threshold=args.threshold,
                    tape_color=args.tape_color,
                    end_line=end_line,
                    end_line_fraction=args.end_zone_fraction,
                    end_tape_color=args.end_tape_color,
                    end_threshold=args.end_threshold,
                )
                extra = {"lap_count": float(lap.lap_count if lap else 0)}
                if end_line is not None:
                    extra["end_line_detected"] = float(end_line.detected)
                    extra["end_row_coverage"] = end_line.row_coverage
                    extra["end_width_span"] = end_line.width_span
                log_camera_snn(
                    frame_idx=step,
                    camera_rgb=frame,
                    debug_rgb=debug_rgb,
                    sensors=sensors,
                    column_darkness=sensor.last_column_darkness,
                    lane_state=f"{status} | {lane_state}",
                    action=action if action is not None else ACTION_STRAIGHT,
                    extra=extra,
                )

            step += 1
            if args.iterations and step >= args.iterations:
                break

            elapsed = time.perf_counter() - loop_start
            precise_sleep(max(period - elapsed, 0.0))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        driver.stop()
        client.disconnect()
        print("Stopped.")


if __name__ == "__main__":
    main()
