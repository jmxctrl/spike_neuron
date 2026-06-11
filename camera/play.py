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
from .sensors import add_lane_sensor_args, build_lane_sensor
from .vision import annotate_debug_frame, interpret_lane_state

logger = logging.getLogger(__name__)

def _steer_label(steer: float) -> str:
    if abs(steer) < 0.08:
        return "FWD"
    return f"{'RIGHT' if steer > 0 else 'LEFT'} {steer:+.2f}"


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
    parser.add_argument("--hz", type=float, default=25.0, help="Control loop rate")
    parser.add_argument(
        "--sensor-window",
        type=int,
        default=2,
        help="Median filter window for lane sensors (lower = faster reaction)",
    )
    parser.add_argument(
        "--action-window",
        type=int,
        default=3,
        help="SNN pathway timesteps averaged for action (lower = faster reaction)",
    )
    parser.add_argument(
        "--p-max",
        type=float,
        default=0.35,
        help="Max spike probability per sensor timestep (higher = stronger input)",
    )
    parser.add_argument(
        "--lif-tau",
        type=float,
        default=10.0,
        help="LIF membrane time constant (lower = faster neuron response)",
    )
    parser.add_argument(
        "--lif-threshold",
        type=float,
        default=0.8,
        help="LIF spike threshold (lower = easier firing)",
    )
    parser.add_argument(
        "--steer-gain",
        type=float,
        default=12000.0,
        help="Scale lateral pool difference into continuous steer",
    )
    parser.add_argument(
        "--steer-deadzone",
        type=float,
        default=0.08,
        help="Steer magnitudes below this become 0 (straight)",
    )
    parser.add_argument("--threshold", type=int, default=60, help="Lane tape threshold (purple side lines)")
    parser.add_argument(
        "--tape-color",
        choices=["auto", "purple", "dark"],
        default="auto",
        help="Side lane lines: auto=purple+black (default)",
    )
    add_lane_sensor_args(parser)
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
    parser.add_argument("--ping-pong", action="store_true", help="Enable end-line 180° turn laps (off by default)")
    parser.add_argument("--turn-speed", type=int, default=40, help="In-place turn speed (lower = less tipping)")
    parser.add_argument(
        "--turn-180-seconds",
        type=float,
        default=3.0,
        help="Spin duration for a full 180° turnaround",
    )
    parser.add_argument(
        "--turn-max-seconds",
        type=float,
        default=5.0,
        help="Maximum spin time (safety cap)",
    )
    parser.add_argument(
        "--turn-settle-seconds",
        type=float,
        default=0.5,
        help="Pause after turn before driving back",
    )
    parser.add_argument("--end-confirm-frames", type=int, default=4, help="Black bar frames before turn")
    parser.add_argument(
        "--tape-lost-confirm-frames",
        type=int,
        default=10,
        help="Frames with no tape visible before turn (corridor end)",
    )
    parser.add_argument(
        "--post-turn-grace-seconds",
        type=float,
        default=8.0,
        help="After 180° turn, ignore end triggers and drive straight this long",
    )
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
    print(f"Sensors: backend={args.sensor_backend}" + (f" model={args.cv_model}" if args.cv_model else ""))
    print(f"Speed:   base={args.base_speed} steer={args.steer_delta}")
    print(
        f"SNN:     sensor_window={args.sensor_window} hz={args.hz} "
        f"action_window={args.action_window} p_max={args.p_max} "
        f"lif_tau={args.lif_tau} lif_threshold={args.lif_threshold} "
        f"steer_gain={args.steer_gain} steer_deadzone={args.steer_deadzone}"
    )
    if args.no_drive:
        print("Mode:    DEBUG (motors disabled)")
    elif args.ping_pong:
        print("Mode:    PING-PONG (turn 180° at end bar, repeat forever)")
    else:
        print("Mode:    CRUISE (lane follow only, no 180° turns)")
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

    if args.sensor_backend == "cv" and not args.cv_model:
        print("--sensor-backend cv requires --cv-model path/to/best.pt", file=sys.stderr)
        sys.exit(1)
    sensor = build_lane_sensor(args)
    inference_config = InferenceConfig(
        p_max=args.p_max,
        action_window=args.action_window,
        lif_tau=args.lif_tau,
        lif_threshold=args.lif_threshold,
        steer_gain=args.steer_gain,
        steer_deadzone=args.steer_deadzone,
    )
    controller = CameraSNNController(weights_path, config=inference_config)
    driver = YahboomRemoteDriver(client, base_speed=args.base_speed, steer_delta=args.steer_delta)

    def _end_line(frame):
        return sensor.detect_end_line(
            frame,
            bottom_fraction=args.end_zone_fraction,
            end_tape_color=args.end_tape_color,
            end_threshold=args.end_threshold,
        )

    lap: LapController | None = None
    if args.ping_pong and not args.no_drive:
        lap = LapController(
            driver,
            turn_speed=args.turn_speed,
            turn_180_seconds=args.turn_180_seconds,
            turn_max_seconds=args.turn_max_seconds,
            turn_settle_seconds=args.turn_settle_seconds,
            end_confirm_frames=args.end_confirm_frames,
            tape_lost_confirm_frames=args.tape_lost_confirm_frames,
            post_turn_grace_seconds=args.post_turn_grace_seconds,
            bottom_fraction=args.end_zone_fraction,
            end_line_fn=_end_line,
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
            steer, sensors = controller.run_inference(sensors)
            lane_state = interpret_lane_state(sensors)
            speed_scale = 1.0

            if lap is not None:
                lap_result = lap.step(frame, steer, sensors)
                steer = lap_result.action
                status = lap_result.status
                speed_scale = lap_result.speed_scale
                end_line = lap.last_end_line
            elif args.ping_pong:
                end_line = _end_line(frame)
                speed_scale = 1.0
            else:
                end_line = None
                speed_scale = 1.0

            if not args.no_drive:
                if steer is not None:
                    if lap is not None and lap_result.slow_center:
                        driver.execute_recovery(steer, speed_scale=speed_scale)
                    elif speed_scale <= 0.05:
                        driver.stop()
                    else:
                        driver.execute_steer(steer, speed_scale=speed_scale)
                elif lap is None or lap.phase.value != "TURNING":
                    driver.stop()
            else:
                client.send_command(RobotCommand(movement="stop"))

            if step % 3 == 0:
                steer_name = "TURN" if steer is None else _steer_label(float(steer))
                print(
                    f"[{step:4d}] {status:22s} {lane_state:32s}  "
                    f"L={sensors[0]:.2f} C={sensors[1]:.2f} R={sensors[2]:.2f}  "
                    f"steer={steer_name}"
                )

            if not args.no_rerun and frame is not None:
                debug_rgb = annotate_debug_frame(
                    frame,
                    sensors,
                    sensor.last_column_darkness,
                    action=steer if steer is not None else ACTION_STRAIGHT,
                    threshold=args.threshold,
                    tape_color=args.tape_color,
                    cv_lane_mask=sensor.lane_mask_for_debug(frame),
                    backend_label=sensor.last_backend_used,
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
                    action=steer if steer is not None else ACTION_STRAIGHT,
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
